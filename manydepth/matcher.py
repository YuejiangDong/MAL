import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.structures.instances import Instances

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, ins_threshold: float = 0.5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.ins_threshold = ins_threshold

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"


    @torch.no_grad()
    def memory_efficient_forward(self, instances_n, instances_m, instances_0):
        """More memory-friendly matching"""
        
        N_n = len(instances_n)
        N_m = len(instances_m)
        N_0 = len(instances_0)

        pred_class_n = instances_n.pred_classes
        pred_class_m = instances_m.pred_classes
        pred_class_0 = instances_0.pred_classes


        class_n = pred_class_n.unsqueeze(1).repeat(1, N_0)
        class_m = pred_class_m.unsqueeze(1).repeat(1, N_0)
        class_0_1 = pred_class_0.repeat(N_n, 1)
        class_0_2 = pred_class_0.repeat(N_m, 1)

        cost_class1 = torch.where(class_n == class_0_1, 0, 1)
        cost_class2 = torch.where(class_m == class_0_2, 0, 1)

        masks_n = instances_n.pred_masks # shape = N_n, H, W
        masks_m = instances_m.pred_masks 
        masks_0 = instances_0.pred_masks # shape = N_0, H, W


        with autocast(enabled=False):
            masks_n = masks_n.flatten(1).float()
            masks_m = masks_m.flatten(1).float()
            masks_0 = masks_0.flatten(1).float()

            # Compute the focal loss between masks
            # cost_mask1 = batch_sigmoid_ce_loss_jit(masks_n, masks_0)
            # cost_mask2 = batch_sigmoid_ce_loss_jit(masks_m, masks_0)

            # Compute the dice loss betwen masks
            cost_dice1 = batch_dice_loss_jit(masks_n, masks_0)
            cost_dice2 = batch_dice_loss_jit(masks_m, masks_0)

        # Final cost matrix
        C1 = (
            # self.cost_mask * cost_mask1
            # + 
            self.cost_class * cost_class1
            + self.cost_dice * cost_dice1
        )
        C2 = (
            # self.cost_mask * cost_mask2
            # + 
            self.cost_class * cost_class2
            + self.cost_dice * cost_dice2
        )
       
        # C = C.reshape(num_queries, -1).cpu()
        C1 = C1.cpu()
        C2 = C2.cpu()


        # print("In matcher, C.shape ", C.shape)
        idx_n, idx_0 = linear_sum_assignment(C1)
        idx_m, idx_1 = linear_sum_assignment(C2)

        id_0 = [0] * N_0
        for i in range(len(idx_0)):
            id_0[idx_0[i]] = i

        id_1 = [0] * N_0
        for i in range(len(idx_1)):
            id_1[idx_1[i]] = i
        
        idx_intersection = set(idx_0) & set(idx_1)

        res0 = []
        res1 = []
        for idx in idx_intersection:
            ix0 = id_0[idx]
            ix1 = id_1[idx]
            res0.append(idx_n[ix0])
            res1.append(idx_m[ix1])
        
        slice_n = torch.as_tensor(res0, dtype=torch.long, device=pred_class_n.device)
        slice_m = torch.as_tensor(res1, dtype=torch.long, device=pred_class_n.device)
        # slice_0 = torch.as_tensor(idx_intersection, dtype=torch.long, device=pred_class_n.device)         

        return slice_n, slice_m

        # indices = []

        # # Iterate through batch size
        # for b in range(bs):
        #     # use -common pixel numbers as cost
        #     # code for ref:
        #     IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        #     for t,trk in enumerate(trackers):
        #         #trk = convert_to_cv2bbox(trk) 
        #         for d,det in enumerate(detections):
        #         #   det = convert_to_cv2bbox(det)
        #             IOU_mat[t,d] = box_iou2(trk,det) 
            
        #     # Produces matches       
        #     # Solve the maximizing the sum of IOU assignment problem using the
        #     # Hungarian algorithm (also known as Munkres algorithm)
            
        #     matched_idx = linear_assignment(-IOU_mat)        

        #     unmatched_trackers, unmatched_detections = [], []
        #     for t,trk in enumerate(trackers):
        #         if(t not in matched_idx[:,0]):
        #             unmatched_trackers.append(t)

        #     for d, det in enumerate(detections):
        #         if(d not in matched_idx[:,1]):
        #             unmatched_detections.append(d)

        #     matches = []
        
        #     # For creating trackers we consider any detection with an 
        #     # overlap less than iou_thrd to signifiy the existence of 
        #     # an untracked object
            
        #     for m in matched_idx:
        #         if(IOU_mat[m[0],m[1]]<iou_thrd):
        #             unmatched_trackers.append(m[0])
        #             unmatched_detections.append(m[1])
        #         else:
        #             matches.append(m.reshape(1,2))
            
        #     if(len(matches)==0):
        #         matches = np.empty((0,2),dtype=int)
        #     else:
        #         matches = np.concatenate(matches,axis=0)
            
        #     return matches, np.array(unmatched_detections), np.array(unmatched_trackers) 
            

    @torch.no_grad()
    def forward(self, outputs1, outputs2, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs1, outputs2, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
