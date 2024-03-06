import torch
from .layers import get_smooth_loss, disp_to_depth
from .pareto import pareto_fn
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import numpy as np

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True) # (nxmx4)
    img = value[:, :, :3]
    #     return img.transpose((2, 0, 1))
    return img


def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
    """ Compute loss masks for each of standard reprojection and depth hint
    reprojection"""

    if identity_reprojection_loss is None:
        # we are not using automasking - standard reprojection loss applied to all pixels
        reprojection_loss_mask = torch.ones_like(reprojection_loss)

    else:
        
        # we are using automasking
        all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        # print(all_losses.shape) # B, 2, H, W
        idxs = torch.argmin(all_losses, dim=1, keepdim=True)
        # print(idxs.shape) # B, 1, H, W
        reprojection_loss_mask = (idxs == 0).float()

    return reprojection_loss_mask

def compute_reprojection_loss(ssim, pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def compute_mono_losses(ssim, inputs, outputs, temporal, has_ins):
    """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
    """
    losses = {}
    total_loss = 0
    total_grad_loss = 0
    reproj_losses = []
    
    # original loss
    loss = 0
    reprojection_losses = []

    source_scale = 0
    scale = 0
    
    # if self.opt.dc:
    #     disp = outputs[("disp", scale)][:self.opt.batch_size]
    # else:
    frame_ids = [-1, 1]
    disp = outputs[("disp", scale)]
    color = inputs[("color", 0, scale)]
    target = inputs[("color", 0, source_scale)]
    for frame_id in frame_ids:
        pred = outputs[("color", frame_id, scale)]
        
        reprojection_losses.append(compute_reprojection_loss(ssim, pred, target))
    
    if temporal and has_ins:
        for frame_id in frame_ids:
            pred = outputs[("syn", frame_id, scale)]

            reprojection_losses.append(compute_reprojection_loss(ssim, pred, target))
    
    reprojection_losses = torch.cat(reprojection_losses, 1)

    identity_reprojection_losses = []
    for frame_id in frame_ids:
        pred = inputs[("color", frame_id, source_scale)]
        identity_reprojection_losses.append(
            compute_reprojection_loss(ssim, pred, target))

    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

    # differently to Monodepth2, compute mins as we go
    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)

    reprojection_loss, frame_idxs = torch.min(reprojection_losses, dim=1, keepdim=True)
    
    identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape).to(reprojection_loss.device) * 0.00001

    # find minimum losses from [reprojection, identity]
    reprojection_loss_mask = compute_loss_masks(reprojection_loss,
                                                        identity_reprojection_loss)
    # standard reprojection loss
    reprojection_loss = reprojection_loss * reprojection_loss_mask
    reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
    
    losses['reproj_loss/{}'.format(scale)] = reprojection_loss

    loss += reprojection_loss
    
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    smooth_loss = get_smooth_loss(norm_disp, color)

    loss += 1e-3 * smooth_loss / (2 ** scale)
    total_loss += loss

    losses["loss/{}".format(scale)] = loss
    losses["loss"] = total_loss
    
    return losses, torch.min(reprojection_losses, dim=1, keepdim=True)[0]

def compute_main_losses(ssim, inputs, outputs, mono_reproj, ensemble_reproj, opt, model, w_list, multi_has_ins):
    """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
    """
    losses = {}
    
    # original loss
    loss = 0
    reprojection_losses = []

    source_scale = 0
    
    disp = outputs[("disp", 0)]
    color = inputs[("color", 0, 0)]
    target = inputs[("color", 0, source_scale)]
    frame_ids = [-1, 1]
    for frame_id in frame_ids:
        pred = outputs[("color", frame_id, 0)]
        # if self.opt.ddad:
        #     img = Image.fromarray(np.uint8(255 * pred[0].permute(1, 2, 0).cpu().detach().numpy())) # no opencv required
        #     img.save(f"./debugout/gen_{frame_id}.png")
        reprojection_losses.append(compute_reprojection_loss(ssim, pred, target))
    if multi_has_ins:
        for frame_id in frame_ids:
            pred = outputs[("syn", frame_id, 0)]
            reprojection_losses.append(compute_reprojection_loss(ssim, pred, target))
    
    
    reprojection_losses = torch.cat(reprojection_losses, 1)

    identity_reprojection_losses = []
    for frame_id in frame_ids:
        pred = inputs[("color", frame_id, source_scale)]
        identity_reprojection_losses.append(compute_reprojection_loss(ssim, pred, target))

    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

    # differently to Monodepth2, compute mins as we go
    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)


    # if self.opt.avg_reprojection:
    #     reprojection_loss = reprojection_losses.mean(1, keepdim=True)
    # else:
    # differently to Monodepth2, compute mins as we go
    reprojection_loss, frame_idxs = torch.min(reprojection_losses, dim=1, keepdim=True)
    multi_reproj = reprojection_loss.clone()
    # add random numbers to break ties
    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(reprojection_loss.device) * 0.00001

    # find minimum losses from [reprojection, identity]
    reprojection_loss_mask = compute_loss_masks(reprojection_loss, identity_reprojection_loss)
        
    # if self.step % 50 == 0:
    #     # if self.step < 10 and batch_idx == 0:
    #     #     self.log_time(batch_idx, duration, losses["loss"].cpu().data)
    #     wandb.log({f"Train/loss": losses["loss"]}, step=self.step)
    #     for scale in range(self.opt.sclm+1):
    #         wandb.log({f"Train/loss_{scale}": losses["loss/{}".format(scale)]}, step=self.step)
    
    # find which pixels to apply reprojection loss to, and which pixels to apply
    # consistency loss to
    reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
    reprojection_loss_mask = (reprojection_loss_mask * outputs['consistency_mask'].unsqueeze(1))
    reprojection_loss_mask = (reprojection_loss_mask * (1 - outputs['augmentation_mask'][:opt.batch_size]))
    consistency_mask = (1 - reprojection_loss_mask).float()

    # standard reprojection loss
    reprojection_loss = reprojection_loss * reprojection_loss_mask
    reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
    
    
    
    # consistency loss:
    # encourage multi frame prediction to be like singe frame where masking is happening
    multi_depth = outputs[("depth", 0, 0)]
    # no gradients for mono prediction!
    mono_depth = outputs[("mono_depth", 0, 0)].detach()
    consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
    consistency_loss = consistency_loss.mean()

    # save for logging to tensorboard
    consistency_target = (mono_depth.detach() * consistency_mask +
                            multi_depth.detach() * (1 - consistency_mask))
    consistency_target = 1 / consistency_target
    outputs["consistency_target/0"] = consistency_target
    losses['consistency_loss/0'] = consistency_loss
        
    losses['reproj_loss/0'] = reprojection_loss

    loss += reprojection_loss + consistency_loss 

    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    smooth_loss = get_smooth_loss(norm_disp, color)

    loss += 1e-3 * smooth_loss / (2 ** 0)
    if ensemble_reproj == None:
        all_reproj = torch.cat([mono_reproj, multi_reproj], dim=1)
        _, frame_idxs = torch.min(all_reproj, dim=1, keepdim=True)
        # # distil_mask = compute_loss_masks(all_reproj_loss, identity_reprojection_loss)
        if opt.dual_distil:
            mono_depth_ = outputs[("mono_depth", 0, 0)]
            distil_depth = torch.where(frame_idxs==0, mono_depth_, multi_depth)
        else:
            distil_depth = torch.where(frame_idxs==0, mono_depth, multi_depth)
    else:   
        all_reproj = torch.cat([mono_reproj, ensemble_reproj, multi_reproj], dim=1)
        _, frame_idxs = torch.min(all_reproj, dim=1, keepdim=True)
        # # distil_mask = compute_loss_masks(all_reproj_loss, identity_reprojection_loss)
        if opt.learn_ens:
            _, ensemble_depth = disp_to_depth(outputs["ens_disp"], opt.min_depth, opt.max_depth)
        else:
            ensemble_depth = (mono_depth + multi_depth) / 2.0
        distil_depth = torch.where(frame_idxs==0, mono_depth, ensemble_depth)
        distil_depth = torch.where(frame_idxs==2, multi_depth, distil_depth)
#     # # print(multi_depth.min(), multi_depth.max(), multi_depth.mean())
#     # # for idx in range(bs):
#     # #     d0 = colorize(reprojection_loss_[idx].squeeze().cpu().detach().numpy(), vmin=0, vmax= 1)
#     # #     im_d0 = Image.fromarray(d0)
#     # #     im_d0.save(f'./debug/reprj_{idx}.jpg')
#     # # exit(0)

    distil_loss = torch.abs(distil_depth - multi_depth) * (1 - consistency_mask)
    distil_loss = distil_loss.mean()
    
    if opt.pareto:
        loss_list = [loss, distil_loss]
        
        c_list = [0.1, 0.3]
        new_w_list = pareto_fn(w_list, c_list, model, 2, loss_list)
        loss = new_w_list[0] * loss + new_w_list[1] * distil_loss
        losses["ori_loss"] = new_w_list[0] * loss 
        losses["distil_loss"] = new_w_list[1] * distil_loss
        losses["w_ori"] = new_w_list[0]
        losses["w_distil"] = new_w_list[1]
    elif opt.loss_blc:
        loss_list = [loss.clone(), distil_loss]
        losses['distil_loss'] = distil_loss
        new_w_list = w_list
        
    else:
        losses['distil_loss'] = distil_loss # 0
        loss += distil_loss
        new_w_list = None
        loss_list = None

    losses["loss/0"] = loss

    losses["loss"] = loss
    
    return losses, new_w_list, loss_list

class LossBalancing():
    def __init__(self, num_loss, num_train_data, bs):
        self.num_loss = num_loss
        self.weight_initialization_done = False
        self.last_rebalancing_iter = 0
        self.previous_total_loss = 0
        self.previous_loss = 0
        
        self.w_list = np.array([1./num_loss, 1./num_loss])
        self.loss_initialize_scale = np.array([1./num_loss, 1./num_loss])
        
        self.train_scores = np.zeros((num_train_data, num_loss))
        self.train_metrics = np.zeros((num_train_data, 7))
        
        self.num_data = num_train_data
        self.bs = bs
        
        self.weight_initialization = True
        self.weight_initialization_done = False
    
    def compute_loss(self, loss_list, index_iter):
        l_custom = torch.zeros(self.bs).cuda(torch.device("cuda:0"))
        for index_batch in range(self.bs):
            index_record = self.bs * index_iter + index_batch

            if index_batch == 0:
                loss = 0

            if index_record < self.num_data:
                for index_loss in range(self.num_loss):
                    loss = loss + self.w_list[index_loss] * loss_list[index_loss]

                for index_loss in range(self.num_loss):
                    self.train_scores[index_record, index_loss] = loss_list[index_loss]

        return loss
    
    def update_weight(self, i, current_lambda_for_adjust):
        temp_train_scores_mean = self.train_scores[self.last_rebalancing_iter*self.bs:(i+1)*self.bs, :].mean(axis=0)
        total_loss = np.sum(temp_train_scores_mean * self.w_list)
        if self.weight_initialization == True and self.weight_initialization_done == False:
            for index_loss in range(self.num_loss):
                self.w_list[index_loss] = (total_loss * self.loss_initialize_scale[index_loss]) / temp_train_scores_mean[index_loss]
                
            # save previous record
            self.weight_initialization_done = True
            self.previous_total_loss = np.sum(temp_train_scores_mean * self.w_list)
            self.previous_loss = temp_train_scores_mean

        elif self.weight_initialization_done == True or self.weight_initialization == False:
            temp_train_scores_mean = self.train_scores[self.last_rebalancing_iter*self.bs:(i+1)*self.bs, :].mean(axis=0)
            total_loss = np.sum(temp_train_scores_mean * self.w_list)
            previous_loss_weights = np.array(self.w_list)
            if self.previous_total_loss > 0:
                for index_loss in range(self.num_loss):
                    adjust_term = 1 + current_lambda_for_adjust * ((total_loss/self.previous_total_loss) * (self.previous_loss[index_loss]/temp_train_scores_mean[index_loss]) - 1)
                    adjust_term = min(max(adjust_term, 1.0/2.0), 2.0/1.0)
                    self.w_list[index_loss] = previous_loss_weights[index_loss] * adjust_term
                    
            # save previous record
            self.previous_total_loss = np.sum(temp_train_scores_mean * self.w_list)
            self.previous_loss = temp_train_scores_mean
        return self.w_list[0], self.w_list[1]

    