import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .resnet_encoder import ResnetEncoder, ResnetEncoderDYJ, ResnetEncoderMatching
from .layers import transformation_from_parameters, disp_to_depth
from .pose_cnn import PoseCNN

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram

class RepDepth(nn.Module):
    def __init__(self, opt):
        super(RepDepth, self).__init__()
        
        self.opt = opt
        
        self.encoder = ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        
        num_ch_enc = self.encoder.num_ch_enc
        
        self.depth = DepthDecoder(
            num_ch_enc, self.opt.scales)#, self.opt.debug)
        
        self.mono_encoder = ResnetEncoder(18, self.opt.weights_init == "pretrained")  
            
        # num_ch_enc = np.array([128, 256, 512, 1024]) # Base Model
        self.mono_depth = DepthDecoder(
            self.mono_encoder.num_ch_enc, self.opt.scales) #, self.opt.debug, dc=self.opt.dc, test_id=self.opt.dec_id)
        
        # posenet
        self.need_pose_dec = False
        
        if self.opt.pose_cnn:
            self.pose_encoder = PoseCNN(num_input_frames=3)
        else:
            self.pose_encoder = ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                num_input_images=2)
            self.need_pose_dec = True
          
        if self.need_pose_dec:
            self.pose = PoseDecoder(self.pose_encoder.num_ch_enc,
                                    num_input_features=1,
                                    num_frames_to_predict_for=2)
        
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
        
        self.freeze_tp = False
        self.freeze_pose = False
        self.dc = self.opt.dc
        
    
    def cross_load_kitti(self, pretrained_folder='./ckpt/KITTI_MR'):
        whole_model = torch.load(pretrained_folder+'/model.pth', map_location='cpu')
        self.load_state_dict(whole_model.state_dict(), strict=False)
    
    def load_manydepth(self, depthbin_tracker, train_cs, pretrained_folder='./ckpt/KITTI_MR'): 
        if train_cs:
            pretrained_folder = './ckpt/CityScapes_MR'
        print("debug ", pretrained_folder)
        encoder_dict = torch.load(pretrained_folder+'/encoder.pth', map_location='cpu')
        self.encoder.load_state_dict(encoder_dict, strict=False)
        
        mono_encoder_dict = torch.load(pretrained_folder+'/mono_encoder.pth', map_location='cpu')
        self.mono_encoder.load_state_dict(mono_encoder_dict, strict=False)
         
        # self.mono_encoder.trans_drop_path = whole_mono_encoder.trans_drop_path
        depth_dict = torch.load(pretrained_folder+'/depth.pth', map_location='cpu')
        self.depth.load_state_dict(depth_dict, strict=False)
        self.mono_depth.load_state_dict(torch.load(pretrained_folder+'/mono_depth.pth', map_location='cpu'), strict=False)
        self.pose_encoder.load_state_dict(torch.load(pretrained_folder+'/pose_encoder.pth', map_location='cpu'))
        self.pose.load_state_dict(torch.load(pretrained_folder+'/pose.pth', map_location='cpu'))
        
        min_depth_bin = encoder_dict.get('min_depth_bin')
        max_depth_bin = encoder_dict.get('max_depth_bin')
        if depthbin_tracker is not None:
            print(min_depth_bin, max_depth_bin)
            depthbin_tracker.load(torch.Tensor([min_depth_bin]), torch.Tensor([max_depth_bin]))
        else:
            return torch.Tensor([min_depth_bin]), torch.Tensor([max_depth_bin])
    
    
    def freeze_tp_net(self):
        for name, param in self.mono_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.mono_depth.named_parameters():
            param.requires_grad = False
        for name, param in self.pose_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.pose.named_parameters():
            param.requires_grad = False
        self.freeze_tp = True
        
        num_param = sum(p.numel() for p in self.mono_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_encoder.parameters())
        print("for mono_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.mono_depth.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_depth.parameters())
        print("for mono_depth ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose_encoder.parameters())
        print("for pose_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose.parameters())
        print("for pose ", num_param, num_total_param)
    
    def freeze_pose_net(self):
        for name, param in self.pose_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.pose.named_parameters():
            param.requires_grad = False
        self.freeze_pose = True
        
        num_param = sum(p.numel() for p in self.pose_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose_encoder.parameters())
        print("for pose_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose.parameters())
        print("for pose ", num_param, num_total_param)
        
            
    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            # f_i = -1, 1
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]

                axisangle, translation = self.pose(pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                
                if self.opt.temporal:
                    if f_i < 0:
                        # Tt-1->t
                        outputs[("cam_T_cam", -1, 0)] = (transformation_from_parameters)(
                            axisangle[:, 0], translation[:, 0], invert=False)
                    else:
                       # Tt+1->t
                        outputs[("cam_T_cam", 1, 0)] = (transformation_from_parameters)(
                            axisangle[:, 0], translation[:, 0], invert=True) 

        # now we need poses for matching - compute without gradients
        to_iter = self.matching_ids # if not self.dc else [0, -1, 1]
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in to_iter}
        with torch.no_grad():
            to_iter = self.matching_ids[1:] #if not self.dc else [-1, 1]
            
            # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
            for fi in to_iter:
                # fi = -1
                if fi < 0:
                    pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                    pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.pose(pose_inputs)
                    pose = (transformation_from_parameters)(
                        axisangle[:, 0], translation[:, 0], invert=True)
                    # print("for -1 ", axisangle[:, 0], translation[:, 0])
                    # if self.dc:
                    #     # Tt->t-1
                    #     pose_inv = (transformation_from_parameters)(
                    #         axisangle[:, 0], translation[:, 0], invert=False)
                    

                    # now find 0->fi pose
                    if fi != -1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                else:
                    pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                    pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.pose(pose_inputs)
                    pose = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                    
                    # if self.dc:
                    #     # Tt->t+1
                    #     pose_inv = (transformation_from_parameters)(
                    #         axisangle[:, 0], translation[:, 0], invert=True)
                    

                    # now find 0->fi pose
                    if fi != 1:
                        pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                # set missing images to 0 pose
                for batch_idx, feat in enumerate(pose_feats[fi]):
                    if feat.sum() == 0:
                        pose[batch_idx] *= 0
                        # if self.dc:
                        #     pose_inc[batch_idx] *= 0

                inputs[('relative_pose', fi)] = pose
                # if self.dc:
                #     inputs[('relative_pose_inv', fi)] = pose
                
        return outputs    
     
    def print_num_param(self):
        num_param = sum(p.numel() for p in self.mono_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_encoder.parameters())
        print("for mono_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.encoder.parameters())
        print("for encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.pose_encoder.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.pose_encoder.parameters())
        print("for pose_encoder ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.depth.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.depth.parameters())
        print("for depth ", num_param, num_total_param)
        num_param = sum(p.numel() for p in self.mono_depth.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.mono_depth.parameters())
        print("for mono_depth ", num_param, num_total_param)
    
        
    def forward(self, inputs, min_depth_bin, max_depth_bin):
        mono_outputs = {}
        outputs = {}
        
        # predict poses
        if self.freeze_tp == False and self.freeze_pose == False:
            # if self.opt.pose_cnn:
            #     pose_pred = self.predict_poses_cnn(inputs)
            # else:
            pose_pred = self.predict_poses_vit(inputs) if not self.need_pose_dec else self.predict_poses(inputs)
        else:
            with torch.no_grad():
                # if self.opt.pose_cnn:
                #     pose_pred = self.predict_poses_cnn(inputs)
                # else:
                pose_pred = self.predict_poses_vit(inputs) if not self.need_pose_dec else self.predict_poses(inputs)
                
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)
        
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
        
        self.device = inputs[('color_aug', 0, 0)].device
        
        batch_size = len(lookup_frames)
        
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        # matching augmentation
        for batch_idx in range(batch_size):
            rand_num = random.random()
            # static camera augmentation -> overwrite lookup frames with current frame
            if rand_num < 0.25:
                replace_frames = \
                    [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                replace_frames = torch.stack(replace_frames, 0)
                lookup_frames[batch_idx] = replace_frames
                
                augmentation_mask[batch_idx] += 1
            # missing cost volume augmentation -> set all poses to 0, the cost volume will
            # skip these frames
            elif rand_num < 0.5:
                relative_poses[batch_idx] *= 0
                augmentation_mask[batch_idx] += 1
    
        outputs['augmentation_mask'] = augmentation_mask
        
        
        
        # predict by teacher network
        if self.freeze_tp == False:
            img_aug = inputs["color_aug", 0, 0]
            feats = self.mono_encoder(img_aug)
            mono_outputs.update(self.mono_depth(feats))
        else:
            with torch.no_grad():
                img_aug = inputs["color_aug", 0, 0]
                feats = self.mono_encoder(img_aug)
                mono_outputs.update(self.mono_depth(feats))
        
        # update multi frame outputs dictionary with single frame outputs
        # aim to compute consistency loss
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]
        
        features, lowest_cost, confidence_mask = self.encoder(
                                            inputs["color_aug", 0, 0],
                                            lookup_frames,
                                            relative_poses,
                                            inputs[('K', 2)],
                                            inputs[('inv_K', 2)],
                                            min_depth_bin=min_depth_bin,
                                            max_depth_bin=max_depth_bin)
                                            
        
        outputs.update(self.depth(features))
        
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                            [self.opt.height, self.opt.width],
                                            mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]
        
        return mono_outputs, outputs
