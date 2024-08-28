import torch
from PIL import Image
import numpy as np

@torch.jit.script
def fill_dynamic_obj(mask, delta_x, delta_y, source, img):
    N, H, W = mask.shape
    
    start_hl = torch.max(torch.zeros_like(delta_x), delta_x)
    end_hl = torch.min(torch.ones_like(delta_x) * H, H+delta_x)
    start_hr = torch.max(torch.zeros_like(delta_x), -delta_x)
    end_hr = torch.min(torch.ones_like(delta_x) * H, H-delta_x)

    start_wl = torch.max(torch.zeros_like(delta_y), delta_y)
    end_wl = torch.min(torch.ones_like(delta_y) * W, W+delta_y)
    start_wr = torch.max(torch.zeros_like(delta_y), -delta_y)
    end_wr = torch.min(W-delta_y, torch.ones_like(delta_y) * W)
    
    chn = img.shape[0]

    source_mv = torch.zeros((N, chn, H, W), device=source.device)
    mask_mv = torch.zeros(mask.shape, dtype=torch.bool, device=mask.device)
    for i in range(len(mask)):
        source_mv[i, :, start_hl[i]:end_hl[i], start_wl[i]:end_wl[i]] = source[:, start_hr[i]:end_hr[i], start_wr[i]:end_wr[i]]
        mask_mv[i, start_hl[i]:end_hl[i], start_wl[i]:end_wl[i]] = mask[i, start_hr[i]:end_hr[i], start_wr[i]:end_wr[i]]

    img_mv = mask_mv.unsqueeze(1).repeat(1,chn,1,1) * source_mv
    # print(img_mv.shape) # 15, 3, H, W
    img_sum = img_mv.sum(dim=0) # 3, H, W

    mask_or = torch.zeros((H, W), dtype=torch.bool, device=mask.device)
    for mask_item in mask_mv:
        mask_or = mask_or | mask_item
    img = torch.where(mask_or, img_sum, img)
    
    return img

@torch.jit.script
def generate_dynamic_instance(grid_h, grid_w, mask_last, mask_next, img_last, img_next, replace:bool):
    mask_or = mask_last | mask_next
    mask_or_ = torch.zeros_like(mask_or[0])
    for mask_item in mask_or:
        mask_or_ = mask_or_ | mask_item
    num, H, W = mask_last.shape

    x = torch.arange(H, device=mask_last.device)
    y = torch.arange(W, device=mask_last.device)
    
    # compute boundaries
    grid_h = grid_h.repeat(num, 1, 1)
    grid_w = grid_w.repeat(num, 1, 1)

    ### new ver
    inf = (H + 1) * (W + 1)

    h_sum_last = (mask_last * grid_h).sum(dim=2) # B, H
    h_nonzero = torch.where(h_sum_last==0, 0, x)
    low_last = h_nonzero.argmax(dim=1)
    h_nonzero = torch.where(h_nonzero == 0, inf, h_nonzero)
    top_last = h_nonzero.argmin(dim=1)

    w_sum_last = (mask_last * grid_w).sum(dim=1) # B, H
    w_nonzero = torch.where(w_sum_last==0, 0, y)
    right_last = w_nonzero.argmax(dim=1)
    w_nonzero = torch.where(w_nonzero == 0, inf, w_nonzero)
    left_last = w_nonzero.argmin(dim=1)
    
    h_sum_next = (mask_next * grid_h).sum(dim=2) # B, H
    h_nonzero = torch.where(h_sum_next==0, 0, x)
    low_next = h_nonzero.argmax(dim=1)
    h_nonzero = torch.where(h_nonzero == 0, inf, h_nonzero)
    top_next = h_nonzero.argmin(dim=1)

    w_sum_next = (mask_next * grid_w).sum(dim=1) # B, H
    w_nonzero = torch.where(w_sum_next==0, 0, y)
    right_next = w_nonzero.argmax(dim=1)
    w_nonzero = torch.where(w_nonzero == 0, inf, w_nonzero)
    left_next = w_nonzero.argmin(dim=1)

    # compute delta, with boundary judgement
    batch_slice = torch.arange(num, device=mask_last.device)
    delta_x = torch.stack([low_next-low_last, top_next-top_last], dim=1)
    delta_x_selected = delta_x[batch_slice, delta_x.abs().argmax(dim=1)]# .squeeze(dim=-1)

    delta_y = torch.stack([right_next-right_last, left_next-left_last], dim=1)
    delta_y_selected = delta_y[batch_slice, delta_y.abs().argmax(dim=1)]# .squeeze(dim=-1)
    disp_x = torch.round(delta_x_selected / 2).long()
    disp_y = torch.round(delta_y_selected / 2).long()
    
    if replace:
        delta_x_last = torch.where(disp_x.abs() < 3, 0, disp_x)
        delta_y_last = torch.where(disp_y.abs() < 3, 0, disp_y)
        delta_x_next = torch.where(disp_x.abs() < 3, 0, -disp_x)
        delta_y_next = torch.where(disp_y.abs() < 3, 0, -disp_y)
    else:
        delta_x_last = disp_x
        delta_y_last = disp_y
        delta_x_next = -disp_x
        delta_y_next = -disp_y
    
    mask = mask_last & (~ mask_next)
    mask_bg = torch.zeros_like(mask[0])
    for ms in mask:
        mask_bg = mask_bg | ms
    img_bg = torch.where(mask_bg, img_next, img_last)
    mask2 = mask_next & (~ mask_last)
    mask_bg2 = torch.zeros_like(mask2[0])
    for ms in mask2:
        mask_bg2 = mask_bg2 | ms
    img_bg2 = torch.where(mask_bg2, img_last, img_next)
      
    image_syn_last = fill_dynamic_obj(mask_last, delta_x_last, delta_y_last, img_last, img_bg)
    ori_last = torch.where(mask_or_, image_syn_last, img_last)

    image_syn_next = fill_dynamic_obj(mask_next, delta_x_next, delta_y_next, img_next, img_bg2)
    ori_next = torch.where(mask_or_, image_syn_next, img_next)
    
    
    return ori_last, ori_next

def image_synthesis(inputs, outputs, scale, thres, ins_model, matcher):
    # predict instances for outputs[("color", frame_id, scale)]
    bs = inputs[("color", 0, 0)].shape[0]
    
    instances = generate_instances(inputs[("color", 0, 0)], ins_model)
    
    syn_last = outputs[("color", -1, scale)].clone()
    syn_next = outputs[("color", 1, scale)].clone()
    
    has_ins = False
    
    H= syn_last.shape[-2]
    W= syn_last.shape[-1]

    x = torch.arange(H, device=syn_last.device)
    y = torch.arange(W, device=syn_last.device)
    grid_h, grid_w = torch.meshgrid(x, y, indexing='ij')

    for b in range(bs):
        instances_cur = instances[b]["instances"][instances[b]["instances"].scores > thres]
        
        if len(instances_cur) == 0:
            continue
        
        img_last = outputs[("color", -1, scale)][b].clone()
        img_next = outputs[("color", 1, scale)][b].clone()

        instances_all = generate_instances(torch.cat([img_last.unsqueeze(0), img_next.unsqueeze(0)], dim=0), ins_model)
        ins_last = instances_all[0]["instances"]
        ins_next = instances_all[1]["instances"]
        slice_last, slice_next = matcher(ins_last, ins_next, instances_cur)
        
        if len(slice_last) + len(slice_next) == 0:
            continue 
        
        has_ins = True

        # dynamic object synthesis
        mask_last = ins_last.pred_masks[slice_last].bool()
        mask_next = ins_next.pred_masks[slice_next].bool()
        
        tmp_last, tmp_next = generate_dynamic_instance(grid_h, grid_w, mask_last, mask_next, img_last, img_next, replace=False)
        syn_last[b] = (tmp_last.unsqueeze(0))
        syn_next[b] = (tmp_next.unsqueeze(0))

    if has_ins:
        outputs[("syn", -1, scale)] = syn_last
        outputs[("syn", 1, scale)] = syn_next
        
    return has_ins
            
def generate_instances(images, ins_model):
    (height, width) = images.shape[-2:]

    # convert rgb to bgr
    permute = [2, 1, 0]
    images = images[:, permute, :, :]

    images = images * 255

    input_ls_dict = [
        {"image": img, "height": height, "width": width}
        for img in images
    ]
    with torch.no_grad():
        pred_instances = ins_model(input_ls_dict)
    
    return pred_instances
