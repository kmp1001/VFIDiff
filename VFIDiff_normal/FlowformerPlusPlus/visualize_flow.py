


import sys

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from FlowformerPlusPlus.configs.submissions import get_cfg
from FlowformerPlusPlus.core.utils.misc import process_cfg
import datasets
from FlowformerPlusPlus.core.utils import flow_viz
from FlowformerPlusPlus.core.utils import frame_utils
import cv2
import math
import os.path as osp
import numpy as np
import cv2
import torch
import glob
import os
import time
import gzip
import pickle
from FlowformerPlusPlus.core.FlowFormer import build_flowformer

from FlowformerPlusPlus.core.utils.utils import InputPadder, forward_interpolate
import itertools
def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model
FLOW_MODEL = build_model()  # torch.nn.DataParallel(...) / DDP(...)
for p in FLOW_MODEL.parameters():
    p.requires_grad = False
FLOW_MODEL = FLOW_MODEL.cuda()
TRAIN_SIZE = [432, 960]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

def compute_flow(model, image1, image2, weights=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
    print(f"preparing image...")
    print(f"root dir = {root_dir}, fn = {fn1}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn1))[0]

    viz_dir = osp.join(viz_root_dir, dirname)
    if not osp.exists(viz_dir):
        os.makedirs(viz_dir)

    viz_fn = osp.join(viz_dir, filename + '.png')

    return image1, image2, viz_fn


def visualize_flow(root_dir, viz_root_dir, model, img_pairs, keep_size):
    weights = None
    for img_pair in img_pairs:
        fn1, fn2 = img_pair
        print(f"processing {fn1}, {fn2}...")

        image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)
        flow = compute_flow(model, image1, image2, weights)
        flow_img = flow_viz.flow_to_image(flow)
        cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])

def process_sintel(sintel_dir):
    img_pairs = []
    for scene in os.listdir(sintel_dir):
        dirname = osp.join(sintel_dir, scene)
        image_list = sorted(glob(osp.join(dirname, '*.png')))
        for i in range(len(image_list)-1):
            img_pairs.append((image_list[i], image_list[i+1]))

    return img_pairs

def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        img1 = osp.join(dirname, f'{idx:06}.png')
        img2 = osp.join(dirname, f'{idx+1:06}.png')
        # img1 = f'{idx:06}.png'
        # img2 = f'{idx+1:06}.png'
        img_pairs.append((img1, img2))

    return img_pairs
def compute_optical_flow1(model,image1,image2, weights=None):
    image_size = image1.shape[1:]
    image1, image2 = image1[None].cuda(), image2[None].cuda()
    hws=hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            flow_pre, _ = model(image1, image2)
            flow_pre = flow_pre.detach()
        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow



import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

def compute_optical_flow(img0: torch.Tensor, img1: torch.Tensor, device=None) -> torch.Tensor:

    global FLOW_MODEL
    if device is None:
        device = img0.device
    img0 = img0.to(device)
    img1 = img1.to(device)

    B, C, H, W = img0.shape
    flows = []
    for b in range(B):
        im0_np = (img0[b].cpu().numpy() * 255).astype('uint8').transpose(1,2,0)
        im1_np = (img1[b].cpu().numpy() * 255).astype('uint8').transpose(1,2,0)
        flow_np = compute_optical_flow1(FLOW_MODEL, torch.from_numpy(im0_np).permute(2,0,1).float().to(device),
                                             torch.from_numpy(im1_np).permute(2,0,1).float().to(device))
        # flow_np: numpy [H,W,2]
        flow = torch.from_numpy(flow_np).permute(2,0,1)  # -> [2,H,W]
        flows.append(flow)
    return torch.stack(flows, dim=0)



def warp_single_step(flow1, flow2, img_chw):

    # [C,H,W] → [H,W,C]
    img = img_chw.permute(1, 2, 0)
    H, W = flow1.shape[:2]
    device = flow1.device

    output = torch.zeros_like(img)

    flow_t = torch.zeros_like(flow1)
    
    gy = torch.arange(0, H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
    gx = torch.arange(0, W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
    grid = torch.stack((gy, gx), dim=2)  # [H,W,2]
    
    dx = grid[..., 0] + flow2[..., 1]
    dy = grid[..., 1] + flow2[..., 0]
    
    sx = torch.floor(dx)
    sy = torch.floor(dy)
    valid_flow = (sx >= 0) & (sx < H-1) & (sy >= 0) & (sy < W-1)
    
    sx_clamped = sx.clamp(0, H-2)
    sy_clamped = sy.clamp(0, W-2)
    
    sx_mat = torch.stack([sx_clamped, sx_clamped+1, sx_clamped, sx_clamped+1], dim=2)
    sy_mat = torch.stack([sy_clamped, sy_clamped, sy_clamped+1, sy_clamped+1], dim=2)

    wx = 1 - torch.abs(sx_mat - dx.unsqueeze(2))
    wy = 1 - torch.abs(sy_mat - dy.unsqueeze(2))
    w = wx * wy  # [H,W,4]
    
    flow_t.zero_()
    for i in range(4):
        ix = sx_mat[..., i].long()
        iy = sy_mat[..., i].long()
        mask = valid_flow.unsqueeze(2).expand(-1, -1, 2)
        flow_t += (w[..., i:i+1] * flow1[ix, iy]) * mask

    diff = flow_t[..., [1,0]] + torch.stack((dx, dy), dim=2) - grid
    valid_consist = valid_flow & (diff.norm(dim=2) < 100)
    
    flow_t = (flow2 - flow_t) / 2.0
    
    dx_final = grid[..., 0] + flow_t[..., 1]
    dy_final = grid[..., 1] + flow_t[..., 0]
    
    valid_final = valid_consist & (dx_final >= 0) & (dx_final < H-1) & (dy_final >= 0) & (dy_final < W-1)
    
    sx_final = torch.floor(dx_final).clamp(0, H-2)
    sy_final = torch.floor(dy_final).clamp(0, W-2)
    x0 = sx_final.long()
    x1 = (sx_final + 1).long().clamp(0, H-1)
    y0 = sy_final.long()
    y1 = (sy_final + 1).long().clamp(0, W-1)
    
    wx = dx_final - sx_final
    wy = dy_final - sy_final
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy
    mask = valid_final.unsqueeze(2).expand(-1, -1, img.shape[2])
    output = (
        w00.unsqueeze(2) * img[x0, y0] +
        w01.unsqueeze(2) * img[x0, y1] +
        w10.unsqueeze(2) * img[x1, y0] +
        w11.unsqueeze(2) * img[x1, y1]
    ) * mask
    
    # [H,W,C] → [C,H,W]
    return output.permute(2, 0, 1)

def warp(
    img0: torch.Tensor,      # [B,C,H,W]
    flow01: torch.Tensor,    # [B,2,H,W] 
    flow10: torch.Tensor,    # [B,2,H,W]
    t: torch.Tensor,         # [B] 
    total_steps: int | torch.Tensor = 13
) -> torch.Tensor:

    B, C, H, W = img0.shape
    device = img0.device
 
    if not torch.is_tensor(total_steps):
        ts = torch.full((B,), int(total_steps), device=device, dtype=torch.int64)
    else:
        ts = total_steps.to(device)
        if ts.dim() == 0:
            ts = ts.expand(B)
        ts = ts.to(torch.int64)
    
    if t.dim() == 0:
        t = t.expand(B)
    
    outputs = []
    
    for b in range(B):
        img_single = img0[b]  # [C,H,W]
        flow_01 = flow01[b].permute(1, 2, 0)  # [H,W,2]
        flow_10 = flow10[b].permute(1, 2, 0)  # [H,W,2]
        t_val = float(t[b].clamp(0.0, 1.0).item()) #  [0,1]
        ts_b  = int(ts[b].item())                  
         
        target_steps = int(round(t_val * ts_b))
        target_steps = max(0, min(target_steps, ts_b))
        step_flow1   = flow_01 / ts_b
        step_flow2   = flow_10 / ts_b
        
        if target_steps == 0:
            outputs.append(img_single)
            continue
        
        result = img_single  
        for i in range(target_steps):
            result = warp_single_step(step_flow1, step_flow2, result)
        
        outputs.append(result)
    
    return torch.stack(outputs, dim=0)
def main():
    model = build_model()
    # read and preprocess
    im1 = frame_utils.read_gen('im1.png'); im7 = frame_utils.read_gen('im7.png')
    im1 = np.array(im1)[...,:3].astype(np.uint8); im7 = np.array(im7)[...,:3].astype(np.uint8)
    img1 = torch.from_numpy(im1).permute(2,0,1).float().cuda()
    img7 = torch.from_numpy(im7).permute(2,0,1).float().cuda()

    # compute flows
    flow1_np = compute_optical_flow1(model, img1, img7)
    flow2_np = compute_optical_flow1(model, img7, img1)
    flow1 = torch.from_numpy(flow1_np).cuda()
    flow2 = torch.from_numpy(flow2_np).cuda()

    # full warp + coords
    full, coord_full = warp_with_coords(flow1, flow2, img1)

    # prepare base coords
    C,H,W = img1.shape
    ys = torch.arange(H,device=img1.device)
    xs = torch.arange(W,device=img1.device)
    gy = ys.repeat(W,1).permute(1,0); gx = xs.repeat(H,1)
    base_coords = torch.stack((gx,gy),dim=2).float()  # [H,W,2]

    base_grid = coords_to_grid(base_coords)
    full_grid = coords_to_grid(coord_full)

    # save all frames
    out_dir = 'interp_steps'; os.makedirs(out_dir, exist_ok=True)
    bimg = img1.unsqueeze(0)  # [1,C,H,W]
    steps = 13
    for k in range(steps+1):
        t = k/steps
        grid_t = base_grid + t*(full_grid - base_grid)
        warped = F.grid_sample(bimg, grid_t, mode='bilinear',
                               padding_mode='border', align_corners=True)[0]
        arr = warped.permute(1,2,0).cpu().numpy()
        arr = np.clip(arr,0,255).astype(np.uint8)[...,::-1]
        cv2.imwrite(f"{out_dir}/interp_{k:02d}.png", arr)

if __name__ == '__main__':
    main()
