#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys 
sys.path.append('autodl-tmp/VFIDiff-journal/ResShift-journal/FlowformerPlusPlus/core')

import os
import time
import math
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import itertools

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from FlowformerPlusPlus.configs.submissions import get_cfg
from FlowformerPlusPlus.core.utils.misc import process_cfg
from FlowformerPlusPlus.core.FlowFormer import build_flowformer
from FlowformerPlusPlus.core.utils import frame_utils
TRAIN_SIZE = [432, 960]

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self.mode = mode
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == "downzero":
            self._pad = [0, pad_wd, 0, pad_ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        if self.mode == "downzero":
            return [F.pad(x, self._pad) for x in inputs]
        else:
            return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

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

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def main(args):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()

    im1 = Image.open(args.image1).convert("RGB")
    im2 = Image.open(args.image2).convert("RGB")
    im1 = np.array(im1).astype(np.uint8)[..., :3]
    im2 = np.array(im2).astype(np.uint8)[..., :3]

    image1 = torch.from_numpy(im1).permute(2, 0, 1).float() / 255.0
    image2 = torch.from_numpy(im2).permute(2, 0, 1).float() / 255.0
    image_size = image1.shape[1:]
    image1, image2 = image1[None].to(device), image2[None].to(device)
    hws = compute_grid_indices(image_size)
    weights = None

    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)
        # image1 -> image2
        flow_pre, _ = model(image2_pad, image1_pad)
        flow_pre = padder.unpad(flow_pre).detach()      # [1, 2, H, W]
        image1 = padder.unpad(image1_pad)
        image2 = padder.unpad(image2_pad)
        flow = flow_pre.permute(0, 2, 3, 1)           # [B, H, W, 2]
    else:                 
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

    B, C, H, W = image1.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=image1.device),
                              torch.arange(W, device=image1.device),
                              indexing='ij')       # yy: [H,W], xx: [H,W]
    base_grid = torch.stack((xx, yy), dim=2)    # [H, W, 2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0
    
    num_steps = 15
    flow_step = flow / num_steps
    
    current = image1.clone()
    
    results = []
    
    for i in range(1, num_steps + 1):
        warped_grid_i = base_grid + flow_step * i
    
        # [-1, 1]
        warped_grid_i[..., 0] = 2.0 * warped_grid_i[..., 0] / (W - 1) - 1.0
        warped_grid_i[..., 1] = 2.0 * warped_grid_i[..., 1] / (H - 1) - 1.0
    
        current = F.grid_sample(
            current, 
            warped_grid_i, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
    
        current_np = current.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        imageio.imwrite(f"interpolated_{i:02d}.png", (current_np * 255).clip(0, 255).astype(np.uint8))
        results.append(current_np)
        
    final_warp = current
    diff_final = torch.abs(final_warp - image2)
    mean_diff = diff_final.mean().item()
    print("Mean absolute difference between final warp and image2:", mean_diff)
    
    diff_final_np = diff_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
    imageio.imwrite("final_diff_image.png", (diff_final_np * 255).clip(0, 255).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="flow and warping test python document")
    parser.add_argument("--image1", type=str, required=True)
    parser.add_argument("--image2", type=str, required=True)
    args = parser.parse_args()
    main(args)
