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

# 分布式相关
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# FlowFormerPlusPlus 模块
from FlowformerPlusPlus.configs.submissions import get_cfg
from FlowformerPlusPlus.core.utils.misc import process_cfg
from FlowformerPlusPlus.core.FlowFormer import build_flowformer
from FlowformerPlusPlus.core.utils import frame_utils
TRAIN_SIZE = [432, 960]

# --------------------- 基础函数部分 ---------------------

# class InputPadder:
#     """ Pads images such that dimensions are divisible by 8 and at least TRAIN_SIZE """
#     def __init__(self, dims, mode='sintel'):
#         self.mode = mode
#         self.B, self.C, self.ht, self.wd = dims
#         min_ht, min_wd = TRAIN_SIZE
#         pad_ht_needed = max(min_ht - self.ht, 0)
#         pad_wd_needed = max(min_wd - self.wd, 0)
#         pad_ht_total = pad_ht_needed + (((self.ht + pad_ht_needed) + 7) // 8 * 8 - (self.ht + pad_ht_needed))
#         pad_wd_total = pad_wd_needed + (((self.wd + pad_wd_needed) + 7) // 8 * 8 - (self.wd + pad_wd_needed))
#         pad_top = pad_ht_total // 2
#         pad_bottom = pad_ht_total - pad_top
#         pad_left = pad_wd_total // 2
#         pad_right = pad_wd_total - pad_left

#         if mode.startswith('kitti'):
#             pad_bottom = max(pad_ht_total, TRAIN_SIZE[0] - self.ht)
#             pad_right = max(pad_wd_total, TRAIN_SIZE[1] - self.wd)
#             pad_top = 0
#             pad_left = 0

#         self._pad = [pad_left, pad_right, pad_top, pad_bottom]

#     def pad(self, *inputs):
#         return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

#     def unpad(self, x):
#         return x[..., self._pad[2]:x.shape[-2]-self._pad[3], self._pad[0]:x.shape[-1]-self._pad[1]]
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

# def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
#     H, W = image_shape
#     patch_H, patch_W = patch_size
#     if min_overlap >= patch_H or min_overlap >= patch_W:
#         raise ValueError("min_overlap 必须小于 patch_size 的每个维度。")
#     hs = list(range(0, H, patch_H - min_overlap))
#     ws = list(range(0, W, patch_W - min_overlap))
#     if hs:
#         hs[-1] = max(H - patch_H, 0)
#     else:
#         hs = [0]
#     if ws:
#         ws[-1] = max(W - patch_W, 0)
#     else:
#         ws = [0]
#     hs = [max(h, 0) for h in hs]
#     ws = [max(w, 0) for w in ws]
#     return [(h, w) for h in hs for w in ws]
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


# def compute_optical_flow(model, image1, image2, sigma=0.05):
#     """
#     计算光流
#     参数：
#       - model: 预训练的 FlowFormer 模型（DDP 封装后的）
#       - image1: 第一个图像张量，[B, 3, H, W]
#       - image2: 第二个图像张量，[B, 3, H, W]
#       - sigma: 权重计算的标准差
#     返回：
#       - flow: 光流张量，[B, 2, H, W]
#     """
#     B, C, H, W = image1.shape
#     device = image1.device
#     flows = torch.zeros((B, 2, H, W), device=device)
#     for b in range(B):
#         img1 = image1[b:b+1]
#         img2 = image2[b:b+1]
#         padder = InputPadder(img1.shape, mode='sintel')
#         img1_padded, img2_padded = padder.pad(img1, img2)
#         padded_H, padded_W = img1_padded.shape[-2:]
#         IMAGE_SIZE = [padded_H, padded_W]
#         hws = compute_grid_indices(IMAGE_SIZE)
#         weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma, device=device)
#         flows_b = torch.zeros((2, padded_H, padded_W), device=device)
#         flow_count_b = torch.zeros((2, padded_H, padded_W), device=device)
#         for idx, (h, w) in enumerate(hws):
#             patch_size = TRAIN_SIZE
#             img1_tile = img1_padded[:, :, h:h+patch_size[0], w:w+patch_size[1]]
#             img2_tile = img2_padded[:, :, h:h+patch_size[0], w:w+patch_size[1]]
#             current_patch_H, current_patch_W = img1_tile.shape[-2:]
#             if current_patch_H != patch_size[0] or current_patch_W != patch_size[1]:
#                 pad_bottom = patch_size[0] - current_patch_H
#                 pad_right = patch_size[1] - current_patch_W
#                 img1_tile = F.pad(img1_tile, (0, pad_right, 0, pad_bottom), mode='constant', value=0.0)
#                 img2_tile = F.pad(img2_tile, (0, pad_right, 0, pad_bottom), mode='constant', value=0.0)
#                 weight = weights[idx][:, :, :current_patch_H, :current_patch_W]
#             else:
#                 weight = weights[idx]
#             flow_pre, _ = model(img1_tile, img2_tile)
#             if current_patch_H != patch_size[0] or current_patch_W != patch_size[1]:
#                 flow_pre = flow_pre[:, :, :current_patch_H, :current_patch_W]
#             flows_b[:, h:h+current_patch_H, w:w+current_patch_W] += flow_pre[0] * weight[0]
#             flow_count_b[:, h:h+current_patch_H, w:w+current_patch_W] += weight[0]
#         flows_b = flows_b / (flow_count_b + 1e-8)
#         flows_b = padder.unpad(flows_b)
#         flows[b] = flows_b[:, :H, :W]
#     return flows


# --------------------- 模型加载（DDP方式） ---------------------

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

# --------------------- 主程序 ---------------------

# def main(args):
#     # 根据环境设置设备（此处设备由 load_model_ddp() 设置）
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = build_model()

#     # 读取并裁剪图片
#     # im1 = Image.open(args.image1).convert("RGB")
#     # im2 = Image.open(args.image2).convert("RGB")
#     im1 = Image.open(args.image1).convert("RGB")
#     im2 = Image.open(args.image2).convert("RGB")
#     im1=np.array(im1).astype(np.uint8)[..., :3]
#     im2=np.array(im2).astype(np.uint8)[..., :3]
#     image1 = torch.from_numpy(im1).permute(2, 0, 1).float()
#     image2 = torch.from_numpy(im2).permute(2, 0, 1).float()
#     image_size = image1.shape[1:]
#     image1, image2 = image1[None].cuda(), image2[None].cuda()
#     hws = compute_grid_indices(image_size)
#     # im1 = center_crop(im1, TRAIN_SIZE)
#     # im2 = center_crop(im2, TRAIN_SIZE)
#     # tensor1 = pil_to_tensor(im1).to(device)
#     # tensor2 = pil_to_tensor(im2).to(device)
#     weights = None
#     if weights is None:     # no tile
#         padder = InputPadder(image1.shape)
#         image1, image2 = padder.pad(image1, image2)

#         flow_pre, _ = model(image1, image2)

#         flow_pre = padder.unpad(flow_pre).detach()
#         # image1    = padder.unpad(image1)
#         # image2    = padder.unpad(image2)
#         # flow = flow_pre.permute(0, 2, 3, 1)
#         flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    
#     else:                   # tile
#         flows = 0
#         flow_count = 0

#         for idx, (h, w) in enumerate(hws):
#             image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
#             image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
#             flow_pre, _ = model(image1_tile, image2_tile)
#             padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
#             flows += F.pad(flow_pre * weights[idx], padding)
#             flow_count += F.pad(weights[idx], padding)

#         flow_pre = flows / flow_count
#         flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

#     # 加载预训练模型（使用 DDP 封装）
#     # print("开始加载模型（DDP方式）……")

#     # # 计算光流
#     # print("开始计算光流……")
#     # flow = compute_optical_flow(model, tensor1, tensor2)
#     # print("光流计算完成。")


#     # —— 构造基础网格 grid —— #
#     B, C, H, W = image1.shape
#     # 生成 y, x 两个坐标矩阵，范围 0…H-1, 0…W-1
#     yy, xx = torch.meshgrid(torch.arange(H, device=image1.device),
#                             torch.arange(W, device=image1.device),
#                             indexing='ij')       # yy: [H,W], xx: [H,W]
#     # 合并成 [H, W, 2]，再扩成 [B, H, W, 2]
#     base_grid = torch.stack((xx, yy), dim=2)    # [H, W, 2]
#     base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    
#     # —— 加上光流，归一化到 [-1,1] —— #
#     # 新坐标 = 原坐标 + flow
#     warped_grid = base_grid + flow              # [B, H, W, 2]
#     # 归一化：x 方向除以 (W-1)/2 并减 1，y 方向除以 (H-1)/2 并减 1
#     warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
#     warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0
    
#     # —— 采样 —— #
#     # align_corners=True 保持坐标系一致；padding_mode 可选 'zeros'/'border'/'reflection'
#     tensor1_warp = F.grid_sample(
#         image1,               # [B, C, H, W]
#         warped_grid,          # [B, H, W, 2], in [-1,1]
#         mode='bilinear',
#         padding_mode='border',
#         align_corners=True
#     )

#     # 计算 warp 后与第二帧之间的差异（绝对差）
#     diff = torch.abs(tensor1_warp - image2)
#     diff_np = diff.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     tensor1_warp_np = tensor1_warp.squeeze(0).permute(1, 2, 0).cpu().numpy()

#     # 保存 warp 后的图像与差异图
#     imageio.imwrite("warped_image.png", (tensor1_warp_np * 255).astype(np.uint8))
#     imageio.imwrite("diff_image.png", (diff_np * 255).astype(np.uint8))
#     print("结果已保存：warped_image.png 和 diff_image.png")

def main(args):
    # 根据环境设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()

    # 读取并处理图片
    im1 = Image.open(args.image1).convert("RGB")
    im2 = Image.open(args.image2).convert("RGB")
    im1 = np.array(im1).astype(np.uint8)[..., :3]
    im2 = np.array(im2).astype(np.uint8)[..., :3]
    # 对输入图像进行归一化（0~1）
    image1 = torch.from_numpy(im1).permute(2, 0, 1).float() / 255.0
    image2 = torch.from_numpy(im2).permute(2, 0, 1).float() / 255.0
    image_size = image1.shape[1:]
    image1, image2 = image1[None].to(device), image2[None].to(device)
    hws = compute_grid_indices(image_size)
    weights = None

    if weights is None:     # no tile，直接计算全图光流
        padder = InputPadder(image1.shape)
        # pad 两张图片
        image1_pad, image2_pad = padder.pad(image1, image2)
        # 使用模型计算正向光流（image1 -> image2）
        flow_pre, _ = model(image2_pad, image1_pad)
        # unpad 得到原始尺寸的结果
        flow_pre = padder.unpad(flow_pre).detach()      # [1, 2, H, W]
        image1 = padder.unpad(image1_pad)
        image2 = padder.unpad(image2_pad)
        # 取负获得近似的反向光流（即 image2 -> image1 的光流）
        flow = flow_pre.permute(0, 2, 3, 1)           # [B, H, W, 2]
    else:                   # tile（此分支未作修改）
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

    # —— 构造基础网格 grid —— #
    B, C, H, W = image1.shape
    # 生成 y, x 两个坐标矩阵，范围 0…H-1, 0…W-1
    yy, xx = torch.meshgrid(torch.arange(H, device=image1.device),
                              torch.arange(W, device=image1.device),
                              indexing='ij')       # yy: [H,W], xx: [H,W]
    # 合并成 [H, W, 2]，再扩成 [B, H, W, 2]
    base_grid = torch.stack((xx, yy), dim=2)    # [H, W, 2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    
    # # —— 加上光流，归一化到 [-1,1] —— #
    # # 注意：由于 flow 为反向光流，此处用 base_grid + flow 得到采样坐标
    warped_grid = base_grid + flow              # [B, H, W, 2]
    # 归一化：x 方向除以 (W-1)/2 并减 1，y 方向除以 (H-1)/2 并减 1
    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0
    
    # # —— 采样 —— #
    # # align_corners=True 保持坐标系一致；padding_mode 可选 'zeros'/'border'/'reflection'
    # tensor1_warp = F.grid_sample(
    #     image1,               # [B, C, H, W]
    #     warped_grid,          # [B, H, W, 2], in [-1,1]
    #     mode='bilinear',
    #     padding_mode='border',
    #     align_corners=True
    # )

    # # 计算 warp 后与第二帧之间的差异（绝对差）
    # diff = torch.abs(tensor1_warp - image2)
    # diff1 = torch.abs(tensor1_warp - image1)
    # mean_diff = diff.mean().item()
    # print("Mean absolute difference between final warp and image2:", mean_diff)
    # diff_np = diff.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # diff_np1 = diff1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # tensor1_warp_np = tensor1_warp.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # print("diff_np2:")
    # print(diff_np)
    # print("diff_np1:")
    # print(diff_np1)
    # # 保存 warp 后的图像与差异图（记得乘回 255）
    # imageio.imwrite("warped_image.png", (tensor1_warp_np * 255).clip(0,255).astype(np.uint8))
    # imageio.imwrite("diff_image.png", (diff_np * 255).clip(0,255).astype(np.uint8))
    # imageio.imwrite("diff1_image.png", (diff_np1 * 255).clip(0,255).astype(np.uint8))
    # print("结果已保存：warped_image.png 和 diff_image.png")
    # 将光流除以15，作为每一步的位移
    num_steps = 15
    flow_step = flow / num_steps
    
    # 初始帧为 image1
    current = image1.clone()
    
    # 存储每一步结果
    results = []
    
    for i in range(1, num_steps + 1):
        # 当前步累加光流：flow_step * i
        warped_grid_i = base_grid + flow_step * i
    
        # 归一化到 [-1, 1]
        warped_grid_i[..., 0] = 2.0 * warped_grid_i[..., 0] / (W - 1) - 1.0
        warped_grid_i[..., 1] = 2.0 * warped_grid_i[..., 1] / (H - 1) - 1.0
    
        # 使用 grid_sample 对当前帧做 warp
        current = F.grid_sample(
            current, 
            warped_grid_i, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
    
        # 保存可视化结果
        current_np = current.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        imageio.imwrite(f"interpolated_{i:02d}.png", (current_np * 255).clip(0, 255).astype(np.uint8))
        results.append(current_np)
    
    # 最终一步 warp 与 image2 对比
    final_warp = current
    diff_final = torch.abs(final_warp - image2)
    mean_diff = diff_final.mean().item()
    print("Mean absolute difference between final warp and image2:", mean_diff)
    
    # 可视化差异
    diff_final_np = diff_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
    imageio.imwrite("final_diff_image.png", (diff_final_np * 255).clip(0, 255).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="光流估计及 warp 测试（DDP版）")
    parser.add_argument("--image1", type=str, required=True, help="第一帧图片路径")
    parser.add_argument("--image2", type=str, required=True, help="第二帧图片路径")
    args = parser.parse_args()
    main(args)
