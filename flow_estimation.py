import sys
sys.path.append('FlowformerPlusPlus')

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
import FlowformerPlusPlus.core.datasets
from FlowformerPlusPlus.core.utils import flow_viz
from FlowformerPlusPlus.core.utils import frame_utils
import cv2
import math
import os.path as osp

from FlowformerPlusPlus.core.FlowFormer import build_flowformer
from FlowformerPlusPlus.core.utils.utils import InputPadder, forward_interpolate
import itertools

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
    for idx, (h_val, w_val) in enumerate(hws):
        weights[:, idx, h_val:h_val+patch_size[0], w_val:w_val+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h_val, w_val) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h_val:h_val+patch_size[0], w_val:w_val+patch_size[1]])
    return patch_weights

def compute_flow(model, image1, image2, weights=None):
    print(f"Computing flow...")

    image_size = image1.shape[1:]

    # Move tensors to GPU if needed
    if not image1.is_cuda:
        image1 = image1.cuda()
    if not image2.is_cuda:
        image2 = image2.cuda()
    
    # Ensure batch dimension exists
    if len(image1.shape) == 3:
        image1 = image1.unsqueeze(0)
    if len(image2.shape) == 3:
        image2 = image2.unsqueeze(0)

    hws = compute_grid_indices(image_size)
    if weights is None:  # no tile weights provided
        try:
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_pre, _ = model(image1, image2)
            flow_pre = padder.unpad(flow_pre)
        except Exception as e:
            print(f"Direct inference failed: {e}")
            print("Trying tiled approach instead...")
            weights = compute_weight(hws, image_size)
            flows = 0
            flow_count = 0
            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
                flow_pre_tile, _ = model(image1_tile, image2_tile)
                padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre_tile * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)
            flow_pre = flows / flow_count
    else:  # using provided weights
        flows = 0
        flow_count = 0
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre_tile, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre_tile * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
        flow_pre = flows / flow_count

    return flow_pre

def warp_image(image, flow):
    """
    Warp an image using optical flow.
    
    Args:
        image: Input image tensor [B, C, H, W]
        flow: Optical flow tensor [B, 2, H, W]
        
    Returns:
        Warped image tensor [B, C, H, W]
    """
    B, C, H, W = image.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), dim=0).float().to(image.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    flow_grid = grid + flow
    flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / (W - 1) - 1.0
    flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / (H - 1) - 1.0
    flow_grid = flow_grid.permute(0, 2, 3, 1)
    warped_image = F.grid_sample(image, flow_grid, padding_mode='border', align_corners=True)
    return warped_image

def compute_flow_and_warp(model, image1, image2, weights=None):
    """
    Compute optical flow and warp the second image.
    
    Args:
        model: FlowFormer model.
        image1: First image tensor [B, 3, H, W] or [3, H, W].
        image2: Second image tensor [B, 3, H, W] or [3, H, W].
        weights: Optional weights for tiled processing.
        
    Returns:
        flow: Optical flow tensor [B, 2, H, W].
        warped_image: Warped second image tensor [B, 3, H, W].
    """
    if len(image1.shape) == 3:
        image1 = image1.unsqueeze(0)
    if len(image2.shape) == 3:
        image2 = image2.unsqueeze(0)
    if not image1.is_cuda:
        image1 = image1.cuda()
    if not image2.is_cuda:
        image2 = image2.cuda()
    
    flow = compute_flow(model, image1, image2, weights)
    warped_image = warp_image(image2, flow)
    return flow, warped_image

def compute_adaptive_image_size(image_size):
    # image_size: (H, W)
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]
    scale = scale0 if scale0 > scale1 else scale1
    # 返回 (W_new, H_new)
    new_size = (int(image_size[1] * scale), int(image_size[0] * scale))
    return new_size

def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
    print(f"Preparing images...")
    print(f"Root dir = {root_dir}, fn = {fn1}")
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
    viz_fn = osp.join(viz_dir, filename + '_flow.png')
    warp_fn = osp.join(viz_dir, filename + '_warp.png')
    return image1, image2, viz_fn, warp_fn

def process_image_paths(model, img1_path, img2_path, output_dir, keep_size=False):
    """
    Process two image paths, compute flow and warp, and save results.
    
    Args:
        model: FlowFormer model.
        img1_path: Path to first image.
        img2_path: Path to second image.
        output_dir: Directory to save output images.
        keep_size: Whether to keep original image size.
        
    Returns:
        flow: Optical flow numpy array [H, W, 2].
        warped_image: Warped second image numpy array [H, W, 3].
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Reading images from {img1_path} and {img2_path}")
    image1 = frame_utils.read_gen(img1_path)
    image2 = frame_utils.read_gen(img2_path)
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    original_size = image1.shape[:2]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        print(f"Resizing images from {image1.shape[:2]} to {dsize[::-1]}")
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    flow, warped_image = compute_flow_and_warp(model, image1, image2)
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
    warped_image_np = warped_image[0].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    if not keep_size:
        warped_image_np = cv2.resize(warped_image_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
    basename = osp.splitext(osp.basename(img1_path))[0]
    flow_viz_path = osp.join(output_dir, f'{basename}_flow.png')
    warp_path = osp.join(output_dir, f'{basename}_warp.png')
    flow_img = flow_viz.flow_to_image(flow_np)
    cv2.imwrite(flow_viz_path, flow_img[:, :, [2,1,0]])
    cv2.imwrite(warp_path, warped_image_np[:, :, [2,1,0]])
    print(f"Saved flow visualization to {flow_viz_path}")
    print(f"Saved warped image to {warp_path}")
    return flow_np, warped_image_np

def build_model():
    print(f"Building model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()
    model.eval()
    return model

def visualize_flow(root_dir, viz_root_dir, model, img_pairs, keep_size):
    for img_pair in img_pairs:
        fn1, fn2 = img_pair
        print(f"Processing {fn1}, {fn2}...")
        image1, image2, viz_fn, warp_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)
        flow, warped_image = compute_flow_and_warp(model, image1, image2)
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
        warped_image_np = warped_image[0].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        flow_img = flow_viz.flow_to_image(flow_np)
        cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])
        cv2.imwrite(warp_fn, warped_image_np[:, :, [2,1,0]])

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
        img_pairs.append((img1, img2))
    return img_pairs


@torch.no_grad()
def compute_optical_flow(model, image1, image2, sigma=0.05):
    """
    外部计算光流返回光流的函数

    参数：
      - model: 预训练的 FlowFormer 模型
      - image1: 第一个图像张量，[B, 3, H, W]
      - image2: 第二个图像张量，[B, 3, H, W]
      - sigma: 权重计算的标准差

    返回：
      - flow: 光流张量，[B, 2, H, W]，尺寸与输入图像一致
    """
    image_size = image1.shape[1:]

    hws = compute_grid_indices(image_size)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow_pre, _ = model(image1, image2)
    flow_pre = padder.unpad(flow_pre)    
    return flow_pre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', default='sintel', choices=['sintel', 'seq', 'pair'])
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
    parser.add_argument('--seq_dir', default='demo_data/mihoyo')
    parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
    parser.add_argument('--end_idx', type=int, default=1200)      # ending index of the image sequence
    parser.add_argument('--viz_root_dir', default='viz_results')
    parser.add_argument('--keep_size', action='store_true')      # 是否保留原始图像尺寸，否则进行自适应 resize
    parser.add_argument('--img1_path', default=None)             # pair 模式下第一个图像路径
    parser.add_argument('--img2_path', default=None)             # pair 模式下第二个图像路径

    args = parser.parse_args()

    root_dir = args.root_dir
    viz_root_dir = args.viz_root_dir
    model = build_model()

    with torch.no_grad():
        if args.eval_type == 'sintel':
            img_pairs = process_sintel(args.sintel_dir)
            visualize_flow(root_dir, viz_root_dir, model, img_pairs, args.keep_size)
        elif args.eval_type == 'seq':
            img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx)
            visualize_flow(root_dir, viz_root_dir, model, img_pairs, args.keep_size)
        elif args.eval_type == 'pair':
            if args.img1_path is None or args.img2_path is None:
                print("Error: For pair mode, both img1_path and img2_path must be provided.")
                sys.exit(1)
            process_image_paths(model, args.img1_path, args.img2_path, viz_root_dir, args.keep_size)
