import sys
sys.path.append('D:\VFI\FlowFormerPlusPlus-main\core')
# from attr import validate
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submissions import get_cfg as get_submission_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
from core.FlowFormer import build_flowformer
from utils.utils import InputPadder, forward_interpolate
import imageio
import itertools

TRAIN_SIZE = [432, 960]

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

import math
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

# @torch.no_grad()
# def create_sintel_submission(model, img1_path, img2_path, output_flow_path='output.flo',
#                                         output_viz_path='flow_viz.png', sigma=0.05):
#     """
#     计算两张图片之间的光流并保存结果。
#
#     参数：
#     - model: 预训练的 FlowFormer 模型。
#     - img1_path: 第一张图片的文件路径。
#     - img2_path: 第二张图片的文件路径。
#     - output_flow_path: 保存光流文件的路径，默认 'output.flo'。
#     - output_viz_path: 保存光流可视化图像的路径，默认 'flow_viz.png'。
#     - sigma: 权重计算的标准差，默认 0.05。
#     """
#
#     from utils.utils import InputPadder  # 确保已导入 InputPadder 类
#
#     # 定义训练时使用的图像块大小
#
#     # 定义要处理的图像尺寸
#     IMAGE_SIZE = [436, 1024]  # 可以根据需要调整
#
#     # 计算图像块的网格索引和权重
#     hws = compute_grid_indices(IMAGE_SIZE)
#     weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
#     model.eval()
#
#     # 加载和预处理图片
#     # def load_image(img_path):
#     #     img = Image.open(img_path).convert('RGB')
#     #     img = np.array(img).astype(np.float32) / 255.0
#     #     img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
#     #     return img.cuda()
#
#     image1 = Image.open(img1_path).convert('RGB')
#     image2 = Image.open(img2_path).convert('RGB')
#
#     # 转换为张量
#     from torchvision import transforms
#     transform = transforms.ToTensor()
#     image1 = transform(image1)
#     image2 = transform(image2)
#     image1, image2 = image1[None].cuda(), image2[None].cuda()
#     flows = 0
#     flow_count = 0
#
#     # 分块处理
#     for idx, (h, w) in enumerate(hws):
#         image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
#         image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
#
#         # 计算光流
#         flow_pre, flow_low = model(image1_tile, image2_tile)
#
#         padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
#         flows += F.pad(flow_pre * weights[idx], padding)
#         flow_count += F.pad(weights[idx], padding)
#         print(flows)
#         print(flow_count)
#
#     # 归一化光流
#     flow_pre = flows / flow_count
#     flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
#
#     # 保存光流文件
#     frame_utils.writeFlow(output_flow_path, flow)
#     print(f"光流文件已保存至 {output_flow_path}")
#
#     # 可视化光流并保存
#     flow_img = flow_viz.flow_to_image(flow)
#     image = Image.fromarray(flow_img)
#     image.save(output_viz_path)
#     print(f"光流可视化图像已保存至 {output_viz_path}")
#
import random
@torch.no_grad()
def compute_optical_flow(model, image_path1, image_path2, sigma=0.05):
    """Compute optical flow between two images using the FlowFormer++ model."""

    IMAGE_SIZE = [436,1024] # Get the size from the input image
    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    img1=frame_utils.read_gen(image_path1)
    img2=frame_utils.read_gen(image_path2)
    img1 = np.array(img1).astype(np.uint8)[..., :3]
    img2 = np.array(img2).astype(np.uint8)[..., :3]
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

    # Prepare the images
    image1, image2 = img1[None].cuda(), img2[None].cuda()

    flows = 0
    flow_count = 0

    # Loop through the patches
    for idx, (h, w) in enumerate(hws):
        image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
        image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
        flow_pre, _ = model(image1_tile, image2_tile)

        padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
        flows += F.pad(flow_pre * weights[idx], padding)
        flow_count += F.pad(weights[idx], padding)

    flow_pre = flows / flow_count
    flow = flow_pre[0].permute(1, 2, 0).cpu().numpy() # Convert to numpy array

    return flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the model checkpoint')
    parser.add_argument('--image1', required=True, help='Path to the first image')
    parser.add_argument('--image2', required=True, help='Path to the second image')
    parser.add_argument('--output', required=True,help='Path to save the output flow file')
    args = parser.parse_args()
    # Load configuration
    cfg = get_submission_cfg()
    cfg.update(vars(args))

    # Initialize the model
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()
    # Create the flow
    flow = compute_optical_flow(model, args.image1, args.image2)
    # Save the flow output
    frame_utils.writeFlow(args.output, flow)
    print(f"Flow saved to {args.output}")
    flow_img = flow_viz.flow_to_image(flow)  # 将光流转换为可视化图像
    flow_image = Image.fromarray(flow_img)  # 将 NumPy 数组转换为 PIL 图像
    flow_image.save(args.output.replace('.flo', '.png'))  # 保存为 PNG 文件
    print(f"Flow visualization saved to {args.output.replace('.flo', '.png')}")

