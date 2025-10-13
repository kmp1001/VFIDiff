# import sys
# sys.path.append('core')

# from PIL import Image
# from glob import glob
# import argparse
# import os
# import time
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from configs.submissions import get_cfg
# from core.utils.misc import process_cfg
# import datasets
# from utils import flow_viz
# from utils import frame_utils
# import cv2
# import math
# import os.path as osp
# import numpy as np
# import cv2
# import torch
# import glob
# import os
# import time
# import gzip
# import pickle
# from core.FlowFormer import build_flowformer

# from utils.utils import InputPadder, forward_interpolate
# import itertools

# TRAIN_SIZE = [432, 960]


# def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
#   if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
#     raise ValueError(
#         f"Overlap should be less than size of patch (got {min_overlap}"
#         f"for patch size {patch_size}).")
#   if image_shape[0] == TRAIN_SIZE[0]:
#     hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
#   else:
#     hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
#   if image_shape[1] == TRAIN_SIZE[1]:
#     ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
#   else:
#     ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

#   # Make sure the final patch is flush with the image boundary
#   hs[-1] = image_shape[0] - patch_size[0]
#   ws[-1] = image_shape[1] - patch_size[1]
#   return [(h, w) for h in hs for w in ws]

# def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
#     patch_num = len(hws)
#     h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
#     h, w = h / float(patch_size[0]), w / float(patch_size[1])
#     c_h, c_w = 0.5, 0.5 
#     h, w = h - c_h, w - c_w
#     weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
#     denorm = 1 / (sigma * math.sqrt(2 * math.pi))
#     weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

#     weights = torch.zeros(1, patch_num, *image_shape)
#     for idx, (h, w) in enumerate(hws):
#         weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
#     weights = weights.cuda()
#     patch_weights = []
#     for idx, (h, w) in enumerate(hws):
#         patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

#     return patch_weights

# def compute_flow(model, image1, image2, weights=None):
#     print(f"computing flow...")

#     image_size = image1.shape[1:]

#     image1, image2 = image1[None].cuda(), image2[None].cuda()

#     hws = compute_grid_indices(image_size)
#     if weights is None:     # no tile
#         padder = InputPadder(image1.shape)
#         image1, image2 = padder.pad(image1, image2)

#         flow_pre, _ = model(image1, image2)

#         flow_pre = padder.unpad(flow_pre)
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

#     return flow

# def compute_adaptive_image_size(image_size):
#     target_size = TRAIN_SIZE
#     scale0 = target_size[0] / image_size[0]
#     scale1 = target_size[1] / image_size[1] 

#     if scale0 > scale1:
#         scale = scale0
#     else:
#         scale = scale1

#     image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

#     return image_size

# def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
#     print(f"preparing image...")
#     print(f"root dir = {root_dir}, fn = {fn1}")

#     image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
#     image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
#     image1 = np.array(image1).astype(np.uint8)[..., :3]
#     image2 = np.array(image2).astype(np.uint8)[..., :3]
#     if not keep_size:
#         dsize = compute_adaptive_image_size(image1.shape[0:2])
#         image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#         image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#     image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
#     image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


#     dirname = osp.dirname(fn1)
#     filename = osp.splitext(osp.basename(fn1))[0]

#     viz_dir = osp.join(viz_root_dir, dirname)
#     if not osp.exists(viz_dir):
#         os.makedirs(viz_dir)

#     viz_fn = osp.join(viz_dir, filename + '.png')

#     return image1, image2, viz_fn

# def build_model():
#     print(f"building  model...")
#     cfg = get_cfg()
#     model = torch.nn.DataParallel(build_flowformer(cfg))
#     model.load_state_dict(torch.load(cfg.model))

#     model.cuda()
#     model.eval()

#     return model

# def visualize_flow(root_dir, viz_root_dir, model, img_pairs, keep_size):
#     weights = None
#     for img_pair in img_pairs:
#         fn1, fn2 = img_pair
#         print(f"processing {fn1}, {fn2}...")

#         image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)
#         flow = compute_flow(model, image1, image2, weights)
#         flow_img = flow_viz.flow_to_image(flow)
#         cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])

# def process_sintel(sintel_dir):
#     img_pairs = []
#     for scene in os.listdir(sintel_dir):
#         dirname = osp.join(sintel_dir, scene)
#         image_list = sorted(glob(osp.join(dirname, '*.png')))
#         for i in range(len(image_list)-1):
#             img_pairs.append((image_list[i], image_list[i+1]))

#     return img_pairs

# def generate_pairs(dirname, start_idx, end_idx):
#     img_pairs = []
#     for idx in range(start_idx, end_idx):
#         img1 = osp.join(dirname, f'{idx:06}.png')
#         img2 = osp.join(dirname, f'{idx+1:06}.png')
#         # img1 = f'{idx:06}.png'
#         # img2 = f'{idx+1:06}.png'
#         img_pairs.append((img1, img2))

#     return img_pairs
# def compute_optical_flow(model,image1,image2, weights=None):
#     image_size = image1.shape[1:]
#     image1, image2 = image1[None].cuda(), image2[None].cuda()
#     hws=hws = compute_grid_indices(image_size)
#     if weights is None:     # no tile
#         padder = InputPadder(image1.shape)
#         image1, image2 = padder.pad(image1, image2)
#         with torch.no_grad():
#             flow_pre, _ = model(image1, image2)
#             flow_pre = flow_pre.detach()
#         flow_pre = padder.unpad(flow_pre)
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

#     return flow




# # warping function of pytorch version on cpu by ASOP
# # flow1 is the forward flow (H, W, 2)
# # flow2 is the backward flow (H, W, 2)
# # img is the input flow (H, W, n)
# # warp_cur = warp(forward_flow, backward_flow, prev)
# # warp_prev = warp(backward_flow, forward_flow, cur)

# def warp(flow1, flow2, img):  # H,W,C  h*w*2
    
#     img = img.permute(1, 2, 0)
#     flow_shape = flow1.shape
#     label_shape = img.shape
#     height, width = flow1.shape[0], flow1.shape[1]

#     output = torch.zeros(label_shape)  # output = np.zeros_like(img, dtype=img.dtype)
#     flow_t = torch.zeros(flow_shape)  # flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

#     # grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
#     h_grid = torch.arange(0, height)
#     w_grid = torch.arange(0, width)
#     h_grid = h_grid.repeat(width, 1).permute(1, 0)  # .unsqueeze(0)
#     w_grid = w_grid.repeat(height, 1)  # .unsqueeze(0)
#     grid = torch.stack((h_grid, w_grid), 0).permute(1, 2, 0)  # float3
#     # grid = torch.cat([h_grid, w_grid],0).permute(1,2,0)

#     dx = grid[:, :, 0] + flow2[:, :, 1].long()
#     dy = grid[:, :, 1] + flow2[:, :, 0].long()
#     sx = torch.floor(dx.float())  # float32 #sx = np.floor(dx).astype(int)
#     sy = torch.floor(dy.float())

#     valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)  # H* W 512 x 512 uint8

#     # sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
#     # sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
#     # sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
#     #                   (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

#     sx_mat = torch.stack((sx, sx + 1, sx, sx + 1), dim=2).clamp(0, height - 1)  # torch.float32
#     sy_mat = torch.stack((sy, sy, sy + 1, sy + 1), dim=2).clamp(0, width - 1)
#     sxsy_mat = torch.abs((1 - torch.abs(sx_mat - dx.float().unsqueeze(0).permute(1, 2, 0))) *
#                          (1 - torch.abs(sy_mat - dy.float().unsqueeze(0).permute(1, 2, 0))))

#     for i in range(4):
#         flow_t = flow_t.long() + sxsy_mat.long()[:, :, i].unsqueeze(0).permute(1, 2, 0) * flow1.long()[
#                                                                                           sx_mat.long()[:, :, i],
#                                                                                           sy_mat.long()[:, :, i], :]

#     valid = valid & (
#                 torch.norm(flow_t.float()[:, :, [1, 0]] + torch.stack((dx.float(), dy.float()), dim=2) - grid.float(),
#                            dim=2, keepdim=True).squeeze(2) < 100)

#     flow_t = (flow2.float() - flow_t.float()) / 2.0
#     dx = grid.long()[:, :, 0] + flow_t.long()[:, :, 1]
#     dy = grid.long()[:, :, 1] + flow_t.long()[:, :, 0]
#     valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)

#     output[valid, :] = img.float()[dx[valid].long(), dy[valid].long(), :]

#     return output.permute(2, 0, 1)  # 3HW


# def main():
#     # prev = cv2.imread('one.png')
#     # cur = cv2.imread('two.png')
#     model = build_model()
#     image1 = frame_utils.read_gen('im1.png')
#     image2 = frame_utils.read_gen('im7.png')
#     image1 = np.array(image1).astype(np.uint8)[..., :3]
#     image2 = np.array(image2).astype(np.uint8)[..., :3]
#     # if not keep_size:
#     #     dsize = compute_adaptive_image_size(image1.shape[0:2])
#     #     image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#     #     image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#     image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
#     image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
#     flow1 = compute_optical_flow(model,image1,image2)
#     flow2 = compute_optical_flow(model, image2, image1)
#     # flow1 = pickle.loads(gzip.GzipFile('forward_0_5.pkl', 'rb').read())  # 'forward_0_5.pkl'
#     # flow2 = pickle.loads(gzip.GzipFile('backward_5_0.pkl', 'rb').read())  # 'backward_5_0.pkl'
#     print("read flow and image")
#     # warp(forward, backward, 0th)  #  0 -> 1
#     # warp(backward, forward, 1th)  #  1 -> 0
#     flow1 = torch.from_numpy(flow1)
#     flow2 = torch.from_numpy(flow2)
#     prev = image1.contiguous()  # -> [H, W, C]
#     cur  = image2.contiguous()  # -> [H, W, C]
#     # 得到 warp 后的 numpy array，形状 (H, W, 3)，值域大概在 [0,255]
#     w0 = warp(flow1, flow2, prev).permute(1, 2, 0).numpy()
#     w1 = warp(flow2, flow1, cur).permute(1, 2, 0).numpy()
    
#     # 1) Clip & 转 uint8（如果它是 float 的话）
#     # w0 = w0.permute(1, 2, 0)
#     # w1 = w1.permute(1, 2, 0)
#     w0 = np.clip(w0, 0, 255).astype(np.uint8)
#     w1 = np.clip(w1, 0, 255).astype(np.uint8)
    
#     # 2) RGB -> BGR
#     w0 = w0[..., ::-1]
#     w1 = w1[..., ::-1]

#     # 3) 写磁盘
#     cv2.imwrite('warp_forward.png', w0)
#     cv2.imwrite('warp_backward.png', w1)
#     # w0 = warp(flow1, flow2, prev).numpy()  # 0->1
#     # w1 = warp(flow2, flow1, cur).numpy()  # 1->0
#     # print("finish warp")
#     # cv2.imwrite('warp_forward.png', w0)
#     # cv2.imwrite('warp_backward.png', w1)
# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--eval_type', default='sintel')
#     # parser.add_argument('--root_dir', default='.')
#     # parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
#     # parser.add_argument('--seq_dir', default='demo_data/mihoyo')
#     # parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
#     # parser.add_argument('--end_idx', type=int, default=1200)    # ending index of the image sequence
#     # parser.add_argument('--viz_root_dir', default='viz_results')
#     # parser.add_argument('--keep_size', action='store_true')     # keep the image size, or the image will be adaptively resized.
#     #
#     # args = parser.parse_args()
#     #
#     # root_dir = args.root_dir
#     # viz_root_dir = args.viz_root_dir
#     #
#     # model = build_model()
#     #
#     # if args.eval_type == 'sintel':
#     #     img_pairs = process_sintel(args.sintel_dir)
#     # elif args.eval_type == 'seq':
#     #     img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx)
#     # with torch.no_grad():
#     #     visualize_flow(root_dir, viz_root_dir, model, img_pairs, args.keep_size)
#     main()


import sys
sys.path.append('autodl-tmp/VFIDiff-journal/ResShift-journal_original/FlowformerPlusPlus')
sys.path.append('autodl-tmp/VFIDiff-journal/ResShift-journal_original/FlowformerPlusPlus/core')

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




# warping function of pytorch version on cpu by ASOP
# flow1 is the forward flow (H, W, 2)
# flow2 is the backward flow (H, W, 2)
# img is the input flow (H, W, n)
# warp_cur = warp(forward_flow, backward_flow, prev)
# warp_prev = warp(backward_flow, forward_flow, cur)

# def warp(flow1, flow2, img):  # C,H,W  h*w*2
    
#     img = img.permute(1, 2, 0)
#     flow_shape = flow1.shape
#     label_shape = img.shape
#     height, width = flow1.shape[0], flow1.shape[1]

#     output = torch.zeros(label_shape)  # output = np.zeros_like(img, dtype=img.dtype)
#     flow_t = torch.zeros(flow_shape)  # flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

#     # grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
#     h_grid = torch.arange(0, height)
#     w_grid = torch.arange(0, width)
#     h_grid = h_grid.repeat(width, 1).permute(1, 0)  # .unsqueeze(0)
#     w_grid = w_grid.repeat(height, 1)  # .unsqueeze(0)
#     grid = torch.stack((h_grid, w_grid), 0).permute(1, 2, 0)  # float3
#     # grid = torch.cat([h_grid, w_grid],0).permute(1,2,0)

#     dx = grid[:, :, 0] + flow2[:, :, 1].long()
#     dy = grid[:, :, 1] + flow2[:, :, 0].long()
#     sx = torch.floor(dx.float())  # float32 #sx = np.floor(dx).astype(int)
#     sy = torch.floor(dy.float())

#     valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)  # H* W 512 x 512 uint8

#     # sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
#     # sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
#     # sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
#     #                   (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

#     sx_mat = torch.stack((sx, sx + 1, sx, sx + 1), dim=2).clamp(0, height - 1)  # torch.float32
#     sy_mat = torch.stack((sy, sy, sy + 1, sy + 1), dim=2).clamp(0, width - 1)
#     sxsy_mat = torch.abs((1 - torch.abs(sx_mat - dx.float().unsqueeze(0).permute(1, 2, 0))) *
#                          (1 - torch.abs(sy_mat - dy.float().unsqueeze(0).permute(1, 2, 0))))

#     for i in range(4):
#         flow_t = flow_t.long() + sxsy_mat.long()[:, :, i].unsqueeze(0).permute(1, 2, 0) * flow1.long()[
#                                                                                           sx_mat.long()[:, :, i],
#                                                                                           sy_mat.long()[:, :, i], :]

#     valid = valid & (
#                 torch.norm(flow_t.float()[:, :, [1, 0]] + torch.stack((dx.float(), dy.float()), dim=2) - grid.float(),
#                            dim=2, keepdim=True).squeeze(2) < 100)

#     flow_t = (flow2.float() - flow_t.float()) / 2.0
#     dx = grid.long()[:, :, 0] + flow_t.long()[:, :, 1]
#     dy = grid.long()[:, :, 1] + flow_t.long()[:, :, 0]
#     valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)

#     output[valid, :] = img.float()[dx[valid].long(), dy[valid].long(), :]

#     return output.permute(2, 0, 1)  # 3HW

# def main():
#     # prev = cv2.imread('one.png')
#     # cur = cv2.imread('two.png')
#     model = build_model()
#     image1 = frame_utils.read_gen('im1.png')
#     image2 = frame_utils.read_gen('im7.png')
#     image1 = np.array(image1).astype(np.uint8)[..., :3]
#     image2 = np.array(image2).astype(np.uint8)[..., :3]
#     # if not keep_size:
#     #     dsize = compute_adaptive_image_size(image1.shape[0:2])
#     #     image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#     #     image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#     image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
#     image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
#     flow1 = compute_optical_flow(model,image1,image2)
#     flow2 = compute_optical_flow(model, image2, image1)
#     # flow1 = pickle.loads(gzip.GzipFile('forward_0_5.pkl', 'rb').read())  # 'forward_0_5.pkl'
#     # flow2 = pickle.loads(gzip.GzipFile('backward_5_0.pkl', 'rb').read())  # 'backward_5_0.pkl'
#     print("read flow and image")
#     # warp(forward, backward, 0th)  #  0 -> 1
#     # warp(backward, forward, 1th)  #  1 -> 0
#     flow1 = torch.from_numpy(flow1)
#     flow2 = torch.from_numpy(flow2)
#     prev = image1.contiguous()  # -> [C, H, W]
#     cur  = image2.contiguous()  # -> [C, H, W]
#     # 预备输出目录
#     out_dir = 'warp_steps'
#     os.makedirs(out_dir, exist_ok=True)
    
#     # 每一步都用 1/13 的光流增量来 warp
#     # step_flow1 = flow1 / 13.
#     # step_flow1 = step_flow1.permute(2, 0, 1)  # -> [2, H, W]
#     # step_flow2 = flow2 / 13.0
#     result = image1.contiguous()  # [C, H, W], float
#     w0 = warp(flow1, flow2, prev).permute(1, 2, 0).numpy()
    
#     for i in range(1, 14):   # 1,2,...,13
#         # warp 上一步结果
#         #实现
        
#         # 转 HWC、numpy、uint8
#         res_hwc = result.permute(1, 2, 0).numpy()      # [H,W,3]
#         res_hwc = np.clip(res_hwc, 0, 255).astype(np.uint8)  # 确保合法
#         # RGB->BGR
#         res_bgr = res_hwc[..., ::-1]
        
#         # 保存成 interp_01.png ... interp_13.png
#         cv2.imwrite(os.path.join(out_dir, f'interp_{i:02d}.png'), res_bgr)
#     # # 得到 warp 后的 numpy array，形状 (H, W, 3)，值域大概在 [0,255]
    
#     # w0 = warp(flow1, flow2, prev).permute(1, 2, 0).numpy()
#     # w1 = warp(flow2, flow1, cur).permute(1, 2, 0).numpy()
    
#     # # 1) Clip & 转 uint8（如果它是 float 的话）
#     # # w0 = w0.permute(1, 2, 0)
#     # # w1 = w1.permute(1, 2, 0)
#     # w0 = np.clip(w0, 0, 255).astype(np.uint8)
#     # w1 = np.clip(w1, 0, 255).astype(np.uint8)
    
#     # # 2) RGB -> BGR
#     # w0 = w0[..., ::-1]
#     # w1 = w1[..., ::-1]

#     # # 3) 写磁盘
#     # cv2.imwrite('warp_forward.png', w0)
#     # cv2.imwrite('warp_backward.png', w1)
#     # w0 = warp(flow1, flow2, prev).numpy()  # 0->1
#     # w1 = warp(flow2, flow1, cur).numpy()  # 1->0
#     # print("finish warp")
#     # cv2.imwrite('warp_forward.png', w0)
#     # cv2.imwrite('warp_backward.png', w1)
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

def compute_optical_flow(img0: torch.Tensor, img1: torch.Tensor, device=None) -> torch.Tensor:
    """
    计算一批图像对之间的正向光流 (img0 -> img1).

    Args:
      img0:  Tensor of shape [B,3,H,W], RGB in [0,1]
      img1:  Tensor of shape [B,3,H,W], RGB in [0,1]
      model: 你的 FlowFormer 模型实例
      device: 可选，cpu 或 cuda

    Returns:
      flow01: Tensor of shape [B,2,H,W], 正向光流字段
    """
    global FLOW_MODEL
    # 确保在同一设备上
    if device is None:
        device = img0.device
    img0 = img0.to(device)
    img1 = img1.to(device)

    B, C, H, W = img0.shape
    flows = []
    # 逐项调用 compute_optical_flow
    for b in range(B):
        # 单张 [3,H,W] 转 numpy 再调用用户实现
        im0_np = (img0[b].cpu().numpy() * 255).astype('uint8').transpose(1,2,0)
        im1_np = (img1[b].cpu().numpy() * 255).astype('uint8').transpose(1,2,0)
        # 调用已有接口
        flow_np = compute_optical_flow1(FLOW_MODEL, torch.from_numpy(im0_np).permute(2,0,1).float().to(device),
                                             torch.from_numpy(im1_np).permute(2,0,1).float().to(device))
        # flow_np: numpy [H,W,2]
        flow = torch.from_numpy(flow_np).permute(2,0,1)  # -> [2,H,W]
        flows.append(flow)
    # 堆叠为 [B,2,H,W]
    return torch.stack(flows, dim=0)
# def warp_with_coords(flow1, flow2, img_chw):
#     """
#     Args:
#       flow1:   [H,W,2] Tensor (forward flow)
#       flow2:   [H,W,2] Tensor (backward flow)
#       img_chw: [C,H,W] Tensor
#     Returns:
#       out_chw:    [C,H,W] warped image
#       coord_full: [H,W,2] float pixel coords in source img for each output
#       mask:       [H,W] 有效区域mask
#     """
#     # --- copy your warp body, but capture dx3, dy3 ---
#     img = img_chw.permute(1,2,0)  # [H,W,C]
#     H, W = flow1.shape[:2]
#     device = flow1.device


#     # build grid coords
#     ys = torch.arange(H, dtype=torch.float32, device=flow1.device)
#     xs = torch.arange(W, dtype=torch.float32, device=flow1.device)
#     # grid_y = ys.repeat(W,1).permute(1,0)
#     # grid_x = xs.repeat(H,1)
#     gy, gx = torch.meshgrid(ys, xs, indexing='ij')
#     grid = torch.stack([gx, gy], dim=-1)  # [H,W,2] as (x,y)
#     # grid = torch.stack((grid_y, grid_x), dim=2)  # [H,W,2] as (y,x)

#     # # backward step
#     # dx = grid[...,0] + flow2[...,1].long()
#     # dy = grid[...,1] + flow2[...,0].long()
#     # sx = torch.floor(dx.float()); sy = torch.floor(dy.float())
#     # valid = (sx>=0)&(sx < H-1)&(sy>=0)&(sy<W-1)

#     # # bilinear for flow_t
#     # sx0 = sx.long().clamp(0,H-1); sy0 = sy.long().clamp(0,W-1)
#     # sx1 = (sx0+1).clamp(0,H-1); sy1 = (sy0+1).clamp(0,W-1)
#     # fx = dx - sx.float(); fy = dy - sy.float()
#     # w00 = (1-fx)*(1-fy); w01 = (1-fx)*fy
#     # w10 = fx*(1-fy);   w11 = fx*fy

#     # # gather flow1 at neighbors
#     # f00 = flow1[sx0, sy0]; f01 = flow1[sx0, sy1]
#     # f10 = flow1[sx1, sy0]; f11 = flow1[sx1, sy1]
#     # flow_t = f00*w00.unsqueeze(2) + f01*w01.unsqueeze(2) + f10*w10.unsqueeze(2) + f11*w11.unsqueeze(2)

#     # # forward consistency
#     # dx2 = grid[...,0] + flow_t[...,1]
#     # dy2 = grid[...,1] + flow_t[...,0]
#     # valid = valid & ((flow2 - flow_t).norm(dim=2) < 1.0)

#     # # residual warp
#     # flow_res = (flow2 - flow_t)*0.5
#     # dx3 = (grid[...,0].float() + flow_res[...,1]).clamp(0,H-1)
#     # dy3 = (grid[...,1].float() + flow_res[...,0]).clamp(0,W-1)

#     # # final bilinear sample into output
#     # sx3 = torch.floor(dx3).long().clamp(0,H-1)
#     # sy3 = torch.floor(dy3).long().clamp(0,W-1)
#     # sx31= (sx3+1).clamp(0,H-1); sy31 = (sy3+1).clamp(0,W-1)
#     # fx3 = dx3 - sx3.float(); fy3 = dy3 - sy3.float()
#     # w003 = (1-fx3)*(1-fy3); w013 = (1-fx3)*fy3
#     # w103 = fx3*(1-fy3);   w113 = fx3*fy3

#     # out = torch.zeros_like(img)
#     # for (xx,yy,ww) in [(sx3,sy3,w003),(sx3,sy31,w013),(sx31,sy3,w103),(sx31,sy31,w113)]:
#     #     out[valid] += img[xx[valid], yy[valid]] * ww[valid].unsqueeze(1)

#     # out_chw = out.permute(2,0,1).contiguous()  # [C,H,W]
#     # coord_full = torch.stack((dy3, dx3), dim=2) # note (x,y) indexing swap
#     # return out_chw, coord_full
#         # Step 1: 使用后向光流 flow2 (img1->img0) 找到对应点
#     # 对于输出位置 (x,y)，找到它在 img0 中的对应位置
#     back_coords = grid + flow2  # [H,W,2]
    
#     # 确保坐标在有效范围内
#     valid_back = (back_coords[..., 0] >= 0) & (back_coords[..., 0] <= W-1) & \
#                  (back_coords[..., 1] >= 0) & (back_coords[..., 1] <= H-1)

#     # Step 2: 在这些对应点处双线性插值获取前向光流
#     # 将坐标clamp到有效范围
#     safe_coords = back_coords.clone()
#     safe_coords[..., 0] = torch.clamp(safe_coords[..., 0], 0, W-1)  # x坐标
#     safe_coords[..., 1] = torch.clamp(safe_coords[..., 1], 0, H-1)  # y坐标
    
#     # 双线性插值参数
#     x0 = torch.floor(safe_coords[..., 0]).long()
#     y0 = torch.floor(safe_coords[..., 1]).long()
#     x1 = torch.clamp(x0 + 1, 0, W-1)
#     y1 = torch.clamp(y0 + 1, 0, H-1)
    
#     # 插值权重
#     wx = safe_coords[..., 0] - x0.float()
#     wy = safe_coords[..., 1] - y0.float()
    
#     # 四个邻居的权重
#     w00 = (1 - wx) * (1 - wy)
#     w01 = (1 - wx) * wy
#     w10 = wx * (1 - wy) 
#     w11 = wx * wy
    
#     # 插值前向光流
#     flow_interp = (flow1[y0, x0] * w00.unsqueeze(-1) +
#                    flow1[y1, x0] * w01.unsqueeze(-1) + 
#                    flow1[y0, x1] * w10.unsqueeze(-1) +
#                    flow1[y1, x1] * w11.unsqueeze(-1))

#     # Step 3: 前向一致性检查
#     # 从当前位置出发，用插值得到的前向光流应该能回到原位置
#     forward_coords = back_coords + flow_interp
#     consistency_error = torch.norm(forward_coords - grid, dim=-1)
#     consistency_valid = consistency_error < 1.0
    
#     # 综合有效性
#     valid_mask = valid_back & consistency_valid

#     # Step 4: 计算最终的采样坐标（混合策略）
#     # 使用残差流进行微调
#     residual_flow = (flow2 - (-flow_interp)) * 0.5  # 残差修正
#     final_coords = back_coords + residual_flow
    
#     # 确保最终坐标在有效范围内
#     final_coords[..., 0] = torch.clamp(final_coords[..., 0], 0, W-1)  # x坐标
#     final_coords[..., 1] = torch.clamp(final_coords[..., 1], 0, H-1)  # y坐标
    
#     # Step 5: 对图像进行双线性采样
#     # 转换为grid_sample需要的格式
#     norm_coords = torch.zeros_like(final_coords)
#     norm_coords[..., 0] = 2.0 * final_coords[..., 0] / (W - 1) - 1.0  # x
#     norm_coords[..., 1] = 2.0 * final_coords[..., 1] / (H - 1) - 1.0  # y
    
#     # 重排为 [1, H, W, 2] 格式
#     sampling_grid = norm_coords.unsqueeze(0)
    
#     # 使用grid_sample进行采样
#     warped_img = F.grid_sample(
#         img_chw.unsqueeze(0),  # [1, C, H, W]
#         sampling_grid,         # [1, H, W, 2]
#         mode='bilinear',
#         padding_mode='border',
#         align_corners=True
#     ).squeeze(0)  # [C, H, W]
    
#     return warped_img, final_coords, valid_mask

# def coords_to_grid(coords):
#     """
#     coords: [H,W,2] pixel coords (x,y)
#     returns [1,H,W,2] in [-1,1] for grid_sample
#     """
#     H,W = coords.shape[:2]
#     gx = 2.0*coords[...,0]/(W-1) - 1.0
#     gy = 2.0*coords[...,1]/(H-1) - 1.0
#     return torch.stack((gx,gy), dim=2).unsqueeze(0)

# def warp(
#     img0: torch.Tensor,      # [B,3,H,W]，图像范围 [0,1]
#     flow01: torch.Tensor,    # [B,2,H,W]，0->1 的光流
#     flow10: torch.Tensor,    # [B,2,H,W]，1->0 的光流
#     t: torch.Tensor          # [B]，每个批次样本的插帧比例，值域 [0,1]
# ) -> torch.Tensor:
#     """
#     对每个样本，在 0->1 的方向上插帧到对应的 t[b]（0<=t<=1）处。

#     Args:
#       img0:   Tensor[B,3,H,W]，范围 [0,1]
#       flow01: Tensor[B,2,H,W]
#       flow10: Tensor[B,2,H,W]
#       t:      Tensor[B], 每个样本的插帧比例

#     Returns:
#       Tensor[B,3,H,W]，插帧结果，范围 [0,1]
#     """
#     B, C, H, W = img0.shape
#     device = img0.device
#     outputs = []

#     for b in range(B):
#         # 获取单个样本
#         img_single = img0[b]  # [C, H, W]
#         flow_01 = flow01[b].permute(1, 2, 0)  # [H, W, 2]
#         flow_10 = flow10[b].permute(1, 2, 0)  # [H, W, 2]
#         t_val = t[b].item()
        
#         # 使用双向一致性warp到目标位置
#         warped_img1, coords, valid_mask = warp_with_coords(flow_01, flow_10, img_single)
        
#         # 在img0和warped_img1之间线性插值
#         # t=0时返回img0，t=1时返回warped_img1
#         interpolated = (1 - t_val) * img_single + t_val * warped_img1
        
#         # 可选：在无效区域使用img0的值
#         interpolated = torch.where(valid_mask.unsqueeze(0), interpolated, img_single)
        
#         outputs.append(interpolated)

#     return torch.stack(outputs, dim=0)

# import torch
# import torch.nn.functional as F

# def warp_single_step(flow1, flow2, img_chw):
#     """
#     完全基于你提供的原算法的单步warp函数
#     不做任何简化或修改

#     Args:
#         flow1: [H,W,2] forward flow 
#         flow2: [H,W,2] backward flow  
#         img_chw: [C,H,W] 输入图像

#     Returns:
#         output: [C,H,W] warped图像
#     """
#     # 完全按照原算法
#     img = img_chw.permute(1, 2, 0)  # [H,W,C]
#     flow_shape = flow1.shape
#     label_shape = img.shape
#     height, width = flow1.shape[0], flow1.shape[1]
#     device = flow1.device
#     output = torch.zeros(label_shape, device=device)  
#     flow_t = torch.zeros(flow_shape, device=device)  
#     # 完全按照原算法构建grid
#     h_grid = torch.arange(0, height, device=device)
#     w_grid = torch.arange(0, width, device=device)
#     h_grid = h_grid.repeat(width, 1).permute(1, 0)  
#     w_grid = w_grid.repeat(height, 1)  
#     grid = torch.stack((h_grid, w_grid), 0).permute(1, 2, 0).float()  
#     # 完全按照原算法
#     dx = grid[:, :, 0] + flow2[:, :, 1].long()
#     dy = grid[:, :, 1] + flow2[:, :, 0].long()
#     sx = torch.floor(dx.float())  
#     sy = torch.floor(dy.float())
#     valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)  
#     # 完全按照原算法的双线性插值
#     sx_mat = torch.stack((sx, sx + 1, sx, sx + 1), dim=2).clamp(0, height - 1)  
#     sy_mat = torch.stack((sy, sy, sy + 1, sy + 1), dim=2).clamp(0, width - 1)
#     sxsy_mat = torch.abs((1 - torch.abs(sx_mat - dx.float().unsqueeze(0).permute(1, 2, 0))) *
#                          (1 - torch.abs(sy_mat - dy.float().unsqueeze(0).permute(1, 2, 0))))
#     # 完全按照原算法的循环
#     for i in range(4):
#         flow_t = flow_t.long() + sxsy_mat.long()[:, :, i].unsqueeze(0).permute(1, 2, 0) * flow1.long()[
#                                                                                           sx_mat.long()[:, :, i],
#                                                                                           sy_mat.long()[:, :, i], :]
#     # 完全按照原算法的一致性检查
#     valid = valid & (
#                 torch.norm(flow_t.float()[:, :, [1, 0]] + torch.stack((dx.float(), dy.float()), dim=2) - grid.float(),
#                            dim=2, keepdim=True).squeeze(2) < 100)
#     # 完全按照原算法的残差处理
#     flow_t = (flow2.float() - flow_t.float()) / 2.0
#     dx = grid.long()[:, :, 0] + flow_t.long()[:, :, 1]
#     dy = grid.long()[:, :, 1] + flow_t.long()[:, :, 0]
#     valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)
#     # 完全按照原算法的最终采样
#     output[valid, :] = img.float()[dx[valid].long(), dy[valid].long(), :]
#     return output.permute(2, 0, 1)  # [C,H,W]
# import torch

import torch

def warp_single_step(flow1, flow2, img_chw):
    """
    原算法基础上，对最终采样做双线性重建 + border 填充（clamp）

    Args:
        flow1: [H,W,2] forward flow 
        flow2: [H,W,2] backward flow  
        img_chw: [C,H,W] 输入图像

    Returns:
        output: [C,H,W] warped 图像
    """

    # [C,H,W] → [H,W,C]
    img = img_chw.permute(1, 2, 0)
    H, W = flow1.shape[:2]
    device = flow1.device
    
    # 输出占位
    output = torch.zeros_like(img)
    
    # 临时累加的流场
    flow_t = torch.zeros_like(flow1)
    
    # 构造网格坐标（float）
    gy = torch.arange(0, H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
    gx = torch.arange(0, W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
    grid = torch.stack((gy, gx), dim=2)  # [H,W,2]
    
    # 第一次 warp：float 保留亚像素
    dx = grid[..., 0] + flow2[..., 1]
    dy = grid[..., 1] + flow2[..., 0]
    
    # floor（不直接 clamp，先检查有效性）
    sx = torch.floor(dx)
    sy = torch.floor(dy)
    
    # 有效性掩码（用于第一步的流场插值）
    valid_flow = (sx >= 0) & (sx < H-1) & (sy >= 0) & (sy < W-1)
    
    # 四邻域坐标（只在有效区域内 clamp）
    sx_clamped = sx.clamp(0, H-2)
    sy_clamped = sy.clamp(0, W-2)
    
    sx_mat = torch.stack([sx_clamped, sx_clamped+1, sx_clamped, sx_clamped+1], dim=2)
    sy_mat = torch.stack([sy_clamped, sy_clamped, sy_clamped+1, sy_clamped+1], dim=2)
    
    # 双线性权重（使用原始坐标计算，避免 clamp 带来的错误权重）
    wx = 1 - torch.abs(sx_mat - dx.unsqueeze(2))
    wy = 1 - torch.abs(sy_mat - dy.unsqueeze(2))
    w = wx * wy  # [H,W,4]
    
    # 只在有效区域累加 flow_t
    flow_t.zero_()
    for i in range(4):
        ix = sx_mat[..., i].long()
        iy = sy_mat[..., i].long()
        # 只在 valid_flow 区域累加
        mask = valid_flow.unsqueeze(2).expand(-1, -1, 2)
        flow_t += (w[..., i:i+1] * flow1[ix, iy]) * mask
    
    # 一致性检查
    diff = flow_t[..., [1,0]] + torch.stack((dx, dy), dim=2) - grid
    valid_consist = valid_flow & (diff.norm(dim=2) < 100)
    
    # 残差修正
    flow_t = (flow2 - flow_t) / 2.0
    
    # 最终采样位置（float）
    dx_final = grid[..., 0] + flow_t[..., 1]
    dy_final = grid[..., 1] + flow_t[..., 0]
    
    # 最终有效性检查
    valid_final = valid_consist & (dx_final >= 0) & (dx_final < H-1) & (dy_final >= 0) & (dy_final < W-1)
    
    # 最终采样：也使用双线性插值
    # 计算四邻域坐标
    sx_final = torch.floor(dx_final).clamp(0, H-2)
    sy_final = torch.floor(dy_final).clamp(0, W-2)
    
    # 四个邻居的坐标
    x0 = sx_final.long()
    x1 = (sx_final + 1).long().clamp(0, H-1)
    y0 = sy_final.long()
    y1 = (sy_final + 1).long().clamp(0, W-1)
    
    # 双线性权重
    wx = dx_final - sx_final
    wy = dy_final - sy_final
    
    # 四个邻居的权重
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy
    
    # 双线性插值采样（只在 valid_final 区域）
    mask = valid_final.unsqueeze(2).expand(-1, -1, img.shape[2])
    output = (
        w00.unsqueeze(2) * img[x0, y0] +
        w01.unsqueeze(2) * img[x0, y1] +
        w10.unsqueeze(2) * img[x1, y0] +
        w11.unsqueeze(2) * img[x1, y1]
    ) * mask
    
    # 可选：对于无效区域，保持原图像素值
    # output = output + img * (~mask)
    
    # [H,W,C] → [C,H,W]
    return output.permute(2, 0, 1)

def warp(
    img0: torch.Tensor,      # [B,C,H,W]
    flow01: torch.Tensor,    # [B,2,H,W] 
    flow10: torch.Tensor,    # [B,2,H,W]
    t: torch.Tensor,         # [B] 或 标量，范围 [0,1]
    total_steps: int | torch.Tensor = 13
) -> torch.Tensor:
    """
    渐进式累积warp - 完全基于你提供的原算法
    接口和之前完全一样
    
    Args:
        img0: 源图像 [B,C,H,W]
        flow01: 前向光流 [B,2,H,W]
        flow10: 后向光流 [B,2,H,W] 
        t: 插值比例 [0,1]
        total_steps: 总步数（对应原算法中的13步）
    
    Returns:
        插值结果 [B,C,H,W]
    """
    B, C, H, W = img0.shape
    device = img0.device
    # —— 将 total_steps 规范成 [B] 的 Int Tensor —— 
    if not torch.is_tensor(total_steps):
        # 用户传了单个 int
        ts = torch.full((B,), int(total_steps), device=device, dtype=torch.int64)
    else:
        # 用户传了 Tensor
        ts = total_steps.to(device)
        if ts.dim() == 0:
            ts = ts.expand(B)
        ts = ts.to(torch.int64)
    
    # 处理t的维度
    if t.dim() == 0:
        t = t.expand(B)
    
    outputs = []
    
    for b in range(B):
        img_single = img0[b]  # [C,H,W]
        flow_01 = flow01[b].permute(1, 2, 0)  # [H,W,2]
        flow_10 = flow10[b].permute(1, 2, 0)  # [H,W,2]
        t_val = float(t[b].clamp(0.0, 1.0).item()) # 确保在 [0,1]
        ts_b  = int(ts[b].item())                  # 本样本的步数
        
        # 计算实际需要的步数
        target_steps = int(round(t_val * ts_b))
        # 防止超范围
        target_steps = max(0, min(target_steps, ts_b))
        step_flow1   = flow_01 / ts_b
        step_flow2   = flow_10 / ts_b
        
        if target_steps == 0:
            # t=0，直接返回原图
            outputs.append(img_single)
            continue
        
        # # 完全按照原算法：每步光流 = 总光流 / 总步数
        # step_flow1 = flow_01 / total_steps  
        # step_flow2 = flow_10 / total_steps
        
        # 完全按照原算法的渐进式warp循环
        result = img_single  # 从原图开始
        for i in range(target_steps):
            # 每一步都用相同的step_flow对上一步结果进行warp
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
