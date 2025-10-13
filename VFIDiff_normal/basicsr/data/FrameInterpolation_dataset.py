# datapipe/datasets/frame_interpolation_dataset.py

import os
import random
import time
from pathlib import Path

import cv2
import torch
from torch.utils import data
import albumentations

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register(suffix='frame_interpolation')
class FrameInterpolationDataset(data.Dataset):
    """
    Dataset for Frame Interpolation Task:
    Reads GT, LQ, and Mid-frame image paths from three separate text files.
    Each line in the text files corresponds to one sample.

    Args:
        opt (dict): Configuration dictionary containing:
            gt_txt_file (str): Path to the text file listing GT image paths.
            lq_txt_file (str): Path to the text file listing LQ image paths.
            mid_txt_file (str): Path to the text file listing Mid-frame image paths.
            io_backend (dict): IO backend configuration.
            use_hflip (bool): Whether to use horizontal flips during training.
            use_rot (bool): Whether to use rotations during training.
            rescale_gt (bool): Whether to rescale GT images.
            crop_pad_size (int): Size to crop or pad images to during training.
            mode (str): 'training' or 'testing'.
            # ... other options as needed
    """

    def __init__(self, opt, mode='training'):
        super(FrameInterpolationDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.rescale_gt = opt.get('rescale_gt', False)
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()


        # Read GT, LQ, and Mid-frame paths from respective text files
        self.gt_paths = self._read_txt(opt['gt_txt_file'])
        self.lq_paths = self._read_txt(opt['lq_txt_file'])
        self.mid_paths = self._read_txt(opt['mid_txt_file'])
        self.im2_paths = self._read_txt(opt['im2_txt_file'])
        self.im3_paths = self._read_txt(opt['im3_txt_file'])
        self.im5_paths = self._read_txt(opt['im5_txt_file'])
        self.im6_paths = self._read_txt(opt['im6_txt_file'])
        # Ensure that all three lists have the same length
        assert len(self.gt_paths) == len(self.lq_paths) == len(self.mid_paths), \
            "GT, LQ, and Mid-frame lists must have the same length."

        # Optionally limit the dataset size
        if 'length' in opt:
            length = opt['length']
            if length < len(self.gt_paths):
                indices = random.sample(range(len(self.gt_paths)), length)
                self.gt_paths = [self.gt_paths[i] for i in indices]
                self.lq_paths = [self.lq_paths[i] for i in indices]
                self.mid_paths = [self.mid_paths[i] for i in indices]
                self.im2_paths = [self.im2_paths[i] for i in indices]
                self.im3_paths = [self.im3_paths[i] for i in indices]
                self.im5_paths = [self.im5_paths[i] for i in indices]
                self.im6_paths = [self.im6_paths[i] for i in indices]


        # Define augmentation transforms for testing
        if self.mode == 'testing':
            self.test_aug = albumentations.Compose([
                #按照保持宽高比的原则，把图像缩放，使得它的短边（最小边）恰好等于 gt_size（opt['gt_size']），长边会按比例放大或缩小到不小于 gt_size。
                albumentations.SmallestMaxSize(max_size=opt['gt_size']),
                #再从缩放后的图像正中间裁下一个大小为 gt_size×gt_size 的正方形。
                albumentations.CenterCrop(opt['gt_size'], opt['gt_size']),
            ])

        # Define augmentation options
        self.use_hflip = opt.get('use_hflip', False)
        self.use_rot = opt.get('use_rot', False)

    def _read_txt(self, txt_file):
        """Reads a text file and returns a list of stripped lines."""
        with open(txt_file, 'r') as f:
            paths = [line.strip() for line in f.readlines()]
        return paths

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load GT, LQ, and Mid-frame images
        gt_path = self.gt_paths[index]
        lq_path = self.lq_paths[index]
        mid_path = self.mid_paths[index]
        im2_path = self.im2_paths[index]
        im3_path = self.im3_paths[index]
        im5_path = self.im5_paths[index]
        im6_path = self.im6_paths[index]


        # Function to load image with retry mechanism
        def load_image(path):
            retry = 3
            while retry > 0:
                try:
                    img_bytes = self.file_client.get(path, 'image')
                    img = imfrombytes(img_bytes, float32=True)
                    return img
                except Exception as e:
                    # logger.warning(f'Error loading {path}: {e}. Retries left: {retry-1}')
                    retry -= 1
                    time.sleep(1)
            raise IOError(f"Failed to load image after 3 retries: {path}")

        img_gt = load_image(gt_path)
        img_lq = load_image(lq_path)
        img_mid = load_image(mid_path)
        img_2 = load_image(im2_path)
        img_3 = load_image(im3_path)
        img_5 = load_image(im5_path)
        img_6 = load_image(im6_path)

        # Apply testing augmentations
        if self.mode == 'testing':
            img_gt = self.test_aug(image=img_gt)['image']
            img_lq = self.test_aug(image=img_lq)['image']
            img_mid = self.test_aug(image=img_mid)['image']
        elif self.mode == 'training':
            # Apply random horizontal flip and rotation
            img_gt, img_lq, img_mid,img_2,img_3,img_5,img_6 = augment([img_gt, img_lq, img_mid,img_2,img_3,img_5,img_6], True, False)

            # Define crop or pad size
            h, w = img_gt.shape[:2]
            crop_pad_size = self.opt.get('crop_pad_size', 400) if not self.rescale_gt else max(min(h, w), self.opt.get('gt_size', 256))

            # -------------------- Pad images if smaller than crop_pad_size -------------------- #
            while h < crop_pad_size or w < crop_pad_size:
                pad_h = min(max(0, crop_pad_size - h), h)
                pad_w = min(max(0, crop_pad_size - w), w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_mid = cv2.copyMakeBorder(img_mid, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_2 = cv2.copyMakeBorder(img_2, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_3 = cv2.copyMakeBorder(img_3, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_5 = cv2.copyMakeBorder(img_5, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_6 = cv2.copyMakeBorder(img_6, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                h, w = img_gt.shape[:2]

            # -------------------- Crop images if larger than crop_pad_size -------------------- #
            if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
                top = random.randint(0, img_gt.shape[0] - crop_pad_size)
                left = random.randint(0, img_gt.shape[1] - crop_pad_size)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, :]
                img_lq = img_lq[top:top + crop_pad_size, left:left + crop_pad_size, :]
                img_mid = img_mid[top:top + crop_pad_size, left:left + crop_pad_size, :]
                img_2 = img_2[top:top + crop_pad_size, left:left + crop_pad_size, :]
                img_3 = img_3[top:top + crop_pad_size, left:left + crop_pad_size, :]
                img_5 = img_5[top:top + crop_pad_size, left:left + crop_pad_size, :]
                img_6 = img_6[top:top + crop_pad_size, left:left + crop_pad_size, :]

            # -------------------- Rescale GT, LQ, and Mid-frame images if required -------------------- #
            if self.rescale_gt and crop_pad_size != self.opt.get('gt_size', 256):
                img_gt = cv2.resize(img_gt, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)
                img_lq = cv2.resize(img_lq, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)
                img_mid = cv2.resize(img_mid, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)
                img_2 = cv2.resize(img_2, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)
                img_3 = cv2.resize(img_3, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)
                img_5 = cv2.resize(img_5, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)
                img_6 = cv2.resize(img_6, (self.opt['gt_size'], self.opt['gt_size']), interpolation=cv2.INTER_AREA)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Convert images from BGR to RGB, HWC to CHW, and numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_mid = img2tensor([img_mid], bgr2rgb=True, float32=True)[0]
        img_2 = img2tensor([img_2], bgr2rgb=True, float32=True)[0]
        img_3 = img2tensor([img_3], bgr2rgb=True, float32=True)[0]
        img_5 = img2tensor([img_5], bgr2rgb=True, float32=True)[0]
        img_6 = img2tensor([img_6], bgr2rgb=True, float32=True)[0]

        return {
            'gt': img_gt,
            'lq': img_lq,
            'mid': img_mid,
            'im2': img_2,
            'im3': img_3,
            'im5': img_5,
            'im6': img_6, 
            'gt_path': gt_path,
            'lq_path': lq_path,
            'mid_path': mid_path
        }
