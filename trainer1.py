
import contextlib
import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn.parallel import DistributedDataParallel as DDP
from FlowformerPlusPlus.core.FlowFormer import build_flowformer  # 确保路径正确
from FlowformerPlusPlus.core.utils.utils import InputPadder, forward_interpolate
from FlowformerPlusPlus.core.utils import frame_utils, flow_viz
# from visualize import compute_optical_flow  # 导入修改后的光流函数
# from softmax_splatting_master.run import compute_optical_flow
# from softmax_splatting_master.run import warp
# from RAFT_SOFT.new_interface import compute_optical_flow_b as compute_optical_flow,warp_b_s as warp,warp_b as warp_ns

# from RAFT.raft_interface import compute_optical_flow,warp
from FlowformerPlusPlus.configs.submissions import get_cfg as get_submission_cfg
# import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import os
environment_vars = ["LOCAL_RANK", "WORLD_SIZE"]
# import cupy
import sys
import torchvision
# sys.path.append('/root/autodl-tmp/VFIDiff-journal/ResShift-journal/RAFT_SOFT')
# sys.path.append('/root/autodl-tmp/VFIDiff-journal/ResShift-journal/RAFT_SOFT/utils')
# from RAFT_SOFT.softmax_splatting import softsplat
# from RIFE.inference_img import batch_interpolate
from FlowformerPlusPlus.visualize_flow import warp,compute_optical_flow, build_model
def print_flow_stats(flow):
    """
    打印光流统计信息
    flow: [B, 2, H, W] 的tensor
    """
    B, _, H, W = flow.shape
    
    for b in range(B):
        # 提取当前样本的光流
        flow_x = flow[b, 0]  # [H, W]
        flow_y = flow[b, 1]  # [H, W]
        
        # 计算统计信息
        x_min, x_max = flow_x.min().item(), flow_x.max().item()
        y_min, y_max = flow_y.min().item(), flow_y.max().item()
        
        print(f"Sample {b}:")
        print(f"  Flow X: min={x_min:.4f}, max={x_max:.4f}")
        print(f"  Flow Y: min={y_min:.4f}, max={y_max:.4f}")
        print(f"  Magnitude: min={torch.sqrt(flow_x**2 + flow_y**2).min().item():.4f}, "
              f"max={torch.sqrt(flow_x**2 + flow_y**2).max().item():.4f}")
        print()
def check_tensor_range(tensor: torch.Tensor, name: str):
    """打印张量的最小值、最大值，并简单判断其范围。"""
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    print(f"{name} range: [{t_min:.4f}, {t_max:.4f}]")

    # 简单判断是否在[0,1]或[-1,1]内
    if t_min >= 0.0 and t_max <= 1.0:
        print(f"  ==> {name} is likely in the [0, 1] range.\n")
    elif t_min >= -1.0 and t_max <= 1.0:
        print(f"  ==> {name} is likely in the [-1, 1] range.\n")
    else:
        print(f"  ==> {name} is outside the typical [0,1] or [-1,1] range.\n")


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        self.tic = None
        self.toc = None
        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()
        # self.model = None
        # self.frozen_model = None

        local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 自动获取当前进程对应的 GPU
        torch.cuda.set_device(local_rank)  # 设置当前 GPU
        if not dist.is_initialized():  # 确保进程组未被初始化
            dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f"cuda:{local_rank}")

        # # model = build_flowformer(cfg)  # 构建模型
        # cfg = get_submission_cfg()
        # cfg.update({
        #     'model': 'FlowformerPlusPlus/checkpoints/sintel.pth'})  # 假设 flow_model_path 在 config 中
        # # model = build_flowformer(cfg).to(device)
        # # model.load_state_dict(torch.load(cfg.model))
        # # self.flow_model = torch.nn.DataParallel(model)
        # # self.flow_model.eval()
        # # 假设 checkpoint 里的 key 带了 module. 前缀：
        # ckpt = torch.load(cfg.model, map_location='cpu')
        # # 1) 统一去掉 module. 前缀
        # new_sd = {}
        # for k, v in ckpt.items():
        #     name = k
        #     if name.startswith('module.'):
        #         name = name[len('module.'):]
        #     new_sd[name] = v
        
        # # 2) 构建裸模型，load 去完 module. 的 state_dict
        # base_model = build_flowformer(cfg).to(device)
        # base_model.load_state_dict(new_sd)
        
        # # 3) （可选）如果你后面想用 DataParallel，就 wrap 一下
        # self.flow_model = torch.nn.DataParallel(base_model)
        # self.flow_model.eval()

    def setup_dist(self):

        if not dist.is_initialized():
            # 仅在没有初始化时进行初始化
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):#????
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module. ' + key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            # torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start + 1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_ " + Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)
            # torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        # self.optimizer = torch.optim.AdamW(list(self.model[0].parameters())+list(self.model[1].parameters()),
        #                                    lr=self.configs.train.lr,
        #                                    weight_decay=self.configs.train.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        # model = [util_common.get_obj_from_str(self.configs.model.target)(**params) for i in range(2)]
        # for i in range(2):
        #     model[i].cuda()
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        # frozen_model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        # frozen_model.cuda()
        
        # assert (self.configs.model.ckpt_path is None and
        #         (self.configs.model.ckpt_path_1 is not None and self.configs.model.ckpt_path_2 is not None)) \
        #         or self.configs.model.ckpt_path is not None

        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
            # for i in range(2):
            #     util_net.reload_model(model[i], ckpt)
        # else:
        #     ckpt_path_1 = self.configs.model.ckpt_path_1
        #     ckpt_path_2 = self.configs.model.ckpt_path_2
        #     if self.rank == 0:
        #         self.logger.info(f"Initializing model from {ckpt_path_1}")
        #         self.logger.info(f"Initializing model from {ckpt_path_2}")
        #     ckpt_1 = torch.load(ckpt_path_1, map_location=f"cuda:{self.rank}")
        #     ckpt_2 = torch.load(ckpt_path_2, map_location=f"cuda:{self.rank}")
        #     assert 'state_dict' in ckpt_1 and 'state_dict' in ckpt_2
        #     ckpt_1 = ckpt_1['state_dict']
        #     ckpt_2 = ckpt_2['state_dict']
        #     util_net.reload_model(model[0], ckpt_1)
        #     util_net.reload_model(model[1] ,ckpt_2)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            # model = [torch.compile(model[i], mode=self.configs.train.compile.mode) for i in range(2)]
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.world_size > 1:
            # self.model = [DDP(model[i], device_ids=[self.rank ,], static_graph=False) for i in range(2)]  # wrap the network
            # self.model = DDP(model, device_ids=[self.rank, ], static_graph=False)  # wrap the network
            self.model = model
        else:
            self.model = model

        # if hasattr(self.configs.model, 'ckpt1_path') and self.configs.model.ckpt1_path is not None:
        #     # print("yes")
        #     ckpt1_path = self.configs.model.ckpt1_path
        #     if self.rank == 0:
        #         self.logger.info(f"Initializing frozen model from {ckpt1_path}")
        #     ckpt1 = torch.load(ckpt1_path, map_location=f"cuda:{self.rank}")
        #     if 'state_dict' in ckpt1:
        #         ckpt1 = ckpt1['state_dict']
        #     util_net.reload_model(frozen_model, ckpt1)
        #     if self.world_size > 1:
        #         # self.model = [DDP(model[i], device_ids=[self.rank ,], static_graph=False) for i in range(2)]  # wrap the network
        #         # self.model = DDP(model, device_ids=[self.rank, ], static_graph=False)  # wrap the network
        #         self.frozen_model = frozen_model
        #     else:
        #         self.frozen_model = frozen_model

        #     # 冻结 frozen 所有参数
        #     for p in frozen_model.parameters():
        #         p.requires_grad = False
        #     frozen_model.eval()

        # else:
        #     # 如果 config 中没有提供 ckpt1_path，则不构建 frozen_model
        #     # print("No")
        #     self.frozen_model = None
        #     if self.rank == 0:
        #         self.logger.warning("No ckpt1_path found in configs.model，frozen_model 未初始化。")

        # EMA ????
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            # self.ema_model = deepcopy(model)
            # self.ema_model = deepcopy(model[0]).cuda()
            # self.ema_state = OrderedDict(
            #     {key :deepcopy(value.data) for key, value in self.model[0].state_dict().items()}
            #     )
            self.ema_state = OrderedDict(
                {key: deepcopy(value.data) for key, value in self.model.state_dict().items()}
            )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {}
        # datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        # print(self.configs.data.get('train', dict))
        datasets['train'] = create_dataset(self.configs.data.get('train', dict))
        print(">>> dataset class is:", type(datasets['train']).__module__,
              type(datasets['train']).__name__)
        # if hasattr(self.configs.data, 'val') and self.rank == 0 :
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))
        # make dataloaders
        if self.world_size > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets['train'],
                # num_replicas=self.num_gpus,
                num_replicas=self.world_size,
                rank=self.rank,
            )
        else:
            sampler = None
        # dataloaders = {'train': _wrap_loader(udata.DataLoader(
        #                 datasets['train'],
        #                 batch_size=self.configs.train.batch[0] // self.num_gpus,
        #                 shuffle=False if self.num_gpus > 1 else True,
        #                 drop_last=True,
        #                 num_workers=min(self.configs.train.num_workers, 4),
        #                 pin_memory=True,
        #                 prefetch_factor=self.configs.train.get('prefetch_factor', 2),
        #                 worker_init_fn=my_worker_init_fn,
        #                 sampler=sampler,
        #                 ))}
        dataloaders = {
            'train': _wrap_loader(udata.DataLoader(
                datasets['train'],
                batch_size=self.configs.train.batch[0] // self.world_size,
                shuffle=(sampler is None),  # 非分布式时启用 shuffle
                sampler=sampler,
                num_workers=min(self.configs.train.num_workers, 4),
                pin_memory=True,
                drop_last=True,
                worker_init_fn=my_worker_init_fn,
                # prefetch_factor=self.configs.train.get('prefetch_factor', 2),
            ))
        }

        if hasattr(self.configs.data, 'val') and self.rank == 0:
            # 定义缩放变换
            resize_transform = transforms.Compose([
                transforms.ToPILImage(),  # 转换为 PIL 图像
                transforms.Resize((256, 256)),  # 缩放至 96×96
                transforms.ToTensor()  # 转换为 PyTorch Tensor
            ])

            # 自定义批量缩放函数
            def resize_batch(batch):
                resized_batch = {
                    'gt': [],
                    'lq': [],
                    'mid': [],
                    # 'im2': [],
                    # 'im3': [],
                    # 'im5': [],
                    # 'im6': [],
                    'gt_path': [],
                    'lq_path': [],
                    'mid_path': []
                }
                for item in batch:
                    # print("image!")
                    # print(item['gt'].shape)
                    # 将图像范围从 [0, 1] 映射到 [0, 255]，然后转换为 uint8
                    gt_image = (item['gt'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    lq_image = (item['lq'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    mid_image = (item['mid'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # im2_image = (item['im2'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # im3_image = (item['im3'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # im5_image = (item['im5'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # im6_image = (item['im6'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    # 对 GT、LQ 和 Mid 图像进行缩放
                    resized_batch['gt'].append(resize_transform(gt_image))
                    resized_batch['lq'].append(resize_transform(lq_image))
                    resized_batch['mid'].append(resize_transform(mid_image))
                    # resized_batch['im2'].append(resize_transform(im2_image))
                    # resized_batch['im3'].append(resize_transform(im3_image))
                    # resized_batch['im5'].append(resize_transform(im5_image))
                    # resized_batch['im6'].append(resize_transform(im6_image))
                    resized_batch['gt_path'].append(item['gt_path'])
                    resized_batch['lq_path'].append(item['lq_path'])
                    resized_batch['mid_path'].append(item['mid_path'])

                # 将列表中的数据堆叠为批次张量
                resized_batch['gt'] = torch.stack(resized_batch['gt'])
                resized_batch['lq'] = torch.stack(resized_batch['lq'])
                resized_batch['mid'] = torch.stack(resized_batch['mid'])
                # resized_batch['im2'] = torch.stack(resized_batch['im2'])
                # resized_batch['im3'] = torch.stack(resized_batch['im3'])
                # resized_batch['im5'] = torch.stack(resized_batch['im5'])
                # resized_batch['im6'] = torch.stack(resized_batch['im6'])

                return resized_batch

            # 包装 DataLoader
            dataloaders['val'] = udata.DataLoader(
                datasets['val'],
                batch_size=self.configs.train.batch[1],
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=resize_batch,  # 应用自定义缩放
            )
            # for idx, data in enumerate(dataloaders['val']):
            #     if 'lq' in data:
            #         print(f"Batch {idx}: LQ Image Sizes - {[img.shape for img in data['lq']]}")
            #     if 'gt' in data:
            #         print(f"Batch {idx}: GT Image Sizes - {[img.shape for img in data['gt']]}")
            #     if idx >= 5:  # 示例：只输出前5个批次
            #         break

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            # num_params = util_net.calculate_parameters(self.model[0]) / 1000**2
            # num_params_1 = util_net.calculate_parameters(self.model[1]) / 1000**2
            num_params = util_net.calculate_parameters(self.model) / 1000 ** 2

            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")
            # self.logger.info(f"Number of parameters_1: {num_params_1:.2f}M")

    # def prepare_data(self, data, dtype=torch.float32, phase='train'):
    #     data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
    #     return data
    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        device = torch.cuda.current_device()
        to_kwargs = dict(device=device, dtype=dtype, non_blocking=True)
        if phase == 'train':
            data = {
                'gt': ((data['gt'] - 0.5) * 2.0).to(**to_kwargs),
                'lq': ((data['lq'] - 0.5) * 2.0).to(**to_kwargs),
                'mid': ((data['mid'] - 0.5) * 2.0).to(**to_kwargs),
                'im2': ((data['im2'] - 0.5) * 2.0).to(**to_kwargs),
                'im3': ((data['im3'] - 0.5) * 2.0).to(**to_kwargs),
                'im5': ((data['im5'] - 0.5) * 2.0).to(**to_kwargs),
                'im6': ((data['im6'] - 0.5) * 2.0).to(**to_kwargs),
                'gt_path': data['gt_path'],
                'lq_path': data['lq_path'],
                'mid_path': data['mid_path']
            }
        else:
            data = {
                'gt': ((data['gt'] - 0.5) * 2.0).to(**to_kwargs),
                'lq': ((data['lq'] - 0.5) * 2.0).to(**to_kwargs),
                'mid': ((data['mid'] - 0.5) * 2.0).to(**to_kwargs),
                'gt_path': data['gt_path'],
                'lq_path': data['lq_path'],
                'mid_path': data['mid_path']
            }
        return data

    # def prepare_data(self, data, dtype=torch.float32, phase='train'):
    #     """确保保留路径信息用于光流缓存查找"""
    #     processed_data = {}
    #
    #     # 处理张量数据
    #     tensor_keys = ['gt', 'lq', 'mid']
    #     for key in tensor_keys:
    #         if key in data:
    #             processed_data[key] = data[key].cuda().to(dtype=dtype)
    #
    #     # 处理路径信息
    #     path_keys = ['gt_path', 'lq_path', 'mid_path']
    #     for key in path_keys:
    #         if key in data:
    #             processed_data[key] = data[key]  # 直接保留路径信息，不做转换
    #
    #     return processed_data
    # def prepare_data(self, data, dtype=torch.float32, phase='train'):
    #     """确保保留路径信息用于光流缓存查找"""
    #     processed_data = {}
    #
    #     # 处理张量数据
    #     tensor_keys = ['gt', 'lq', 'mid']
    #     for key in tensor_keys:
    #         if key in data and data[key] is not None:
    #             processed_data[key] = data[key].cuda().to(dtype=dtype)
    #
    #     # 处理路径信息
    #     path_keys = ['gt_path', 'lq_path', 'mid_path']
    #     for key in path_keys:
    #         if key in data and data[key] is not None:
    #             processed_data[key] = data[key]  # 保留路径信息
    #
    #     # 添加断言以确保训练阶段保留了必要的路径字段
    #     if phase == 'train':
    #         assert 'lq_path' in processed_data, "训练数据缺少 'lq_path'"
    #         assert 'gt_path' in processed_data, "训练数据缺少 'gt_path'"
    #
    #     return processed_data
    #
    # def get_cached_flow(self, lq_path, gt_path):
    #     """获取缓存的光流"""
    #     key = f"{hash(lq_path + '_' + gt_path)}"
    #     flow_path = self.flow_cache_dir / f"{key}.pth"
    #
    #     if flow_path.exists():
    #         flows = torch.load(flow_path, map_location=f'cuda:{self.rank}')  # 确保加载到正确的GPU
    #         # 验证路径匹配
    #         if flows['lq_path'] == lq_path and flows['gt_path'] == gt_path:
    #             return flows['flow0'].cuda(self.rank), flows['flow1'].cuda(self.rank)
    #     return None, None

    def validation(self):
        pass

    def train(self):
        self.init_logger()  # setup logger: self.logger
        # torch.backends.cudnn.enabled = False
        self.build_model()  # build model: self.model, self.loss
        self.setup_optimizaton()  # setup optimization: self.optimzer, self.sheduler
        self.resume_from_ckpt()  # resume if necessary
        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler
        # torch.autograd.set_detect_anomaly(True)
        # for i in range(2):
        #     self.model[i].train()
        self.model.train()

        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])

        # 使用 tqdm 包装训练循环
        progress_bar = tqdm(range(self.iters_start, self.configs.train.iterations), desc="Training", unit="iter")

        for ii in progress_bar:
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)

            # validation phase
            if 'val' in self.dataloaders and (ii + 1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            # update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii + 1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii + 1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii + 1)

            # 更新 tqdm 的描述信息
            if self.rank == 0:
                log_str = f"Iter {self.current_iters}/{self.configs.train.iterations} | "
                log_str += f"Loss: {self.loss_mean['mse'][0]:.4f}  | "
                progress_bar.set_postfix_str(log_str)

        # close the tensorboard
        self.close_logger()

    #
    # def train(self):
    #     self.init_logger()       # setup logger: self.logger
    #
    #     self.build_model()       # build model: self.model, self.loss
    #
    #     self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler
    #
    #     self.resume_from_ckpt()  # resume if necessary
    #
    #     self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler
    #
    #     self.model.train()
    #     num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
    #     for ii in range(self.iters_start, self.configs.train.iterations):
    #         self.current_iters = ii + 1
    #
    #         # prepare data
    #         data = self.prepare_data(next(self.dataloaders['train']))
    #
    #         # training phase
    #         self.training_step(data)
    #
    #         # validation phase
    #         if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
    #             self.validation()
    #
    #         #update learning rate
    #         self.adjust_lr()
    #
    #         # save checkpoint
    #         if (ii+1) % self.configs.train.save_freq == 0:
    #             self.save_ckpt()
    #
    #         if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
    #             self.sampler.set_epoch(ii+1)
    #
    #     # close the tensorboard
    #     self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    # def save_ckpt(self):
    #     if self.rank == 0:
    #         for i in range(2):
    #             ckpt_path = self.ckpt_dir / 'model_{:d}_{}.pth'.format(self.current_iters,i+1)
    #             ckpt = {
    #                     'iters_start': self.current_iters,
    #                     'log_step': {phase :self.log_step[phase] for phase in ['train', 'val']},
    #                     'log_step_img': {phase :self.log_step_img[phase] for phase in ['train', 'val']},
    #                     'state_dict': self.model[i].state_dict(),
    #                     }
    #             if self.amp_scaler is not None:
    #                 ckpt['amp_scaler'] = self.amp_scaler.state_dict()
    #             torch.save(ckpt, ckpt_path)
    #         if hasattr(self, 'ema_rate'):
    #             ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
    #             torch.save(self.ema_state, ema_ckpt_path)
    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            ckpt = {
                'iters_start': self.current_iters,
                'log_step': {phase: self.log_step[phase] for phase in ['train', 'val']},
                'log_step_img': {phase: self.log_step_img[phase] for phase in ['train', 'val']},
                'state_dict': self.model.state_dict(),
            }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.world_size > 1:
                # model_state = {key[7:] :value for key, value in self.ema_state.items()}
                model_state = self.ema_state
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.world_size > 1:
            dist.barrier()
        if self.rank == 0:
            # source_state = self.model[0].state_dict()
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1 - rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        if self.rank != 0:
            return  # 仅 rank 0 记录图像

        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True)
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1, 2, 0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                f"{phase}-{tag}-{self.log_step_img[phase]}",
                im_tensor,
                self.log_step_img[phase],
            )
        if add_global_step:
            self.log_step_img[phase] += 1

    # def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
    #     """
    #     Args:
    #         im_tensor: b x c x h x w tensor
    #         im_tag: str
    #         phase: 'train' or 'val'
    #         nrow: number of displays in each row
    #     """
    #     assert self.tf_logging or self.local_logging
    #     im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
    #     if self.local_logging:
    #         im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
    #         im_np = im_tensor.cpu().permute(1,2,0).numpy()
    #         util_image.imwrite(im_np, im_path)
    #     if self.tf_logging:
    #         self.writer.add_image(
    #                 f"{phase}-{tag}-{self.log_step_img[phase]}",
    #                 im_tensor,
    #                 self.log_step_img[phase],
    #                 )
    #     if add_global_step:
    #         self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def load_model(self, model, ckpt_path=None):
        if self.rank == 0:
            self.logger.info(f'Loading from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            self.logger.info('Loaded Done')

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False


class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                eta_min=self.configs.train.lr_min,
            )

    def build_model(self):
        super().build_model()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder

        else:
            self.autoencoder = None

        # LPIPS metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LIIPS Metric: {lpips_net}...")

        # lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
        lpips_loss = lpips.LPIPS(net=lpips_net).to(self.rank)  # 自动映射到正确设备

        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

        if hasattr(self, 'lpips_loss'):
            self.lpips_loss = self.lpips_loss.to(f"cuda:{self.rank}")

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b * 10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    # def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            # micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            micro_data = {key: value[jj:jj + micro_batchsize] if isinstance(value, torch.Tensor) else value[jj:jj + micro_batchsize] for key, value in data.items()}
            last_batch = (jj + micro_batchsize >= current_batchsize)

            # tt = torch.randint(
            #         0, self.base_diffusion.num_timesteps,
            #         size=(micro_data['gt'].shape[0],),
            #         device=f"cuda:{self.rank}",
            #         )
            tt = torch.randint(
                1,
                self.base_diffusion.num_timesteps,
                size=(micro_data['gt'].shape[0],),
                device=f"cuda:{self.rank}",
            )
            batch_size = micro_data['gt'].shape[0]
            # tt = torch.full(
            #     (batch_size,),
            #     6, # 7                          # 强制所有 time‐step 都是 6
            #     dtype=torch.int64,
            #     device=f"cuda:{self.rank}",
            # )


            mid = self.base_diffusion.num_timesteps // 2

            # 如果随机生成的 tt 中没有出现 mid，就把第 0 个元素强行设置为 mid
            # if not (tt == mid).any():
            #     tt[0] = mid
            # if not (tt == self.base_diffusion.num_timesteps-1).any():
            #     tt[1] = self.base_diffusion.num_timesteps-1
            latent_downsamping_sf = 2 ** (len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                size=(micro_data['gt'].shape[0], noise_chn,) + (latent_resolution,) * 2,
                device=micro_data['gt'].device,
            )
            # flow0 = compute_optical_flow(((micro_data['gt']+1)*127.5), ((micro_data['lq']+1)*127.5))
            # flow1 = compute_optical_flow(((micro_data['lq']+1)*127.5), ((micro_data['gt']+1)*127.5))
            flowa0 = compute_optical_flow((micro_data['gt']+1)*127.5,(micro_data['mid']+1)*127.5)
            flowa1 = compute_optical_flow((micro_data['mid']+1)*127.5,(micro_data['gt']+1)*127.5)
            flowb0 = compute_optical_flow((micro_data['mid']+1)*127.5,(micro_data['lq']+1)*127.5)
            flowb1 = compute_optical_flow((micro_data['lq']+1)*127.5,(micro_data['mid']+1)*127.5)

            # print_flow_stats(flowa0)
            # print_flow_stats(flowa1)
            # print_flow_stats(flowb0)
            # print_flow_stats(flowb1)
            

            if self.configs.model.params.cond_lq:
                # model_kwargs = {'lq': micro_data['lq'], 'flow0': flow0, 'flow1': flow1,'flow0_original': flow0, 'flow1_original': flow1,"tenMean":tenMean,"tenStd":tenStd,"tenOne_n":tenOne_n, "tenTwo_n":tenTwo_n,"first_stage_model":self.autoencoder}
                model_kwargs = {'lq': micro_data['lq'], "first_stage_model": self.autoencoder}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                # model_kwargs = {'flow0': flow0, 'flow1': flow1,'flow0_original': flow0, 'flow1_original': flow1,"tenMean":tenMean,"tenStd":tenStd,"tenOne_n":tenOne_n, "tenTwo_n":tenTwo_n,"first_stage_model":self.autoencoder}
                model_kwargs = {"first_stage_model": self.autoencoder}

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                # self.flow_model, 
                micro_data['gt'],
                micro_data['lq'],
                micro_data['mid'],
                micro_data['im2'],
                micro_data['im3'],
                micro_data['im5'],
                micro_data['im6'],
                tt,
                flowa0,
                flowa1,
                flowb0,
                flowb1,
                rank=self.rank,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.world_size <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate)

                # losses_1, z0_pred_1, z_t_1 = self.backward_step(compute_losses[0], micro_data, num_grad_accumulate)
                # losses_2, z0_pred_2, z_t_2 = self.backward_step(compute_losses[1], micro_data, num_grad_accumulate)
            else:
                # with contextlib.ExitStack() as stack:
                #     for model in self.model:
                #         stack.enter_context(model.no_sync())
                #     losses_1, z0_pred_1, z_t_1 = self.backward_step(compute_losses[0], micro_data, num_grad_accumulate)
                #     losses_2, z0_pred_2, z_t_2 = self.backward_step(compute_losses[1], micro_data, num_grad_accumulate)
                # with self.model.no_sync():
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate)

            # make logging
            if last_batch:
                self.log_step_train(losses, tt.clone().detach(), micro_data, z_t, z0_pred.detach())
                # self.log_step_train(losses_1,losses_2, tt.clone().detach(), micro_data, z_t_1, z0_pred_1.detach())

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        # for i in range(2):
        #     self.model[i].zero_grad()
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [2, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                # self.loss_mean1 = {key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                #                   for key in loss1.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                # for key, value in loss1.items():
                #     index = record_steps[jj] - 1
                #     mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                #     current_loss = torch.sum(value.detach() * mask)
                #     self.loss_mean1[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                # for key in loss1.keys():
                #     self.loss_mean1[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                    self.current_iters,
                    self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                        current_record,
                        self.loss_mean['loss'][jj].item(),
                        self.loss_mean['mse'][jj].item(),
                    )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                    self.base_diffusion._scale_input(z_t, tt),
                    self.autoencoder,
                )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                x_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                )
                self.logging_image(x_pred, tag='x_pred(t-1)', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            # if self.current_iters % self.configs.train.save_freq == 0:
            #     self.toc = time.time()
            #     elaplsed = (self.toc - self.tic)
            #     self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
            #     self.logger.info("="*100)
            if self.current_iters % self.configs.train.save_freq == 0:
                if self.tic is not None:
                    self.toc = time.time()
                    elaplsed = (self.toc - self.tic)
                    self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                else:
                    self.logger.warning("'tic' is None. Elapsed time cannot be calculated.")
                self.logger.info("=" * 100)

    def validation(self, phase='val'):

        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                # for i in range(2):
                #     self.model[i].eval()
                self.model.eval()

            indices = [0, 2, 5, 8, 11]

            # indices = np.linspace(
            #     0,
            #     self.base_diffusion.num_timesteps,
            #     self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
            #     endpoint=False,
            #     dtype=np.int64,
            # ).tolist()
            # if not (self.base_diffusion.num_timesteps - 1) in indices:
            #     indices.append(self.base_diffusion.num_timesteps - 2)
            # indices.append(self.base_diffusion.num_timesteps)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            mean_psnr_mid = 0
            mean_lpips_mid = 0
            mid_step = self.base_diffusion.num_timesteps // 2

            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                # check_tensor_range(data, "val")
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                    im_mid = data['mid']  # 读取中间帧真值
                else:
                    im_lq = data['lq']
                    im_mid = data['mid']
                    # 若无gt则可能无mid，但根据需要判定

                num_iters = 0
                # flow0 = compute_optical_flow((im_gt+1)*127.5,(im_lq+1)*127.5)
                # flow1 = compute_optical_flow((im_lq+1)*127.5,(im_gt+1)*127.5)
                flowa0 = compute_optical_flow((im_gt+1)*127.5,(im_mid+1)*127.5)
                flowa1 = compute_optical_flow((im_mid+1)*127.5,(im_gt+1)*127.5)
                flowb0 = compute_optical_flow((im_mid+1)*127.5,(im_lq+1)*127.5)
                flowb1 = compute_optical_flow((im_lq+1)*127.5,(im_mid+1)*127.5)

                # flowa0 = F.interpolate(flowa0, scale_factor=1 / 4, mode='bicubic') / 4
                # flowa1 = F.interpolate(flowa1, scale_factor=1 / 4, mode='bicubic') / 4
                # flowb0 = F.interpolate(flowb0, scale_factor=1 / 4, mode='bicubic') / 4
                # flowb1 = F.interpolate(flowb1, scale_factor=1 / 4, mode='bicubic') / 4
                if self.configs.model.params.cond_lq:
                    # lq_latent = self.autoencoder.encode(data['lq'])
                    model_kwargs = {'lq': data['lq'],'flowa0':flowa0,'flowa1':flowa1,'flowb0':flowb0,'flowb1':flowb1}
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = {'flowa0':flowa0,'flowa1':flowa1,'flowb0':flowb0,'flowb1':flowb1}
                self.autoencoder.to(data['lq'].device)
                z_y = self.autoencoder.encode(data['lq'])
                z_start = self.autoencoder.encode(data['gt'])
                model_kwargs['first_stage_model'] = self.autoencoder
                tt = torch.tensor(
                    [self.base_diffusion.num_timesteps - 1, ] * im_lq.shape[0],
                    dtype=torch.int64,
                ).cuda()
                # assert torch.all(tt >= 0) and torch.all(
                #     tt < self.base_diffusion.num_timesteps), f"Invalid time steps: {tt}"

                # for sample in self.base_diffusion.p_sample_loop_progressive(
                #         y=im_lq,
                #         model=self.ema_model if self.configs.train.use_ema_val else self.model,
                #         first_stage_model=self.autoencoder,
                #         noise=None,
                #         clip_denoised=True if self.autoencoder is None else False,
                #         model_kwargs=model_kwargs,
                #         device=f"cuda:{self.rank}",
                #         progress=False,
                # ):
                #     sample_decode = {}
                #     if num_iters in indices:
                #         # 对当前时间步的图像进行去噪处理
                #         # 然后再进行解码
                #         for key, value in sample.items():
                #             if key in ['sample', ]:
                #                 for step in range(tt[0]):  # 计算当前需要的去噪次数
                #                     value = self.model(value, tt, **model_kwargs)
                #                 sample_decode[key] = self.base_diffusion.decode_first_stage(
                #                     value,
                #                     self.autoencoder,
                #                 ).clamp(-1.0, 1.0)
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        start=im_gt,
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                ):
                    # tt -= 1
                    # num_iters += 1
                    sample_decode = {}

                    # 当前步数（num_iters）是否在我们需要可视化/评估的 indices
                    if num_iters in indices:
                        # 先做你原有的 decode/可视化/指标计算
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                step_left = tt[0].item()  # 还剩多少步要走才能到0
                                if step_left > 0:
                                    cur = torch.full((value.shape[0],), tt[0].item(), device=value.device).detach()
                                    while cur[0] > 1:  # 一直到0
                                        # print(f"t: {cur}, x_t shape: {value.shape}")
                                        out = self.base_diffusion.p_sample1(
                                            model=self.model,
                                            x=value,
                                            y=z_y,  # 例如你编码后的lq
                                            t=cur,
                                            clip_denoised=True if self.autoencoder is None else False,
                                            model_kwargs=model_kwargs,
                                        )
                                        value = out['sample']
                                        
                                        # with torch.no_grad():
                                        #     decoded = self.base_diffusion.decode_first_stage(value, self.autoencoder).clamp(-1, 1)
                                        # decoded = decoded * 0.5 + 0.5  # 归一化到 [0,1]    
                                        # decoded = rearrange(decoded, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                                        # step_idx = cur[0].item()
                                        # # 4. 调用 logging_image 保存
                                        # #    add_global_step=True 会让 self.log_step_img[phase] 自增，
                                        # #    保证 tag-<step>.png 不会被覆盖
                                        # self.logging_image(
                                        #     decoded,
                                        #     tag=f"decoded_step_{step_left:02d}_{step_idx:02d}",
                                        #     phase=phase,
                                        #     add_global_step=True,
                                        #     nrow=decoded.size(0),
                                        # )
                                        #保存decoded
                                        cur -= 1
                                        
                                # 跑完了以后 x_cur 就是完全到 t=0 的结果
                                # check_tensor_range(self.base_diffusion.decode_first_stage(value, self.autoencoder), "sample_decode['sample'] before")

                                sample_decode[key] = self.base_diffusion.decode_first_stage(value,
                                                                                            self.autoencoder).clamp(-1,
                                                                                                                    1)
                                # check_tensor_range(sample_decode[key], "sample_decode['sample'] after")

                        im_sr_progress = sample_decode['sample']
                        if num_iters == mid_step-1:
                            # 计算中间帧的PSNR和LPIPS
                            if 'gt' in data:
                                # check_tensor_range(sample_decode['sample'], "sample_decode['sample']")
                                # check_tensor_range(im_mid, "im_mid")
                                psnr_mid = util_image.batch_PSNR(
                                    sample_decode['sample'] * 0.5 + 0.5,
                                    im_mid * 0.5 + 0.5,
                                    ycbcr=self.configs.train.val_y_channel,
                                )
                                mean_psnr_mid += psnr_mid
                                self.lpips_loss = self.lpips_loss.to('cuda:0')
                                sample_decode['sample'].to('cuda:0')
                                im_mid.to('cuda:0')
                                lpips_mid = self.lpips_loss(
                                    sample_decode['sample'],
                                    im_mid,
                                ).sum().item()
                                mean_lpips_mid += lpips_mid

                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)

                    num_iters += 1
                    tt -= 1
                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                        sample_decode['sample'] * 0.5 + 0.5,
                        im_gt * 0.5 + 0.5,
                        ycbcr=self.configs.train.val_y_channel,
                    )
                    mean_lpips += self.lpips_loss(
                        sample_decode['sample'],
                        im_gt,
                    ).sum().item()

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii + 1:02d}/{num_iters_epoch:02d}...')
                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                        im_sr_all,
                        tag='progress',
                        phase=phase,
                        add_global_step=False,
                        nrow=len(indices),
                    )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                mean_psnr_mid /= len(self.datasets[phase])
                mean_lpips_mid /= len(self.datasets[phase])
                self.logger.info(
                    f'Validation Metric: PSNR_x0={mean_psnr:5.2f}, LPIPS_x0={mean_lpips:6.4f}, PSNR_mid={mean_psnr_mid:5.2f}, LPIPS_mid={mean_lpips_mid:6.4f}...')
                self.logging_metric(mean_psnr, tag='PSNR_x0', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS_x0', phase=phase, add_global_step=True)
                # 新增对中间帧的logging
                self.logging_metric(mean_psnr_mid, tag='PSNR_mid', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips_mid, tag='LPIPS_mid', phase=phase, add_global_step=True)

            self.logger.info("=" * 100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()


class TrainerDifIRLPIPS(TrainerDifIR):
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        # diffusion loss
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                z0_pred,
                self.autoencoder,
            )  # f16
            self.current_x0_pred = x0_pred.detach()

            # classification loss
            losses["lpips"] = self.lpips_loss(
                x0_pred.clamp(-1.0, 1.0),
                micro_data['gt'],
            ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)

            losses["mse"] *= loss_coef[0]
            losses["lpips"] *= loss_coef[1]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    assert value.shape == mask.shape
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, MSE/LPIPS: '.format(
                    self.current_iters,
                    self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                        current_record,
                        self.loss_mean['mse'][jj].item(),
                        self.loss_mean['lpips'][jj].item(),
                    )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                    self.base_diffusion._scale_input(z_t, tt),
                    self.autoencoder,
                )
                self.logging_image(x_t, tag='middle', phase=phase, add_global_step=False)
                self.logging_image(self.current_x0_pred, tag='x0_pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("=" * 100)


def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

