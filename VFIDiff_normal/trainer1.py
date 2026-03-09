
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

from FlowformerPlusPlus.core.FlowFormer import build_flowformer  
from FlowformerPlusPlus.core.utils.utils import InputPadder, forward_interpolate
from FlowformerPlusPlus.core.utils import frame_utils, flow_viz

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

from FlowformerPlusPlus.visualize_flow import warp,compute_optical_flow, build_model
def print_flow_stats(flow):
    """
    flow: [B, 2, H, W] 的tensor
    """
    B, _, H, W = flow.shape
    
    for b in range(B):
       
        flow_x = flow[b, 0]  # [H, W]
        flow_y = flow[b, 1]  # [H, W]
        
      
        x_min, x_max = flow_x.min().item(), flow_x.max().item()
        y_min, y_max = flow_y.min().item(), flow_y.max().item()
        
        print(f"Sample {b}:")
        print(f"  Flow X: min={x_min:.4f}, max={x_max:.4f}")
        print(f"  Flow Y: min={y_min:.4f}, max={y_max:.4f}")
        print(f"  Magnitude: min={torch.sqrt(flow_x**2 + flow_y**2).min().item():.4f}, "
              f"max={torch.sqrt(flow_x**2 + flow_y**2).max().item():.4f}")
        print()
def check_tensor_range(tensor: torch.Tensor, name: str):
  
    t_min = tensor.min().item()
    t_max = tensor.max().item()

 
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

        local_rank = int(os.environ.get('LOCAL_RANK', 0))  
        torch.cuda.set_device(local_rank) 
        if not dist.is_initialized():  
            dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f"cuda:{local_rank}")


    def setup_dist(self):

        if not dist.is_initialized():
        
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
    
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
    
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
    
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
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

        # EMA ????
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key: deepcopy(value.data) for key, value in self.model.state_dict().items()}
            )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        datasets = {}
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
           
            resize_transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize((256, 256)),  
                transforms.ToTensor() 
            ])

         
            def resize_batch(batch):
                resized_batch = {
                    'gt': [],
                    'lq': [],
                    'mid': [],
                    'gt_path': [],
                    'lq_path': [],
                    'mid_path': []
                }
                for item in batch:
                    gt_image = (item['gt'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    lq_image = (item['lq'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    mid_image = (item['mid'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


                    resized_batch['gt'].append(resize_transform(gt_image))
                    resized_batch['lq'].append(resize_transform(lq_image))
                    resized_batch['mid'].append(resize_transform(mid_image))
                    resized_batch['gt_path'].append(item['gt_path'])
                    resized_batch['lq_path'].append(item['lq_path'])
                    resized_batch['mid_path'].append(item['mid_path'])


                resized_batch['gt'] = torch.stack(resized_batch['gt'])
                resized_batch['lq'] = torch.stack(resized_batch['lq'])
                resized_batch['mid'] = torch.stack(resized_batch['mid'])

                return resized_batch

            #  DataLoader
            dataloaders['val'] = udata.DataLoader(
                datasets['val'],
                batch_size=self.configs.train.batch[1],
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=resize_batch,  
            )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000 ** 2
            self.logger.info(f"Number of parameters: {num_params:.2f}M")
            # self.logger.info(f"Number of parameters_1: {num_params_1:.2f}M")

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

            if self.rank == 0:
                log_str = f"Iter {self.current_iters}/{self.configs.train.iterations} | "
                log_str += f"Loss: {self.loss_mean['mse'][0]:.4f}  | "
                progress_bar.set_postfix_str(log_str)

        # close the tensorboard
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

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
            return  #  rank 0

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
            tt = torch.randint(
                1,
                self.base_diffusion.num_timesteps,
                size=(micro_data['gt'].shape[0],),
                device=f"cuda:{self.rank}",
            )
            batch_size = micro_data['gt'].shape[0]


            mid = self.base_diffusion.num_timesteps // 2

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

            flowa0 = compute_optical_flow((micro_data['gt']+1)*127.5,(micro_data['mid']+1)*127.5)
            flowa1 = compute_optical_flow((micro_data['mid']+1)*127.5,(micro_data['gt']+1)*127.5)
            flowb0 = compute_optical_flow((micro_data['mid']+1)*127.5,(micro_data['lq']+1)*127.5)
            flowb1 = compute_optical_flow((micro_data['lq']+1)*127.5,(micro_data['mid']+1)*127.5)
            

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

            else:
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
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
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
                    im_mid = data['mid']
                else:
                    im_lq = data['lq']
                    im_mid = data['mid']
             

                num_iters = 0
                flowa0 = compute_optical_flow((im_gt+1)*127.5,(im_mid+1)*127.5)
                flowa1 = compute_optical_flow((im_mid+1)*127.5,(im_gt+1)*127.5)
                flowb0 = compute_optical_flow((im_mid+1)*127.5,(im_lq+1)*127.5)
                flowb1 = compute_optical_flow((im_lq+1)*127.5,(im_mid+1)*127.5)

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
                
                    sample_decode = {}

                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                step_left = tt[0].item()  i
                                if step_left > 0:
                                    cur = torch.full((value.shape[0],), tt[0].item(), device=value.device).detach()
                                    while cur[0] > 1:
                                        # print(f"t: {cur}, x_t shape: {value.shape}")
                                        out = self.base_diffusion.p_sample1(
                                            model=self.model,
                                            x=value,
                                            y=z_y, 
                                            t=cur,
                                            clip_denoised=True if self.autoencoder is None else False,
                                            model_kwargs=model_kwargs,
                                        )
                                        value = out['sample']
                                        

                                        cur -= 1
                                        
                                # check_tensor_range(self.base_diffusion.decode_first_stage(value, self.autoencoder), "sample_decode['sample'] before")

                                sample_decode[key] = self.base_diffusion.decode_first_stage(value,
                                                                                            self.autoencoder).clamp(-1,
                                                                                                                    1)
                                # check_tensor_range(sample_decode[key], "sample_decode['sample'] after")

                        im_sr_progress = sample_decode['sample']
                        if num_iters == mid_step-1:
                            if 'gt' in data:
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

