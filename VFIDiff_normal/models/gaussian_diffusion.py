import enum
import math
import sys
# sys.path.append('D:\VFI\ResShift-journal\FlowformerPlusPlus\core')
# sys.path.append('D:\VFI\ResShift-journal\FlowformerPlusPlus\configs')
import torch as th
import numpy as np
import torch.nn
import torch.nn.functional as F
from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from ldm.models.autoencoder import AutoencoderKLTorch
from FlowformerPlusPlus.visualize_flow import compute_optical_flow,warp
# from FlowformerPlusPlus.core.FlowFormer import build_flowformer  # 确保路径正确
# from FlowformerPlusPlus.core.utils.utils import InputPadder, forward_interpolate
# from FlowformerPlusPlus.core.utils import frame_utils, flow_viz
# from visualize import compute_optical_flow  # 导入修改后的光流函数
# from FlowformerPlusPlus.core.FlowFormer import build_flowformer
# from FlowformerPlusPlus.configs.submissions import get_cfg as get_submission_cfg
import os


def normalize_01(t):
    """
    把张量归一化到 [0,1]。
    默认按整个张量范围；若想按通道/批次，可加 dim 参数。
    """
    t = t.float()
    t_min = t.amin()  # 全局最小值
    t_max = t.amax()  # 全局最大值
    return (t - t_min) / (t_max - t_min + 1e-8)


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


import torch as th
import torch.nn.functional as F

sys.path.append('/root/autodl-tmp/VFIDiff-journal/ResShift-journal')
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.utils import save_image
from PIL import Image
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64
        ) ** 2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        # etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    PREVIOUS_X = enum.auto()  # the model predicts epsilon
    RESIDUAL = enum.auto()  # the model predicts epsilon
    EPSILON_SCALE = enum.auto()  # the model predicts epsilon


class LossType(enum.Enum):
    MSE = enum.auto()  # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL


class ModelVarTypeDDPM(enum.Enum):
    """
    What is used as the model's output variance.
    """

    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # if not torch.is_tensor(timesteps) or timesteps.dtype != torch.long:
    #     timesteps = timesteps.long()
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    # res = th.gather(th.from_numpy(arr).to(device=timesteps.device).float(), 0, timesteps.unsqueeze(0)).squeeze(0)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
            self,
            *,
            sqrt_etas,
            kappa,
            model_mean_type,
            loss_type,
            sf=1,
            scale_factor=None,
            normalize_input=True,
            latent_flag=True,
    ):
        self.lambda_mid = 1.0
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas ** 2
        # assert len(self.etas.shape) == 1, "etas must be 1-D"
        # assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas) ** 2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1 - self.etas) * self.sqrt_etas)
            ) ** 2
        else:
            raise NotImplementedError(model_mean_type)

        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse
        local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 自动获取当前进程对应的 GPU
        torch.cuda.set_device(local_rank)  # 设置当前 GPU
        device = torch.device(f"cuda:{local_rank}")

    def q_mean_variance(self, x_start, y, t, model_kwargs=None):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        # mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        # if model_kwargs is not None and 'flow0' in model_kwargs:
        #     flow0 = model_kwargs['flow0']

            # flow0 = compute_optical_flow(self.flow_model, x_start, y)
        # with th.no_grad():
        #     flow0 = compute_optical_flow(self.flow_model, x_start, y)  # [B, 2, H, W]
        scale = (t.float() / (self.num_timesteps - 1)).view(-1, 1, 1, 1)  # [B,1,1,1]
        t_new = t.float() / (self.num_timesteps - 1)
        x_start1 = self.decode_first_stage(
            x_start,
            first_stage_model=model_kwargs["first_stage_model"],
            consistencydecoder=None,
        )
        y1 = self.decode_first_stage(
            y,
            first_stage_model=model_kwargs["first_stage_model"],
            consistencydecoder=None,
        )
        # tenOne, tenTwo, tenMean, tenStd=normalize(x_start1, y1)
        # mean = warp(tenOne, tenTwo, flow0, t_new, tenMean, tenStd)
        mean = warp(tenOne, flow0)  # TODO:确认范围
        mean = self.encode_first_stage(mean, model_kwargs["first_stage_model"], up_sample=False)
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa ** 2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, mid, t, first_stage_model=None, noise=None, model_kwargs=None, init_x=None, init_y=None,flowa0=None, flowa1=None, flowb0=None, flowb1=None):
        # 1) 统一设备
        device = next(first_stage_model.parameters()).device
        # out_dir = os.path.abspath('debug_vis/q_sample')
        # os.makedirs(out_dir, exist_ok=True)

        # 2) 把输入都搬到这个设备
        x_start = x_start.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        init_x = init_x.to(device, non_blocking=True)
        init_y = init_y.to(device, non_blocking=True)
        B = x_start.shape[0]
        # model_kwargs['flow0'] = model_kwargs['flow0'].to(device, non_blocking=True)
        # model_kwargs['flow1'] = model_kwargs['flow1'].to(device, non_blocking=True)
        flowa0 = flowa0.to(device)
        flowa1 = flowa1.to(device)
        flowb0 = flowb0.to(device)
        flowb1 = flowb1.to(device)
        if noise is None:
            noise = x_start.new_empty_like(x_start).normal_()
        else:
            noise = noise.to(device, non_blocking=True)
        # assert noise.shape == x_start.shape

        # 3) 计算 t 的比例张量
        # t_new = (t.float() / (self.num_timesteps - 1)).to(device)
            # 根据时间步分组
        first_half_mask = t <= 6  # t: 0,1,2,3,4,5,6
        second_half_mask = t >= 7  # t: 7,8,9,10,11,12

        # 初始化结果张量
        warp_x_start = torch.zeros_like(init_x, dtype=init_x.dtype)
        # print(f"warp_x_start:{x_start.dtype}")

        with torch.no_grad():
            # 处理前半段 (t: 0-6)
            if first_half_mask.any():
                t_normalized = t[first_half_mask].float() / 6.0  # 归一化到 [0, 1]
                warp_result_1 = warp(
                    init_x[first_half_mask],
                    flowa0[first_half_mask],
                    flowa1[first_half_mask],
                    t_normalized, 6
                ).to(warp_x_start.dtype)
                # print(f"warp_result_1:{warp_result_1.dtype}")
                warp_x_start[first_half_mask] = warp_result_1.to(device)

            # 处理后半段 (t: 7-12)
            if second_half_mask.any():
                t_normalized = (t[second_half_mask] - 6).float() / 6.0  # 归一化到 [0, 1]
                warp_result_2 = warp(
                    mid[second_half_mask],
                    flowb0[second_half_mask],
                    flowb1[second_half_mask],
                    t_normalized, 6
                ).to(warp_x_start.dtype)
                warp_x_start[second_half_mask] = warp_result_2.to(device)
        
        # mask1 = torch.where(t <= 6, t/6, 0)
        # mask2 = torch.where(t >= 7, (t-6)/6, 0)
        #
        # # 4) 先做光流插值，再搬到 device
        # with torch.no_grad():
        #     warp_x_start = warp(init_x, flowa0, flowa1, mask1, 6).to(device)
        #     warp_x_start = warp(mid, flowb0, flowb1, mask2, 6).to(device)
            # warp_norm = warp_x_start.mul(0.5).add(0.5).clamp(0,1)
            # # #为了debug设计的w
            # save_image(
            #     warp_norm,  # warp_norm
            #     os.path.join(out_dir, f"warp_pixel_t{int(t[0].item()):03d}.png"),
            #     nrow=warp_x_start.shape[0],
            #     normalize=True
            # )
        # 5) 编码
            warp_z = self.encode_first_stage(warp_x_start, first_stage_model, up_sample=False).detach()

        # 6) 将噪声项加回来
        eps_factor = _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape).to(device)
        return warp_z + eps_factor * noise, warp_x_start

    def q_posterior_mean_variance(self, x_start, x_t, t, y, model_kwargs=None, input1=None, first_stage_model=None):
        """
        这里的x_start是我们预测的结果，准确说是x_{t-1}
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        # if model_kwargs is not None and 'flow0' in model_kwargs:
        #     flow0 = model_kwargs['flow0']
        #     flow1 = model_kwargs['flow1']

        # assert x_start.shape == x_t.shape
        B = x_t.shape[0]  # 获取批量大小

        posterior_mean = (
                (_extract_into_tensor(self.etas, t - 1, x_start.shape) / _extract_into_tensor(self.etas, t,
                                                                                              x_start.shape)) * input1 +
                ((_extract_into_tensor(self.etas, t, x_start.shape) - _extract_into_tensor(self.etas, t - 1,
                                                                                           x_start.shape)) / _extract_into_tensor(
                    self.etas, t, x_start.shape)) * x_start
        )

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        # assert (
        #         posterior_mean.shape[0]
        #         == posterior_variance.shape[0]
        #         == posterior_log_variance_clipped.shape[0]
        #         == x_start.shape[0]
        # )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_mean_variance1(self, x_start, x_t, t, y, model_kwargs=None):
        """
        这里的x_start是我们预测的结果，准确说是x_{t-1}
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        # assert x_start.shape == x_t.shape
        B = x_t.shape[0]  # 获取批量大小
        posterior_mean = (
                (_extract_into_tensor(self.etas, t - 1, x_start.shape) / _extract_into_tensor(self.etas, t,
                                                                                              x_start.shape)) * x_t
                + ((_extract_into_tensor(self.etas, t, x_start.shape) - _extract_into_tensor(self.etas, t - 1,
                                                                                             x_start.shape)) / _extract_into_tensor(
            self.etas, t, x_start.shape)) * x_start

        )

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        # assert (
        #         posterior_mean.shape[0]
        #         == posterior_variance.shape[0]
        #         == posterior_log_variance_clipped.shape[0]
        #         == x_start.shape[0]
        # )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, start, z_t, y, t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            first_stage_model=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = z_t.shape[:2]
        # assert t.shape == (B,)
        device = t.device

        # t_new = 1.0 / t
        model_kwargs['flowa0'] = model_kwargs['flowa0'].to(device)
        model_kwargs['flowa1'] = model_kwargs['flowa1'].to(device)
        model_kwargs['flowb0'] = model_kwargs['flowb0'].to(device)
        model_kwargs['flowb1'] = model_kwargs['flowb1'].to(device)

        # 为每个样本确定应该使用哪个光流阶段
        first_half_mask = t <= 6  # 前半段样本使用flowa
        second_half_mask = t >= 7  # 后半段样本使用flowb

        with torch.no_grad():
            x_t = self.decode_first_stage(
                z_t,
                first_stage_model=first_stage_model,
                consistencydecoder=None,
            ).clamp(-1.0, 1.0)
            x_t = x_t.to(device)

            # 初始化结果
            warp_t = torch.zeros_like(x_t, dtype=x_t.dtype)
            step_flow = th.zeros(x_t.shape[0], 2, x_t.shape[2], x_t.shape[3], 
                     dtype=x_t.dtype, device=x_t.device)

            # 前半段样本：用flowa向前一步
            if first_half_mask.any():
                # 获取前半段样本的当前时间步
                current_t = t[first_half_mask].float()  # 例如：[1, 2, 3, 4, 5, 6]

                # 计算光流的切分比例
                # flowa是从t=0到t=6的光流，我们需要从当前t到t-1的部分
                flow_ratio1 = (current_t / 6.0).view(-1, 1, 1, 1)  # 当前位置在总光流中的比例
                step_ratio1 = 1.0/current_t

                # 切分光流：从完整光流中提取当前步骤需要的部分
                step_flow_a1 = model_kwargs['flowa1'][first_half_mask]  * flow_ratio1
                step_flow_a0 = model_kwargs['flowa0'][first_half_mask] * flow_ratio1
                step_flow[first_half_mask] = step_flow_a1 * (step_ratio1.view(-1, 1, 1, 1)) 

                warp_result_a = warp(
                    x_t[first_half_mask],
                    step_flow_a1,  # 切分后的单步光流
                    step_flow_a0,  # 切分后的单步光流
                    step_ratio1,  # 1/t
                    current_t, #改动
                ).to(warp_t.dtype)
                warp_t[first_half_mask] = warp_result_a

            # 后半段样本：用flowb向前一步
            if second_half_mask.any():
                # 获取前半段样本的当前时间步
                current_t = (t[second_half_mask] - 6).float()  # 例如：[1, 2, 3, 4, 5, 6]

                # 计算光流的切分比例
                flow_ratio2 = (current_t / 6.0).view(-1, 1, 1, 1)  # 当前位置在总光流中的比例
                step_ratio2 = 1.0/current_t

                # 切分光流：从完整光流中提取当前步骤需要的部分
                step_flow_b1 = model_kwargs['flowb1'][second_half_mask]  * flow_ratio2
                step_flow_b0 = model_kwargs['flowb0'][second_half_mask] * flow_ratio2
                step_flow[second_half_mask] = step_flow_b1 * (step_ratio2.view(-1, 1, 1, 1))

                warp_result_b = warp(
                    x_t[second_half_mask],
                    step_flow_b1,  # 切分后的单步光流
                    step_flow_b0,  # 切分后的单步光流
                    step_ratio2,  # 1/t
                    current_t,  #改动
                ).to(warp_t.dtype)
                warp_t[second_half_mask] = warp_result_b

            warp_t = self.encode_first_stage(warp_t, first_stage_model, up_sample=False)
        # device = t.device
        warp_t.to(device)
        warp_t_1 = self._scale_input(warp_t, (t - 1).to(device)).to(device)
        # warp_t = th.cat((warp_t_1, th.ones_like(warp_t_1)), dim=1)
        t_before = (t-1).to(device)
        model_kwargs = {key: value.to(device) for key, value in model_kwargs.items()}
        # # 假设 scaled_input_b2 的形状是 [batch_size, 3, height, width]
        # # 创建一个 2 通道的零张量
        # zeros_2ch = th.zeros(warp_t_1.shape[0], 2, warp_t_1.shape[2], warp_t_1.shape[3], 
        #                      dtype=warp_t_1.dtype, device=warp_t_1.device)
        step_flow = F.interpolate(step_flow, scale_factor=1 / 4, mode='bicubic') / 4
        warp_t_input = th.cat((z_t, step_flow), dim=1)
        model_output = model(warp_t_input, t_before, **model_kwargs)
        model_variance = _extract_into_tensor(self.posterior_variance, t, z_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, z_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:  # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:  # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=z_t, y=y, t=t, eps=model_output)
            )  # predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=z_t, y=y, t=t, eps=model_output)
            )  # predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=z_t, t=t, y=y, model_kwargs=model_kwargs, input1=warp_t_1,
            first_stage_model=first_stage_model
        )

        # assert (
        #         model_mean.shape == model_log_variance.shape == pred_xstart.shape == z_t.shape
        # )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_mean_variance1(
            self, model, x_t, y, t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        # assert t.shape == (B,)

        input_b2 = x_t
        scaled_input_b2 = self._scale_input(input_b2, t)
        # scaled_input_b2 = th.cat((scaled_input_b2, th.zeros_like(scaled_input_b2)), dim=1)
        zeros_2ch = th.zeros(scaled_input_b2.shape[0], 2, scaled_input_b2.shape[2], scaled_input_b2.shape[3], 
                     dtype=scaled_input_b2.dtype, device=scaled_input_b2.device)
        scaled_input_b2 = th.cat((scaled_input_b2, zeros_2ch), dim=1)
        origin_kwargs = {key: value.to(scaled_input_b2.device) for key, value in model_kwargs.items() if
                         key not in ['tenMean', 'tenStd', 'tenOne_n', 'tenTwo_n', 'first_stage_model', 'occ']}
        if hasattr(model, 'model'):
            model.model.to(t.device)
        else:
            model.to(t.device)
        
        model_output = model(scaled_input_b2, t, **origin_kwargs)
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:  # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:  # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )  # predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )  # predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance1(
            x_start=pred_xstart, x_t=x_t, t=t, y=y, model_kwargs=model_kwargs
        )

        # assert (
        #         model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        # assert x_t.shape == eps.shape
        return (
                x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        # assert x_t.shape == eps.shape
        return (
                x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        # assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
                x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(self, model, start, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                 noise_repeat=False,
                 first_stage_model=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        if t[0] != 0:
            out = self.p_mean_variance(
                model,
                start,
                x,
                y,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                first_stage_model=first_stage_model
            )
            device = out["mean"].device
            noise = th.randn_like(x).to(device)
            if noise_repeat:
                noise = noise[0,].repeat(x.shape[0], 1, 1, 1).to(device)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            ).to(device)  # no noise when t == 0
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        else:
            sample = x
            return {"sample": sample}
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def p_sample1(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance1(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def p_sample_loop(
            self,
            y,
            model,
            first_stage_model=None,
            consistencydecoder=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model.

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                y,
                model,
                first_stage_model=first_stage_model,
                noise=noise,
                noise_repeat=noise_repeat,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample["sample"]
        with th.no_grad():
            out = self.decode_first_stage(
                final,
                first_stage_model=first_stage_model,
                consistencydecoder=consistencydecoder,
            )
        return out

    # 验证时用到的,逐步生成并返回每一步的中间结果，方便监控和调试
    def p_sample_loop_progressive(
            self, start, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        with torch.no_grad():
            z_y = self.encode_first_stage(y, first_stage_model, up_sample=False)
        # generating noise
        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            if i == 0:
                continue
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    start,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                    first_stage_model=first_stage_model
                )
                yield out
                z_sample = out["sample"]

    def p_sample_loop_progressive1(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample(), including intermediate frames.
        """
        if device is None:
            device = next(model.parameters()).device
        # assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # Define the mid-step for capturing the intermediate frame
        mid_step = self.num_timesteps // 2
        mid_frame = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                # Capture the intermediate frame at the mid-step
                if i == mid_step:
                    mid_frame = out["sample"]

                yield out
                img = out["sample"]

        # Return the final frame and the intermediate frame
        return img, mid_frame

    def decode_first_stage1(self, z_sample, first_stage_model=None, consistencydecoder=None, scale_factor=None):
        batch_size = z_sample.shape[0]
        data_dtype = z_sample.dtype

        if consistencydecoder is None:
            model = first_stage_model
            decoder = first_stage_model.decode
            model_dtype = next(model.parameters()).dtype
        else:
            model = consistencydecoder
            decoder = consistencydecoder
            model_dtype = next(model.ckpt.parameters()).dtype

        if first_stage_model is None:
            return z_sample
        else:
            # 使用传入的 scale_factor 或者默认的 self.scale_factor
            effective_scale_factor = scale_factor if scale_factor is not None else self.scale_factor
            z_sample = 1 / effective_scale_factor * z_sample
            if consistencydecoder is None:
                out = decoder(z_sample.type(model_dtype))
            else:
                with th.cuda.amp.autocast():
                    out = decoder(z_sample)
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def decode_first_stage(self, z_sample, first_stage_model=None, consistencydecoder=None):
        batch_size = z_sample.shape[0]
        data_dtype = z_sample.dtype

        if consistencydecoder is None:
            model = first_stage_model
            decoder = first_stage_model.decode
            model_dtype = next(model.parameters()).dtype
        else:
            model = consistencydecoder
            decoder = consistencydecoder
            model_dtype = next(model.ckpt.parameters()).dtype

        if first_stage_model is None:
            return z_sample
        else:
            z_sample = 1 / self.scale_factor * z_sample
            if consistencydecoder is None:
                out = decoder(z_sample.type(model_dtype))
            else:
                with th.cuda.amp.autocast():
                    out = decoder(z_sample)
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        data_dtype = y.dtype
        model_dtype = next(first_stage_model.parameters()).dtype
        if up_sample and self.sf != 1:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            if not model_dtype == data_dtype:
                y = y.type(model_dtype)
            with th.no_grad():
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
            if not model_dtype == data_dtype:
                out = out.type(data_dtype)
            return out

    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps - 1, ] * y.shape[0], device=y.device).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def _p_sample_no_warp(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        一个简化版的p_sample，用于“无warp”反向去噪。
        在这个函数里，我们不再调用warp(x, flow1)。
        """
        if model_kwargs is None:
            model_kwargs = {}
        # 无warp时，直接把warp去掉或让flow1=0
        # 这里演示“彻底去掉warp”，只做普通后验
        out = self._p_mean_variance_no_warp(
            model=model,
            x_t=x,
            y=y,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def _q_posterior_mean_variance_no_warp(self, x_start, x_t, t, y, model_kwargs=None):
        """
        这里的x_start是我们预测的结果，准确说是x_{t-1}
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        if model_kwargs is None:
            model_kwargs = {}
        # assert x_start.shape == x_t.shape
        B = x_t.shape[0]  # 获取批量大小
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        # assert (
        #         posterior_mean.shape[0]
        #         == posterior_variance.shape[0]
        #         == posterior_log_variance_clipped.shape[0]
        #         == x_start.shape[0]
        # )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _p_mean_variance_no_warp(self, model, x_t, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        无warp版本，用于从 x_t -> x_{t-1} 的均值与方差。
        """
        if model_kwargs is None:
            model_kwargs = {}

        # 普通模型forward, 不带warp
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        # variance/log_variance 可以复用 posterior_variance 逻辑
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # 预测 x_0
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = process_xstart(self._predict_xstart_from_residual(y=y, residual=model_output))
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output))
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output))
        else:
            raise NotImplementedError(self.model_mean_type)
        model_mean, _, _ = self._q_posterior_mean_variance_no_warp(
            x_start=pred_xstart, x_t=x_t, t=t, y=y, model_kwargs=model_kwargs
        )

        # assert (
        #         model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def training_losses(self, model, x_start, y, mid, im2, im3, im5, im6, t, flowa0, flowa1, flowb0, flowb1, first_stage_model=None, model_kwargs=None,noise=None,rank=None):
        if model_kwargs is None:
            model_kwargs = {}
        device = next(first_stage_model.parameters()).device
        # device = next(model.parameters()).device
        x_start = x_start.to(device)
        y = y.to(device)
        mid = mid.to(device)
        im2 = im2.to(device)
        im3 = im3.to(device)
        im5 = im5.to(device)
        im6 = im6.to(device)
        t = t.to(device)
        B = x_start.shape[0]
        # 为了debug设计w
        # out_dir = os.path.abspath('debug_vis/training')
        # os.makedirs(out_dir, exist_ok=True)
        # for name, img in [('x_start', x_start), ('y', y)]:
        #     img_norm = img.mul(0.5).add(0.5).clamp(0,1)
        #     save_image(
        #         img_norm,
        #         os.path.join(out_dir, f"orig_{name}.png"),
        #         nrow=B, normalize=True
        #     )
        # model_kwargs['flow0'] = model_kwargs['flow0'].to(device)
        # model_kwargs['flow1'] = model_kwargs['flow1'].to(device)
        flowa0 = flowa0.to(device)
        flowa1 = flowa1.to(device)
        flowb0 = flowb0.to(device)
        flowb1 = flowb1.to(device)
        with torch.no_grad():
            z_y = self.encode_first_stage(y, first_stage_model, up_sample=False).to(device)
            z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False).to(device)
            # mid1 = self.encode_first_stage(mid, first_stage_model, up_sample=False).to(device)
            # 如果没有提供噪声，则生成与 z_start 形状相同的随机噪声
            if noise is None:
                noise = th.randn_like(z_start).to(device)
            # 通过 q_sample 函数获取 z_t
            z_t,x_t = self.q_sample(z_start, z_y, mid, t, first_stage_model=first_stage_model, noise=noise,
                                model_kwargs=model_kwargs, init_x=x_start, init_y=y,flowa0 = flowa0, flowa1 = flowa1, flowb0 = flowb0, flowb1 = flowb1)
            B = z_t.shape[0]  # 批量大小
            # 为 model_output 分配空间（与 z_t 形状相同）
            model_output = th.empty_like(z_t)
            # 为每个样本确定应该使用哪个光流阶段
            first_half_mask = t <= 6  # 前半段样本使用flowa
            second_half_mask = t >= 7  # 后半段样本使用flowb
            z_t1 = self.decode_first_stage(
                z_t,
                first_stage_model=first_stage_model,
                consistencydecoder=None,
            ).clamp(-1.0, 1.0)
            # z_t11 = z_t1.mul(0.5).add(0.5).clamp(0,1)
            # # 为了debug设计w
            # save_image(
            #     z_t11,
            #     os.path.join(out_dir, f"after_qsample_t{int(t[0].item()):03d}.png"),
            #     nrow=B, normalize=False
            # )
            # with torch.no_grad():
            # 初始化结果
            # warp_t = torch.zeros_like(z_t1,dtype=z_t1.dtype)
            step_flow = th.zeros(x_start.shape[0], 2, x_start.shape[2], x_start.shape[3], 
         dtype=x_start.dtype, device=x_start.device)

            # 前半段样本：用flowa向前一步
            if first_half_mask.any():
                # 获取前半段样本的当前时间步
                current_t = t[first_half_mask].float()  # 例如：[1, 2, 3, 4, 5, 6]

                # 计算光流的切分比例
                # flowa是从t=0到t=6的光流，我们需要从当前t到t-1的部分
                flow_ratio1 = (current_t / 6.0).view(-1, 1, 1, 1)  # 当前位置在总光流中的比例
                step_ratio1 = 1.0/current_t

                # 切分光流：从完整光流中提取当前步骤需要的部分
                step_flow_a1 = flowa1[first_half_mask]  * flow_ratio1
                step_flow_a0 = flowa0[first_half_mask] * flow_ratio1
                step_flow[first_half_mask] = step_flow_a1 * (step_ratio1.view(-1, 1, 1, 1))

                # warp_result_a = warp(
                #     z_t1[first_half_mask],
                #     step_flow_a1,  # 切分后的单步光流
                #     step_flow_a0,  # 切分后的单步光流
                #     step_ratio1,  # 1/t
                #     current_t,  #改动
                # ).to(warp_t.dtype)
                # warp_t[first_half_mask] = warp_result_a

            # 后半段样本：用flowb向前一步
            if second_half_mask.any():
                # 获取前半段样本的当前时间步
                current_t = (t[second_half_mask] - 6).float()  # 例如：[1, 2, 3, 4, 5, 6]

                # 计算光流的切分比例
                flow_ratio2 = (current_t / 6.0).view(-1, 1, 1, 1)  # 当前位置在总光流中的比例
                step_ratio2 = 1.0/current_t

                # 切分光流：从完整光流中提取当前步骤需要的部分
                step_flow_b1 = flowb1[second_half_mask]  * flow_ratio2
                step_flow_b0 =flowb0[second_half_mask] * flow_ratio2
                step_flow[second_half_mask] = step_flow_b1 * (step_ratio2.view(-1, 1, 1, 1))

                # warp_result_b = warp(
                #     z_t1[second_half_mask],
                #     step_flow_b1,  # 切分后的单步光流
                #     step_flow_b0,  # 切分后的单步光流
                #     step_ratio2,  # 1/t
                #     current_t,  #改动
                # ).to(warp_t.dtype)
                # warp_t[second_half_mask] = warp_result_b

            # warp_y_11 = warp_t.mul(0.5).add(0.5).clamp(0,1)
            # # 为了debug设计w
            # save_image(
            #     warp_y_11,
            #     os.path.join(out_dir, f"reverse_warp_px_t{int(t[0].item()):03d}.png"),
            #     nrow=B, normalize=True
            # )

            # warp_t = self.encode_first_stage(warp_t, first_stage_model, up_sample=False)
            # scaled_input_all = self._scale_input(warp_t, t - 1)
            # # 将噪声项加回来
            # eps_factor1 = _extract_into_tensor(self.sqrt_etas * self.kappa, t, z_start.shape).to(device)
            # scaled_input_all = scaled_input_all+eps_factor1*noise
            step_flow = F.interpolate(step_flow, scale_factor=1 / 4, mode='bicubic') / 4
            scaled_input_input = th.cat((z_t, step_flow), dim=1)
        # 4) 先做光流插值，再搬到 device
        # with torch.no_grad():
            # 根据时间步分组
            first_half_mask = t <= 7  # t: 0,1,2,3,4,5,6
            second_half_mask = t >= 8  # t: 7,8,9,10,11,12

            # 初始化结果张量

            warp_x_start = torch.zeros_like(x_start,dtype=x_start.dtype)

            # 处理前半段 (t: 0-6)
            if first_half_mask.any():
                t_normalized = (t[first_half_mask] - 1).float() / 6.0  # 归一化到 [0, 1]
                warp_result_1 = warp(

                    x_start[first_half_mask],
                    flowa0[first_half_mask],
                    flowa1[first_half_mask],
                    t_normalized, 6
                ).to(warp_x_start.dtype)
                warp_x_start[first_half_mask] = warp_result_1.to(device)

            # 处理后半段 (t: 7-12)
            if second_half_mask.any():
                t_normalized = (t[second_half_mask] - 7).float() / 6.0  # 归一化到 [0, 1]
                warp_result_2 = warp(
                    
                    mid[second_half_mask],
                    flowb0[second_half_mask],
                    flowb1[second_half_mask],
                    t_normalized, 6
                ).to(warp_x_start.dtype)
                warp_x_start[second_half_mask] = warp_result_2.to(device)
 
            # warp_x_start1 = warp_x_start.mul(0.5).add(0.5).clamp(0,1)
            # #为了debug设计w
            # save_image(
            #     warp_x_start1,
            #     os.path.join(out_dir, f"reverse1_warp_px_t{int(t[0].item()):03d}.png"),
            #     nrow=B, normalize=True
            # )
            warp_x_start = self.encode_first_stage(warp_x_start, first_stage_model, up_sample=False)
        # target = warp_x_start + eps_factor1 * noise

        # input_all = th.cat((scaled_input_all, th.ones_like(scaled_input_all)), dim=1)
        origin_kwargs = {key: value.to(t.device) for key, value in model_kwargs.items() if
                         key not in ['first_stage_model']}

        model.model.to(t.device)
        # t_1 = (t-1).to(device)
        model_output = model(scaled_input_input, t, **origin_kwargs).float()

        if self.model_mean_type == ModelMeanType.START_X:
            target = warp_x_start
            # target = self._scale_input(warp_z_start_0, t - 1)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            target = (z_y - z_start)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            target = noise
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            target = noise * self.kappa * _extract_into_tensor(self.sqrt_etas, t,
                                                               noise.shape)
        else:
            raise NotImplementedError(self.model_mean_type)

        # posterior_mean = (
        #         (_extract_into_tensor(self.etas, t - 1, z_start.shape) / _extract_into_tensor(self.etas, t,
        #                                                                                       z_start.shape)) * scaled_input_all
        #         + ((_extract_into_tensor(self.etas, t, z_start.shape) - _extract_into_tensor(self.etas, t - 1,
        #                                                                                      z_start.shape)) / _extract_into_tensor(
        #     self.etas, t, z_start.shape)) * warp_x_start

        # )

        # model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, z_t.shape)
        # nonzero_mask = (
        #     (t != 0).float().view(-1, *([1] * (len(scaled_input_all.shape) - 1)))
        # ).to(device)  # no noise when t == 0
        # z_sample = posterior_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise


        # 改进：对于刚得到的z_sampl做去噪，z_sample对应的是t-1,假设t-1代表【0,3,4,5,6】，0的样本无需再做p_sample1,剩余的分别做2,3,4,6次p_sample1,因为p_sample1只接受batch_size大小，因此下面代码不能改变长度，最后产生最终的结果。例子如下，请给出代码，此外，得到的结果要分别与相应时间步骤的真值作比较，注意，我们只比较时间步在0(gt),2(im6),4(im5),6(mid),8(im3),10(im2),12(lq) 这7个时间步的，对于该例，0的和gt做L1损失，3的没有对应不做，4的结果和im5做L1损失，以此类推，最终损失取平均值（实际有比较的样本数量，这个例子中0,4,6是，因此除以3）

        # z_sample 是刚采样得到的、处于 step = t−1 的隐向量，shape [B,C,H,W]
        steps_per_sample = t.long()      # [B], e.g. {0,2,4,6,8,10,12}
        max_steps = int(steps_per_sample.max().item())

        # 预先编码好我们要比较的真值 latent
        gt_latents = {
            0:  z_start,  2: self.encode_first_stage(im6, first_stage_model, up_sample=False),
            4:  self.encode_first_stage(im5, first_stage_model, up_sample=False),
            6:  self.encode_first_stage(mid, first_stage_model, up_sample=False),
            8:  self.encode_first_stage(im3, first_stage_model, up_sample=False),
            10: self.encode_first_stage(im2, first_stage_model, up_sample=False),
            12: z_y
        }
        compare_steps = list(gt_latents.keys())  # [0,2,4,6,8,10,12]

        # 当前的 latent，整批
        latent_current   = z_t.clone()           # [B,C,H,W]
        # 用来存每个样本“它自己那一刻”的 snapshot
        latent_snapshots = torch.zeros_like(latent_current)

        # 1) 先把“0 次调用”对应的结果（也就是初始 z_sample）存下
        mask0 = steps_per_sample == 0               # [B]
        if mask0.any():
            latent_snapshots[mask0] = latent_current[mask0]
        value = z_t.clone() 


        # 2) 依次做 max_steps 次全批次去噪
        for i in range(max_steps):
            # 该轮每个样本还剩多少步要去
            t_cur = (steps_per_sample - i).clamp(min=0).long()
            latent_current = self.p_sample1(
                model=model,
                x=value,
                y=z_y,
                t=t_cur,
                clip_denoised=True if first_stage_model is None else False,
                model_kwargs=origin_kwargs,
            )
            value = latent_current['sample']
            # 这一轮后，恰好完成“i+1 次调用”的样本
            mask_i1 = steps_per_sample == (i + 1)
            if mask_i1.any():
                latent_snapshots[mask_i1] = value[mask_i1]

        # 3) 统一计算 L1：只对那些 steps_per_sample 在 compare_steps 中的样本做
        l1_sum = 0.0
        count  = 0
        for step in compare_steps:
            mask = steps_per_sample == step
            if mask.any():
                pred   = latent_snapshots[mask]          # [M,C,H,W]
                target1 = gt_latents[step][mask]          # [M,C,H,W]
                # sum over all像素 & channels
                # l1_sum += F.l1_loss(pred, target1, reduction='sum')
                # count  += mask.sum().item()
                # print(f"step:{step},loss:{F.l1_loss(pred, target1, reduction='mean') * mask.sum().item()},count:{mask.sum().item()}")
                l1_sum += F.l1_loss(pred, target1, reduction='mean') * mask.sum().item()
                count  += mask.sum().item()

        loss_flow_interpolation = l1_sum / max(count, 1)
        # print(target.shape[0])
        # print(model_output.shape[0])

        # ── 每个样本的整体 MSE ───────────────────────────
        loss_mse_per_sample = (target - model_output).abs()  \
            .view(B, -1) \
            .mean(dim=1)  # [B] 逐样本 MSE

        non_mid_loss_total = loss_mse_per_sample.mean()
        # print(f"I am average loss:{non_mid_loss_total}")
        # print(f"I am supervised loss:{loss_flow_interpolation}")
        losses = 0.5*non_mid_loss_total + 0.5*loss_flow_interpolation
        # losses = non_mid_loss_total


        terms = {"mse": losses}

        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.model_mean_type)

        # return terms, pred_zstart, pred_zstart
        return terms, z_t, pred_zstart

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                inputs = inputs.to(t.device)
                # the variance of latent code is around 1.0
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa ** 2 + 1).to(t.device)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm


class GaussianDiffusionDDPM:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarTypeDDPM determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            scale_factor=None,
            sf=1,
    ):
        self.model_mean_type = model_mean_type  # EPSILON
        self.model_var_type = model_var_type  # LEARNED_RANGE
        self.scale_factor = scale_factor  # scale factor in latent space default True
        self.sf = sf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        # assert len(betas.shape) == 1, "betas must be 1-D"
        # assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        # assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, y, t, model_kwargs=None):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarTypeDDPM.LEARNED, ModelVarTypeDDPM.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarTypeDDPM.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarTypeDDPM.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarTypeDDPM.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:  # predict x_{t-1}
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:  # predict x_0
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )  # predict \eps
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            first_stage_model=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            first_stage_model=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device).long()
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, first_stage_model=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_start = self.encode_first_stage(x_start, first_stage_model)
        if noise is None:
            noise = th.randn_like(z_start)
        z_t = self.q_sample(z_start, t, noise=noise)

        terms = {}

        model_output = model(z_t, t, **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=z_start, x_t=z_t, t=t
            )[0],
            ModelMeanType.START_X: z_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == z_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        if self.model_mean_type == ModelMeanType.START_X:  # predict x_0
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)  # q(x_t|x_0)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def _scale_input(self, inputs, t):
        return inputs

    def decode_first_stage(self, z_sample, first_stage_model=None):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            with th.no_grad():
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample)
                return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with th.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
                return out.type(ori_dtype)

