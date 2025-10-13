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
# from softmax_splatting_master.run import 
# from softmax_splatting_master.run import normalize
# from softmax_splatting_master.run import compute_optical_flow
# from RAFT_SOFT.new_interface import compute_optical_flow_b as compute_optical_flow,_b_s as ,_b as _ns
# from RAFT_SOFT.new_interface import _b as 1
from RIFE.inference_img import batch_interpolate
from RIFE.inference_img import double_interpolate


# from RAFT.raft_interface import ,compute_optical_flow,normalize


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
        if model_kwargs is not None and 'flow0' in model_kwargs:
            flow0 = model_kwargs['flow0']

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
        # mean = (tenOne, tenTwo, flow0, t_new, tenMean, tenStd)
        mean = (tenOne, flow0)  # TODO:确认范围
        mean = self.encode_first_stage(mean, model_kwargs["first_stage_model"], up_sample=False)
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa ** 2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, t, noise=None):
        # 1) 统一设备
        device = x_start.device

        # 2) 把输入都搬到这个设备
        x_start = x_start.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if noise is None:
            noise = x_start.new_empty_like(x_start).normal_()
        else:
            noise = noise.to(device, non_blocking=True)
        # assert noise.shape == x_start.shape

        # 6) 将噪声项加回来
        eps_factor = _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape).to(device)
        return x_start + eps_factor * noise

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
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
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
            self, model, start, x_t, y, t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            first_stage_model=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        # scaled_input_b2 = th.cat((self._scale_input(x_t, t),th.zeros_like(x_t)),dim=1)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t,y=y, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
        # """
        # Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        # the initial x, x_0.

        # :param model: the model, which takes a signal and a batch of timesteps
        #               as input.
        # :param x_t: the [N x C x ...] tensor at time t.
        # :param y: the [N x C x ...] tensor of degraded inputs.
        # :param t: a 1-D Tensor of timesteps.
        # :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        # :param denoised_fn: if not None, a function which applies to the
        #     x_start prediction before it is used to sample. Applies before
        #     clip_denoised.
        # :param model_kwargs: if not None, a dict of extra keyword arguments to
        #     pass to the model. This can be used for conditioning.
        # :return: a dict with the following keys:
        #          - 'mean': the model mean output.
        #          - 'variance': the model variance output.
        #          - 'log_variance': the log of 'variance'.
        #          - 'pred_xstart': the prediction for x_0.
        # """
        # if model_kwargs is None:
        #     model_kwargs = {}

        # B, C = z_t.shape[:2]
        # # assert t.shape == (B,)

        # t_new = 1.0 / t
        # with torch.no_grad():
        #     x_t = self.decode_first_stage(
        #         z_t,
        #         first_stage_model=first_stage_model,
        #         consistencydecoder=None,
        #     ).clamp(-1.0, 1.0)
        #     x_t = x_t * 0.5 + 0.5
        #     _t = batch_interpolate(x_t, start, t_new)
        #     _t = self.encode_first_stage(_t, first_stage_model, up_sample=False)
        # device = t.device
        # _t.to(device)
        # _t_1 = self._scale_input(_t, (t - 1).to(device)).to(device)
        # _t = th.cat((_t_1, th.ones_like(_t_1)), dim=1)

        # model_kwargs = {key: value.to(device) for key, value in model_kwargs.items()}
        # model_output = model(_t, t, **model_kwargs)
        # model_variance = _extract_into_tensor(self.posterior_variance, t, z_t.shape)
        # model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, z_t.shape)

        # def process_xstart(x):
        #     if denoised_fn is not None:
        #         x = denoised_fn(x)
        #     if clip_denoised:
        #         return x.clamp(-1, 1)
        #     return x

        # if self.model_mean_type == ModelMeanType.START_X:  # predict x_0
        #     pred_xstart = process_xstart(model_output)
        # elif self.model_mean_type == ModelMeanType.RESIDUAL:  # predict x_0
        #     pred_xstart = process_xstart(
        #         self._predict_xstart_from_residual(y=y, residual=model_output)
        #     )
        # elif self.model_mean_type == ModelMeanType.EPSILON:
        #     pred_xstart = process_xstart(
        #         self._predict_xstart_from_eps(x_t=z_t, y=y, t=t, eps=model_output)
        #     )  # predict \eps
        # elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
        #     pred_xstart = process_xstart(
        #         self._predict_xstart_from_eps_scale(x_t=z_t, y=y, t=t, eps=model_output)
        #     )  # predict \eps
        # else:
        #     raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        # model_mean, _, _ = self.q_posterior_mean_variance(
        #     x_start=pred_xstart, x_t=z_t, t=t, y=y, model_kwargs=model_kwargs, input1=_t_1,
        #     first_stage_model=first_stage_model
        # )

        # # assert (
        # #         model_mean.shape == model_log_variance.shape == pred_xstart.shape == z_t.shape
        # # )
        # return {
        #     "mean": model_mean,
        #     "variance": model_variance,
        #     "log_variance": model_log_variance,
        #     "pred_xstart": pred_xstart,
        # }

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


            # return {"sample": sample}
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
            # if i == 0:
            #     continue
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

    def _p_sample_no_(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        一个简化版的p_sample，用于“无”反向去噪。
        在这个函数里，我们不再调用(x, flow1)。
        """
        if model_kwargs is None:
            model_kwargs = {}
        # 无时，直接把去掉或让flow1=0
        # 这里演示“彻底去掉”，只做普通后验
        out = self._p_mean_variance_no_(
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

    def _q_posterior_mean_variance_no_(self, x_start, x_t, t, y, model_kwargs=None):
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

    def _p_mean_variance_no_(self, model, x_t, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        无版本，用于从 x_t -> x_{t-1} 的均值与方差。
        """
        if model_kwargs is None:
            model_kwargs = {}

        # 普通模型forward, 不带
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
        model_mean, _, _ = self._q_posterior_mean_variance_no_(
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


    def training_losses(self, model, model1, x_start, y, t, first_stage_model=None, clip_denoised=True, model_kwargs=None,noise=None,autoencoder=None, rank=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param up_sample_lq: Upsampling low-quality image before encoding
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_start)

        B = z_y.shape[0]
        
        t_max = torch.full(t.shape, self.num_timesteps - 1, dtype=t.dtype, device=t.device)
        z_t = self.q_sample(z_start, z_y, t_max, noise=noise)
        # 当前 time-step 计数，从 T-1 开始
        cur_t = t_max.clone()
    
        # 循环，直到所有 cur_t[j] <= target_t[j]
        while (cur_t > t).any():
            # 一次 p_sample1，输入全 batch
            out = self.p_sample1(
                model=model1,
                x=z_t,
                y=z_y,
                t=cur_t,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )
            x_next = out['sample']
    
            # 只更新那些还没到目标步的样本
            mask = (cur_t > t).view(B, *[1] * (z_t.ndim - 1))  # [B,1,1,1,...]
            z_t = torch.where(mask, x_next, z_t)
    
            # 全 batch 时间步都减 1
            cur_t = cur_t - 1


        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
            # scaled_input_b2 = th.cat((self._scale_input(z_t, t),th.zeros_like(z_t)),dim=1)
            model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
            target = {
                ModelMeanType.START_X: z_start,
                ModelMeanType.RESIDUAL: z_y - z_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == z_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["mse"] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            terms["mse"] *= weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output)
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.model_mean_type)

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



