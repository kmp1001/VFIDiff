import enum
import math
import os
import sys

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from FlowformerPlusPlus.visualize_flow import compute_optical_flow, warp
from ldm.models.autoencoder import AutoencoderKLTorch

sys.path.append('/root/autodl-tmp/VFIDiff-journal/ResShift-journal')


def normalize_01(t):
    """
    Normalize a tensor to [0, 1].
    By default, normalization uses the global tensor range.
    """
    t = t.float()
    t_min = t.amin()
    t_max = t.amax()
    return (t - t_min) / (t_max - t_min + 1e-8)


def check_tensor_range(tensor: torch.Tensor, name: str):
    """Print the min/max values of a tensor and roughly judge its range."""
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    print(f"{name} range: [{t_min:.4f}, {t_max:.4f}]")

    if t_min >= 0.0 and t_max <= 1.0:
        print(f"  ==> {name} is likely in the [0, 1] range.\n")
    elif t_min >= -1.0 and t_max <= 1.0:
        print(f"  ==> {name} is likely in the [-1, 1] range.\n")
    else:
        print(f"  ==> {name} is outside the typical [0,1] or [-1,1] range.\n")


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
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
    kwargs=None,
):
    """
    Get a pre-defined eta schedule for the given name.
    """
    if schedule_name == 'exponential':
        power = kwargs.get('power', None)
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(
            1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start)
        )
        base = np.ones([num_diffusion_timesteps]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknown schedule_name {schedule_name}")

    return sqrt_etas


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()       # The model predicts x_0
    EPSILON = enum.auto()       # The model predicts epsilon
    PREVIOUS_X = enum.auto()    # The model predicts x_{t-1}
    RESIDUAL = enum.auto()      # The model predicts residual
    EPSILON_SCALE = enum.auto() # The model predicts scaled epsilon


class LossType(enum.Enum):
    MSE = enum.auto()           # Simplified MSE
    WEIGHTED_MSE = enum.auto()  # Weighted MSE derived from KL


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
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
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

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # Weight for the MSE loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = (
                0.5 / self.posterior_variance_clipped * (self.alpha / self.etas) ** 2
            )
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                kappa * self.alpha / ((1 - self.etas) * self.sqrt_etas)
            ) ** 2
        else:
            raise NotImplementedError(model_mean_type)

        self.weight_loss_mse = weight_loss_mse
        local_rank = int(os.environ.get('LOCAL_RANK', 0))  # Automatically get the GPU for the current process
        torch.cuda.set_device(local_rank)                  # Set the current GPU
        device = torch.device(f"cuda:{local_rank}")

    def q_mean_variance(self, x_start, y, t, model_kwargs=None):
        """
        Get the distribution q(x_t | x_0).
        """
        scale = (t.float() / (self.num_timesteps - 1)).view(-1, 1, 1, 1)
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
        mean = warp(tenOne, flow0)  # TODO: verify the value range
        mean = self.encode_first_stage(mean, model_kwargs["first_stage_model"], up_sample=False)
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa ** 2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(
        self,
        x_start,
        y,
        mid,
        t,
        first_stage_model=None,
        noise=None,
        model_kwargs=None,
        init_x=None,
        init_y=None,
        flowa0=None,
        flowa1=None,
        flowb0=None,
        flowb1=None,
    ):
        # Unify device placement
        device = next(first_stage_model.parameters()).device

        # Move all inputs to the same device
        x_start = x_start.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        init_x = init_x.to(device, non_blocking=True)
        init_y = init_y.to(device, non_blocking=True)
        B = x_start.shape[0]

        flowa0 = flowa0.to(device)
        flowa1 = flowa1.to(device)
        flowb0 = flowb0.to(device)
        flowb1 = flowb1.to(device)

        if noise is None:
            noise = x_start.new_empty_like(x_start).normal_()
        else:
            noise = noise.to(device, non_blocking=True)

        first_half_mask = t <= 6
        second_half_mask = t >= 7

        warp_x_start = torch.zeros_like(init_x, dtype=init_x.dtype)

        with torch.no_grad():
            # Process the first half (t: 0-6)
            if first_half_mask.any():
                t_normalized = t[first_half_mask].float() / 6.0
                warp_result_1 = warp(
                    init_x[first_half_mask],
                    flowa0[first_half_mask],
                    flowa1[first_half_mask],
                    t_normalized,
                    6,
                ).to(warp_x_start.dtype)
                warp_x_start[first_half_mask] = warp_result_1.to(device)

            # Process the second half (t: 7-12)
            if second_half_mask.any():
                t_normalized = (t[second_half_mask] - 6).float() / 6.0
                warp_result_2 = warp(
                    mid[second_half_mask],
                    flowb0[second_half_mask],
                    flowb1[second_half_mask],
                    t_normalized,
                    6,
                ).to(warp_x_start.dtype)
                warp_x_start[second_half_mask] = warp_result_2.to(device)

            warp_z = self.encode_first_stage(warp_x_start, first_stage_model, up_sample=False).detach()

        # Add the noise term back
        eps_factor = _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape).to(device)
        return warp_z + eps_factor * noise, warp_x_start

    def q_posterior_mean_variance(self, x_start, x_t, t, y, model_kwargs=None, input1=None, first_stage_model=None):
        """
        Here x_start is the predicted result, more precisely x_{t-1}.
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        B = x_t.shape[0]

        posterior_mean = (
            (_extract_into_tensor(self.etas, t - 1, x_start.shape) / _extract_into_tensor(self.etas, t, x_start.shape)) * input1
            + (
                (_extract_into_tensor(self.etas, t, x_start.shape) - _extract_into_tensor(self.etas, t - 1, x_start.shape))
                / _extract_into_tensor(self.etas, t, x_start.shape)
            ) * x_start
        )

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_mean_variance1(self, x_start, x_t, t, y, model_kwargs=None):
        """
        Here x_start is the predicted result, more precisely x_{t-1}.
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        B = x_t.shape[0]
        posterior_mean = (
            (_extract_into_tensor(self.etas, t - 1, x_start.shape) / _extract_into_tensor(self.etas, t, x_start.shape)) * x_t
            + (
                (_extract_into_tensor(self.etas, t, x_start.shape) - _extract_into_tensor(self.etas, t - 1, x_start.shape))
                / _extract_into_tensor(self.etas, t, x_start.shape)
            ) * x_start
        )

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model,
        start,
        z_t,
        y,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        first_stage_model=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = z_t.shape[:2]
        device = t.device

        model_kwargs['flowa0'] = model_kwargs['flowa0'].to(device)
        model_kwargs['flowa1'] = model_kwargs['flowa1'].to(device)
        model_kwargs['flowb0'] = model_kwargs['flowb0'].to(device)
        model_kwargs['flowb1'] = model_kwargs['flowb1'].to(device)

        first_half_mask = t <= 6
        second_half_mask = t >= 7

        with torch.no_grad():
            x_t = self.decode_first_stage(
                z_t,
                first_stage_model=first_stage_model,
                consistencydecoder=None,
            ).clamp(-1.0, 1.0)
            x_t = x_t.to(device)

            warp_t = torch.zeros_like(x_t, dtype=x_t.dtype)
            step_flow = th.zeros(
                x_t.shape[0], 2, x_t.shape[2], x_t.shape[3],
                dtype=x_t.dtype, device=x_t.device
            )

            # First-half samples: use flowa to move one step forward
            if first_half_mask.any():
                current_t = t[first_half_mask].float()
                flow_ratio1 = (current_t / 6.0).view(-1, 1, 1, 1)
                step_ratio1 = 1.0 / current_t

                step_flow_a1 = model_kwargs['flowa1'][first_half_mask] * flow_ratio1
                step_flow_a0 = model_kwargs['flowa0'][first_half_mask] * flow_ratio1
                step_flow[first_half_mask] = step_flow_a1 * step_ratio1.view(-1, 1, 1, 1)

                warp_result_a = warp(
                    x_t[first_half_mask],
                    step_flow_a1,
                    step_flow_a0,
                    step_ratio1,
                    current_t,
                ).to(warp_t.dtype)
                warp_t[first_half_mask] = warp_result_a

            # Second-half samples: use flowb to move one step forward
            if second_half_mask.any():
                current_t = (t[second_half_mask] - 6).float()
                flow_ratio2 = (current_t / 6.0).view(-1, 1, 1, 1)
                step_ratio2 = 1.0 / current_t

                step_flow_b1 = model_kwargs['flowb1'][second_half_mask] * flow_ratio2
                step_flow_b0 = model_kwargs['flowb0'][second_half_mask] * flow_ratio2
                step_flow[second_half_mask] = step_flow_b1 * step_ratio2.view(-1, 1, 1, 1)

                warp_result_b = warp(
                    x_t[second_half_mask],
                    step_flow_b1,
                    step_flow_b0,
                    step_ratio2,
                    current_t,
                ).to(warp_t.dtype)
                warp_t[second_half_mask] = warp_result_b

            warp_t = self.encode_first_stage(warp_t, first_stage_model, up_sample=False)

        warp_t.to(device)
        warp_t_1 = self._scale_input(warp_t, (t - 1).to(device)).to(device)
        t_before = (t - 1).to(device)
        model_kwargs = {key: value.to(device) for key, value in model_kwargs.items()}

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

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = process_xstart(self._predict_xstart_from_residual(y=y, residual=model_output))
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=z_t, y=y, t=t, eps=model_output))
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(self._predict_xstart_from_eps_scale(x_t=z_t, y=y, t=t, eps=model_output))
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart,
            x_t=z_t,
            t=t,
            y=y,
            model_kwargs=model_kwargs,
            input1=warp_t_1,
            first_stage_model=first_stage_model,
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_mean_variance1(
        self,
        model,
        x_t,
        y,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]

        input_b2 = x_t
        scaled_input_b2 = self._scale_input(input_b2, t)
        zeros_2ch = th.zeros(
            scaled_input_b2.shape[0],
            2,
            scaled_input_b2.shape[2],
            scaled_input_b2.shape[3],
            dtype=scaled_input_b2.dtype,
            device=scaled_input_b2.device,
        )
        scaled_input_b2 = th.cat((scaled_input_b2, zeros_2ch), dim=1)
        origin_kwargs = {
            key: value.to(scaled_input_b2.device)
            for key, value in model_kwargs.items()
            if key not in ['tenMean', 'tenStd', 'tenOne_n', 'tenTwo_n', 'first_stage_model', 'occ']
        }

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

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = process_xstart(self._predict_xstart_from_residual(y=y, residual=model_output))
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output))
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output))
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance1(
            x_start=pred_xstart,
            x_t=x_t,
            t=t,
            y=y,
            model_kwargs=model_kwargs,
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        return (
            x_t
            - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
            - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        return (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        return y - residual

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t
            - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
            - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(
        self,
        model,
        start,
        x,
        y,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        noise_repeat=False,
        first_stage_model=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
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
                first_stage_model=first_stage_model,
            )
            device = out["mean"].device
            noise = th.randn_like(x).to(device)
            if noise_repeat:
                noise = noise[0].repeat(x.shape[0], 1, 1, 1).to(device)
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device)
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        else:
            sample = x
            return {"sample": sample}

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def p_sample1(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.
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
            noise = noise[0].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
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

    # Used during validation: generate step by step and return intermediate results for monitoring and debugging
    def p_sample_loop_progressive(
        self,
        start,
        y,
        model,
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
        Generate samples from the model and yield intermediate samples from each timestep.
        """
        if device is None:
            device = next(model.parameters()).device
        with torch.no_grad():
            z_y = self.encode_first_stage(y, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
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
                    first_stage_model=first_stage_model,
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
        Generate samples from the model and yield intermediate samples from each timestep.
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

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
                if i == mid_step:
                    mid_frame = out["sample"]

                yield out
                img = out["sample"]

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
            # Use the provided scale_factor, or fall back to self.scale_factor
            effective_scale_factor = scale_factor if scale_factor is not None else self.scale_factor
            z_sample = 1 / effective_scale_factor * z_sample
            if consistencydecoder is None:
                out = decoder(z_sample.type(model_dtype))
            else:
                with th.cuda.amp.autocast():
                    out = decoder(z_sample)
            if model_dtype != data_dtype:
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
            if model_dtype != data_dtype:
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
            if model_dtype != data_dtype:
                y = y.type(model_dtype)
            with th.no_grad():
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
            if model_dtype != data_dtype:
                out = out.type(data_dtype)
            return out

    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e. q(x_T|x_0) ~= N(x_T|y, ~)
        """
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps - 1] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def _p_sample_no_warp(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        A simplified p_sample used for reverse denoising without warp.
        In this function, warp(x, flow1) is no longer used.
        """
        if model_kwargs is None:
            model_kwargs = {}

        out = self._p_mean_variance_no_warp(
            model=model,
            x_t=x,
            y=y,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def _q_posterior_mean_variance_no_warp(self, x_start, x_t, t, y, model_kwargs=None):
        """
        Here x_start is the predicted result, more precisely x_{t-1}.
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        if model_kwargs is None:
            model_kwargs = {}
        B = x_t.shape[0]
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _p_mean_variance_no_warp(self, model, x_t, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        No-warp version used to compute the mean and variance from x_t -> x_{t-1}.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Standard model forward without warp
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        # Reuse posterior_variance logic for variance/log_variance
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # Predict x_0
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
            x_start=pred_xstart,
            x_t=x_t,
            t=t,
            y=y,
            model_kwargs=model_kwargs,
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def training_losses(
        self,
        model,
        x_start,
        y,
        mid,
        im2,
        im3,
        im5,
        im6,
        t,
        flowa0,
        flowa1,
        flowb0,
        flowb1,
        first_stage_model=None,
        model_kwargs=None,
        noise=None,
        rank=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        device = next(first_stage_model.parameters()).device

        x_start = x_start.to(device)
        y = y.to(device)
        mid = mid.to(device)
        im2 = im2.to(device)
        im3 = im3.to(device)
        im5 = im5.to(device)
        im6 = im6.to(device)
        t = t.to(device)
        B = x_start.shape[0]

        flowa0 = flowa0.to(device)
        flowa1 = flowa1.to(device)
        flowb0 = flowb0.to(device)
        flowb1 = flowb1.to(device)

        with torch.no_grad():
            z_y = self.encode_first_stage(y, first_stage_model, up_sample=False).to(device)
            z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False).to(device)

            if noise is None:
                noise = th.randn_like(z_start).to(device)

            z_t, x_t = self.q_sample(
                z_start,
                z_y,
                mid,
                t,
                first_stage_model=first_stage_model,
                noise=noise,
                model_kwargs=model_kwargs,
                init_x=x_start,
                init_y=y,
                flowa0=flowa0,
                flowa1=flowa1,
                flowb0=flowb0,
                flowb1=flowb1,
            )
            B = z_t.shape[0]
            model_output = th.empty_like(z_t)

            first_half_mask = t <= 6
            second_half_mask = t >= 7
            z_t1 = self.decode_first_stage(
                z_t,
                first_stage_model=first_stage_model,
                consistencydecoder=None,
            ).clamp(-1.0, 1.0)

            step_flow = th.zeros(
                x_start.shape[0], 2, x_start.shape[2], x_start.shape[3],
                dtype=x_start.dtype, device=x_start.device
            )

            # First-half samples: use flowa to move one step forward
            if first_half_mask.any():
                current_t = t[first_half_mask].float()
                flow_ratio1 = (current_t / 6.0).view(-1, 1, 1, 1)
                step_ratio1 = 1.0 / current_t

                step_flow_a1 = flowa1[first_half_mask] * flow_ratio1
                step_flow_a0 = flowa0[first_half_mask] * flow_ratio1
                step_flow[first_half_mask] = step_flow_a1 * step_ratio1.view(-1, 1, 1, 1)

            # Second-half samples: use flowb to move one step forward
            if second_half_mask.any():
                current_t = (t[second_half_mask] - 6).float()
                flow_ratio2 = (current_t / 6.0).view(-1, 1, 1, 1)
                step_ratio2 = 1.0 / current_t

                step_flow_b1 = flowb1[second_half_mask] * flow_ratio2
                step_flow_b0 = flowb0[second_half_mask] * flow_ratio2
                step_flow[second_half_mask] = step_flow_b1 * step_ratio2.view(-1, 1, 1, 1)

            step_flow = F.interpolate(step_flow, scale_factor=1 / 4, mode='bicubic') / 4
            scaled_input_input = th.cat((z_t, step_flow), dim=1)

            first_half_mask = t <= 7
            second_half_mask = t >= 8

            warp_x_start = torch.zeros_like(x_start, dtype=x_start.dtype)

            # Process the first half (t: 0-6)
            if first_half_mask.any():
                t_normalized = (t[first_half_mask] - 1).float() / 6.0
                warp_result_1 = warp(
                    x_start[first_half_mask],
                    flowa0[first_half_mask],
                    flowa1[first_half_mask],
                    t_normalized,
                    6,
                ).to(warp_x_start.dtype)
                warp_x_start[first_half_mask] = warp_result_1.to(device)

            # Process the second half (t: 7-12)
            if second_half_mask.any():
                t_normalized = (t[second_half_mask] - 7).float() / 6.0
                warp_result_2 = warp(
                    mid[second_half_mask],
                    flowb0[second_half_mask],
                    flowb1[second_half_mask],
                    t_normalized,
                    6,
                ).to(warp_x_start.dtype)
                warp_x_start[second_half_mask] = warp_result_2.to(device)

            warp_x_start = self.encode_first_stage(warp_x_start, first_stage_model, up_sample=False)

        origin_kwargs = {
            key: value.to(t.device)
            for key, value in model_kwargs.items()
            if key not in ['first_stage_model']
        }

        model.model.to(t.device)
        model_output = model(scaled_input_input, t, **origin_kwargs).float()

        if self.model_mean_type == ModelMeanType.START_X:
            target = warp_x_start
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            target = z_y - z_start
        elif self.model_mean_type == ModelMeanType.EPSILON:
            target = noise
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            target = noise * self.kappa * _extract_into_tensor(self.sqrt_etas, t, noise.shape)
        else:
            raise NotImplementedError(self.model_mean_type)

        # z_sample is the newly sampled latent at step = t-1, shape [B, C, H, W]
        steps_per_sample = t.long()
        max_steps = int(steps_per_sample.max().item())

        # Pre-encode the ground-truth latents we want to compare against
        gt_latents = {
            0: z_start,
            2: self.encode_first_stage(im6, first_stage_model, up_sample=False),
            4: self.encode_first_stage(im5, first_stage_model, up_sample=False),
            6: self.encode_first_stage(mid, first_stage_model, up_sample=False),
            8: self.encode_first_stage(im3, first_stage_model, up_sample=False),
            10: self.encode_first_stage(im2, first_stage_model, up_sample=False),
            12: z_y,
        }
        compare_steps = list(gt_latents.keys())

        latent_current = z_t.clone()
        latent_snapshots = torch.zeros_like(latent_current)

        # First store the result for samples requiring 0 denoising steps
        mask0 = steps_per_sample == 0
        if mask0.any():
            latent_snapshots[mask0] = latent_current[mask0]
        value = z_t.clone()

        # Denoise the whole batch for max_steps rounds
        for i in range(max_steps):
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

            mask_i1 = steps_per_sample == (i + 1)
            if mask_i1.any():
                latent_snapshots[mask_i1] = value[mask_i1]

        # Compute L1 only for samples whose steps are in compare_steps
        l1_sum = 0.0
        count = 0
        for step in compare_steps:
            mask = steps_per_sample == step
            if mask.any():
                pred = latent_snapshots[mask]
                target1 = gt_latents[step][mask]
                l1_sum += F.l1_loss(pred, target1, reduction='mean') * mask.sum().item()
                count += mask.sum().item()

        loss_flow_interpolation = l1_sum / max(count, 1)

        loss_mse_per_sample = (target - model_output).abs().view(B, -1).mean(dim=1)
        non_mid_loss_total = loss_mse_per_sample.mean()

        losses = 0.5 * non_mid_loss_total + 0.5 * loss_flow_interpolation

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

        return terms, z_t, pred_zstart

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                inputs = inputs.to(t.device)
                # The variance of latent code is around 1.0
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
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.scale_factor = scale_factor
        self.sf = sf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and related terms
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, y, t, model_kwargs=None):
        """
        Get the distribution q(x_t | x_0).
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
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

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of x_0.
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
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # model_var_values is in [-1, 1] for [min_var, max_var]
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
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

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
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
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.
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
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
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
        Generate samples from the model and yield intermediate samples from each timestep.
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
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but re-derive it here
        # in case x_start or x_prev prediction was used
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
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
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12 reversed
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
        Use DDIM to sample from the model and yield intermediate samples from each timestep.
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
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=z_start, x_t=z_t, t=t)[0],
            ModelMeanType.START_X: z_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == z_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in bits-per-dim.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
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
