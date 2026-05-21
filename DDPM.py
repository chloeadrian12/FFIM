from diffusers import DDPMScheduler
import torch
import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
import albumentations as A
from torchvision import transforms
from PIL import Image

albu_pre_val = A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512, p=1.0),
        A.CenterCrop(height=512, width=512, p=1.0),
        ],
        p=1.0)

imagenet_norm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

class DDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def parallel_noise(V, N, eps=1e-8):

    # bs, c, h, w = V.shape
    bs, c, h, w = V.shape
    d = c * h * w
    V_flat = V.view(bs, d)                  # [bs, d]
    N_flat = N.view(bs, d)          # [bs, d]

    V_norm = V_flat / (torch.norm(V_flat, dim=1, keepdim=True) + eps)  # [bs, d]

    proj_coeff = torch.sum(N_flat * V_norm, dim=1, keepdim=True)    # [bs, 1]

    N_P_flat = proj_coeff * V_norm  # [bs, d]

    N_P_norm = torch.norm(N_P_flat, dim=1, keepdim=True) + eps
    N_orig_norm = torch.norm(N_flat, dim=1, keepdim=True) + eps
    scale_factor = N_orig_norm / N_P_norm
    N_P_flat = N_P_flat * scale_factor

    return N_P_flat.view(bs, c, h, w)



def orthogonal_noise(V, N, epsilon_raw, eps=1e-8):

    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    N_flat = N.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_norm = V_flat / (torch.norm(V_flat, dim=1, keepdim=True) + eps)
    N_norm = N_flat / (torch.norm(N_flat, dim=1, keepdim=True) + eps)

    basis = torch.stack([V_norm, N_norm], dim=1)

    proj_coeff = torch.einsum("bd,bpd->bp", epsilon_flat, basis)

    projection = torch.einsum("bp,bpd->bd", proj_coeff, basis)
    epsilon_ortho_flat = epsilon_flat - projection

    epsilon_parallel = projection.view(bs, c, h, w)
    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)

    return epsilon_ortho, epsilon_parallel

def orthogonal_noise_V(V, epsilon_raw, eps=1e-8):
    #  default L2
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_norm = V_flat / (torch.norm(V_flat, dim=1, keepdim=True) + eps)

    proj_coeff = torch.einsum("bd,bd->b", epsilon_flat, V_norm).unsqueeze(1)

    projection = proj_coeff * V_norm

    epsilon_ortho_flat = epsilon_flat - projection

    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection

def orthogonal_noise_V_mvss(V, epsilon_raw, eps=1e-8):
    #  default L2
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_norm = V_flat / (torch.norm(V_flat, dim=1, keepdim=True) + eps)

    proj_coeff = torch.einsum(
        "bd,bd->b",
        epsilon_flat.float(),
        V_norm.float()
    ).unsqueeze(1)

    projection = proj_coeff * V_norm

    epsilon_ortho_flat = epsilon_flat - projection

    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection

def orthogonal_noise_L1(V, epsilon_raw, eps=1e-8):
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_norm = V_flat / (torch.norm(V_flat, p=1, dim=1, keepdim=True) + eps)

    proj_coeff = torch.sum(epsilon_flat * torch.sign(V_norm), dim=1, keepdim=True)
    projection = proj_coeff * torch.sign(V_norm) / d

    epsilon_ortho_flat = epsilon_flat - projection
    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection


def orthogonal_noise_cosine(V, epsilon_raw, eps=1e-8):
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_norm = torch.norm(V_flat, dim=1, keepdim=True)
    epsilon_norm = torch.norm(epsilon_flat, dim=1, keepdim=True)

    cosine_sim = torch.einsum("bd,bd->b", V_flat, epsilon_flat) / (V_norm.squeeze() * epsilon_norm.squeeze() + eps)

    proj_coeff = cosine_sim.unsqueeze(1) * epsilon_norm
    projection = proj_coeff * V_flat / (V_norm + eps)

    epsilon_ortho_flat = epsilon_flat - projection
    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection


def orthogonal_noise_Linf(V, epsilon_raw, eps=1e-8):
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_abs_max, _ = torch.max(torch.abs(V_flat), dim=1, keepdim=True)
    V_norm = V_flat / (V_abs_max + eps)

    proj_coeff = torch.einsum("bd,bd->b", epsilon_flat, V_norm).unsqueeze(1)
    projection = proj_coeff * V_norm

    epsilon_ortho_flat = epsilon_flat - projection
    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection


def orthogonal_noise_kernel(V, epsilon_raw, eps=1e-8):
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_norm = V_flat / (torch.norm(V_flat, dim=1, keepdim=True) + eps)

    V_norm_expanded = V_norm.unsqueeze(2)  # [bs, d, 1]
    VVT = torch.bmm(V_norm_expanded, V_norm_expanded.transpose(1, 2))  # [bs, d, d]

    identity = torch.eye(d, device=V.device).unsqueeze(0).repeat(bs, 1, 1)
    projection_matrix = identity - VVT

    epsilon_ortho_flat = torch.bmm(projection_matrix, epsilon_flat.unsqueeze(2)).squeeze(2)
    projection = epsilon_flat - epsilon_ortho_flat

    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection


def orthogonal_noise_weight_norm(V, epsilon_raw, eps=1e-8):
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    V_direction = V_flat / (torch.norm(V_flat, dim=1, keepdim=True) + eps)
    V_magnitude = torch.norm(V_flat, dim=1, keepdim=True)

    proj_coeff = torch.einsum("bd,bd->b", epsilon_flat, V_direction).unsqueeze(1)

    weights = torch.softmax(torch.abs(V_flat), dim=1)
    weighted_proj = proj_coeff * weights * V_magnitude

    projection = weighted_proj * V_direction
    epsilon_ortho_flat = epsilon_flat - projection

    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection.view(bs, c, h, w)

    return epsilon_ortho, projection


def orthogonal_noise_chunk(V, epsilon_raw, eps=1e-8, chunk_size=64):
    bs, c, h, w = V.shape
    d = c * h * w

    V_flat = V.view(bs, d)
    epsilon_flat = epsilon_raw.view(bs, d)

    n_chunks = (d + chunk_size - 1) // chunk_size
    epsilon_ortho_chunks = []
    projection_chunks = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, d)

        V_chunk = V_flat[:, start_idx:end_idx]
        epsilon_chunk = epsilon_flat[:, start_idx:end_idx]

        V_chunk_norm = V_chunk / (torch.norm(V_chunk, dim=1, keepdim=True) + eps)
        proj_coeff = torch.einsum("bd,bd->b", epsilon_chunk, V_chunk_norm).unsqueeze(1)
        projection_chunk = proj_coeff * V_chunk_norm

        epsilon_ortho_chunk = epsilon_chunk - projection_chunk

        epsilon_ortho_chunks.append(epsilon_ortho_chunk)
        projection_chunks.append(projection_chunk)

    epsilon_ortho_flat = torch.cat(epsilon_ortho_chunks, dim=1)
    projection_flat = torch.cat(projection_chunks, dim=1)

    epsilon_ortho = epsilon_ortho_flat.view(bs, c, h, w)
    projection = projection_flat.view(bs, c, h, w)

    return epsilon_ortho, projection

def adjust_similarity(V, N, target_sim=0.5):

    bs, c, h, w = V.shape
    d = c * h * w

    V = V.view(bs, d)
    N = N.view(bs, d)

    V_norm = V / torch.linalg.norm(V, dim=1, keepdim=True)
    N_norm = N / torch.linalg.norm(N, dim=1, keepdim=True)

    current_sim = F.cosine_similarity(V_norm, N_norm, dim=1)

    current_sim = float(current_sim)

    if target_sim > current_sim:
        noise_adj = N_norm + (target_sim - current_sim) * V_norm * torch.linalg.norm(N_norm)
    else:
        proj = N_norm - (N_norm * V_norm).sum(dim=1, keepdim=True) * V_norm
        proj_norm = proj / torch.norm(proj, dim=1, keepdim=True)
        noise_adj = N_norm * torch.norm(proj_norm, p=2, dim=1, keepdim=True) * (1 - target_sim**2)**0.5

    return noise_adj.view(bs, c, h, w)

def fgsm(model, pipe, I_V, Var, mask, generator, device, epsilon=0.03):

    for param in model.parameters():
        param.requires_grad = False
    condition_kwargs = {}
    I = pipe.vae.decode(
                I_V / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
                **condition_kwargs
            )[0]
    image, has_nsfw_concept = pipe.run_safety_checker(I, device, 'torch.float32')

    do_denormalize = [True] * image.shape[0]
    image = image.detach()
    I_i = pipe.image_processor.postprocess(image, output_type="pil",
                                             do_denormalize=do_denormalize)[0]

    I_var = pipe.vae.decode(
                Var / pipe.vae.config.scaling_factor, return_dict=False, generator=generator,
                **condition_kwargs
            )[0]
    image, has_nsfw_concept = pipe.run_safety_checker(I_var, device, 'torch.float32')

    do_denormalize = [True] * image.shape[0]
    image = image.detach()
    I_var1 = pipe.image_processor.postprocess(image, output_type="pil",
                                         do_denormalize=do_denormalize)[0]

    I_np = np.array(I_i)
    I_vnp = np.array(I_var1)
    mk = np.array(mask)
    preprocessed_I = albu_pre_val(image=I_np)
    preprocessed_Iv = albu_pre_val(image=I_vnp)
    preprocessed_mask = albu_pre_val(image=mk)
    I_i = preprocessed_I['image']
    I_v = preprocessed_Iv['image']
    m = preprocessed_mask['image']

    I_t = imagenet_norm(I_i)
    Iv_t = imagenet_norm(I_v)
    I_b = I_t.unsqueeze(0)
    Iv_b = Iv_t.unsqueeze(0)
    I_p = torch.cat([I_b, Iv_b])
    I_p = torch.cat([I_p,I_p], dim=0)

    I_p = I_p.to(device).requires_grad_(True) # .detach()

    m = mask_norm(m)
    m = m.unsqueeze(0)
    m = m.to(device)
    m = F.interpolate(m, size=(I_p.size()[2] // 16, I_p.size()[3] // 16), mode='nearest')
    _, noise = model(image_input=I_p, isTrain=False)
    I_n = noise[0]
    I_vn = noise[1]
    m = m[0]
    m = m[0:1,:,:]
    m = m.repeat(320, 1, 1)
    f = I_vn[m == 1]
    bg = I_n[m == 0]
    indices = torch.linspace(0, bg.size()[0] - 1, steps=f.size()[0] + 1).round().long()
    bg_gm = torch.stack([
        bg[indices[i]:indices[i + 1]].mean()
        for i in range(f.size()[0])
    ])
    cos = -F.cosine_similarity(f, bg_gm, dim=0).mean()

    model.zero_grad()
    grad = torch.autograd.grad(cos, I_p, retain_graph=False)[0]
    print(cos)
    del noise, I_n, I_vn, f, bg, bg_gm, cos

    image_grad = grad[1].data
    with torch.no_grad():
        perturbation = epsilon * torch.sign(image_grad)
        perturbation = perturbation.unsqueeze(0)
        o_var, p_var = orthogonal_noise(I, perturbation, I_var)
        var_latent = pipe._encode_vae_image(image=p_var, generator=generator)
    torch.cuda.empty_cache()

    return var_latent

def scale_by_max_abs(small_sample, large_sample):

    max_abs_large = torch.max(torch.abs(large_sample))
    max_abs_small = torch.max(torch.abs(small_sample))

    if max_abs_small == 0:
        return small_sample

    scale_factor = max_abs_large / max_abs_small

    scaled_small = small_sample * scale_factor
    return scaled_small

def linear_range_scaling(small_sample, large_sample):
    min_large = torch.min(large_sample)
    max_large = torch.max(large_sample)
    min_small = torch.min(small_sample)
    max_small = torch.max(small_sample)

    if max_small == min_small:
        return torch.full_like(small_sample, min_large)

    scaled_small = (small_sample - min_small) * (max_large - min_large) / (max_small - min_small) + min_large
    return scaled_small

class OrthogonalDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Parastep(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            init_latent: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            noise_ortho, noise_para = orthogonal_noise(image_latent, init_latent, variance_noise)
            x = noise_ortho
            noise_ortho = linear_range_scaling(noise_para, noise_ortho)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * noise_ortho
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * noise_ortho
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * noise_ortho

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def Orthstep(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            init_latent: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            noise_ortho, noise_para = orthogonal_noise(image_latent, init_latent, variance_noise)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * noise_ortho
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * noise_ortho
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * noise_ortho

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
                variance_noise,
                noise_ortho
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


    def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            init_latent: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,

            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


    def adjstep(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            init_latent: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            adjn = adjust_similarity(image_latent, variance_noise)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * adjn
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * adjn
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * adjn

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def fgsmstep(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            init_latent: torch.Tensor,
            mask: torch.Tensor,
            model,
            pipe,
            seed1_g=None,
            seed2_g=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=seed2_g, device=device, dtype=model_output.dtype
            )

            # noise_ortho = variance_noise
            noise_ortho, noise_para = orthogonal_noise(image_latent, init_latent, variance_noise)
            # noise_ortho = fgsm(model, pipe, variance_noise, init_latent, mask, seed1_g, device)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * noise_ortho
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * noise_ortho
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * noise_ortho

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def OVstep(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            mask: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep
        a = 0.9

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":   # default
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            noise_ortho, noise_para = orthogonal_noise_V(image_latent, variance_noise)
            noise_ortho = (1 - a) * mask * noise_ortho + a * noise_ortho

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * noise_ortho
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * noise_ortho
            else:  # default
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * noise_ortho

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
                variance_noise,
                noise_ortho
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


    def mvss_OVstep(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            image_latent: torch.Tensor,
            mask: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        t = timestep
        a = 0.9

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":   # default
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            noise_ortho, noise_para = orthogonal_noise_V_mvss(image_latent, variance_noise)
            noise_ortho = (1 - a) * mask * noise_ortho + a * noise_ortho  #  weight

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * noise_ortho
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * noise_ortho
            else:  # default
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * noise_ortho

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
                variance_noise,
                noise_ortho
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)