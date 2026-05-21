from contextlib import nullcontext
import torch
from PIL import Image
import numpy as np
import os
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler
from datasets import load_dataset, Features, Array3D, Value, Dataset
from DDPM import OrthogonalDDPMScheduler
import pyarrow as pa
import pyarrow.parquet as pq
import random
from tqdm import tqdm
import argparse
from torch.nn import functional as F
from test_docker.src.config import update_config
import pyarrow.parquet as pq
from test_docker.src.config import _C as config
from test_docker.metrics import computeLocalizationMetrics, computeDetectionMetrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='DDPM_noise', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)


args = parser.parse_args()
update_config(config, args)

def process(pipe, step, device, image, mk, pt, generator):
    h = pipe.unet.config.sample_size * pipe.vae_scale_factor
    w = pipe.unet.config.sample_size * pipe.vae_scale_factor
    prompt_embeds = None
    num_inference_steps = step
    latents = None

    with torch.no_grad():
        pipe.check_inputs(
            pt,
            image,
            mk,
            h,
            w,
            1,
            None,
            "pil",
            None,
            None,
            None,
            None,
            None,
            None,
            None, )

        if pt is not None and isinstance(pt, str):  # bs由prompt的个数决定
            batch_size = 1
        elif pt is not None and isinstance(pt, list):
            batch_size = len(pt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            pt,
            device,
            1,
            True,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        pipe.scheduler.set_timesteps(num_inference_steps, device=device,)
        timesteps = pipe.scheduler.timesteps

        timesteps, num_inference_steps = pipe.get_timesteps(
            num_inference_steps=num_inference_steps, strength=1, device=device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * 1)
        # embedding image and mask
        original_image = image
        init_image = pipe.image_processor.preprocess(
            image, height=h, width=w, crops_coords=None, resize_mode='default'
        )

        latents_outputs = pipe.prepare_latents(
            batch_size * 1,
            pipe.vae.config.latent_channels,
            h,
            w,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=True,  # strength = 1.0时为True
            return_noise=True,
            return_image_latents=True,  # 默认unet通道=4时True
        )

        latents, noise, image_latents = latents_outputs

        mask_condition = pipe.mask_processor.preprocess(
            mk, height=h, width=w, resize_mode='default', crops_coords=None
        )
        masked_image = init_image * (mask_condition < 0.5)

        mask, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * 1,
            h,
            w,
            prompt_embeds.dtype,
            device,
            generator,
            True,
        )

        # v2 执行验证
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if pipe.vae.config.latent_channels + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {pipe.vae.config.latent_channels} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {pipe.vae.config.latent_channels + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)

        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * 1)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

    return{
            "prompt_embeds": prompt_embeds,
            "latents": latents,
            "image_latents": image_latents,
            "mask": mask,
            "masked_image_latents": masked_image_latents,
            "extra_step_kwargs": extra_step_kwargs,
            "timestep_cond": timestep_cond,
            "timesteps": timesteps,
            "num_inference_steps": num_inference_steps
        }

def weight_mask(m: torch.Tensor, k: int = 3) -> torch.Tensor:

    pad = (k - 1) // 2
    padded = torch.nn.functional.pad(
        m,
        pad=(pad, pad, pad, pad),
        mode='reflect'
    )

    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    b, c, h, w, *_ = patches.shape
    patches_flat = patches.reshape(b, c, h, w, -1)

    probs = torch.softmax(patches_flat, dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-6), dim=-1)

    emin = entropy.view(b, c, -1).min(dim=-1)[0].view(b, c, 1, 1)
    emax = entropy.view(b, c, -1).max(dim=-1)[0].view(b, c, 1, 1)

    enorm = (entropy - emin) / (emax - emin + 1e-6)

    wm = m * enorm
    return wm

def load_seeds(path="seeds.txt"):
    with open(path, "r") as f:
        return [int(x.strip()) for x in f if x.strip()]

def load_trufor_model(config, weight_path, device):
    print(f"=> loading model from {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")
    from test_docker.src.models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(f"cuda:{device}").eval()
    return model

def preprocess_mask_for_trufor(mask_pil, device):
    # 原逻辑：16x16 mask + flatten bool
    m = torch.tensor(np.array(mask_pil), dtype=torch.float32) / 255.0  # [H,W]
    m = m.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    m16 = F.interpolate(m, size=(16, 16), mode="nearest")
    m16 = (m16 > 0.5).float()
    mask_flat = m16.view(-1).bool()  # [256]
    return mask_flat

@torch.inference_mode()
def detection_fast(model, image_pil, mask_flat, device):
    img = torch.tensor(np.array(image_pil.convert("RGB")).transpose(2,0,1), dtype=torch.float32) / 255.0
    img = img.unsqueeze(0).to(device)  # [1,3,H,W]

    _, _, _, _, _, _, x_rgb_out, _ = model(img)
    features = x_rgb_out[3]  # [B,C,16,16]
    B, C, H, W = features.shape
    feat_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)  # [256, C]

    f1 = feat_flat[mask_flat]
    f0 = feat_flat[~mask_flat]

    # 让 f0 和 f1 数量对齐（你原来是均匀分段求均值）
    group_size = max(1, len(f0) // len(f1))
    f0_grouped = f0[:len(f1) * group_size].view(len(f1), group_size, -1).mean(dim=1)

    cos = F.cosine_similarity(f1, f0_grouped, dim=1).mean()
    return float(cos)

def guided(model, image, mask, device,alpha=0.2):
    # alpha = alpha * (t / total_timesteps)

    with torch.no_grad():
        img_RGB = np.array(image.convert("RGB"))
        img_RGB = img_RGB.transpose(2, 0, 1)  # [C, H, W]
        img_RGB = torch.tensor(img_RGB, dtype=torch.float32) / 255.0
        img_RGB = img_RGB.unsqueeze(0).to(device).requires_grad_(True)

    mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    mask = mask.to(device)

    with torch.enable_grad():
        _, _, _, _, _, _, x_rgb_out, _ = model(img_RGB)

        with torch.no_grad():
            # 创建下采样mask - 不可导但必要
            m16 = F.interpolate(mask, size=(16, 16), mode='nearest')
            m16 = (m16 > 0.5).float()  # 二值化
            mask_flat = m16.view(-1).bool()

        features = x_rgb_out[3]
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
        f1 = features_flat[mask_flat]
        f0 = features_flat[~mask_flat]
        group_size = max(1, len(f0) // len(f1))
        f0_grouped = f0[:len(f1) * group_size].view(len(f1), group_size, -1).mean(dim=1)
        cos = F.cosine_similarity(f1, f0_grouped, dim=1).mean()
        cos.backward(retain_graph=True)
        img_grad = img_RGB.grad.detach()
        grad_norm = torch.norm(img_grad)
        if grad_norm > 1e-8:
            img_grad = img_grad / grad_norm

        # 只更新mask区域
        mask_expanded = F.interpolate(mask, size=img_RGB.shape[2:], mode='bilinear')
        img_grad_masked = img_grad * mask_expanded

        # 应用修正到输入图像
        img_modified = img_RGB.detach() - alpha * img_grad_masked
        with torch.no_grad():
            img_out = torch.clamp(img_modified, 0, 1)


        del img_RGB, mask, mask_expanded, mask_flat, img_grad, img_modified, img_grad_masked, features, features_flat
        del f0, f1, f0_grouped, x_rgb_out, cos
        return img_out

def noise_fast(seed, devices, save_path, step, get_step, model):
    device = f"cuda:{devices}"
    torch.cuda.set_device(devices)

    # 固定随机
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 读 seeds 只做一次
    all_seeds = load_seeds("seeds.txt")

    # pipeline（fp16 + 关 safety）
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "models/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to(f"cuda:{devices}")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe.scheduler = OrthogonalDDPMScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device=f"cuda:{devices}").manual_seed(seed)

    device = pipe.device

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    with open('/data/chy/data/anicoco/anicocopt.txt', 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    path_res = '/data/chy/data/anicoco/test'
    os.makedirs(path_res, exist_ok=True)

    # 推理上下文：fp16 推荐 autocast
    autocast_ctx = torch.cuda.amp.autocast if device.startswith("cuda") else nullcontext

    for line in tqdm(lines):
        img_path, mk_path, _ = line.split(',', 2)
        pt = line.split(';')[-1]

        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mk_path).convert("L")
        n = os.path.basename(img_path)[:-4]

        # 一次性准备扩散需要的输入
        with torch.inference_mode(), autocast_ctx(dtype=torch.float16):
            pro_data = process(pipe, step, device, image_pil, mask_pil, pt, generator=generator)

        timesteps = pro_data['timesteps']
        num_inference_steps = pro_data['num_inference_steps']
        prompt_embeds = pro_data['prompt_embeds']
        mask = pro_data['mask']
        masked_image_latents = pro_data['masked_image_latents']
        timestep_cond = pro_data['timestep_cond']
        image_latents = pro_data['image_latents']
        latents0 = pro_data['latents'].detach()  # 初始 latents 保存一份

        # wm 每张图只算一次
        with torch.inference_mode():
            a_mask, _ = mask.chunk(2)
            wm = weight_mask(a_mask)

        # TruFor mask_flat 每张图只算一次
        mask_flat = preprocess_mask_for_trufor(mask_pil, device)

        # 抽 10 个seed
        seeds = random.sample(all_seeds, 10)

        # 逐 seed 推理（进一步提速可以做 batch，但要看你的 OVstep 是否支持）
        for idx in seeds:
            latents = latents0.clone()

            g = torch.Generator(device=device).manual_seed(int(idx))

            with torch.inference_mode(), autocast_ctx(dtype=torch.float16):
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                    noise_pred = pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                    if i <= int(get_step) - 1:
                        latents = pipe.scheduler.OVstep(
                            noise_pred, t, latents, image_latents, wm,
                            generator=generator, return_dict=False
                        )[0]
                    else:
                        latents = pipe.scheduler.OVstep(
                            noise_pred, t, latents, image_latents, wm,
                            generator=g, return_dict=False
                        )[0]

                # decode
                init_mask, _ = mask.chunk(2)
                b = (1 - init_mask) * image_latents + init_mask * latents
                image = pipe.vae.decode(b / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]

            # detection（model 已经在 device 上，不要再 to(device)）
            try:
                cos = detection_fast(model, image, mask_flat, device)
            except Exception as e:
                print(e, "\nERROR:", img_path)
                cos = 0.0

            if cos > 0.5:
                img_out = guided(model, image, mask_pil, device)  # 你 guided 也建议做同样的“别搬模型”
                img_out = img_out.detach().cpu().numpy()[0].transpose(1,2,0)
                Image.fromarray((img_out * 255).astype(np.uint8)).save(f'{path_res}/{n}_{seed}_{idx}.png')
            else:
                image.save(f'{path_res}/{n}_{seed}_{idx}.png')


    print(f"Results saved to {path_res}")

def main():
    model_state_file = './test_docker/weights/trufor.pth.tar'
    model = load_trufor_model(config, model_state_file, 1)
    noise_fast(3216383462, 1, "OVG", 50, 25, model)


if __name__ == "__main__":
    main()
