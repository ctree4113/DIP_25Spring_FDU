import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import gc
import glob
import torch
import torchvision
import random
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from diffbir.sampler import SpacedSampler
from diffbir.utils.cond_fn import Guidance
from diffbir.model import ControlLDM, Diffusion
from diffbir.pipeline import pad_to_multiples_of
from diffbir.utils.common import instantiate_from_config
from torchvision.transforms import ToTensor, Resize, InterpolationMode


@torch.no_grad()
def main(args) -> None:
    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.inference.result_folder, exist_ok=True)

    # enable memory optimization
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.inference.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    print(
        f"strictly load pretrained SD weight from {cfg.inference.sd_path}\n"
        f"unused weights: {unused}\n"
        f"missing weights: {missing}"
    )

    cldm.load_controlnet_from_ckpt(torch.load(cfg.inference.controlnet_path, map_location="cpu"))
    print(
        f"strictly load controlnet weight from checkpoint: {cfg.inference.controlnet_path}"
    )

    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    guidance: Guidance = instantiate_from_config(cfg.guidance)

    # Image processing configuration with strict size limits
    max_size = min(cfg.inference.get('max_image_size', 512), 512)
    min_size = cfg.inference.get('resize_shorter_edge', 512)
    
    print(f"Baseline AlignOp - Image size constraints: max={max_size}, min={min_size}")

    # Get all image files
    all_image_names = [
        os.path.basename(name)
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for name in glob.glob(os.path.join(cfg.inference.image_folder, ext))
    ]

    # Apply sampling if specified in config or args
    sample_size = getattr(args, 'sample_size', None) or cfg.inference.get('sample_size', None)
    
    if sample_size and sample_size < len(all_image_names):
        print(f"Sampling {sample_size} images from {len(all_image_names)} total images")
        random.seed(42)  # For reproducible results
        image_names = random.sample(all_image_names, sample_size)
    else:
        image_names = all_image_names
        
    print(f"Processing {len(image_names)} images with baseline AlignOp")

    for i, image_name in enumerate(tqdm(image_names, desc="Processing images")):
        # clear gpu cache before each image
        torch.cuda.empty_cache()
        gc.collect()
        
        if (i + 1) % 10 == 0:
            print(f"Processing: {image_name} ({i+1}/{len(image_names)})")
        
        try:
            image = cv2.imread(os.path.join(cfg.inference.image_folder, image_name))
            if image is None:
                print(f"Warning: Cannot load image {image_name}, skipping...")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = ToTensor()(image).unsqueeze(0)

            _, _, h, w = image.shape
            
            # Strict size constraints - ensure image is within memory limits
            if max(h, w) > max_size:
                scale_factor = max_size / max(h, w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                # Ensure dimensions are reasonable
                new_h = min(new_h, max_size)
                new_w = min(new_w, max_size)
                image = Resize((new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(image)
            
            # Ensure minimum size
            _, _, h_, w_ = image.shape
            if min(h_, w_) < min_size:
                scale_factor = min_size / min(h_, w_)
                new_h = min(int(h_ * scale_factor), max_size)
                new_w = min(int(w_ * scale_factor), max_size)
                image = Resize((new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(image)
            
            # Final padding
            _, _, h_, w_ = image.shape
            image = pad_to_multiples_of(image, multiple=64).to(device)

            cond = cldm.prepare_condition(
                image,
                ['remove dense fog', ],
            )

            z = sampler.accsamp(
                model=cldm,
                device=device,
                steps=cfg.inference.get('num_inference_steps', 50),
                x_size=cond['c_img'].shape,
                cond=cond,
                uncond=None,
                cond_fn=guidance,
                hazy=image,
                diffusion=diffusion,
                cfg_scale=cfg.inference.get('cfg_scale', 1.0),
                progress=True,  # show progress
                proportions=[cfg.inference.tau, cfg.inference.omega]
            )

            result = (cldm.vae_decode(z) + 1) / 2
            result = result[:, :, :h_, :w_].clip(0., 1.)
            result = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(result)
            torchvision.utils.save_image(result.squeeze(0),
                                         os.path.join(cfg.inference.result_folder, f'{image_name[:-4]}.png'))
            
            # clear intermediate variables
            del image, cond, z, result
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            # Clear GPU memory on error
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print(f"Baseline processing completed! Results saved in {cfg.inference.result_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of images to sample for evaluation")
    args = parser.parse_args()
    main(args)
