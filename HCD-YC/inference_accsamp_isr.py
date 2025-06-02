import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import cv2
import gc
import glob
import torch
import torchvision
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

    # Enable memory optimization
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Create model
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.inference.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    print(
        f"Loaded pretrained SD weight from {cfg.inference.sd_path}\n"
        f"Unused weights: {unused}\n"
        f"Missing weights: {missing}"
    )

    cldm.load_controlnet_from_ckpt(torch.load(cfg.inference.controlnet_path, map_location="cpu"))
    print(f"Loaded ControlNet weight from: {cfg.inference.controlnet_path}")

    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    guidance: Guidance = instantiate_from_config(cfg.guidance)

    # Read ISR-AlignOp configuration from config file
    isr_config = cfg.isr_alignop
    isr_mode = isr_config.mode
    verbose = isr_config.get('verbose', False)
    
    # Extract mode-specific configuration
    mode_config = isr_config[isr_mode]
    
    # Prepare ISR parameters based on mode
    if isr_mode == "basic":
        isr_params = {
            "kernel_size": mode_config.kernel_size,
            "stride": mode_config.stride,
            "low_memory": mode_config.low_memory,
            "eps": mode_config.get('eps', 1e-6)
        }
    elif isr_mode == "adaptive":
        isr_params = {
            "kernel_size": mode_config.kernel_size,
            "stride": mode_config.stride,
            "quality_threshold": mode_config.quality_threshold,
            "low_memory": mode_config.low_memory,
            "eps": mode_config.get('eps', 1e-6)
        }
    elif isr_mode == "multi_scale":
        isr_params = {
            "scales": mode_config.scales,
            "stride": mode_config.stride,
            "low_memory": mode_config.low_memory,
            "eps": mode_config.get('eps', 1e-6)
        }
    else:
        raise ValueError(f"Unknown ISR mode: {isr_mode}")
    
    print(f"ISR-AlignOp Configuration:")
    print(f"  Mode: {isr_mode}")
    print(f"  Parameters: {isr_params}")
    print(f"  Verbose: {verbose}")

    # Image processing configuration
    resize_config = cfg.inference
    rescaler = Resize(
        resize_config.get('resize_shorter_edge', 384), 
        interpolation=InterpolationMode.BICUBIC, 
        antialias=True
    )

    image_names = [
        os.path.basename(name)
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for name in glob.glob(os.path.join(cfg.inference.image_folder, ext))
    ]

    print(f"Found {len(image_names)} images to process")

    for image_name in tqdm(image_names, desc="Processing images"):
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        if verbose:
            print(f"\nProcessing: {image_name}")
        
        # Load and preprocess image
        image = cv2.imread(os.path.join(cfg.inference.image_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).unsqueeze(0)

        _, _, h, w = image.shape
        if verbose:
            print(f"Original image size: {h}x{w}")
        
        # Resize if necessary
        max_size = resize_config.get('max_image_size', 1024)
        if h > max_size or w > max_size:
            scale_factor = max_size / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            image = Resize((new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(image)
            if verbose:
                print(f"Resized to: {new_h}x{new_w}")
        
        if h < resize_config.get('resize_shorter_edge', 384) or w < resize_config.get('resize_shorter_edge', 384):
            image = rescaler(image)
        
        _, _, h_, w_ = image.shape
        image = pad_to_multiples_of(image, multiple=resize_config.get('padding_multiple', 64)).to(device)

        # Prepare conditioning
        cond = cldm.prepare_condition(
            image,
            ['remove dense fog'],
        )

        # Run AccSamp with ISR-AlignOp enhancement
        if verbose:
            print(f"Running AccSamp-ISR with {isr_mode} mode...")
        
        z = sampler.accsamp_isr(
            model=cldm,
            device=device,
            steps=resize_config.get('num_inference_steps', 30),
            x_size=cond['c_img'].shape,
            cond=cond,
            uncond=None,
            cond_fn=guidance,
            hazy=image,
            diffusion=diffusion,
            cfg_scale=resize_config.get('cfg_scale', 1.0),
            progress=True,
            proportions=[cfg.inference.tau, cfg.inference.omega],
            isr_mode=isr_mode,
            isr_config=isr_params,
            verbose=verbose
        )

        # Decode and save result
        result = (cldm.vae_decode(z) + 1) / 2
        result = result[:, :, :h_, :w_].clip(0., 1.)
        result = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(result)
        
        # Save result with ISR mode identifier
        output_name = f'{image_name[:-4]}_isr_{isr_mode}.png'
        torchvision.utils.save_image(
            result.squeeze(0),
            os.path.join(cfg.inference.result_folder, output_name)
        )
        
        if verbose:
            print(f"Saved result: {output_name}")
        
        # Clear intermediate variables
        del image, cond, z, result
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Processing completed! Results saved in {cfg.inference.result_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to ISR-AlignOp configuration file")
    
    args = parser.parse_args()
    main(args) 