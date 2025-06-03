import os
# adjust as needed
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import glob
import torch
import torchvision
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from diffbir.sampler import SpacedSampler
from diffbir.model import ControlLDM, Diffusion
from diffbir.pipeline import pad_to_multiples_of
from diffbir.utils.common import instantiate_from_config
from diffbir.utils.text_guidance import TextPromptPool, FogAnalyzer, TextCondProcessor
from torchvision.transforms import ToTensor, Resize, InterpolationMode


@torch.no_grad()
def main(args) -> None:
    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.inference.result_folder, exist_ok=True)

    # Initialize text guidance components
    prompt_pool = TextPromptPool()
    fog_analyzer = FogAnalyzer(device=device)
    
    # lightweight text processor for inference
    text_processor = None
    if cfg.inference.get('use_text_processor', True):
        text_processor = TextCondProcessor(embed_dim=1024).to(device)
        text_processor.eval()

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.inference.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    print(
        f"Loaded pretrained SD weights from {cfg.inference.sd_path}"
    )

    cldm.load_controlnet_from_ckpt(torch.load(cfg.inference.controlnet_path, map_location="cpu"))
    print(
        f"Loaded controlnet weights from checkpoint: {cfg.inference.controlnet_path}"
    )

    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    rescaler = Resize(512, interpolation=InterpolationMode.BICUBIC, antialias=True)

    # Handle both PASCAL VOC format and simple directory format
    image_folder = cfg.inference.image_folder
    
    # Check if it's PASCAL VOC format (has JPEGImages subdirectory)
    jpeg_images_path = os.path.join(image_folder, 'JPEGImages')
    if os.path.exists(jpeg_images_path):
        print(f"Detected PASCAL VOC format dataset, using JPEGImages directory: {jpeg_images_path}")
        search_folder = jpeg_images_path
    else:
        print(f"Using simple directory format: {image_folder}")
        search_folder = image_folder

    image_names = [
        os.path.basename(name)
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for name in glob.glob(os.path.join(search_folder, ext))
    ]
    
    print(f"Found {len(image_names)} images to process")

    for image_name in tqdm(image_names):
        image = cv2.imread(os.path.join(search_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).unsqueeze(0)

        _, _, h, w = image.shape
        if h < 512 or w < 512:
            image = rescaler(image)
        _, _, h_, w_ = image.shape

        image = pad_to_multiples_of(image, multiple=64).to(device)
        
        # intelligent prompt selection
        if cfg.inference.get('use_dynamic_prompt', True):
            try:
                optimal_prompt = fog_analyzer.select_optimal_prompt(image, prompt_pool)
                prompts = [optimal_prompt]
                if cfg.inference.get('verbose', False):
                    print(f"Selected prompt for {image_name}: {optimal_prompt}")
            except Exception as e:
                print(f"Failed to analyze {image_name}, using default prompt: {e}")
                prompts = ['remove dense fog']
        else:
            prompts = cfg.inference.get('default_prompts', ['remove dense fog'])

        # prepare conditions
        cond = cldm.prepare_condition(image, prompts)
        
        # enhanced text condition processing
        if text_processor is not None:
            try:
                with torch.no_grad():
                    enhanced_text = text_processor(cond['c_txt'])
                    cond['c_txt'] = enhanced_text
            except Exception as e:
                print(f"Text processing failed for {image_name}, using original text: {e}")

        # adaptive sampling steps based on image complexity
        sample_steps = cfg.inference.get('steps', 50)
        if cfg.inference.get('adaptive_steps', False):
            # estimate complexity based on gradient
            gray = torch.mean(image, dim=1, keepdim=True)
            grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            complexity = torch.mean(grad_x) + torch.mean(grad_y)
            
            if complexity > 0.15:  # high complexity
                sample_steps = min(50, sample_steps + 10)
            elif complexity < 0.05:  # low complexity
                sample_steps = max(20, sample_steps - 10)

        z = sampler.sample(
            model=cldm,
            device=device,
            steps=sample_steps,
            x_size=cond['c_img'].shape,
            cond=cond,
            uncond=None,
            cfg_scale=1.,
            progress=False,
        )

        result = (cldm.vae_decode(z) + 1) / 2
        result = result[:, :, :h_, :w_].clip(0., 1.)
        result = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(result)
        torchvision.utils.save_image(result.squeeze(0), os.path.join(cfg.inference.result_folder, f'{image_name[:-4]}.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
