import os
# adjust as needed
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

    # 初始化文本引导相关组件
    prompt_pool = TextPromptPool()
    fog_analyzer = FogAnalyzer(device=device)
    text_processor = TextCondProcessor(embed_dim=1024).to(device)

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

    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    rescaler = Resize(512, interpolation=InterpolationMode.BICUBIC, antialias=True)

    image_names = [
        os.path.basename(name)
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for name in glob.glob(os.path.join(cfg.inference.image_folder, ext))
    ]

    for image_name in tqdm(image_names):
        image = cv2.imread(os.path.join(cfg.inference.image_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).unsqueeze(0)

        _, _, h, w = image.shape
        if h < 512 or w < 512:
            image = rescaler(image)
        _, _, h_, w_ = image.shape

        image = pad_to_multiples_of(image, multiple=64).to(device)
        
        # 使用雾气分析器生成最优提示词
        if cfg.inference.get('use_dynamic_prompt', True):
            optimal_prompt = fog_analyzer.select_optimal_prompt(image, prompt_pool)
            prompts = [optimal_prompt]
            print(f"分析图像 {image_name} - 使用提示词: {optimal_prompt}")
        else:
            prompts = ['remove dense fog']

        cond = cldm.prepare_condition(
            image,
            prompts,
        )
        
        # 使用文本条件优化处理
        if cfg.inference.get('use_text_processor', True):
            cond['c_crossattn'] = text_processor(cond['c_txt'])
            cond['c_txt'] = cond['c_crossattn']

        z = sampler.sample(
            model=cldm,
            device=device,
            steps=50,
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