import os
# adjust as needed
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import cv2
import glob
import copy
import torch
import torchvision
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from diffbir.sampler import SpacedSampler
from diffbir.model import ControlLDM, Diffusion
from diffbir.pipeline import pad_to_multiples_of
from diffbir.utils.common import instantiate_from_config
from torchvision.transforms import ToTensor, Resize, InterpolationMode


@torch.no_grad()
def main(args) -> None:
    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.inference.result_folder, exist_ok=True)

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

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    cldm.eval().to(device)
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
        prompt = "hazy, foggy, misty, obscure, smoggy."

        cond = cldm.prepare_condition(image, [prompt, ])
        uncond = {
            'c_img': torch.zeros_like(cond['c_img']),
            'c_txt': copy.deepcopy(cond['c_txt']),
        }

        z = sampler.sample(
            model=cldm,
            device=device,
            steps=50,
            x_size=cond['c_img'].shape,
            cond=cond,
            uncond=uncond,
            cfg_scale=cfg.inference.w,
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