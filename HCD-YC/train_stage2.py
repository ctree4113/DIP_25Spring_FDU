import os
# adjust as needed
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
from diffbir.utils.losses import HierarchicalCycleLoss, AsymmetricCycleLoss, TextImageAlignmentLoss
from diffbir.utils.text_guidance import TextPromptPool, FogAnalyzer, TextCondProcessor


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # initialize the hierarchical cycle consistency loss
    hier_cycle_loss = HierarchicalCycleLoss(
        pixel_weight=cfg.train.get('pixel_weight', 1.0),
        feature_weight=cfg.train.get('feature_weight', 0.5),
        semantic_weight=cfg.train.get('semantic_weight', 0.3),
        region_weight=cfg.train.get('region_weight', 0.2),
        adaptive_weight=cfg.train.get('adaptive_weight', True)
    ).to(device)
    
    # initialize the asymmetric cycle loss
    asym_cycle_loss = AsymmetricCycleLoss(
        forward_weight=cfg.train.get('forward_weight', 0.7),
        backward_weight=cfg.train.get('backward_weight', 0.3)
    ).to(device)

    # initialize the text prompt pool and fog analyzer
    prompt_pool = TextPromptPool()
    fog_analyzer = FogAnalyzer(device=device)
    text_processor = TextCondProcessor(embed_dim=1024).to(device)
    
    # initialize the text-image alignment loss
    text_align_loss = TextImageAlignmentLoss(device=device).to(device)

    # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    cldm.train().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    cycle_loss_log = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            clean, hazy, prompt = batch
            clean = clean.contiguous().float()
            hazy = hazy.contiguous().float()
            
            # analyze fog and select optimal prompt
            if cfg.train.get('use_dynamic_prompt', True) and global_step > cfg.train.get('dynamic_prompt_start', 1000):
                with torch.no_grad():
                    # convert hazy image to [0,1] range for analysis
                    hazy_norm = (hazy + 1) / 2
                    optimal_prompt = fog_analyzer.select_optimal_prompt(hazy_norm, prompt_pool)
                    # replace prompts for entire batch
                    prompt = [optimal_prompt] * hazy.shape[0]

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(clean)
                cond = pure_cldm.prepare_condition(hazy, prompt)
                
                # enhanced text condition processing
                if cfg.train.get('use_text_processor', True) and global_step > cfg.train.get('text_processor_start', 1000):
                    # process text embeddings
                    enhanced_text = text_processor(cond['c_txt'])
                    cond['c_txt'] = enhanced_text
                
                cond['c_img'] = cond['c_img'].contiguous().float()

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            # primary diffusion loss
            diffusion_loss = diffusion.p_losses(cldm, z_0, t, cond)
            
            # progressive training strategy
            cycle_enabled = global_step > cfg.train.get('cycle_start_step', 500)
            text_align_enabled = (cfg.train.get('use_text_processor', True) and 
                                global_step > cfg.train.get('text_processor_start', 1000))
            
            total_loss = diffusion_loss
            
            # hierarchical cycle consistency loss
            if cycle_enabled:
                with torch.no_grad():
                    # efficient dehazing for cycle loss computation
                    sample_steps = max(10, 50 - global_step // 100)  # decrease steps over time
                    dehazed_z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=sample_steps,
                        x_size=z_0.shape,
                        cond=cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=False,
                    )
                    
                    # decode to [0,1] range
                    dehazed_imgs = torch.clamp((pure_cldm.vae_decode(dehazed_z) + 1) / 2, 0, 1)
                
                # normalize inputs for loss computation
                clean_norm = torch.clamp((clean + 1) / 2, 0, 1)
                hazy_norm = torch.clamp((hazy + 1) / 2, 0, 1)
                
                # hierarchical cycle loss
                cycle_loss, cycle_details = hier_cycle_loss(
                    clean=clean_norm, 
                    hazy=hazy_norm, 
                    dehazed=dehazed_imgs
                )
                
                # asymmetric cycle loss  
                asym_loss, asym_details = asym_cycle_loss(
                    x_clean=clean_norm, 
                    x_hazy=hazy_norm, 
                    cycle_clean=dehazed_imgs
                )
                
                # progressive weight scheduling
                cycle_progress = min(1.0, (global_step - cfg.train.get('cycle_start_step', 500)) / 
                                   max(1, cfg.train.get('cycle_ramp_steps', 1500)))
                
                cycle_weight = cycle_progress * cfg.train.get('cycle_weight', 0.5)
                asym_weight = cycle_progress * cfg.train.get('asym_weight', 0.2)
                
                total_loss = total_loss + cycle_weight * cycle_loss + asym_weight * asym_loss
                
                # log cycle losses
                cycle_loss_log.append(cycle_loss.item())
                
                # text-image alignment loss
                if text_align_enabled:
                    try:
                        text_alignment_loss, text_details = text_align_loss(
                            dehazed_imgs=dehazed_imgs,
                            clean_imgs=clean_norm, 
                            text_embeds=cond['c_txt']
                        )
                        
                        # progressive text alignment weight
                        text_progress = min(1.0, (global_step - cfg.train.get('text_processor_start', 1000)) / 
                                          max(1, cfg.train.get('text_align_ramp_steps', 2000)))
                        text_weight = text_progress * cfg.train.get('text_align_weight', 0.1)
                        
                        total_loss = total_loss + text_weight * text_alignment_loss
                        
                        # log text alignment loss  
                        if accelerator.is_main_process and global_step % cfg.train.log_every == 0:
                            writer.add_scalar("loss/text_align", text_alignment_loss.item(), global_step)
                            for key, val in text_details.items():
                                writer.add_scalar(f"loss/text_{key}", val, global_step)
                                
                    except Exception as e:
                        if accelerator.is_main_process:
                            print(f"Text alignment loss computation failed at step {global_step}: {e}")
            
            # optimize
            opt.zero_grad()
            accelerator.backward(total_loss)
            
            # gradient clipping for stability
            if cfg.train.get('grad_clip', None):
                accelerator.clip_grad_norm_(cldm.parameters(), cfg.train.grad_clip)
                
            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(total_loss.item())
            epoch_loss.append(total_loss.item())
            pbar.update(1)
            
            # improved progress display
            loss_str = f"Loss: {total_loss.item():.4f}"
            if cycle_enabled:
                loss_str += f", Cycle: {cycle_loss.item():.4f}"
            if text_align_enabled:
                loss_str += f", Text: {text_alignment_loss.item():.4f}"
                
            pbar.set_description(f"Epoch: {epoch:04d}, Step: {global_step:07d}, {loss_str}")

            # enhanced logging
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                
                # cycle loss logging
                if len(cycle_loss_log) > 0:
                    avg_cycle_loss = sum(cycle_loss_log) / len(cycle_loss_log)
                    cycle_loss_log.clear()
                    if accelerator.is_main_process:
                        writer.add_scalar("loss/cycle_consistency", avg_cycle_loss, global_step)
                
                if accelerator.is_main_process:
                    writer.add_scalar("loss/total_loss", avg_loss, global_step)
                    writer.add_scalar("loss/diffusion_loss", diffusion_loss.item(), global_step)
                    
                    # log training progress weights
                    if cycle_enabled:
                        writer.add_scalar("weight/cycle_weight", cycle_weight, global_step)
                    if text_align_enabled:
                        writer.add_scalar("weight/text_weight", text_weight, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_clean, log_hazy = clean[:N], hazy[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_clean), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    
                    # if the cycle consistency training is enabled, also show the cycle result
                    if global_step > cfg.train.get('cycle_start_step', 500):
                        dehazed_imgs = (pure_cldm.vae_decode(z) + 1) / 2  # [0,1] range
                        cycle_clean = dehazed_imgs
                    
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/clean", (log_clean + 1) / 2),
                            ("image/hazy", (log_hazy + 1) / 2),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                            
                        if global_step > cfg.train.get('cycle_start_step', 500):
                            writer.add_image(
                                "cycle/dehazed", 
                                make_grid(cycle_clean, nrow=4), 
                                global_step
                            )
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/total_loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
