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
from diffbir.utils.losses import HierarchicalCycleLoss, AsymmetricCycleLoss


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

    # 初始化循环一致性损失
    hier_cycle_loss = HierarchicalCycleLoss(
        pixel_weight=cfg.train.get('pixel_weight', 1.0),
        feature_weight=cfg.train.get('feature_weight', 1.0),
        semantic_weight=cfg.train.get('semantic_weight', 0.5),
        region_weight=cfg.train.get('region_weight', 0.2),
        adaptive_weight=cfg.train.get('adaptive_weight', True),
        use_dcp=cfg.train.get('use_dcp', True)
    ).to(device)
    
    # 非对称循环损失
    asym_cycle_loss = AsymmetricCycleLoss(
        forward_weight=cfg.train.get('forward_weight', 0.7),
        backward_weight=cfg.train.get('backward_weight', 0.3)
    ).to(device)

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
            gt, lq, prompt, idf = batch
            gt = gt.contiguous().float()
            lq = lq.contiguous().float()

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                cond = pure_cldm.prepare_condition(lq, prompt)
                for i in range(len(lq)):
                    if idf[i] == 'uncond':
                        cond['c_img'][i] = torch.zeros_like(cond['c_img'][i])
                cond['c_img'] = cond['c_img'].contiguous().float()

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            # 1. Standard diffusion loss
            diffusion_loss = diffusion.p_losses(cldm, z_0, t, cond)
            
            # 2. Simple cycle consistency loss without full sampling
            if global_step > cfg.train.get('cycle_start_step', 500):
                # Use simple forward pass through network for faster training
                with torch.no_grad():
                    # Generate approximate dehazed result using single step
                    noise = torch.randn_like(z_0)
                    t_sample = torch.randint(200, 400, (z_0.shape[0],), device=device)  # mid-range timestep
                    x_t = diffusion.q_sample(z_0, t_sample, noise)
                    
                    # Single denoising step for approximate result
                    pred_noise = cldm(x_t, t_sample, **cond)
                    alpha_t = diffusion.sqrt_alphas_cumprod[t_sample].view(-1, 1, 1, 1)
                    beta_t = diffusion.sqrt_one_minus_alphas_cumprod[t_sample].view(-1, 1, 1, 1)
                    approx_z = (x_t - beta_t * pred_noise) / alpha_t
                    
                    # Decode to get approximate dehazed image
                    approx_dehazed = (pure_cldm.vae_decode(approx_z) + 1) / 2  # [0,1] range
                    
                # Estimate fog density for adaptive weighting
                gt_norm = (gt + 1) / 2  # Convert to [0,1] range
                lq_norm = torch.clamp(lq, 0, 1)  # Ensure LQ is in [0,1] range
                
                # Compute hierarchical cycle consistency loss
                cycle_consistency_loss, loss_details = hier_cycle_loss(
                    clean=gt_norm, 
                    hazy=lq_norm, 
                    dehazed=approx_dehazed
                )
                
                # Compute asymmetric cycle loss
                asymmetric_loss, _ = asym_cycle_loss(gt_norm, lq_norm, approx_dehazed)
                
                # Progressive weight scheduling
                cycle_weight = min(1.0, global_step / cfg.train.get('cycle_max_step', 2000)) * cfg.train.get('cycle_weight', 0.5)
                asym_weight = min(1.0, global_step / cfg.train.get('cycle_max_step', 2000)) * cfg.train.get('asym_weight', 0.3)
                
                # Ensure losses are scalars
                if not torch.is_tensor(asymmetric_loss) or asymmetric_loss.numel() > 1:
                    asymmetric_loss = asymmetric_loss.mean()
                if not torch.is_tensor(cycle_consistency_loss) or cycle_consistency_loss.numel() > 1:
                    cycle_consistency_loss = cycle_consistency_loss.mean()
                    
                loss = diffusion_loss + cycle_weight * cycle_consistency_loss + asym_weight * asymmetric_loss
                
                # Log cycle loss
                cycle_loss_log.append(cycle_consistency_loss.item())
            else:
                loss = diffusion_loss
            
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                
                # 记录循环损失
                if len(cycle_loss_log) > 0:
                    avg_cycle_loss = sum(cycle_loss_log) / len(cycle_loss_log)
                    cycle_loss_log.clear()
                    if accelerator.is_main_process:
                        writer.add_scalar("loss/cycle_loss", avg_cycle_loss, global_step)
                
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
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
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
