import os
# adjust as needed
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
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

    # 初始化层级循环一致性损失
    hier_cycle_loss = HierarchicalCycleLoss(
        pixel_weight=cfg.train.get('pixel_weight', 1.0),
        feature_weight=cfg.train.get('feature_weight', 1.0),
        semantic_weight=cfg.train.get('semantic_weight', 0.5),
        region_aware=cfg.train.get('region_aware', True),
        adaptive_weight=cfg.train.get('adaptive_weight', True)
    ).to(device)
    
    # 非对称循环损失
    asym_cycle_loss = AsymmetricCycleLoss(
        forward_weight=cfg.train.get('forward_weight', 0.7),
        backward_weight=cfg.train.get('backward_weight', 0.3)
    ).to(device)

    # 初始化文本提示池和雾气分析器
    prompt_pool = TextPromptPool()
    fog_analyzer = FogAnalyzer(device=device)
    text_processor = TextCondProcessor(embed_dim=1024).to(device)
    
    # 初始化文本-图像对齐损失
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
            
            # 分析雾图并选择最优提示词
            if cfg.train.get('use_dynamic_prompt', True) and global_step > cfg.train.get('dynamic_prompt_start', 1000):
                with torch.no_grad():
                    # 将hazy从[-1,1]转换到[0,1]范围进行分析
                    hazy_norm = (hazy + 1) / 2
                    optimal_prompts = [fog_analyzer.select_optimal_prompt(hazy_norm, prompt_pool) for _ in range(hazy.shape[0])]
                    # 替换批次中的提示词
                    prompt = optimal_prompts

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(clean)
                cond = pure_cldm.prepare_condition(hazy, prompt)
                
                # 如果启用文本条件优化处理
                if cfg.train.get('use_text_processor', True) and global_step > cfg.train.get('text_processor_start', 1000):
                    # 处理文本条件嵌入 - 修复：使用c_txt而不是c_crossattn
                    cond['c_crossattn'] = text_processor(cond['c_txt'])
                    # 保持向后兼容，也将c_txt设置为处理后的值
                    cond['c_txt'] = cond['c_crossattn']
                
                cond['c_img'] = cond['c_img'].contiguous().float()

            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            # 1. 常规扩散损失
            diffusion_loss = diffusion.p_losses(cldm, z_0, t, cond)
            
            # 2. 层级循环一致性损失
            if global_step > cfg.train.get('cycle_start_step', 500):  # 在预热阶段后启用循环损失
                with torch.no_grad():
                    # 使用当前模型进行去雾
                    dehazed_z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=20,  # 使用较少步数提高训练效率
                        x_size=(z_0.shape[0], *z_0.shape[1:]),
                        cond=cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=False,
                    )
                    
                    # 解码为RGB图像
                    dehazed_imgs = (pure_cldm.vae_decode(dehazed_z) + 1) / 2  # 转换到[0,1]范围
                    
                    # 使用去雾后的图像作为输入，再生成雾图 (反向循环)
                    if cfg.train.get('use_backward_cycle', True):
                        # 将去雾图像送入Stage1模型生成雾图
                        # 注意：这里需要有一个Stage1模型的推理逻辑，如果有的话可以加载
                        # 简化起见，我们这里只做正向循环，反向循环需要额外的代码
                        pass
                
                # 估计雾气密度 (简单用亮度估计)
                hazy_gray = 0.299 * hazy[:, 0:1] + 0.587 * hazy[:, 1:2] + 0.114 * hazy[:, 2:3]
                hazy_density = F.adaptive_avg_pool2d(hazy_gray, 1)  # 全局平均池化
                
                # 计算层级循环一致性损失
                clean_norm = (clean + 1) / 2  # 转换到[0,1]范围
                hazy_norm_cycle = (hazy + 1) / 2
                cycle_consistency_loss, loss_details = hier_cycle_loss(
                    clean=clean_norm, 
                    hazy=hazy_norm_cycle, 
                    dehazed=dehazed_imgs,
                    hazy_density=hazy_density
                )
                
                # 计算非对称循环损失
                asymmetric_loss, _ = asym_cycle_loss(clean_norm, hazy_norm_cycle, dehazed_imgs)
                
                # 权重递增，在训练过程中逐渐增加循环损失的权重
                cycle_weight = min(1.0, global_step / cfg.train.get('cycle_max_step', 2000)) * cfg.train.get('cycle_weight', 1.0)
                asym_weight = min(1.0, global_step / cfg.train.get('cycle_max_step', 2000)) * cfg.train.get('asym_weight', 0.5)
                
                # 总损失
                # 确保每个损失都是标量
                if not torch.is_tensor(asymmetric_loss) or asymmetric_loss.numel() > 1:
                    asymmetric_loss = asymmetric_loss.mean()
                loss = diffusion_loss + cycle_weight * cycle_consistency_loss + asym_weight * asymmetric_loss
                
                # 计算文本-图像对齐损失(如果启用)
                if cfg.train.get('use_text_processor', True) and global_step > cfg.train.get('text_processor_start', 1000):
                    # 使用处理后的文本条件或原始文本条件
                    text_condition = cond.get('c_crossattn', cond['c_txt'])
                    text_alignment_loss, text_loss_details = text_align_loss(
                        dehazed_imgs,
                        clean_norm, 
                        text_condition
                    )
                    
                    # 权重递增，在训练过程中逐渐增加文本对齐损失的权重
                    text_align_weight = min(1.0, global_step / cfg.train.get('text_align_max_step', 3000)) * cfg.train.get('text_align_weight', 0.3)
                    
                    # 添加到总损失中
                    loss = loss + text_align_weight * text_alignment_loss
                    
                    # 记录损失
                    if accelerator.is_main_process:
                        writer.add_scalar("loss/text_align_loss", text_loss_details['text_align_loss'], global_step)
                
                # 记录循环损失
                cycle_loss_log.append(cycle_consistency_loss.mean().item())
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
                    
                    # 如果启用了循环一致性训练，也展示循环结果
                    if global_step > cfg.train.get('cycle_start_step', 500):
                        dehazed_imgs = (pure_cldm.vae_decode(z) + 1) / 2  # [0,1]范围
                        
                        # 将去雾图像转回雾图（如果有Stage1模型）
                        # 这里简化处理，只展示去雾结果
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
                        
                        # 如果有循环结果，也记录
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
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
