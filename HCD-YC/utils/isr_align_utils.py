import torch
from typing import Tuple, Optional
from .align_utils import tiled_align


def isr_tiled_align(
    hazy_image: torch.Tensor,
    early_pred_a: torch.Tensor,
    early_pred_b: torch.Tensor,
    kernel_size: int = 37,
    stride: int = 10,
    eps: float = 1e-6,
    low_memory: bool = False,
    unfold_threshold: int = 200_000_000,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ISR-AlignOp: Iterative Statistical Refinement for AlignOp
    
    Performs two-step iterative refinement process to generate initial dehazing estimate:
    1. Initial coarse alignment: using very early (noisy) prediction early_pred_a
    2. Iterative refinement alignment: using slightly later (clearer) prediction early_pred_b
    
    Args:
        hazy_image: [B, C, H, W] Input hazy image
        early_pred_a: [B, C, H, W] Very early diffusion prediction (at τ_a, more noisy)
        early_pred_b: [B, C, H, W] Slightly later diffusion prediction (at τ_b, clearer)
        kernel_size: Patch size k
        stride: Sliding window stride d
        eps: Numerical stability term
        low_memory: Whether to use low memory mode
        unfold_threshold: Maximum number of elements threshold for unfold operation
        verbose: Whether to print detailed information
        
    Returns:
        estimate_1: Result of first coarse alignment step
        estimate_2: Final result of second refinement alignment step
    """
    
    if verbose:
        print("ISR-AlignOp: Starting iterative statistical refinement...")
        print(f"  Hazy image shape: {hazy_image.shape}")
        print(f"  Early pred A shape: {early_pred_a.shape}")
        print(f"  Early pred B shape: {early_pred_b.shape}")
    
    # Step 1: Initial coarse alignment
    # E_1 = AlignOp(I_hazy, P_a)
    if verbose:
        print("  Step 1: Initial coarse alignment with early prediction A...")
    
    estimate_1 = tiled_align(
        hazy_image=hazy_image,
        pred_image=early_pred_a,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        low_memory=low_memory,
        unfold_threshold=unfold_threshold
    )
    
    if verbose:
        print(f"  Step 1 completed. Estimate 1 shape: {estimate_1.shape}")
    
    # Step 2: Iterative refinement alignment
    # E_2 = AlignOp(E_1, P_b)
    if verbose:
        print("  Step 2: Iterative refinement alignment with prediction B...")
    
    estimate_2 = tiled_align(
        hazy_image=estimate_1,  # Use first step result as "content" image
        pred_image=early_pred_b,
        kernel_size=kernel_size,
        stride=stride,
        eps=eps,
        low_memory=low_memory,
        unfold_threshold=unfold_threshold
    )
    
    if verbose:
        print(f"  Step 2 completed. Final estimate shape: {estimate_2.shape}")
        print("ISR-AlignOp: Iterative refinement completed!")
    
    return estimate_1, estimate_2


def adaptive_isr_tiled_align(
    hazy_image: torch.Tensor,
    early_pred_a: torch.Tensor,
    early_pred_b: torch.Tensor,
    kernel_size: int = 37,
    stride: int = 10,
    eps: float = 1e-6,
    low_memory: bool = False,
    unfold_threshold: int = 200_000_000,
    quality_threshold: float = 0.1,
    verbose: bool = False
) -> torch.Tensor:
    """
    Adaptive ISR-AlignOp: Automatically selects ISR or standard AlignOp based on image quality
    
    Args:
        hazy_image: [B, C, H, W] Input hazy image
        early_pred_a: [B, C, H, W] Very early diffusion prediction
        early_pred_b: [B, C, H, W] Slightly later diffusion prediction
        quality_threshold: Quality threshold to decide whether to use ISR
        Other parameters same as isr_tiled_align
        
    Returns:
        final_estimate: Final dehazing estimate
    """
    
    # Simple quality assessment: compute difference between two predictions
    pred_diff = torch.mean(torch.abs(early_pred_a - early_pred_b))
    
    if verbose:
        print(f"Adaptive ISR-AlignOp: Prediction difference = {pred_diff.item():.4f}")
        print(f"Quality threshold = {quality_threshold}")
    
    if pred_diff > quality_threshold:
        # Large difference, use ISR for refinement
        if verbose:
            print("Using ISR-AlignOp for refinement...")
        estimate_1, estimate_2 = isr_tiled_align(
            hazy_image, early_pred_a, early_pred_b,
            kernel_size, stride, eps, low_memory, unfold_threshold, verbose
        )
        return estimate_2
    else:
        # Small difference, use better prediction for standard alignment
        if verbose:
            print("Using standard AlignOp with better prediction...")
        return tiled_align(
            hazy_image=hazy_image,
            pred_image=early_pred_b,  # Use better prediction
            kernel_size=kernel_size,
            stride=stride,
            eps=eps,
            low_memory=low_memory,
            unfold_threshold=unfold_threshold
        )


def multi_scale_isr_align(
    hazy_image: torch.Tensor,
    early_pred_a: torch.Tensor,
    early_pred_b: torch.Tensor,
    scales: list = [37, 25, 15],  # Multi-scale patch sizes
    stride: int = 10,
    eps: float = 1e-6,
    low_memory: bool = False,
    unfold_threshold: int = 200_000_000,
    verbose: bool = False
) -> torch.Tensor:
    """
    Multi-scale ISR-AlignOp: Performs ISR alignment at multiple scales and fuses results
    
    Args:
        hazy_image: [B, C, H, W] Input hazy image
        early_pred_a, early_pred_b: Early diffusion predictions
        scales: List of multiple patch scales
        Other parameters same as isr_tiled_align
        
    Returns:
        fused_estimate: Multi-scale fused final estimate
    """
    
    if verbose:
        print(f"Multi-scale ISR-AlignOp with scales: {scales}")
    
    estimates = []
    weights = []
    
    for i, scale in enumerate(scales):
        if verbose:
            print(f"  Processing scale {scale}...")
        
        _, estimate = isr_tiled_align(
            hazy_image, early_pred_a, early_pred_b,
            kernel_size=scale, stride=stride, eps=eps,
            low_memory=low_memory, unfold_threshold=unfold_threshold,
            verbose=False
        )
        
        estimates.append(estimate)
        
        # Assign higher weights to larger patches (more stable statistics)
        weight = scale / sum(scales)
        weights.append(weight)
        
        if verbose:
            print(f"    Scale {scale} weight: {weight:.3f}")
    
    # Weighted fusion of multi-scale results
    fused_estimate = torch.zeros_like(estimates[0])
    for estimate, weight in zip(estimates, weights):
        fused_estimate += weight * estimate
    
    if verbose:
        print(f"Multi-scale fusion completed. Final estimate shape: {fused_estimate.shape}")
    
    return fused_estimate
