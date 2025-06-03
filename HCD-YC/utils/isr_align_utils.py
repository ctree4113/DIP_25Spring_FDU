import torch
from typing import Tuple, Optional
from .align_utils import tiled_align


def validate_isr_inputs(hazy_image: torch.Tensor, early_pred_a: torch.Tensor, early_pred_b: torch.Tensor, verbose: bool = False):
    """Validate ISR inputs and check for potential issues"""
    if verbose:
        print(f"  ISR Input Validation:")
        print(f"    Hazy image - shape: {hazy_image.shape}, range: [{hazy_image.min():.3f}, {hazy_image.max():.3f}]")
        print(f"    Early pred A - shape: {early_pred_a.shape}, range: [{early_pred_a.min():.3f}, {early_pred_a.max():.3f}]")
        print(f"    Early pred B - shape: {early_pred_b.shape}, range: [{early_pred_b.min():.3f}, {early_pred_b.max():.3f}]")
    
    # Check for NaN or inf values
    if torch.isnan(hazy_image).any() or torch.isinf(hazy_image).any():
        raise ValueError("Hazy image contains NaN or inf values")
    if torch.isnan(early_pred_a).any() or torch.isinf(early_pred_a).any():
        raise ValueError("Early prediction A contains NaN or inf values")
    if torch.isnan(early_pred_b).any() or torch.isinf(early_pred_b).any():
        raise ValueError("Early prediction B contains NaN or inf values")
    
    # Ensure proper value ranges
    if hazy_image.min() < -1.1 or hazy_image.max() > 1.1:
        if verbose:
            print(f"    Warning: Hazy image range unusual: [{hazy_image.min():.3f}, {hazy_image.max():.3f}]")
    
    return True


def stabilize_result(result: torch.Tensor, reference: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """Stabilize ISR result to prevent extreme values"""
    # Clamp to reasonable range
    result = torch.clamp(result, -1.5, 1.5)
    
    # Check for potential issues
    if torch.isnan(result).any() or torch.isinf(result).any():
        if verbose:
            print("    Warning: ISR result contains NaN/inf, falling back to reference")
        return reference.clone()
    
    # Ensure result is not completely uniform (likely error)
    if result.std() < 1e-6:
        if verbose:
            print("    Warning: ISR result too uniform, blending with reference")
        result = 0.7 * result + 0.3 * reference
    
    return result


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
    
    # Validate inputs
    validate_isr_inputs(hazy_image, early_pred_a, early_pred_b, verbose)
    
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
    
    # Stabilize intermediate result
    estimate_1 = stabilize_result(estimate_1, hazy_image, verbose)
    
    if verbose:
        print(f"  Step 1 completed. Estimate 1 - range: [{estimate_1.min():.3f}, {estimate_1.max():.3f}]")
    
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
    
    # Stabilize final result
    estimate_2 = stabilize_result(estimate_2, estimate_1, verbose)
    
    if verbose:
        print(f"  Step 2 completed. Final estimate - range: [{estimate_2.min():.3f}, {estimate_2.max():.3f}]")
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
    
    # Validate inputs first
    validate_isr_inputs(hazy_image, early_pred_a, early_pred_b, verbose)
    
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
        result = tiled_align(
            hazy_image=hazy_image,
            pred_image=early_pred_b,  # Use better prediction
            kernel_size=kernel_size,
            stride=stride,
            eps=eps,
            low_memory=low_memory,
            unfold_threshold=unfold_threshold
        )
        return stabilize_result(result, hazy_image, verbose)


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
    
    # Validate inputs first
    validate_isr_inputs(hazy_image, early_pred_a, early_pred_b, verbose)
    
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
    
    # Final stabilization
    fused_estimate = stabilize_result(fused_estimate, hazy_image, verbose)
    
    if verbose:
        print(f"Multi-scale fusion completed. Final estimate range: [{fused_estimate.min():.3f}, {fused_estimate.max():.3f}]")
    
    return fused_estimate
