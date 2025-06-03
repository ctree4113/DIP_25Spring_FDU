# HCD-YC: Hierarchical Cycle-consistent Hazing-Dehazing with YCbCr Fusion and ISR-AlignOp

![Python 3.10](https://img.shields.io/badge/python-3.10-g) ![pytorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-blue.svg)

Enhanced real-world image dehazing framework building upon Learning Hazing-to-Dehazing with four synergistic innovations: YCbCr-assisted haze representation, hierarchical cycle-consistency learning, refined text guidance, and ISR-AlignOp for statistical alignment.

**GitHub:** https://github.com/ctree4113/DIP_25Spring_FDU

## Method Overview

### Core Innovations
1. **YCbCr-Assisted Haze Representation** - Dual-branch processing for RGB and YCbCr features, leveraging structural advantages of YCbCr color space
2. **Hierarchical Cycle-Consistency Learning** - Multi-level consistency constraints at pixel, feature, and semantic levels with adaptive weighting based on haze density
3. **Refined Text Guidance** - Dynamic prompt selection and enhanced text-image alignment mechanisms
4. **ISR-AlignOp** - Iterative Statistical Refinement using multiple diffusion predictions for two-step optimization

### Network Architecture
```
HCD-YC = Stage1(YC-HazeGen) + Stage2(YC-DiffDehaze)
├── YCbCr Dual-branch ControlNet
├── Hierarchical Cycle-consistency Loss
├── Dynamic Text Conditioning
└── ISR-AlignOp Enhanced Sampling
```

## Environment Setup

### Installation
```bash
# Clone repository
git clone https://github.com/ctree4113/DIP_25Spring_FDU.git
cd HCD-YC

# Create environment
conda create -n hcd_yc python=3.10
conda activate hcd_yc
pip install -r requirements.txt
```

### Download Pre-trained Model Weights
Place the following pre-trained models in the `weights/` directory:

| Model File | Description | Download Link |
|------------|-------------|---------------|
| `v2-1_512-ema-pruned.ckpt` | Stable Diffusion v2.1 base model | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) |

## Dataset Preparation

### Training Data
1. **Stage1 (HazeGen Training)**
   - **URHI Dataset**: Download URHI portion from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-β)
   - **RIDCP Synthetic Data**: Clear images + depth maps for haze generation
   ```
   datasets/
   ├── URHI/          # Real hazy images (.jpeg, .png)
   └── RIDCP/         # Synthetic data
       ├── rgb_500/   # Clear images (.jpg)
       └── depth_500/ # Depth maps (.npy)
   ```

2. **Stage2 (DiffDehaze Training)**
   - Use Stage1-trained HazeGen to generate hazy data from OTS clear images
   - Or use pre-trained weights for direct inference

### Inference Data
```
inputs/     # Input hazy images for dehazing
outputs/    # Dehazing output results
```

## Quick Start

### Inference
```bash
# Standard dehazing
python inference_stage2.py --config configs/inference/stage2.yaml

# ISR-AlignOp enhanced dehazing (Recommended)
python inference_accsamp_isr.py --config configs/inference/stage2_accsamp_isr.yaml

# Haze generation
python inference_stage1.py --config configs/inference/stage1.yaml
```

## Training

### Stage 1: YC-HazeGen Training
```bash
# Single GPU
python train_stage1.py --config configs/train/stage1.yaml

# Multi-GPU (Recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_stage1.py --config configs/train/stage1.yaml
```

### Stage 2: YC-DiffDehaze Training
```bash
# Single GPU
python train_stage2.py --config configs/train/stage2.yaml

# Multi-GPU (Recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_stage2.py --config configs/train/stage2.yaml
```

### Training Configuration
Configure accelerate before first training:
```bash
accelerate config
```

Key training parameters:
- **batch_size**: 32 (adjust based on GPU memory)
- **learning_rate**: 5e-5
- **train_steps**: 5000 (Stage1), 10000 (Stage2)
- **cycle_weight**: 0.5 (hierarchical cycle-consistency weight)

## ISR-AlignOp Configuration

### Three Operation Modes

#### 1. Adaptive Mode (Recommended)
```yaml
isr_alignop:
  mode: "adaptive"
  adaptive:
    kernel_size: 41
    stride: 10
    quality_threshold: 0.12
    low_memory: true
```

#### 2. Basic Mode (Fastest)
```yaml
isr_alignop:
  mode: "basic"
  basic:
    kernel_size: 39
    stride: 12
    low_memory: true
```

#### 3. Multi-scale Mode (Highest Quality)
```yaml
isr_alignop:
  mode: "multi_scale"
  multi_scale:
    scales: [43, 31, 19]
    stride: 8
    low_memory: true
```

## Evaluation

### Comprehensive Evaluation System
```bash
# Quick evaluation with 200 randomly sampled images
python evaluation_hcdyc.py --rtts_path datasets/RTTS --output_dir evaluation_outputs --sample_size 200

# Full dataset evaluation
python evaluation_hcdyc.py --rtts_path datasets/RTTS --output_dir evaluation_outputs

# Legacy evaluation script
python eval_metrics.py --method_folder outputs/ --dataset_folder test_images/
```

### Supported Evaluation Metrics

#### No-Reference Image Quality Assessment
| Metric | Range | Direction | Description |
|--------|-------|-----------|-------------|
| **FADE** | 0-5 | Lower Better | Fog Aware Density Evaluator - atmospheric visibility assessment |
| **Q-Align** | 1-5 | Higher Better | Large model-based visual quality assessment |
| **LIQE** | 0-1 | Higher Better | Learnable Image Quality Evaluator - perceptual quality |
| **CLIPIQA** | 0-1 | Higher Better | CLIP-based Image Quality Assessment |
| **ManIQA** | 0-1 | Higher Better | Multi-dimensional Attention Network for IQA |
| **MUSIQ** | 0-100 | Higher Better | Multi-Scale Image Quality assessment |
| **BRISQUE** | 0-100 | Lower Better | Blind/Referenceless Image Spatial Quality Evaluator |

#### Implementation Details
- **FADE**: Combines contrast analysis, histogram features, and edge density for fog assessment
- **Q-Align**: Multi-feature quality assessment including brightness, contrast, sharpness, and saturation
- **LIQE**: Vision-language correspondence for perceptual quality evaluation
- **CLIPIQA**: Leverages CLIP embeddings for semantic quality assessment
- **ManIQA**: Multi-dimensional attention mechanisms for comprehensive quality analysis
- **MUSIQ**: Multi-scale features for robust quality prediction
- **BRISQUE**: Spatial domain natural scene statistics

## Project Structure

```
HCD-YC/
├── configs/                         # Configuration files
│   ├── train/                       # Training configurations
│   │   ├── stage1.yaml              # Stage1 HazeGen training config
│   │   └── stage2.yaml              # Stage2 DiffDehaze training config
│   └── inference/                   # Inference configurations
│       ├── stage1.yaml              # Haze generation config
│       ├── stage2.yaml              # Standard dehazing config
│       └── stage2_accsamp_isr.yaml  # ISR-enhanced dehazing config
├── diffbir/                         # Core model implementation
│   ├── model/                       # Model architectures
│   │   ├── cldm.py                  # ControlLDM implementation
│   │   ├── controlnet.py            # YCbCr ControlNet
│   │   ├── gaussian_diffusion.py    # Diffusion model
│   │   └── vae.py                   # VAE encoder/decoder
│   ├── sampler/                     # Sampling algorithms
│   │   └── spaced_sampler.py        # ISR-AlignOp implementation
│   ├── utils/                       # Utility functions
│   │   ├── common.py                # Common utilities
│   │   └── losses.py                # Loss function implementations
│   └── pipeline.py                  # Inference pipeline
├── utils/                           # Additional utilities
│   ├── isr_align_utils.py           # ISR alignment utilities
│   ├── align_utils.py               # Basic alignment functions
│   ├── data_utils.py                # Data processing utilities
│   ├── dcp_utils.py                 # Dark Channel Prior utilities
│   └── ssim_utils.py                # SSIM calculation utilities
├── datasets.py                      # Dataset loading and processing
├── train_stage1.py                  # Stage1 HazeGen training script
├── train_stage2.py                  # Stage2 DiffDehaze training script
├── inference_stage1.py              # Haze generation inference
├── inference_stage2.py              # Standard dehazing inference
├── inference_accsamp.py             # AccSamp inference
├── inference_accsamp_isr.py         # ISR-enhanced inference (Recommended)
├── eval_metrics.py                  # Comprehensive evaluation metrics
├── evaluation_hcdyc.py              # HCD-YC evaluation system
├── requirements.txt                 # Python dependencies
├── weights/                         # Model checkpoints directory
├── inputs/                          # Input images directory
├── outputs/                         # Output results directory
├── datasets/                        # Training datasets directory
├── experiment/                      # Training experiment logs
├── evaluation_outputs/              # Evaluation results directory
└── assets/                          # Project assets and documentation
```

## Performance Benchmarks

### Computational Efficiency
- **Memory Requirements**: 24GB GPU supports 1024×1024 images
- **Processing Speed**: ~20 seconds per image on RTX 4090
- **Training Time**: 2-3 days per stage on 4×RTX 4090

### ISR-AlignOp Mode Comparison
| Mode | Quality | Speed | Memory | Use Case |
|------|---------|-------|--------|----------|
| Basic | Good | Fastest | Lowest | Simple scenes, quick processing |
| Adaptive | Excellent | Fast | Medium | Mixed conditions, recommended |
| Multi-scale | Best | Slower | Highest | Complex scenes, highest quality |

## Troubleshooting

### Memory Issues
- Enable `low_memory: true` in configuration
- Reduce `max_image_size` for large images
- Use `basic` ISR mode instead of `multi_scale`

### Training Issues
- Check dataset path configuration
- Verify model weight files are downloaded completely
- Run `accelerate config` before multi-GPU training

### Quality Optimization
- Use `multi_scale` ISR mode for complex scenes
- Adjust `quality_threshold` parameters
- Verify pre-trained weight paths

## Acknowledgments

This work builds upon the excellent [Learning Hazing-to-Dehazing](https://arxiv.org/abs/2503.19262) framework. We thank the authors for their outstanding contributions.

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.