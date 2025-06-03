# Baseline Method Evaluation Report (Learning Hazing to Dehazing)

## Evaluation Overview

This report evaluates the baseline method 'Learning Hazing to Dehazing' with standard AlignOp on RTTS dataset using the same evaluation framework as the HCD-YC enhanced method for fair comparison.

**Evaluation Sample Size:** 200 randomly selected images

## Baseline Method Features

The baseline evaluation uses:
- **Standard AlignOp**: Original alignment operation without ISR enhancement
- **AccSamp sampling**: τ=0.8, ω=0.6 (original paper settings)
- **Standard ControlNet**: Without YCbCr fusion enhancements
- **Basic guidance**: WeightedSSIMGuidance with scale=0.1

## Evaluation Metrics

Using the following no-reference image quality assessment metrics:

- **FADE** (lower better): Fog density assessment
- **Q-Align** (higher better): Large model-based visual quality assessment
- **LIQE** (higher better): Vision-language correspondence blind image quality assessment
- **CLIPIQA** (higher better): CLIP-based image quality assessment
- **ManIQA** (higher better): Multi-dimensional attention network quality assessment
- **MUSIQ** (higher better): Multi-scale image quality assessment
- **BRISQUE** (lower better): Spatial domain no-reference quality assessment

## Quantitative Results

| Method | FADE | Q-Align | LIQE | CLIPIQA | ManIQA | MUSIQ | BRISQUE |
|--------|------|---------|------|---------|--------|-------|----------|
| Learning_H2D_Baseline | 1.9876 | 3.1532 | 1.8875 | 0.3602 | 0.2466 | 51.3133 | 32.0978 |

## Method Description

### Learning_H2D_Baseline
Learning Hazing to Dehazing - Baseline AlignOp (dehazing)

## Baseline Performance Notes

- **Standard AlignOp**: Uses single-step alignment without iterative refinement
- **No YCbCr Enhancement**: Standard RGB processing without color space advantages
- **Basic Guidance**: Lower guidance strength compared to ISR-enhanced version
- **Reference Performance**: Serves as baseline for comparison with HCD-YC improvements
