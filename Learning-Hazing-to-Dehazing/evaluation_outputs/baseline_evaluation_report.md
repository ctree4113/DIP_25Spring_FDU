# Baseline Method Evaluation Report (Learning Hazing to Dehazing)

## Evaluation Overview

This report evaluates the baseline method 'Learning Hazing to Dehazing' on RTTS dataset.

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
| Learning_H2D_Stage2 | 1.9932 | 3.1715 | 3.1522 | 0.4689 | 0.3829 | 64.4703 | 14.8371 |

## Method Description

### Learning_H2D_Stage2
Learning Hazing to Dehazing - Stage2 (dehazing)

