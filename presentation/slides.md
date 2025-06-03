---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: HCD-YC - Hierarchical Cycle-consistent Hazing-Dehazing 
info: |
  ## HCD-YC: Hierarchical Cycle-consistent Hazing-Dehazing with YCbCr Fusion and ISR-AlignOp
  Real-world image dehazing with advanced deep learning techniques
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
---

# HCD-YC: 

## Hierarchical Cycle-consistent Hazing-Dehazing with YCbCr Fusion and ISR-AlignOp

## for Real-World Image Dehazing

<div @click="$slidev.nav.next" class="mt-12 py-1">
  Yi Cui
</div>

<div @click="$slidev.nav.next" class="mt-12 py-1">
  22307130038
</div>

<div class="abs-br m-6 text-xl">
  <a href="https://github.com/ctree4113/DIP_25Spring_FDU" target="_blank" class="slidev-icon-btn">
    <carbon:logo-github />
  </a>
</div>

---
layout: two-cols
layoutClass: gap-6
---

# Problem Statement

**Real-World Image Dehazing**

**Core Challenges**

**Physical Model Limitations**
- Atmospheric scattering: $I(x) = J(x)t(x) + A(1-t(x))$
- Ill-posed parameter estimation

**Domain Gap Problem**
- Synthetic vs. real atmospheric conditions
- Performance degradation on real scenes

**Information Loss Challenge**
- Heavily hazed images lose critical details
- Enhancement methods lack generative recovery

::right::

**Existing Method Limitations**

**Color Space Constraints**
- RGB-only processing misses scattering properties

**Cycle Consistency Deficiency**
- Pixel-level consistency only
- Missing multi-level structural preservation

**Static Adaptation Issues**  
- Fixed text prompts for varying conditions
- Single early prediction dependency

**Limited Generative Application**
- Heavy reliance on pre-training quality
- Insufficient diffusion utilization in dehazing

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: image-right
image: /flowchart_base.jpg
backgroundSize: contain
---

# Baseline Method

## Learning H2D Framework

**Two-Stage Pipeline:**

**Stage 1: HazeGen**
- Diffusion-based realistic haze generation
- IRControlNet conditional injection
- Domain adaptation via blended sampling

**Stage 2: DiffDehaze**  
- AccSamp: Accelerated sampling with early termination
- AlignOp: Statistical alignment in local patches
- Adaptive guidance based on haze density

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Our Framework

## Overview

<div class="w-full h-full flex justify-center items-center">
  <img src="/flowchart.jpg" class="rounded shadow-lg max-w-full max-h-full object-contain" alt="HCD-YC Framework">
</div>

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: two-cols
layoutClass: gap-6
---

## YCbCr-Assisted Haze Representation

**Color Space Transformation (ITU-R BT.601):**
$$\begin{bmatrix} Y \\ C_b \\ C_r \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.1687 & -0.3313 & 0.5 \\ 0.5 & -0.4187 & -0.0813 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$

**Dual-Branch Architecture:**
- **RGB Branch**: ResNet-style processing with skip connections
- **YCbCr Branch**: Parallel processing on converted space
- **Cross-Attention Fusion**: $F_{fused} = \alpha F_{RGB} + (1-\alpha) \text{Attn}_{RGB \rightarrow YCbCr}$

::right::

## Refined Text Guidance

**Dynamic Prompt Selection:**
- Multi-stage analysis: atmospheric conditions, haze density, scene type
- **Prompt Categories**:
  - Density: "clear light fog", "remove moderate haze", "eliminate dense fog"
  - Scene: "dehaze outdoor landscape", "clear urban scene"  
  - Type: "remove uniform haze", "clear patchy fog"

**Contrastive Alignment Loss:**
$$\mathcal{L}_{align} = -\log\left(\frac{\exp(s(E_t, E_v)/\tau)}{\sum_{j} \exp(s(E_t, E_v^j)/\tau)}\right)$$

$s(\cdot,\cdot)$ is similarity, $E_t$, $E_v$ are text/visual embeddings

---
layout: two-cols
layoutClass: gap-6
---

## Hierarchical Cycle-Consistency Learning

**Multi-Level Consistency Framework:**
$$\begin{align*}
\mathcal{L}_{cycle} &= \lambda_p(\rho) \mathcal{L}_{pixel} + \lambda_f(\rho) \mathcal{L}_{feature} \\
&\quad + \lambda_s(\rho) \mathcal{L}_{semantic} + \lambda_r \mathcal{L}_{region}
\end{align*}$$

**Haze Density Estimation:**
$$\rho = \frac{\rho_{lum} + \rho_{var} + \rho_{edge}}{3}$$

where $\rho_{lum} = 1 - \frac{\text{mean}(Y)}{\text{max}(Y)}$, $\rho_{var} = \frac{\text{var}(Y)}{\text{mean}(Y)}$, $\rho_{edge} = \frac{\text{edges}}{\text{pixels}}$

**Region Loss (DCP Integration):**
$$\mathcal{L}_{region} = \frac{1}{N} \sum_{i=1}^{N} \|DCP(I_i) - DCP(\hat{I}_i)\|_1$$

::right::

**Cycle Loss Components:**

**Forward Cycle:** $\mathcal{J} \xrightarrow{G_H} \mathcal{I} \xrightarrow{G_D} \hat{\mathcal{J}}$
$$\mathcal{L}_{forward} = \|\mathcal{J} - \hat{\mathcal{J}}\|_1 + \lambda_{vgg} \mathcal{L}_{VGG}(\mathcal{J}, \hat{\mathcal{J}})$$

**Backward Cycle:** $\mathcal{I} \xrightarrow{G_D} \mathcal{J} \xrightarrow{G_H} \hat{\mathcal{I}}$  
$$\mathcal{L}_{backward} = \|\mathcal{I} - \hat{\mathcal{I}}\|_1 + \lambda_{vgg} \mathcal{L}_{VGG}(\mathcal{I}, \hat{\mathcal{I}})$$

**Total Cycle Loss:**
$$\mathcal{L}_{total} = \mathcal{L}_{forward} + \mathcal{L}_{backward} + \mathcal{L}_{region}$$

**Adaptive Weighting:**
- $\lambda_p(\rho) = 0.8 + 0.4\rho$ (pixel importance)
- $\lambda_f(\rho) = 0.6 - 0.2\rho$ (feature emphasis)  
- $\lambda_s(\rho) = 0.4 + 0.3\rho$ (semantic consistency)
- $\lambda_r = 0.1$ (region constraint)

**Region-Aware Processing:** High-texture regions via gradient analysis ($\tau = 1.5$)

---

## ISR-AlignOp: Iterative Statistical Refinement

**Two-Step Iterative Strategy:**
- **Early Prediction** ($\tau_a = 0.3\tau$): Captures global structure
- **Later Prediction** ($\tau_b = 0.7\tau$): Refines local features

<div class="flex justify-center mt-6">
  <img src="/isr_alignop.jpg" class="rounded shadow-lg max-w-4xl max-h-60 object-contain" alt="ISR-AlignOp Architecture">
</div>

---
layout: two-cols
layoutClass: gap-6
---

# Experiment Setup

## Datasets

**Training Data:**
- **Stage 1 (YC-HazeGen)**: ~4,800 real-world hazy images (URHI split, RESIDE dataset)
- **RIDCP Synthetic Data**: Clear images + depth maps for realistic haze generation
- **Stage 2 (YC-DiffDehaze)**: HazeGen-generated pairs from OTS split (RESIDE)

**Evaluation Data:**
- **RTTS split**: 4,322 diverse real-world images with varying atmospheric conditions

::right::

## Train & Eval Setup

**Progressive Training Strategy:**
- **0-500 steps**: Warmup with base diffusion loss
- **1,500+ steps**: Dynamic prompt selection + text enhancement
- **5,000+ steps**: Hierarchical cycle consistency activation

**Evaluation Setup:**
- **RTTS dataset**: 4,322 real-world images (PASCAL VOC format)
- **No-reference metrics**: FADE, Q-Align, LIQE, CLIPIQA, ManIQA, MUSIQ, BRISQUE
- **ISR-AlignOp**: Adaptive mode (41×41, stride 10, threshold 0.12), 80 steps, τ=0.7

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: center
---

# Quantitative Results

*Preliminary results from 20,000 training iterations*

<div class="flex justify-center">
<div class="overflow-x-auto">

| Method | FADE ↓ | Q-Align ↑  | LIQE ↑ | CLIPIQA ↑ | ManIQA ↑ | MUSIQ ↑ | BRISQUE ↓ |
|:-------|:------:|:----------:|:------:|:---------:|:--------:|:-------:|:---------:|
| Learning H2D | 1.9876 | 3.1532 | 1.8875 | 0.3602 | 0.2466 | 51.3133 | 32.0978 |
| **HCD-YC (Ours)** | **1.8554** | **3.2173** | **2.4201** | **0.4246** | **0.3219** | **58.6616** | **29.9930** |
| **Improvement** | **↓6.6%** | **↑2.0%** | **↑28.2%** | **↑17.9%** | **↑30.5%** | **↑14.3%** | **↓6.5%** |

</div>
</div>

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: center
---

# Visual Comparison

<div class="max-w-7xl mx-auto">

<div class="flex justify-center gap-8 mb-4">
  <div class="flex items-center gap-2">
    <div class="w-4 h-4 border-2 border-red-300 rounded"></div>
    <span class="text-sm font-medium text-gray-700">Hazy Input</span>
  </div>
  <div class="flex items-center gap-2">
    <div class="w-4 h-4 border-2 border-blue-300 rounded"></div>
    <span class="text-sm font-medium text-gray-700">HCD-YC Dehazed</span>
  </div>
</div>

<div class="grid grid-cols-6 gap-2 mb-3">
  <img src="/inputs/1.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 1">
  <img src="/inputs/2.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 2">
  <img src="/inputs/3.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 3">
  <img src="/inputs/4.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 4">
  <img src="/inputs/5.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 5">
  <img src="/inputs/6.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 6">
</div>

<div class="grid grid-cols-6 gap-2 mb-4">
  <img src="/outputs/1.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 1">
  <img src="/outputs/2.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 2">
  <img src="/outputs/3.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 3">
  <img src="/outputs/4.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 4">
  <img src="/outputs/5.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 5">
  <img src="/outputs/6.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 6">
</div>

<div class="grid grid-cols-6 gap-2 mb-3">
  <img src="/inputs/7.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 7">
  <img src="/inputs/8.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 8">
  <img src="/inputs/9.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 9">
  <img src="/inputs/10.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 10">
  <img src="/inputs/11.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 11">
  <img src="/inputs/12.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-red-300" alt="Hazy 12">
</div>

<div class="grid grid-cols-6 gap-2">
  <img src="/outputs/7.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 7">
  <img src="/outputs/8.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 8">
  <img src="/outputs/9.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 9">
  <img src="/outputs/10.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 10">
  <img src="/outputs/11.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 11">
  <img src="/outputs/12.png" class="rounded shadow-lg h-22 object-cover w-full border-2 border-blue-300" alt="Clear 12">
</div>

</div>

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: center
class: text-center
---
# Thank You

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>
