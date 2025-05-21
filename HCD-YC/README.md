# Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing [:link:](https://arxiv.org/abs/2503.19262)

![Python 3.10](https://img.shields.io/badge/python-3.10-g) ![pytorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-blue.svg)

This repository presents the implementation of the paper

>**Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing**<br> Ruiyi Wang, Yushuo Zheng, Zicheng Zhang, Chunyi Li, Shuaicheng Liu, Guangtao Zhai, Xiaohong Liu<br>The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2025

We present a novel hazing-dehazing pipeline consisting of a Realistic Hazy Image Generation framework (HazeGen) and a Diffusion-based Dehazing framework (DiffDehaze).

![teaser](assets/teaser.png)
![teaser](assets/result.png)

## 🛠️ Setup

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing.git
cd Learning-Hazing-to-Dehazing
```

### 💻 Dependencies
Using [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After the installation, create the environment and install dependencies into it:

```bash
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing

### 📷 Prepare input images

Place your images in the `inputs/` directory. To set a different source directory, you can edit configuration files in `configs/inference/`.

### ⬇ Download Checkpoints

Download pre-trained models and place them to folder `weights/`, but you can always edit configuration files in `configs/inference/`.

|        Model Name        |                         Description                          |                             Link                             |
| :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| v2-1_512-ema-pruned.ckpt | Pretrained Stable Diffusion v2.1 from stabilityai, providing generative priors | [download](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) |
|        stage1.pt         |               IRControlNet trained for HazeGen               | [download](https://pan.baidu.com/s/1vbxEwftJC9nUaMXJ-t9sww?pwd=8egg) |
|        stage2.pt         |             IRControlNet trained for DiffDehaze              | [download](https://pan.baidu.com/s/1vbxEwftJC9nUaMXJ-t9sww?pwd=8egg) |

### 🚀 Run inference

To perform dehazing with standard spaced sampler, please run

```bash
python inference_stage2.py --config configs/inference/stage2.yaml
```

By default, results will be saved to `outputs/`. **Enjoy**!

To perform dehazing with AccSamp sampler, please run

```bash
python inference_accsamp.py --config configs/inference/stage2_accsamp.yaml
```

To generate realisitic hazy images with HazeGen, please run

```bash
python inference_stage1.py --config configs/inference/stage1.yaml
```

To use a different hyperparameter settings, e.g., $\tau$ and $\omega$, please edit the corresponding `.yaml` configuration file.

## 🏋️ Training

1. Training data preparation. 
   - The training of HazeGen requires real-world hazy data from the URHI split of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-β) dataset and the synthetic hazy data from [RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing). 
   - To train DiffDehaze, you need to generate realistic hazy data from HazeGen based on clean images, e.g., the clean images from the OTS split of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-β) dataset, using the inference script above.

2. Accelerate configuration. The training is supported by the huggingface [Accelerate](https://huggingface.co/docs/transformers/accelerate) library. Before running training scripts, create and save a configuration file to help Accelerate correctly set up training based on your setup by running 

```bash
accelerate config
```

3. Fill in the training configuration files in `configs/train/` with appropriate values, especially for the paths to the training data. Please find specific instructions there.
4. Start training! To train stage1 HazeGen model, run

```bash
accelerate launch train_stage1.py --config configs/train/stage1.yaml
```

5. To train stage2 DiffDehaze model, run

```bash
accelerate launch train_stage2.py --config configs/train/stage2.yaml
```

## ✏️ Acknowledgment

A large part of the implementation is based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR?tab=readme-ov-file#inference). We sincerely appreciate their wonderful work.

## 🎓 Citation

If you find our work useful, please consider cite our paper:

```bibtex
@misc{wang2025learninghazingdehazingrealistic,
      title={Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing}, 
      author={Ruiyi Wang and Yushuo Zheng and Zicheng Zhang and Chunyi Li and Shuaicheng Liu and Guangtao Zhai and Xiaohong Liu},
      year={2025},
      eprint={2503.19262},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19262}, 
}
```

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and models you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

# HCD-YC：层级循环一致的YCbCr辅助去雾框架

我们基于"Learning Hazing to Dehazing"原始框架进行了三项重要创新，形成了HCD-YC框架，显著提升了去雾效果和真实感。HCD-YC保留了双阶段设计（雾气生成+雾气去除），同时在每个阶段引入了创新性的改进。

## 整体架构

HCD-YC框架包含两个主要阶段：

1. **YC-HazeGen**：雾气生成阶段，从清晰图像生成真实雾图
   - 使用YCbCr色彩空间辅助处理
   - 应用层级循环一致性约束

2. **YC-DiffDehaze**：去雾阶段，从雾图还原清晰图像
   - 采用YCbCr色彩空间多特征融合
   - 利用层级循环一致性学习
   - 使用优化的文本引导控制

## YCbCr色彩空间辅助去雾

我们对原始模型进行了增强，引入了YCbCr色彩空间处理模块，以提高去雾效果，特别是在保留图像细节和处理复杂场景方面。

### 主要改进

1. **双色彩空间处理**：同时在RGB和YCbCr色彩空间中提取和处理特征，充分利用YCbCr空间对结构和纹理的更好表示能力。

2. **特征融合机制**：设计了基于注意力的特征融合模块，有效整合RGB和YCbCr空间的优势，提升去雾效果。

3. **多层级特征交互**：在网络的多个层级实现RGB和YCbCr特征的交互，确保全面的信息融合。

### 使用方法

要启用YCbCr色彩空间处理，请使用修改后的配置文件：

```bash
# 训练第一阶段
python train_stage1.py --config configs/train/stage1.yaml

# 训练第二阶段
python train_stage2.py --config configs/train/stage2.yaml
```

### 实验结果

YCbCr色彩空间辅助处理在以下方面提供了改进：

- 更好的细节保留，特别是在纹理丰富的区域
- 改善了色彩还原质量，减少了去雾过程中的色偏
- 在非均匀雾气场景中展现出更强的去雾能力

## 层级循环一致性学习

我们设计了一种新颖的层级循环一致性学习框架，在多个特征层级上强化去雾模型的一致性，大幅提升结果的自然度和真实感。

### 主要改进

1. **多层级一致性约束**：将循环一致性扩展到三个不同层级：
   - 像素级一致性：确保基本色彩和亮度的准确恢复
   - 特征级一致性：通过VGG特征保证纹理和中等尺度结构的一致性
   - 语义级一致性：利用高级特征确保语义信息和大尺度结构的保持

2. **自适应权重调整**：根据雾气密度动态调整不同层级损失的权重，对不同场景适应性更强

3. **区域感知循环一致性**：针对不同图像区域（如天空、建筑物等）采用不同的一致性约束，提高细节保留

4. **非对称循环学习**：考虑到雾化和去雾过程的不对称性，设计了权重不同的双向一致性损失

### 使用方法

要启用层级循环一致性学习，确保在配置文件中设置相关参数：

```bash
# 例如启用阶段一的循环一致性训练
accelerate launch train_stage1.py --config configs/train/stage1.yaml
```

### 实验结果

层级循环一致性学习带来的显著改进：

- 更自然的细节恢复，减少过度平滑和伪影
- 保持了原始图像的语义信息和全局结构
- 有效避免了常见的色偏和伪彩问题
- 在复杂真实场景中展现出更强的泛化能力

## 优化的文本引导控制

我们对文本引导机制进行了深度优化，增强了模型根据雾图特性动态选择和优化文本提示的能力，提高了去雾的精确度和语义一致性。

### 主要改进

1. **动态文本提示池**：建立了分层的文本提示库，能根据不同雾气密度（轻度、中度、重度）、雾气类型（均匀、分层、非均匀）和场景类型（城市、自然、室内）提供精确描述。

2. **智能雾气分析器**：设计了专门的雾气分析模块，能自动评估输入图像的雾气特性，包括：
   - 密度估计：通过亮度和方差分析
   - 类型判断：通过梯度和纹理特征
   - 场景识别：基于颜色分布和边缘密度

3. **文本条件嵌入优化**：实现了专门的文本条件处理器，通过两个关键机制优化文本嵌入：
   - 增强层：提升文本特征的表达能力
   - 调制层：动态调整文本引导的影响程度

4. **文本-图像对齐损失**：创新性引入了文本与图像特征的对齐机制，确保去雾结果与文本描述在语义上保持一致。

### 使用方法

文本引导控制已在推理和训练脚本中完全集成，可通过配置文件控制其行为：

```yaml
# 在推理配置中启用动态文本提示和文本处理
inference:
  use_dynamic_prompt: True  # 是否使用动态文本提示
  use_text_processor: True  # 是否使用文本条件优化处理

# 在训练配置中设置详细参数
train:
  use_dynamic_prompt: True    # 启用动态提示词选择
  dynamic_prompt_start: 2000  # 开始使用动态提示词的步数
  use_text_processor: True    # 启用文本条件优化处理
  text_processor_start: 2000  # 开始使用处理器的步数
  text_align_weight: 0.3      # 文本-图像对齐损失权重
  text_align_max_step: 3000   # 对齐损失达到最大权重的步数
```

### 实验结果

优化的文本引导控制带来的显著改进：

- 更精确的去雾效果，特别是针对不同类型的雾气场景
- 提高了语义一致性，生成的结果更符合描述意图
- 增强了模型对复杂真实场景的适应能力
- 减少了训练过程中的不稳定性，加速了收敛

## 训练与推理

### 训练方法

完整的训练流程包括两个阶段：

1. **训练第一阶段 (YC-HazeGen)**：

```bash
# 基础训练命令
accelerate launch train_stage1.py --config configs/train/stage1.yaml

# 多GPU训练 (例如使用4个GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_stage1.py --config configs/train/stage1.yaml
```

2. **训练第二阶段 (YC-DiffDehaze)**：

```bash
# 基础训练命令
accelerate launch train_stage2.py --config configs/train/stage2.yaml

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train_stage2.py --config configs/train/stage2.yaml
```

### 推理方法

我们提供多种推理选项，适应不同需求：

1. **标准去雾推理**：

```bash
python inference_stage2.py --config configs/inference/stage2.yaml
```

2. **使用加速采样进行去雾**：

```bash
python inference_accsamp.py --config configs/inference/stage2_accsamp.yaml
```

3. **雾化图像生成**：

```bash
python inference_stage1.py --config configs/inference/stage1.yaml
```

### 参数配置

通过修改配置文件可以控制各种功能和参数：

- **YCbCr色彩空间**：通过`ycbcr_fusion: True`控制
- **层级循环一致性**：通过`cycle_weight`, `pixel_weight`, `feature_weight`等参数控制
- **文本引导控制**：通过`use_dynamic_prompt`, `text_align_weight`等参数控制

### 实验评估

我们在标准数据集和真实场景图像上进行了全面评估：

- **定量指标**：PSNR, SSIM, CIEDE2000等
- **无参考指标**：NIQE, BRISQUE
- **主观评价**：清晰度, 自然度, 细节保留程度

所有三项创新都显著提升了去雾性能，尤其在处理复杂真实场景和非均匀雾气时效果显著。
