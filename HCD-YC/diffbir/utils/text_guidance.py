import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union


class TextPromptPool:
    """
    提示词池，包含多种针对不同雾气场景的精确描述
    """

    def __init__(self):
        # 基础去雾提示
        self.base_prompts = [
            "remove dense fog",  # 基线使用的原始提示
            "clear up the hazy scene",
            "enhance visibility by removing fog",
            "restore the clear view from haziness",
        ]

        # 针对不同雾气密度的提示
        self.density_prompts = {
            "light": [
                "remove light fog", 
                "clear up the slightly hazy scene",
                "enhance mild foggy condition"
            ],
            "medium": [
                "remove moderate fog", 
                "clear up the moderately hazy scene",
                "recover details from medium fog"
            ],
            "heavy": [
                "remove dense fog", 
                "clear up the extremely hazy scene",
                "restore clarity from heavy fog"
            ]
        }

        # 针对不同雾气类型的提示
        self.type_prompts = {
            "uniform": [
                "remove uniform fog layer",
                "clear up evenly distributed haze",
                "enhance visibility in uniform foggy condition"
            ],
            "layered": [
                "remove layered fog", 
                "clear up stratified haze",
                "resolve depth-varying fog layers"
            ],
            "non_uniform": [
                "remove patchy fog", 
                "clear up unevenly distributed haze",
                "enhance visibility in non-uniform foggy condition"
            ]
        }

        # 针对不同场景的提示
        self.scene_prompts = {
            "outdoor_urban": [
                "remove fog from urban scene", 
                "clear up hazy city view", 
                "enhance visibility of buildings and streets"
            ],
            "outdoor_nature": [
                "remove fog from natural landscape", 
                "clear up hazy mountain view",
                "enhance visibility of trees and vegetation" 
            ],
            "indoor": [
                "remove indoor haze", 
                "clear up foggy indoor scene",
                "enhance visibility of interior details"
            ]
        }


class FogAnalyzer:
    """
    分析雾图特性，确定最适合的文本提示
    """

    def __init__(self, device='cuda'):
        self.device = device
        
    def analyze_fog_density(self, image: torch.Tensor) -> str:
        """
        分析雾气密度，返回密度类别：'light', 'medium', 'heavy'
        参数:
            image: [B, C, H, W] 雾图，值范围为[-1, 1]或[0, 1]
        """
        # 确保图像在[0, 1]范围内
        if image.min() < 0:
            image = (image + 1) / 2
            
        # 计算亮度通道
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # 计算平均亮度和亮度方差
        avg_brightness = torch.mean(gray, dim=[1, 2, 3])
        var_brightness = torch.var(gray, dim=[1, 2, 3])
        
        # 基于亮度和方差确定雾气密度
        # 高亮度低方差通常表示重雾
        batch_density = []
        for i in range(len(avg_brightness)):
            if avg_brightness[i] > 0.6:
                density = "heavy"
            elif avg_brightness[i] > 0.45:
                density = "medium"
            else:
                density = "light"
                
            # 方差很低时通常是大面积雾气
            if var_brightness[i] < 0.05:
                density = "heavy"
                
            batch_density.append(density)
            
        # 返回批次中最常见的密度类别
        if len(batch_density) == 1:
            return batch_density[0]
        else:
            # 统计最常见的密度
            from collections import Counter
            counter = Counter(batch_density)
            return counter.most_common(1)[0][0]
    
    def analyze_fog_type(self, image: torch.Tensor) -> str:
        """
        分析雾气类型，返回类型：'uniform', 'layered', 'non_uniform'
        参数:
            image: [B, C, H, W] 雾图，值范围为[-1, 1]或[0, 1]
        """
        # 确保图像在[0, 1]范围内
        if image.min() < 0:
            image = (image + 1) / 2
            
        # 计算亮度通道
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # 计算垂直方向的梯度
        # 使用简单的差分近似梯度
        vertical_grad = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        vertical_grad_mean = torch.mean(vertical_grad, dim=[1, 2, 3])
        
        # 计算水平方向的梯度
        horizontal_grad = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        horizontal_grad_mean = torch.mean(horizontal_grad, dim=[1, 2, 3])
        
        # 计算局部区域的标准差来评估非均匀性
        # 使用池化操作计算局部区域统计
        patch_size = 16
        patches = F.unfold(gray, kernel_size=patch_size, stride=patch_size)
        patch_std = torch.std(patches, dim=1)
        patch_std_var = torch.var(patch_std, dim=1)
        
        batch_type = []
        for i in range(len(gray)):
            # 判断雾气类型
            # 高垂直梯度通常表示分层雾
            if vertical_grad_mean[i] > 0.04 and vertical_grad_mean[i] > 1.5 * horizontal_grad_mean[i]:
                fog_type = "layered"
            # 局部标准差变化大通常表示非均匀雾
            elif patch_std_var[i] > 0.01:
                fog_type = "non_uniform"
            else:
                fog_type = "uniform"
                
            batch_type.append(fog_type)
            
        # 返回批次中最常见的类型
        if len(batch_type) == 1:
            return batch_type[0]
        else:
            from collections import Counter
            counter = Counter(batch_type)
            return counter.most_common(1)[0][0]
    
    def detect_scene(self, image: torch.Tensor) -> str:
        """
        检测场景类型，返回：'outdoor_urban', 'outdoor_nature', 'indoor'
        参数:
            image: [B, C, H, W] 雾图，值范围为[-1, 1]或[0, 1]
        注意：这是一个简化版本，实际应用中可以使用更复杂的场景分类器
        """
        # 使用简单的颜色分布和纹理特征来粗略估计场景类型
        # 实际应用中可以使用预训练的场景分类器
        
        # 确保图像在[0, 1]范围内
        if image.min() < 0:
            image = (image + 1) / 2
            
        batch_size = image.shape[0]
        scene_types = []
        
        for i in range(batch_size):
            img = image[i]
            
            # 计算RGB通道平均值
            r_mean = torch.mean(img[0])
            g_mean = torch.mean(img[1])
            b_mean = torch.mean(img[2])
            
            # 计算边缘密度
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            grad_x = torch.abs(gray[:, 1:] - gray[:, :-1])
            grad_y = torch.abs(gray[1:, :] - gray[:-1, :])
            edge_density = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
            
            # 简单规则判断场景类型
            # 蓝色比例高，且绿色偏高，可能是自然场景
            if b_mean > 0.4 and g_mean > r_mean:
                scene = "outdoor_nature"
            # 边缘密度高，且颜色分布比较均匀，可能是城市场景
            elif edge_density > 0.1 and abs(r_mean - g_mean) < 0.05:
                scene = "outdoor_urban"
            # 其他情况可能是室内场景
            else:
                scene = "indoor"
                
            scene_types.append(scene)
        
        # 返回批次中最常见的场景类型
        if len(scene_types) == 1:
            return scene_types[0]
        else:
            from collections import Counter
            counter = Counter(scene_types)
            return counter.most_common(1)[0][0]
    
    def select_optimal_prompt(self, image: torch.Tensor, prompt_pool: TextPromptPool) -> str:
        """
        根据图像特性选择最优文本提示
        参数:
            image: [B, C, H, W] 雾图
            prompt_pool: 文本提示池
        返回:
            选择的提示词
        """
        # 分析雾气特性
        density = self.analyze_fog_density(image)
        fog_type = self.analyze_fog_type(image)
        scene = self.detect_scene(image)
        
        # 选择对应的提示词
        density_prompts = prompt_pool.density_prompts[density]
        type_prompts = prompt_pool.type_prompts[fog_type]
        scene_prompts = prompt_pool.scene_prompts.get(scene, prompt_pool.base_prompts)
        
        # 从各类别中随机选择提示词
        # 在实际应用中，可以基于更复杂的策略选择最优提示词
        density_prompt = np.random.choice(density_prompts)
        type_prompt = np.random.choice(type_prompts)
        scene_prompt = np.random.choice(scene_prompts)
        
        # 构建最终提示词 - 结合密度、类型和场景
        combined_prompt = f"{density_prompt} in {scene}. {type_prompt}."
        
        return combined_prompt


class TextCondProcessor(nn.Module):
    """
    优化文本条件嵌入的处理方式
    """
    
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 文本条件加强层
        self.enhance = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 文本条件调制层
        self.modulation = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, text_cond):
        """
        处理文本条件嵌入
        参数:
            text_cond: 文本条件嵌入 [B, seq_len, embed_dim]
        """
        # 增强文本特征
        enhanced = self.enhance(text_cond)
        
        # 生成调制系数
        mod_weights = self.modulation(text_cond)
        
        # 应用调制
        modulated = text_cond * mod_weights + enhanced * (1 - mod_weights)
        
        return modulated 