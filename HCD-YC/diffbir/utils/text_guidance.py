import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextPromptPool:
    """
    提示词池，包含多种针对不同雾气场景的精确描述
    """

    def __init__(self):
        # baseline prompt
        self.base_prompts = [
            "remove dense fog",  # baseline prompt
            "clear up the hazy scene",
            "enhance visibility by removing fog",
            "restore the clear view from haziness",
        ]

        # prompt for different fog densities
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

        # prompt for different fog types
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

        # prompt for different scenes
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
    analyze fog image characteristics, determine the most suitable text prompt
    """

    def __init__(self, device='cuda'):
        self.device = device
        
    def analyze_fog_density(self, image: torch.Tensor) -> str:
        """
        analyze fog density, return density category: 'light', 'medium', 'heavy'
        Args:
            image: [B, C, H, W] fog image, value range [-1, 1] or [0, 1]
        """
        # ensure the image is in [0, 1] range
        if image.min() < 0:
            image = (image + 1) / 2
            
        # calculate the brightness channel
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # calculate the average brightness and brightness variance
        avg_brightness = torch.mean(gray, dim=[1, 2, 3])
        var_brightness = torch.var(gray, dim=[1, 2, 3])
        
        # determine the fog density based on brightness and variance, high brightness and low variance usually represent heavy fog
        batch_density = []
        for i in range(len(avg_brightness)):
            if avg_brightness[i] > 0.6:
                density = "heavy"
            elif avg_brightness[i] > 0.45:
                density = "medium"
            else:
                density = "light"
                
            # when the variance is very low, it usually represents large area fog
            if var_brightness[i] < 0.05:
                density = "heavy"
                
            batch_density.append(density)
            
        # return the most common density category
        if len(batch_density) == 1:
            return batch_density[0]
        else:
            # count the most common density category
            from collections import Counter
            counter = Counter(batch_density)
            return counter.most_common(1)[0][0]
    
    def analyze_fog_type(self, image: torch.Tensor) -> str:
        """
        analyze fog type, return type: 'uniform', 'layered', 'non_uniform'
        Args:
            image: [B, C, H, W] fog image, value range [-1, 1] or [0, 1]
        """
        # ensure the image is in [0, 1] range
        if image.min() < 0:
            image = (image + 1) / 2
            
        # calculate the brightness channel
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # calculate the vertical gradient, use simple difference approximation gradient
        vertical_grad = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        vertical_grad_mean = torch.mean(vertical_grad, dim=[1, 2, 3])
        
        # calculate the horizontal gradient
        horizontal_grad = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        horizontal_grad_mean = torch.mean(horizontal_grad, dim=[1, 2, 3])
        
        # calculate the local region standard deviation to evaluate non-uniformity, use pooling operation to calculate local region statistics
        patch_size = 16
        patches = F.unfold(gray, kernel_size=patch_size, stride=patch_size)
        patch_std = torch.std(patches, dim=1)
        patch_std_var = torch.var(patch_std, dim=1)
        
        batch_type = []
        for i in range(len(gray)):
            # determine the fog type, high vertical gradient usually represents layered fog
            if vertical_grad_mean[i] > 0.04 and vertical_grad_mean[i] > 1.5 * horizontal_grad_mean[i]:
                fog_type = "layered"
            # local standard deviation change large usually represents non-uniform fog
            elif patch_std_var[i] > 0.01:
                fog_type = "non_uniform"
            else:
                fog_type = "uniform"
                
            batch_type.append(fog_type)
            
        # return the most common type
        if len(batch_type) == 1:
            return batch_type[0]
        else:
            from collections import Counter
            counter = Counter(batch_type)
            return counter.most_common(1)[0][0]
    
    def detect_scene(self, image: torch.Tensor) -> str:
        """
        detect scene type, return: 'outdoor_urban', 'outdoor_nature', 'indoor'
        Args:
            image: [B, C, H, W] fog image, value range [-1, 1] or [0, 1]
        """
        # use simple color distribution and texture features to roughly estimate the scene type
        
        # ensure the image is in [0, 1] range
        if image.min() < 0:
            image = (image + 1) / 2
            
        batch_size = image.shape[0]
        scene_types = []
        
        for i in range(batch_size):
            img = image[i]
            
            # calculate the average value of RGB channels
            r_mean = torch.mean(img[0])
            g_mean = torch.mean(img[1])
            b_mean = torch.mean(img[2])
            
            # calculate the edge density
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            grad_x = torch.abs(gray[:, 1:] - gray[:, :-1])
            grad_y = torch.abs(gray[1:, :] - gray[:-1, :])
            edge_density = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
            
            # simple rules to determine the scene type
            # blue channel is high, and green channel is high, it may be a natural scene
            if b_mean > 0.4 and g_mean > r_mean:
                scene = "outdoor_nature"
            # edge density is high, and color distribution is uniform, it may be a urban scene
            elif edge_density > 0.1 and abs(r_mean - g_mean) < 0.05:
                scene = "outdoor_urban"
            # other cases may be indoor scene
            else:
                scene = "indoor"
                
            scene_types.append(scene)
        
        # return the most common scene type
        if len(scene_types) == 1:
            return scene_types[0]
        else:
            from collections import Counter
            counter = Counter(scene_types)
            return counter.most_common(1)[0][0]
    
    def select_optimal_prompt(self, image: torch.Tensor, prompt_pool: TextPromptPool) -> str:
        """
        select the optimal text prompt based on image characteristics
        Args:
            image: [B, C, H, W] fog image
            prompt_pool: text prompt pool
        Returns:
            selected prompt
        """
        # analyze fog characteristics
        density = self.analyze_fog_density(image)
        fog_type = self.analyze_fog_type(image)
        scene = self.detect_scene(image)
        
        # select the corresponding prompt
        density_prompts = prompt_pool.density_prompts[density]
        type_prompts = prompt_pool.type_prompts[fog_type]
        scene_prompts = prompt_pool.scene_prompts.get(scene, prompt_pool.base_prompts)
        
        # select the prompt from each category
        density_prompt = np.random.choice(density_prompts)
        type_prompt = np.random.choice(type_prompts)
        scene_prompt = np.random.choice(scene_prompts)
        
        # build the final prompt - combine density, type and scene
        combined_prompt = f"{density_prompt} in {scene}. {type_prompt}."
        
        return combined_prompt


class TextCondProcessor(nn.Module):
    """
    optimize the text condition embedding processing
    """
    
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # text condition enhancement layer
        self.enhance = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # text condition modulation layer
        self.modulation = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, text_cond):
        """
        process the text condition embedding
        Args:
            text_cond: text condition embedding [B, seq_len, embed_dim]
        """
        # enhance the text feature
        enhanced = self.enhance(text_cond)
        
        # generate modulation coefficients
        mod_weights = self.modulation(text_cond)
        
        # apply modulation
        modulated = text_cond * mod_weights + enhanced * (1 - mod_weights)
        
        return modulated
