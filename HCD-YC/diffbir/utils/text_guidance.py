import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPromptPool:
    """
    Enhanced text prompt pool with comprehensive descriptions for different fog scenarios
    """

    def __init__(self):
        # base prompts for general dehazing
        self.base_prompts = [
            "remove dense fog",
            "clear up the hazy scene", 
            "enhance visibility by removing fog",
            "restore clear view from haziness",
            "eliminate atmospheric haze"
        ]

        # density-specific prompts with improved descriptions
        self.density_prompts = {
            "light": [
                "clear light fog and enhance visibility",
                "remove mild atmospheric haze gently",
                "enhance slightly foggy image clarity",
                "improve visibility through light morning fog"
            ],
            "medium": [
                "remove moderate fog and restore clarity", 
                "clear medium density haze effectively",
                "restore visibility from moderate atmospheric fog",
                "enhance image through medium haze layer"
            ],
            "heavy": [
                "remove dense thick fog completely", 
                "clear extremely heavy haze and restore details",
                "restore from thick impenetrable fog",
                "eliminate dense atmospheric fog entirely"
            ]
        }

        # fog distribution type prompts with spatial awareness
        self.distribution_prompts = {
            "uniform": [
                "clear uniform fog layer evenly",
                "remove evenly distributed atmospheric haze"
            ],
            "patchy": [
                "clear patchy fog areas selectively", 
                "remove unevenly distributed haze patches"
            ],
            "depth_varying": [
                "clear depth-varying fog layers",
                "remove layered atmospheric haze with depth awareness"
            ]
        }

        # enhanced scene-aware prompts
        self.scene_prompts = {
            "outdoor": [
                "enhance outdoor landscape visibility",
                "clear outdoor atmospheric haze naturally"
            ],
            "urban": [
                "remove urban fog and pollution effects",
                "clear city scene visibility and air quality"  
            ],
            "nature": [
                "clear natural landscape fog preserving colors",
                "enhance mountain forest visibility naturally"
            ]
        }


class FogAnalyzer:
    """
    Enhanced fog characteristic analyzer with improved robustness and precision
    """

    def __init__(self, device='cuda'):
        self.device = device
        
        # adaptive thresholds for better classification
        self.density_thresholds = {
            'brightness': [0.4, 0.65],  # [light_to_medium, medium_to_heavy]
            'contrast': [0.12, 0.06],   # [light_to_medium, medium_to_heavy] 
            'edge': [0.15, 0.08]        # [light_to_medium, medium_to_heavy]
        }
        
    def analyze_fog_density(self, image: torch.Tensor) -> str:
        """
        Enhanced fog density analysis with multi-metric approach
        Args:
            image: [B, C, H, W] input image, range [0, 1] or [-1, 1]
        """
        # normalize to [0, 1] range
        if image.min() < 0:
            image = (image + 1) / 2
            
        # calculate luminance using ITU-R BT.709 standards
        luminance = 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
        
        # enhanced brightness analysis
        mean_lum = torch.mean(luminance, dim=[2, 3])
        std_lum = torch.std(luminance, dim=[2, 3])
        
        # improved contrast metric with local analysis
        # use Michelson contrast
        max_lum = torch.max(luminance.view(image.shape[0], -1), dim=1)[0]
        min_lum = torch.min(luminance.view(image.shape[0], -1), dim=1)[0]
        michelson_contrast = (max_lum - min_lum) / (max_lum + min_lum + 1e-8)
        
        # enhanced edge density with sobel operator
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        lum_padded = F.pad(luminance, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(lum_padded, sobel_x)
        grad_y = F.conv2d(lum_padded, sobel_y)
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        edge_density = torch.mean(edge_magnitude, dim=[2, 3])
        
        # multi-metric decision making
        # ensure all tensors have same shape for consistent indexing
        mean_lum_flat = mean_lum.squeeze()
        michelson_contrast_flat = michelson_contrast
        edge_density_flat = edge_density.squeeze()
        
        # handle scalar case (single image) by ensuring at least 1D
        if mean_lum_flat.dim() == 0:
            mean_lum_flat = mean_lum_flat.unsqueeze(0)
        if michelson_contrast_flat.dim() == 0:
            michelson_contrast_flat = michelson_contrast_flat.unsqueeze(0)
        if edge_density_flat.dim() == 0:
            edge_density_flat = edge_density_flat.unsqueeze(0)
        
        # initialize score tensors with same shape
        brightness_scores = torch.zeros_like(mean_lum_flat)
        contrast_scores = torch.zeros_like(mean_lum_flat) 
        edge_scores = torch.zeros_like(mean_lum_flat)
        
        # brightness classification
        brightness_scores[mean_lum_flat > self.density_thresholds['brightness'][1]] = 2  # heavy
        brightness_scores[(mean_lum_flat > self.density_thresholds['brightness'][0]) & 
                         (mean_lum_flat <= self.density_thresholds['brightness'][1])] = 1  # medium
        
        # contrast classification (inverse relationship)
        contrast_scores[michelson_contrast_flat < self.density_thresholds['contrast'][1]] = 2  # heavy
        contrast_scores[(michelson_contrast_flat >= self.density_thresholds['contrast'][1]) & 
                       (michelson_contrast_flat < self.density_thresholds['contrast'][0])] = 1  # medium
        
        # edge density classification (inverse relationship) 
        edge_scores[edge_density_flat < self.density_thresholds['edge'][1]] = 2  # heavy
        edge_scores[(edge_density_flat >= self.density_thresholds['edge'][1]) & 
                   (edge_density_flat < self.density_thresholds['edge'][0])] = 1  # medium
        
        # weighted voting system
        final_scores = 0.4 * brightness_scores + 0.3 * contrast_scores + 0.3 * edge_scores
        
        # determine most common classification
        if torch.mean(final_scores) >= 1.5:
            return "heavy"
        elif torch.mean(final_scores) >= 0.8:
            return "medium"
        else:
            return "light"
    
    def analyze_fog_distribution(self, image: torch.Tensor) -> str:
        """
        Enhanced fog distribution analysis with better spatial understanding
        Args:
            image: [B, C, H, W] input image
        """
        if image.min() < 0:
            image = (image + 1) / 2
            
        # convert to grayscale with proper weighting
        gray = 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
        
        # multi-scale patch analysis for better distribution understanding
        patch_sizes = [16, 32, 64]
        distribution_scores = []
        
        for patch_size in patch_sizes:
            if gray.shape[2] >= patch_size and gray.shape[3] >= patch_size:
                patches = F.unfold(gray, kernel_size=patch_size, stride=patch_size//2)
                patch_means = torch.mean(patches, dim=1)
                patch_vars = torch.var(patches, dim=1)
                
                # analyze variance distribution
                var_of_vars = torch.var(patch_vars, dim=1)
                distribution_scores.append(var_of_vars)
        
        if distribution_scores:
            avg_distribution_score = torch.mean(torch.stack(distribution_scores), dim=0)
        else:
            avg_distribution_score = torch.tensor(0.01, device=self.device)
        
        # enhanced gradient analysis for depth variation
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        
        # analyze gradient patterns - fix dimension issues
        # grad_x shape: [B, C, H, W-1], grad_y shape: [B, C, H-1, W]
        horizontal_grad_var = torch.var(torch.mean(grad_x, dim=3), dim=2)  # mean over width, var over height
        vertical_grad_var = torch.var(torch.mean(grad_y, dim=2), dim=2)    # mean over height, var over width
        depth_variation = torch.mean(horizontal_grad_var + vertical_grad_var)
        
        # enhanced classification with adaptive thresholds
        uniform_threshold = 0.003
        depth_threshold = 0.02
        
        if torch.mean(avg_distribution_score) < uniform_threshold:
            return "uniform"
        elif depth_variation > depth_threshold:
            return "depth_varying" 
        else:
            return "patchy"
    
    def detect_scene_type(self, image: torch.Tensor) -> str:
        """
        Enhanced scene type detection with improved feature analysis
        Args:
            image: [B, C, H, W] input image
        """
        if image.min() < 0:
            image = (image + 1) / 2
            
        # enhanced color analysis
        rgb_means = torch.mean(image, dim=[2, 3])
        rgb_stds = torch.std(image, dim=[2, 3])
        
        # color distribution analysis
        r_mean, g_mean, b_mean = rgb_means[0]
        r_std, g_std, b_std = rgb_stds[0]
        
        # enhanced edge and texture analysis
        gray = torch.mean(image, dim=1, keepdim=True)
        
        # compute local binary patterns for texture
        lbp_kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                                 dtype=torch.float32, device=self.device)
        gray_padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        lbp_response = F.conv2d(gray_padded, lbp_kernel)
        texture_complexity = torch.std(lbp_response)
        
        # edge density with multiple scales
        edge_densities = []
        for kernel_size in [3, 5, 7]:
            # create proper sobel kernels
            pad_size = kernel_size // 2
            gray_pad = F.pad(gray, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            
            # use standard sobel operators
            if kernel_size == 3:
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            else:
                # for larger kernels, use simplified edge detection
                sobel_x = torch.ones(1, 1, 1, kernel_size, device=self.device) * -1
                sobel_x[:, :, :, kernel_size//2] = 0
                sobel_x[:, :, :, kernel_size//2+1:] = 1
                
                sobel_y = torch.ones(1, 1, kernel_size, 1, device=self.device) * -1
                sobel_y[:, :, kernel_size//2, :] = 0
                sobel_y[:, :, kernel_size//2+1:, :] = 1
            
            grad_x = F.conv2d(gray_pad, sobel_x)
            grad_y = F.conv2d(gray_pad, sobel_y)
            
            # ensure dimensions match for addition
            min_h = min(grad_x.shape[2], grad_y.shape[2])
            min_w = min(grad_x.shape[3], grad_y.shape[3])
            grad_x = grad_x[:, :, :min_h, :min_w]
            grad_y = grad_y[:, :, :min_h, :min_w]
            
            edge_density = torch.mean(torch.sqrt(grad_x**2 + grad_y**2 + 1e-6))
            edge_densities.append(edge_density)
        
        avg_edge_density = torch.mean(torch.stack(edge_densities))
        
        # enhanced classification logic
        # nature scenes: high green component, moderate texture
        if (g_mean > r_mean * 1.1 and g_mean > b_mean * 1.1 and 
            texture_complexity > 0.15 and avg_edge_density > 0.08):
            return "nature"
        
        # urban scenes: high edge density, balanced colors
        elif (avg_edge_density > 0.12 and texture_complexity > 0.2 and
              abs(r_mean - g_mean) < 0.1 and abs(g_mean - b_mean) < 0.1):
            return "urban"
        
        # outdoor general scenes
        else:
            return "outdoor"
    
    def select_optimal_prompt(self, image: torch.Tensor, prompt_pool: TextPromptPool) -> str:
        """
        Enhanced prompt selection with intelligent rule-based combination
        """
        density = self.analyze_fog_density(image)
        distribution = self.analyze_fog_distribution(image)
        scene = self.detect_scene_type(image)
        
        # intelligent prompt combination strategy
        density_prompts = prompt_pool.density_prompts[density]
        distribution_prompts = prompt_pool.distribution_prompts[distribution]
        scene_prompts = prompt_pool.scene_prompts.get(scene, prompt_pool.base_prompts)
        
        # enhanced deterministic selection strategy
        if density == "heavy":
            # for heavy fog, prioritize strong dehazing with scene context
            if scene == "urban":
                base_prompt = "remove dense urban fog and pollution effects completely"
            elif scene == "nature":
                base_prompt = "clear thick natural fog preserving landscape details"
            else:
                base_prompt = density_prompts[0]  # "remove dense thick fog completely"
        elif density == "medium":
            # for medium fog, balanced approach with distribution awareness
            if distribution == "patchy":
                base_prompt = "remove moderate patchy fog areas selectively"
            elif distribution == "depth_varying":
                base_prompt = "clear medium density layered fog with depth awareness"
            else:
                base_prompt = density_prompts[0]  # "remove moderate fog and restore clarity"
        else:
            # for light fog, gentle approach with scene preservation
            if scene == "nature":
                base_prompt = "gently clear light morning fog preserving natural colors"
            else:
                base_prompt = density_prompts[0]  # "clear light fog and enhance visibility"
        
        # add distribution-specific refinements
        if distribution == "patchy" and density != "medium":
            base_prompt = f"{base_prompt}, focus on uneven fog patches"
        elif distribution == "depth_varying" and density != "medium":
            base_prompt = f"{base_prompt}, handle layered depth variations"
        
        # add scene-specific context for challenging cases
        if density in ["medium", "heavy"] and scene == "urban":
            base_prompt = f"{base_prompt} in complex urban environment"
        elif density in ["medium", "heavy"] and scene == "nature":
            base_prompt = f"{base_prompt} preserving natural landscape beauty"
            
        return base_prompt


class TextCondProcessor(nn.Module):
    """
    Enhanced text condition processor with improved architectural design
    """
    
    def __init__(self, embed_dim=1024, reduction_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim // reduction_ratio
        
        # enhanced processing architecture with residual connections
        self.enhance_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # improved attention mechanism with multiple heads
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # learnable residual connection weights
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.05))
        
        # output projection for final refinement
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

    def forward(self, text_cond):
        """
        Enhanced text condition processing with multi-stage refinement
        Args:
            text_cond: [B, seq_len, embed_dim] or [B, embed_dim]
        """
        # handle different input shapes
        original_shape = text_cond.shape
        if len(text_cond.shape) == 2:
            text_cond = text_cond.unsqueeze(1)  # [B, 1, embed_dim]
        
        # self-attention for context refinement with residual connection
        attended, attention_weights = self.attention(text_cond, text_cond, text_cond)
        attended = text_cond + self.beta * attended
        
        # feature enhancement with improved architecture
        enhanced = self.enhance_proj(attended)
        
        # multiple residual connections for better gradient flow
        output = text_cond + self.alpha * enhanced + self.beta * attended
        
        # final projection for output refinement
        output = self.output_proj(output)
        
        # restore original shape if needed
        if len(original_shape) == 2:
            output = output.squeeze(1)
        
        return output
