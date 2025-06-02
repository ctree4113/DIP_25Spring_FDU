import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """Lightweight VGG feature extractor for perceptual losses"""
    
    def __init__(self, layer_ids=(8, 17, 26), use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_ids = layer_ids
        self.use_input_norm = use_input_norm
        
        # load pre-trained VGG19 model
        vgg19 = models.vgg19(pretrained=True).features
        
        # only keep the layers we need
        self.feature_layers = nn.ModuleList()
        current_idx = 0
        for i, layer in enumerate(vgg19):
            self.feature_layers.append(layer)
            if i in layer_ids:
                current_idx += 1
            if i >= max(layer_ids):
                break
        
        # freeze parameters
        for param in self.feature_layers.parameters():
            param.requires_grad = False
        
        # normalization constants
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
    
    def forward(self, x):
        """Extract VGG features efficiently"""
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        features = []
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.layer_ids:
                features.append(x)
        
        return features


def compute_dark_channel(image, patch_size=15):
    """
    Compute dark channel prior for enhanced fog density estimation
    Args:
        image: [B, C, H, W] input image, range [0, 1]
        patch_size: local window size for dark channel computation
    Returns:
        dark_channel: [B, 1, H, W] dark channel map
    """
    if patch_size % 2 == 0:
        patch_size += 1
    pad = patch_size // 2
    
    # compute minimum across RGB channels
    min_channel = torch.min(image, dim=1, keepdim=True)[0]
    
    # apply minimum filter (dark channel prior)
    dark_channel = -F.max_pool2d(-min_channel, patch_size, stride=1, padding=pad)
    
    return dark_channel


def estimate_fog_density(image, use_dcp=True):
    """
    Enhanced fog density estimation combining multiple metrics with DCP
    Args:
        image: [B, C, H, W] input image, range [0, 1]
        use_dcp: whether to include dark channel prior in estimation
    Returns:
        density: [B, 1, 1, 1] normalized fog density [0, 1]
    """
    # convert to grayscale
    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image[:, 0:1]
    
    # brightness metric (high brightness = more fog)
    brightness = torch.mean(gray, dim=[2, 3], keepdim=True)
    
    # contrast metric (low contrast = more fog)
    contrast = torch.std(gray, dim=[2, 3], keepdim=True)
    contrast_norm = 1.0 - torch.clamp(contrast / 0.2, 0, 1)
    
    # edge density metric (less edges = more fog)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                          dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                          dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(gray_padded, sobel_x)
    grad_y = F.conv2d(gray_padded, sobel_y)
    edge_density = torch.mean(torch.sqrt(grad_x**2 + grad_y**2 + 1e-6), dim=[2, 3], keepdim=True)
    edge_norm = 1.0 - torch.clamp(edge_density / 0.3, 0, 1)
    
    if use_dcp and image.shape[1] == 3:
        # dark channel prior metric (high dark channel = less fog)
        dark_channel = compute_dark_channel(image)
        dcp_density = 1.0 - torch.mean(dark_channel, dim=[2, 3], keepdim=True)
        
        # combine all metrics with optimized weights
        fog_density = 0.3 * brightness + 0.25 * contrast_norm + 0.2 * edge_norm + 0.25 * dcp_density
    else:
        # combine metrics without DCP
        fog_density = 0.4 * brightness + 0.3 * contrast_norm + 0.3 * edge_norm
    
    fog_density = torch.clamp(fog_density, 0, 1)
    
    return fog_density


def compute_gradient_magnitude(image):
    """Compute gradient magnitude for texture analysis"""
    if len(image.shape) == 4 and image.shape[1] == 3:
        # convert to grayscale
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image[:, 0:1] if len(image.shape) == 4 else image.unsqueeze(1)
    
    # sobel operators
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                          dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                          dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    # compute gradients
    padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_y)
    
    # compute magnitude
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
    return magnitude


class HierarchicalCycleLoss(nn.Module):
    """Enhanced hierarchical cycle consistency loss with DCP-aware adaptive weighting"""
    
    def __init__(self, pixel_weight=1.0, feature_weight=0.5, semantic_weight=0.3, 
                 region_weight=0.2, adaptive_weight=True, use_dcp=True):
        super(HierarchicalCycleLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        self.semantic_weight = semantic_weight
        self.region_weight = region_weight
        self.adaptive_weight = adaptive_weight
        self.use_dcp = use_dcp
        
        # lightweight feature extractor
        self.feature_extractor = VGGFeatureExtractor()
        
        # enhanced adaptive weights with more sophisticated architecture
        if adaptive_weight:
            self.density_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3),
                nn.Softmax(dim=-1)
            )
            
            # initialize with balanced weights
            with torch.no_grad():
                self.density_mlp[-2].weight.fill_(0.0)
                self.density_mlp[-2].bias.copy_(torch.tensor([0.4, 0.3, 0.3]))
        
    def forward(self, clean, hazy, dehazed, fog_density=None):
        """
        Compute hierarchical cycle consistency loss with enhanced DCP integration
        Args:
            clean: [B, C, H, W] original clear image
            hazy: [B, C, H, W] hazy image  
            dehazed: [B, C, H, W] dehazed result
            fog_density: [B, 1, 1, 1] fog density (optional)
        """
        # estimate fog density with DCP if not provided
        if fog_density is None:
            fog_density = estimate_fog_density(hazy, use_dcp=self.use_dcp)
        
        # pixel-level loss
        pixel_loss = F.l1_loss(dehazed, clean, reduction='mean')
        
        # feature-level losses with improved normalization
        clean_features = self.feature_extractor(clean)
        dehazed_features = self.feature_extractor(dehazed)
        
        feature_losses = []
        for cf, df in zip(clean_features, dehazed_features):
            # L2 normalization for stable comparison
            cf_norm = F.normalize(cf, p=2, dim=1)
            df_norm = F.normalize(df, p=2, dim=1)
            feature_losses.append(F.mse_loss(df_norm, cf_norm))
        
        # depth-aware weighted feature loss
        depth_weights = [0.2, 0.3, 0.5]  # give more weight to deeper features
        feature_loss = sum(w * fl for w, fl in zip(depth_weights, feature_losses))
        
        # semantic loss using deepest features
        semantic_loss = feature_losses[-1]
        
        # enhanced region-aware loss for texture preservation
        region_loss = self.compute_region_loss(clean, dehazed)
        
        # enhanced adaptive weighting based on fog density
        if self.adaptive_weight:
            density_scalar = torch.mean(fog_density).unsqueeze(0)
            adaptive_weights = self.density_mlp(density_scalar)
            
            # apply density-dependent weight modulation
            pixel_w = self.pixel_weight * adaptive_weights[0]
            feature_w = self.feature_weight * adaptive_weights[1] 
            semantic_w = self.semantic_weight * adaptive_weights[2]
        else:
            pixel_w = self.pixel_weight
            feature_w = self.feature_weight
            semantic_w = self.semantic_weight
        
        # combine losses with enhanced weighting
        total_loss = (pixel_w * pixel_loss + 
                     feature_w * feature_loss + 
                     semantic_w * semantic_loss +
                     self.region_weight * region_loss)
        
        loss_dict = {
            'pixel_loss': pixel_loss.item(),
            'feature_loss': feature_loss.item(), 
            'semantic_loss': semantic_loss.item(),
            'region_loss': region_loss.item(),
            'fog_density': torch.mean(fog_density).item()
        }
        
        return total_loss, loss_dict
    
    def compute_region_loss(self, clean, dehazed):
        """Enhanced region-aware loss focusing on high-texture areas"""
        # compute gradient magnitude for texture detection
        clean_grad = compute_gradient_magnitude(clean)
        
        # adaptive threshold based on image statistics
        grad_mean = torch.mean(clean_grad)
        grad_std = torch.std(clean_grad)
        threshold = grad_mean + 0.5 * grad_std
        
        # create texture mask
        texture_mask = (clean_grad > threshold).float()
        
        # apply mask to loss computation with normalization
        if texture_mask.sum() > 0:
            masked_diff = torch.abs(dehazed - clean) * texture_mask
            region_loss = masked_diff.sum() / (texture_mask.sum() + 1e-6)
        else:
            region_loss = torch.tensor(0.0, device=clean.device)
        
        return region_loss


class AsymmetricCycleLoss(nn.Module):
    """Enhanced asymmetric cycle loss with quality-aware weighting"""
    
    def __init__(self, forward_weight=0.8, backward_weight=0.2):
        super(AsymmetricCycleLoss, self).__init__()
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight
        
    def forward(self, x_clean, x_hazy, cycle_clean, cycle_hazy=None):
        """Compute asymmetric cycle consistency loss with quality awareness"""
        # forward cycle: clear -> hazy -> clear (more important)
        forward_loss = F.l1_loss(cycle_clean, x_clean)
        
        total_loss = self.forward_weight * forward_loss
        loss_dict = {'forward_loss': forward_loss.item()}
        
        # backward cycle: hazy -> clear -> hazy (less important)
        if cycle_hazy is not None:
            backward_loss = F.l1_loss(cycle_hazy, x_hazy)
            total_loss += self.backward_weight * backward_loss
            loss_dict['backward_loss'] = backward_loss.item()
            
        return total_loss, loss_dict


class TextImageAlignmentLoss(nn.Module):
    """
    Enhanced text-image alignment loss with improved semantic understanding
    """
    
    def __init__(self, device='cuda'):
        super(TextImageAlignmentLoss, self).__init__()
        self.device = device
        
        # enhanced feature extractor for semantic alignment
        self.feature_extractor = VGGFeatureExtractor(layer_ids=(17, 26))
        
        # improved projection layers with residual connections
        self.img_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # learnable temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, dehazed_imgs, clean_imgs, text_embeds):
        """
        Compute enhanced text-image alignment loss
        Args:
            dehazed_imgs: [B, C, H, W] dehazed results
            clean_imgs: [B, C, H, W] clean targets
            text_embeds: [B, seq_len, dim] or [B, dim] text embeddings
        """
        # basic reconstruction loss
        recon_loss = F.l1_loss(dehazed_imgs, clean_imgs)
        
        # extract high-level semantic features
        dehazed_features = self.feature_extractor(dehazed_imgs)
        clean_features = self.feature_extractor(clean_imgs)
        
        # use deepest feature for semantic comparison
        dehazed_semantic = dehazed_features[-1]
        clean_semantic = clean_features[-1]
        
        # project image features to embedding space
        dehazed_embed = self.img_proj(dehazed_semantic)
        clean_embed = self.img_proj(clean_semantic)
        
        # handle text embedding dimensions
        if len(text_embeds.shape) == 3:
            text_embeds = torch.mean(text_embeds, dim=1)
        
        # project text to same embedding space
        text_embed = self.text_proj(text_embeds)
        
        # contrastive alignment with learnable temperature
        dehazed_text_sim = F.cosine_similarity(dehazed_embed, text_embed) / self.temperature
        clean_text_sim = F.cosine_similarity(clean_embed, text_embed) / self.temperature
        
        # alignment loss: dehazed should be as aligned as clean
        align_loss = F.mse_loss(dehazed_text_sim, clean_text_sim)
        
        # semantic consistency loss with improved metric
        semantic_loss = F.mse_loss(F.normalize(dehazed_embed, p=2, dim=-1), 
                                 F.normalize(clean_embed, p=2, dim=-1))
        
        # combine losses with balanced weights
        total_loss = recon_loss + 0.15 * align_loss + 0.1 * semantic_loss
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'align_loss': align_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'temperature': self.temperature.item()
        }
        
        return total_loss, loss_dict
