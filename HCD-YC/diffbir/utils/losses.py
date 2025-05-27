import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """VGG network feature extractor, for feature-level cycle consistency"""
    
    def __init__(self, layer_ids=(2, 7, 16, 25, 34), use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_ids = layer_ids
        self.use_input_norm = use_input_norm
        
        # load pre-trained VGG19 model
        self.vgg19 = models.vgg19(pretrained=True).features
        
        # freeze parameters
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        # set mean and std
        if self.use_input_norm:
            # mean and std used in VGG pre-training
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
    
    def forward(self, x):
        """
        extract VGG features
        Args:
            x: input image, range [0, 1]
        """
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        features = []
        for i in range(max(self.layer_ids) + 1):
            x = self.vgg19[i](x)
            if i in self.layer_ids:
                features.append(x)
        
        return features


class HierarchicalCycleLoss(nn.Module):
    """hierarchical cycle consistency loss"""
    
    def __init__(self, pixel_weight=1.0, feature_weight=1.0, semantic_weight=1.0, region_aware=True, adaptive_weight=True):
        super(HierarchicalCycleLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        self.semantic_weight = semantic_weight
        self.region_aware = region_aware
        self.adaptive_weight = adaptive_weight
        
        # VGG feature extractor, for feature-level loss
        self.feature_extractor = VGGFeatureExtractor()
        
    def forward(self, clean, hazy, dehazed, hazy_density=None):
        """
        calculate hierarchical cycle consistency loss
        Args:
            clean: original clear image [B, C, H, W]
            hazy: hazy image [B, C, H, W]
            dehazed: dehazed image [B, C, H, W]
            hazy_density: hazy density [B, 1, 1, 1], for adaptive weight adjustment
        Returns:
            total_loss: total loss
            loss_dict: detailed loss dictionary
        """
        # pixel-level cycle consistency loss (L1)
        pixel_loss = F.l1_loss(dehazed, clean)
        
        # feature-level cycle consistency loss (VGG features)
        clean_features = self.feature_extractor(clean)
        dehazed_features = self.feature_extractor(dehazed)
        
        feature_loss = 0
        for cf, df in zip(clean_features, dehazed_features):
            feature_loss += F.mse_loss(df, cf)
        feature_loss /= len(clean_features)
        
        # semantic-level cycle consistency loss (using high-level features)
        # use the deepest layer of VGG as semantic representation
        semantic_loss = F.mse_loss(dehazed_features[-1], clean_features[-1])
        
        # region-aware cycle consistency loss (optional)
        region_loss = 0
        if self.region_aware:
            # identify texture regions based on gradient information
            clean_grad = gradient_magnitude(clean)
            high_texture_mask = (clean_grad > clean_grad.mean() * 1.5).float()
            
            # use weighted loss for high-texture regions
            region_pixel_loss = F.l1_loss(
                dehazed * high_texture_mask, 
                clean * high_texture_mask,
                reduction='sum'
            ) / (high_texture_mask.sum() + 1e-6)
            
            region_loss = region_pixel_loss
        
        # adaptive weight adjustment (optional)
        if self.adaptive_weight and hazy_density is not None:
            # ensure hazy_density has correct shape [B,1,1,1]
            if len(hazy_density.shape) == 4 and hazy_density.shape[1:] == (1, 1, 1):
                # calculate density factor for each sample
                density_factor = torch.sigmoid(10 * (hazy_density - 0.5))  # map density to [0,1] range
                
                # create scalar weights
                adaptive_pixel_weight = self.pixel_weight * (1 - density_factor.mean() * 0.5)
                adaptive_feature_weight = self.feature_weight * (1 + density_factor.mean() * 0.5)
                adaptive_semantic_weight = self.semantic_weight * (1 + density_factor.mean() * 0.5)
            else:
                print(f"Warning: hazy_density shape {hazy_density.shape} is not [B,1,1,1], using default weights")
                adaptive_pixel_weight = self.pixel_weight
                adaptive_feature_weight = self.feature_weight
                adaptive_semantic_weight = self.semantic_weight
        else:
            adaptive_pixel_weight = self.pixel_weight
            adaptive_feature_weight = self.feature_weight
            adaptive_semantic_weight = self.semantic_weight
        
        # calculate total loss
        total_loss = (
            adaptive_pixel_weight * pixel_loss +
            adaptive_feature_weight * feature_loss +
            adaptive_semantic_weight * semantic_loss
        )
        
        if self.region_aware:
            total_loss += 0.5 * region_loss
        
        # ensure total_loss is a scalar
        if not torch.is_tensor(total_loss) or total_loss.numel() > 1:
            total_loss = total_loss.mean()
        
        # return total loss and detailed loss dictionary
        loss_dict = {
            'pixel_loss': pixel_loss.item(),
            'feature_loss': feature_loss.item(),
            'semantic_loss': semantic_loss.item(),
        }
        
        if self.region_aware:
            loss_dict['region_loss'] = region_loss.item()
            
        return total_loss, loss_dict


def gradient_magnitude(x):
    """calculate gradient magnitude, for texture region identification"""
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).to(x.device)
    
    # calculate grayscale image
    if x.shape[1] == 3:  # RGB image
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    else:  # single channel image
        gray = x
    
    # calculate gradient in x and y directions
    pad = nn.ReflectionPad2d(1)
    gray = pad(gray)
    
    grad_x = F.conv2d(gray, sobel_x)
    grad_y = F.conv2d(gray, sobel_y)
    
    # calculate gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    return grad_magnitude


class AsymmetricCycleLoss(nn.Module):
    """asymmetric cycle learning loss function, for asymmetric fog-to-clear process"""
    
    def __init__(self, forward_weight=0.7, backward_weight=0.3):
        super(AsymmetricCycleLoss, self).__init__()
        self.forward_weight = forward_weight    # clear -> hazy -> clear
        self.backward_weight = backward_weight  # hazy -> clear -> hazy
        
    def forward(self, x_clean, x_hazy, cycle_clean, cycle_hazy=None):
        """
        calculate asymmetric cycle loss
        Args:
            x_clean: original clear image
            x_hazy: hazy image
            cycle_clean: hazy image after clear -> hazy -> clear
            cycle_hazy: clear image after hazy -> clear -> hazy
        """
        # forward cycle consistency (clear -> hazy -> clear)
        forward_loss = F.l1_loss(cycle_clean, x_clean)
        
        loss = self.forward_weight * forward_loss
        
        # if cycle_hazy is provided, calculate backward cycle consistency
        if cycle_hazy is not None:
            backward_loss = F.l1_loss(cycle_hazy, x_hazy)
            loss += self.backward_weight * backward_loss
            
            # ensure loss is a scalar
            if not torch.is_tensor(loss) or loss.numel() > 1:
                loss = loss.mean()
                
            return loss, {'forward_loss': forward_loss.item(), 'backward_loss': backward_loss.item()}
        
        # ensure loss is a scalar
        if not torch.is_tensor(loss) or loss.numel() > 1:
            loss = loss.mean()
            
        return loss, {'forward_loss': forward_loss.item()}


class TextImageAlignmentLoss(nn.Module):
    """
    text-image alignment loss, for optimizing the guidance of text prompts on the dehazing process
    """
    
    def __init__(self, device='cuda'):
        super(TextImageAlignmentLoss, self).__init__()
        # text guidance loss weight
        self.text_weight = 0.5
        
        # similarity loss based on VGG features
        self.feature_extractor = VGGFeatureExtractor(layer_ids=(16, 25, 34))
        self.device = device
        
    def forward(self, dehazed_imgs, clean_imgs, text_embeds):
        """
        calculate text-image alignment loss
        Args:
            dehazed_imgs: dehazed image [B, C, H, W], range [0, 1]
            clean_imgs: clean reference image [B, C, H, W], range [0, 1]
            text_embeds: text embedding [B, seq_len, dim]
        Returns:
            loss: total loss
            loss_dict: detailed loss dictionary
        """
        # image reconstruction loss (basic loss)
        recon_loss = F.l1_loss(dehazed_imgs, clean_imgs)
        
        # feature alignment loss
        dehazed_features = self.feature_extractor(dehazed_imgs)
        clean_features = self.feature_extractor(clean_imgs)
        
        # use the highest layer feature for semantic matching
        dehazed_semantic = dehazed_features[-1]
        clean_semantic = clean_features[-1]
        
        # text-image consistency
        # calculate the mean representation of text embedding
        text_mean = torch.mean(text_embeds, dim=1)  # [B, dim]
        
        # project to feature space
        B, C, H, W = dehazed_semantic.shape
        dehazed_semantic_flat = dehazed_semantic.view(B, C, -1).mean(dim=2)  # [B, C] -> [B, 512]
        clean_semantic_flat = clean_semantic.view(B, C, -1).mean(dim=2)  # [B, C] -> [B, 512]
        
        # simple solution: reduce the dimension of text embedding to 512, or only compare low-dimensional features
        if text_mean.shape[1] != dehazed_semantic_flat.shape[1]:
            # use PCA-style dimensionality reduction or simple truncation
            text_projected = text_mean[:, :dehazed_semantic_flat.shape[1]]  # simple truncation to 512 dimensions
        else:
            text_projected = text_mean
        
        # calculate the distance difference between semantic and text, the semantic of dehazed image should be close to the semantic of clean image
        dehazed_text_dist = F.cosine_similarity(dehazed_semantic_flat, text_projected)
        clean_text_dist = F.cosine_similarity(clean_semantic_flat, text_projected)
        
        text_align_loss = F.mse_loss(dehazed_text_dist, clean_text_dist)
        
        # total loss
        total_loss = recon_loss + self.text_weight * text_align_loss
        
        if not torch.is_tensor(total_loss) or total_loss.numel() > 1:
            total_loss = total_loss.mean()
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'text_align_loss': text_align_loss.item()
        }
        
        return total_loss, loss_dict
