import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """VGG网络特征提取器，用于特征级循环一致性"""
    
    def __init__(self, layer_ids=(2, 7, 16, 25, 34), use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_ids = layer_ids
        self.use_input_norm = use_input_norm
        
        # 加载预训练的VGG19模型
        self.vgg19 = models.vgg19(pretrained=True).features
        
        # 冻结参数
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        # 设置均值和标准差
        if self.use_input_norm:
            # VGG预训练时使用的均值和标准差
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
    
    def forward(self, x):
        """
        提取VGG特征
        Args:
            x: 输入图像，范围为[0, 1]
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
    """层级循环一致性损失"""
    
    def __init__(self, pixel_weight=1.0, feature_weight=1.0, semantic_weight=1.0, region_aware=True, adaptive_weight=True):
        super(HierarchicalCycleLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        self.semantic_weight = semantic_weight
        self.region_aware = region_aware
        self.adaptive_weight = adaptive_weight
        
        # VGG特征提取器，用于特征级损失
        self.feature_extractor = VGGFeatureExtractor()
        
    def forward(self, clean, hazy, dehazed, hazy_density=None):
        """
        计算层级循环一致性损失
        Args:
            clean: 原始清晰图像 [B, C, H, W]
            hazy: 雾化后的图像 [B, C, H, W]
            dehazed: 去雾后的图像 [B, C, H, W]
            hazy_density: 雾气密度 [B, 1, 1, 1]，用于自适应权重调整
        Returns:
            total_loss: 总损失
            loss_dict: 详细损失字典
        """
        # 1. 像素级循环一致性损失 (L1)
        pixel_loss = F.l1_loss(dehazed, clean)
        
        # 2. 特征级循环一致性损失 (VGG特征)
        clean_features = self.feature_extractor(clean)
        dehazed_features = self.feature_extractor(dehazed)
        
        feature_loss = 0
        for cf, df in zip(clean_features, dehazed_features):
            feature_loss += F.mse_loss(df, cf)
        feature_loss /= len(clean_features)
        
        # 3. 语义级循环一致性损失 (使用高级特征)
        # 使用VGG最深层特征作为语义表示
        semantic_loss = F.mse_loss(dehazed_features[-1], clean_features[-1])
        
        # 4. 区域感知的循环一致性 (可选)
        region_loss = 0
        if self.region_aware:
            # 根据梯度信息识别纹理区域
            clean_grad = gradient_magnitude(clean)
            high_texture_mask = (clean_grad > clean_grad.mean() * 1.5).float()
            
            # 对高纹理区域使用加权损失
            region_pixel_loss = F.l1_loss(
                dehazed * high_texture_mask, 
                clean * high_texture_mask,
                reduction='sum'
            ) / (high_texture_mask.sum() + 1e-6)
            
            region_loss = region_pixel_loss
        
        # 5. 自适应权重调整 (可选)
        if self.adaptive_weight and hazy_density is not None:
            # 根据雾气密度动态调整权重
            # 雾气越重，像素级损失权重越低，特征级和语义级损失权重越高
            density_factor = torch.sigmoid(10 * (hazy_density - 0.5))  # 将密度映射到[0,1]范围
            
            adaptive_pixel_weight = self.pixel_weight * (1 - density_factor * 0.5)
            adaptive_feature_weight = self.feature_weight * (1 + density_factor * 0.5)
            adaptive_semantic_weight = self.semantic_weight * (1 + density_factor * 0.5)
        else:
            adaptive_pixel_weight = self.pixel_weight
            adaptive_feature_weight = self.feature_weight
            adaptive_semantic_weight = self.semantic_weight
        
        # 计算总损失
        total_loss = (
            adaptive_pixel_weight * pixel_loss +
            adaptive_feature_weight * feature_loss +
            adaptive_semantic_weight * semantic_loss
        )
        
        if self.region_aware:
            total_loss += 0.5 * region_loss
        
        # 返回总损失和详细损失字典
        loss_dict = {
            'pixel_loss': pixel_loss.item(),
            'feature_loss': feature_loss.item(),
            'semantic_loss': semantic_loss.item(),
        }
        
        if self.region_aware:
            loss_dict['region_loss'] = region_loss.item()
            
        return total_loss, loss_dict


def gradient_magnitude(x):
    """计算图像梯度幅值，用于识别纹理区域"""
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).to(x.device)
    
    # 计算灰度图
    if x.shape[1] == 3:  # RGB图像
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    else:  # 单通道图像
        gray = x
    
    # 计算x和y方向的梯度
    pad = nn.ReflectionPad2d(1)
    gray = pad(gray)
    
    grad_x = F.conv2d(gray, sobel_x)
    grad_y = F.conv2d(gray, sobel_y)
    
    # 计算梯度幅值
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    return grad_magnitude


class AsymmetricCycleLoss(nn.Module):
    """非对称循环学习损失函数，适应雾化-去雾过程的不对称性"""
    
    def __init__(self, forward_weight=0.7, backward_weight=0.3):
        super(AsymmetricCycleLoss, self).__init__()
        self.forward_weight = forward_weight  # 清晰->雾气->清晰
        self.backward_weight = backward_weight  # 雾气->清晰->雾气
        
    def forward(self, x_clean, x_hazy, cycle_clean, cycle_hazy=None):
        """
        计算非对称循环损失
        Args:
            x_clean: 原始清晰图像
            x_hazy: 雾化后的图像
            cycle_clean: 雾化后再去雾的图像
            cycle_hazy: 去雾后再雾化的图像 (如果有)
        """
        # 前向循环一致性 (清晰->雾气->清晰)
        forward_loss = F.l1_loss(cycle_clean, x_clean)
        
        loss = self.forward_weight * forward_loss
        
        # 如果提供了cycle_hazy，则计算反向循环一致性
        if cycle_hazy is not None:
            backward_loss = F.l1_loss(cycle_hazy, x_hazy)
            loss += self.backward_weight * backward_loss
            return loss, {'forward_loss': forward_loss.item(), 'backward_loss': backward_loss.item()}
        
        return loss, {'forward_loss': forward_loss.item()}


class TextImageAlignmentLoss(nn.Module):
    """
    文本-图像对齐损失，用于优化文本提示对去雾过程的引导能力
    """
    
    def __init__(self, device='cuda'):
        super(TextImageAlignmentLoss, self).__init__()
        # 文本引导损失权重
        self.text_weight = 0.5
        
        # 基于VGG特征的相似度损失
        self.feature_extractor = VGGFeatureExtractor(layer_ids=(16, 25, 34))
        self.device = device
        
    def forward(self, dehazed_imgs, clean_imgs, text_embeds):
        """
        计算文本-图像对齐损失
        Args:
            dehazed_imgs: 去雾后的图像 [B, C, H, W]，范围[0, 1]
            clean_imgs: 干净的参考图像 [B, C, H, W]，范围[0, 1]
            text_embeds: 文本嵌入 [B, seq_len, dim]
        Returns:
            loss: 总损失
            loss_dict: 损失详情
        """
        # 1. 图像重建损失 (基础loss)
        recon_loss = F.l1_loss(dehazed_imgs, clean_imgs)
        
        # 2. 特征对齐损失
        dehazed_features = self.feature_extractor(dehazed_imgs)
        clean_features = self.feature_extractor(clean_imgs)
        
        # 选择最高层特征用于语义匹配
        dehazed_semantic = dehazed_features[-1]
        clean_semantic = clean_features[-1]
        
        # 3. 文本-图像一致性
        # 计算文本嵌入的均值表示
        text_mean = torch.mean(text_embeds, dim=1)  # [B, dim]
        
        # 投影到特征空间
        B, C, H, W = dehazed_semantic.shape
        dehazed_semantic_flat = dehazed_semantic.view(B, C, -1).mean(dim=2)  # [B, C]
        clean_semantic_flat = clean_semantic.view(B, C, -1).mean(dim=2)  # [B, C]
        
        # 计算语义与文本的距离差异
        # 我们希望去雾图像的语义与文本的距离接近清晰图像的语义与文本的距离
        dehazed_text_dist = F.cosine_similarity(dehazed_semantic_flat, text_mean)
        clean_text_dist = F.cosine_similarity(clean_semantic_flat, text_mean)
        
        text_align_loss = F.mse_loss(dehazed_text_dist, clean_text_dist)
        
        # 总损失
        total_loss = recon_loss + self.text_weight * text_align_loss
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'text_align_loss': text_align_loss.item()
        }
        
        return total_loss, loss_dict 