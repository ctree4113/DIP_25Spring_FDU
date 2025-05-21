from packaging import version
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from .util import checkpoint, zero_module, exists, default
from .config import Config, AttnMode


# CrossAttn precision handling
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # 确保heads和dim_head是有效值
        self.heads = max(1, heads)
        self.dim_head = max(32, dim_head)
        
        inner_dim = self.dim_head * self.heads
        context_dim = default(context_dim, query_dim)

        self.scale = self.dim_head**-0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        b, n, d = x.shape
        if context is None:
            context = x
            
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(
            lambda t: t.reshape(b, t.shape[1], h, self.dim_head).permute(0, 2, 1, 3),
            (q, k, v),
        )

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", sim, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # 确保heads和dim_head是有效值
        self.heads = max(1, heads)
        self.dim_head = max(32, dim_head)
        
        inner_dim = self.dim_head * self.heads
        context_dim = default(context_dim, query_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        heads = self.heads
        dim_head = self.dim_head
        
        # 更安全的reshape逻辑
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, t.shape[1], dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, out.shape[1], dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], heads * dim_head)
        )
        return self.to_out(out)


class SDPCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # 确保heads和dim_head是有效值
        self.heads = max(1, heads)
        self.dim_head = max(32, dim_head)
        
        inner_dim = self.dim_head * self.heads
        context_dim = default(context_dim, query_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        heads = self.heads
        dim_head = self.dim_head
        
        # 更安全的reshape逻辑
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, t.shape[1], dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = F.scaled_dot_product_attention(q, k, v)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, out.shape[1], dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], heads * dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        AttnMode.VANILLA: CrossAttention,  # vanilla attention
        AttnMode.XFORMERS: MemoryEfficientCrossAttention,
        AttnMode.SDP: SDPCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[Config.attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class FeatureAttention(nn.Module):
    """多特征融合的注意力模块，用于融合RGB和YCbCr特征"""
    
    def __init__(self, dim, num_heads=8, head_dim=32):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, num_heads)  # 确保头数至少为1
        self.head_dim = max(32, head_dim)   # 确保每个头的维度至少为32
        self.scale = self.head_dim ** -0.5
        
        # RGB和YCbCr特征的注意力投影
        self.rgb_proj = nn.Linear(dim, self.num_heads * self.head_dim * 3)
        self.ycbcr_proj = nn.Linear(dim, self.num_heads * self.head_dim * 3)
        
        # 输出投影
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, dim)
        
    def forward(self, rgb_feat, ycbcr_feat):
        """
        融合RGB和YCbCr特征
        Args:
            rgb_feat: [B, C, H, W] RGB特征
            ycbcr_feat: [B, C, H, W] YCbCr特征
        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 检查输入形状是否匹配
        if rgb_feat.shape != ycbcr_feat.shape:
            # 确保空间维度匹配
            if rgb_feat.shape[2:] != ycbcr_feat.shape[2:]:
                # 使用较小的空间尺寸
                min_h = min(rgb_feat.shape[2], ycbcr_feat.shape[2])
                min_w = min(rgb_feat.shape[3], ycbcr_feat.shape[3])
                rgb_feat = rgb_feat[:, :, :min_h, :min_w]
                ycbcr_feat = ycbcr_feat[:, :, :min_h, :min_w]
        
        # 检查通道维度是否匹配预期的self.dim
        if rgb_feat.shape[1] != self.dim or ycbcr_feat.shape[1] != self.dim:
            # 调整通道维度，对两个特征使用相同的通道数
            min_channels = min(min(rgb_feat.shape[1], ycbcr_feat.shape[1]), self.dim)
            rgb_feat = rgb_feat[:, :min_channels]
            ycbcr_feat = ycbcr_feat[:, :min_channels]
            
            # 如果需要，使用临时层处理调整后的维度
            if min_channels != self.dim:
                temp_rgb_proj = nn.Linear(min_channels, self.num_heads * self.head_dim * 3).to(rgb_feat.device)
                temp_ycbcr_proj = nn.Linear(min_channels, self.num_heads * self.head_dim * 3).to(ycbcr_feat.device)
                temp_out_proj = nn.Linear(self.num_heads * self.head_dim, min_channels).to(rgb_feat.device)
            else:
                temp_rgb_proj = self.rgb_proj
                temp_ycbcr_proj = self.ycbcr_proj
                temp_out_proj = self.out_proj
        else:
            temp_rgb_proj = self.rgb_proj
            temp_ycbcr_proj = self.ycbcr_proj
            temp_out_proj = self.out_proj
        
        B, C, H, W = rgb_feat.shape
        
        # 重塑特征为序列形式
        rgb_feat = rgb_feat.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        ycbcr_feat = ycbcr_feat.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # 计算QKV
        rgb_qkv = temp_rgb_proj(rgb_feat).reshape(B, -1, 3, self.num_heads, self.head_dim)
        rgb_q, rgb_k, rgb_v = rgb_qkv.unbind(dim=2)
        
        ycbcr_qkv = temp_ycbcr_proj(ycbcr_feat).reshape(B, -1, 3, self.num_heads, self.head_dim)
        ycbcr_q, ycbcr_k, ycbcr_v = ycbcr_qkv.unbind(dim=2)
        
        # 交叉注意力: RGB_q 与 YCbCr_k
        attn1 = torch.einsum('bnhd,bmhd->bnmh', rgb_q, ycbcr_k) * self.scale
        attn1 = attn1.softmax(dim=2)
        out1 = torch.einsum('bnmh,bmhd->bnhd', attn1, ycbcr_v)
        
        # 交叉注意力: YCbCr_q 与 RGB_k
        attn2 = torch.einsum('bnhd,bmhd->bnmh', ycbcr_q, rgb_k) * self.scale
        attn2 = attn2.softmax(dim=2)
        out2 = torch.einsum('bnmh,bmhd->bnhd', attn2, rgb_v)
        
        # 融合双向交叉注意力结果
        out = out1 + out2
        out = out.reshape(B, -1, self.num_heads * self.head_dim)
        out = temp_out_proj(out)
        
        # 重塑回原始形状
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        return out
