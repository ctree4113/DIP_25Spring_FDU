import torch
import torch as th
import torch.nn as nn

from .util import conv_nd, linear, zero_module, timestep_embedding, exists, rgb_to_ycbcr
from .attention import SpatialTransformer, FeatureAttention
from .unet import (
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
    UNetModel,
)


class ControlledUnetModel(UNetModel):

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        **kwargs,
    ):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_head_channels=64,
        num_heads_upsample=8,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        dims, in_channels + hint_channels, model_channels, 3, padding=1
                    )
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            # ensure num_res_blocks is a list
            if isinstance(num_res_blocks, int):
                res_blocks = num_res_blocks
            else:
                res_blocks = num_res_blocks[level]
                
            for _ in range(res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or _ < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = zero_module(nn.Conv2d(ch, ch, 1))
        self._feature_size += ch

    def make_zero_conv(self, channels):
        """create a zero-initialized convolution layer"""
        return zero_module(nn.Conv2d(channels, channels, 1, padding=0))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []

        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        for i in range(len(self.input_blocks)):
            h = self.input_blocks[i](h, emb, context)
            outs.append(self.zero_convs[i](h))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h))

        return outs


class YCbCrControlNet(ControlNet):
    """
    extend ControlNet to support YCbCr color space processing
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_head_channels=64,
        num_heads_upsample=8,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        ycbcr_fusion=True,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            hint_channels=hint_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
            disable_self_attentions=disable_self_attentions,
            num_attention_blocks=num_attention_blocks,
            disable_middle_self_attn=disable_middle_self_attn,
            use_linear_in_transformer=use_linear_in_transformer,
        )
        
        self.ycbcr_fusion = ycbcr_fusion
        
        # time_embed_dim, usually this value is 4 * model_channels
        self.time_embed_dim = 4 * model_channels
        
        if ycbcr_fusion:
            # YCbCr processing branch
            self.ycbcr_input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                    )
                ]
            )
            
            # copy original input block configuration
            input_block_chans = [model_channels]
            ch = model_channels
            ds = 1
            
            for level, mult in enumerate(channel_mult):
                # ensure num_res_blocks is a list
                if isinstance(num_res_blocks, int):
                    res_blocks = num_res_blocks
                else:
                    res_blocks = num_res_blocks[level]
                
                for _ in range(res_blocks):
                    layers = [
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = mult * model_channels
                    if ds in attention_resolutions:
                        if num_heads_upsample == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = num_heads_upsample
                            dim_head = ch // num_heads
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disable_self_attentions[level] if disable_self_attentions is not None else False,
                                use_linear=use_linear_in_transformer
                            )
                        )
                    self.ycbcr_input_blocks.append(TimestepEmbedSequential(*layers))
                    input_block_chans.append(ch)
                if level != len(channel_mult) - 1:
                    self.ycbcr_input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                self.time_embed_dim,
                                dropout,
                                out_channels=ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                        )
                        if resblock_updown
                        else TimestepEmbedSequential(
                            Downsample(
                                ch, conv_resample, dims=dims, out_channels=ch
                            )
                        )
                    )
                    ch = ch
                    input_block_chans.append(ch)
                    ds *= 2
            
            # YCbCr middle block
            self.ycbcr_middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    self.time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                ) if not use_spatial_transformer else SpatialTransformer(
                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer
                ),
                ResBlock(
                    ch,
                    self.time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
            
            # YCbCr middle block output
            self.ycbcr_middle_block_out = zero_module(nn.Conv2d(ch, ch, 1))
            
            # feature fusion module
            self.feature_fusions = nn.ModuleList()
            
            # feature fusion level indices
            # set fusion at the last block of each resolution, and the first block after each downsampling
            self.fusion_indices = []
            current_idx = 1
            for level, mult in enumerate(channel_mult):
                # ensure num_res_blocks is a list
                if isinstance(num_res_blocks, int):
                    res_blocks = num_res_blocks
                else:
                    res_blocks = num_res_blocks[level]
                
                fusion_level_idx = current_idx + res_blocks - 1  # last block of each resolution
                self.fusion_indices.append(fusion_level_idx)
                current_idx = fusion_level_idx + 1
                
                if level < len(channel_mult) - 1:
                    # first block after each downsampling
                    self.fusion_indices.append(current_idx)
                    current_idx += 1
            
            # create feature fusion module
            for idx in self.fusion_indices:
                ch = input_block_chans[idx]
                self.feature_fusions.append(
                    FeatureAttention(
                        dim=ch,
                        num_heads=max(1, ch // 64),  # determine number of attention heads based on channel count
                        head_dim=64
                    )
                )
            
            # middle block feature fusion
            self.middle_fusion = FeatureAttention(
                dim=ch,
                num_heads=max(1, ch // 64),  # determine number of attention heads based on channel count
                head_dim=64
            )

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # process RGB features
        rgb_hint = hint
        rgb_h = torch.cat([x, rgb_hint], dim=1)
        rgb_hs = []
        rgb_h = self.input_blocks[0](rgb_h, emb, context)
        rgb_hs.append(rgb_h)
        
        if self.ycbcr_fusion:
            # convert to YCbCr color space and process
            # assume hint's value range is [-1, 1], need to convert to [0, 1]
            hint_norm = (hint + 1) / 2
            
            # ensure input channel count is 3 (RGB)
            if hint_norm.shape[1] == 3:
                ycbcr_hint = rgb_to_ycbcr(hint_norm) * 2 - 1  # convert back to [-1, 1] range
            else:
                ycbcr_hint = hint
            
            # YCbCr features processing
            ycbcr_h = torch.cat([x, ycbcr_hint], dim=1)
            ycbcr_hs = []
            ycbcr_h = self.ycbcr_input_blocks[0](ycbcr_h, emb, context)
            ycbcr_hs.append(ycbcr_h)
        
        # feature fusion
        fusions = []
        
        # remaining forward processing
        for i in range(1, len(self.input_blocks)):
            rgb_h = self.input_blocks[i](rgb_h, emb, context)
            rgb_hs.append(rgb_h)
            
            if self.ycbcr_fusion:
                ycbcr_h = self.ycbcr_input_blocks[i](ycbcr_h, emb, context)
                ycbcr_hs.append(ycbcr_h)
                
                # only perform feature fusion at the corresponding level
                if i in self.fusion_indices:
                    fusion_idx = self.fusion_indices.index(i)
                    
                    # ensure feature dimensions match
                    if rgb_h.shape[1] != ycbcr_h.shape[1]:
                        # use smaller dimension
                        min_channels = min(rgb_h.shape[1], ycbcr_h.shape[1])
                        rgb_h_resized = rgb_h[:, :min_channels]
                        ycbcr_h_resized = ycbcr_h[:, :min_channels]
                        
                        # use adjusted features for fusion
                        feature_fusion = self.feature_fusions[fusion_idx]
                        # ensure FeatureAttention's dim parameter matches input feature dimension
                        if feature_fusion.dim != min_channels:
                            temp_fusion = FeatureAttention(min_channels, 
                                                         feature_fusion.num_heads, 
                                                         feature_fusion.head_dim).to(rgb_h.device)
                            fused_h = temp_fusion(rgb_h_resized, ycbcr_h_resized)
                        else:
                            fused_h = feature_fusion(rgb_h_resized, ycbcr_h_resized)
                    else:
                        fused_h = self.feature_fusions[fusion_idx](rgb_h, ycbcr_h)
                    
                    fusions.append(fused_h)
                    
                    # use fused features to update RGB feature stream
                    rgb_h = fused_h
        
        # middle block processing
        rgb_h = self.middle_block(rgb_h, emb, context)
        
        if self.ycbcr_fusion:
            ycbcr_h = self.ycbcr_middle_block(ycbcr_h, emb, context)
            
            # middle block feature fusion
            if rgb_h.shape[1] != ycbcr_h.shape[1]:
                min_channels = min(rgb_h.shape[1], ycbcr_h.shape[1])
                rgb_h_resized = rgb_h[:, :min_channels]
                ycbcr_h_resized = ycbcr_h[:, :min_channels]
                
                # create temporary fusion layer with same dimension
                if self.middle_fusion.dim != min_channels:
                    temp_fusion = FeatureAttention(min_channels, 
                                                 self.middle_fusion.num_heads, 
                                                 self.middle_fusion.head_dim).to(rgb_h.device)
                    middle_fused = temp_fusion(rgb_h_resized, ycbcr_h_resized)
                else:
                    middle_fused = self.middle_fusion(rgb_h_resized, ycbcr_h_resized)
            else:
                middle_fused = self.middle_fusion(rgb_h, ycbcr_h)
                
            rgb_h = middle_fused
        
        # standard ControlNet output format: zero_conv results for each block
        outs = []
        
        # process each input block
        for i in range(len(self.input_blocks)):
            # get corresponding features and apply zero_conv
            h_i = rgb_hs[i]
            out_i = self.zero_convs[i](h_i)
            outs.append(out_i)
        
        # process middle block
        middle_out = self.middle_block_out(rgb_h)
        outs.append(middle_out)
        
        # return control signal list
        return outs
