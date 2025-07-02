import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# [核心修改] 從的 swinir_multi.py 文件中導入所有需要的基礎模块
from .network_swinir_multi_enhanced import (
    # Mlp,  # RSTB 内部使用，无需直接导入
    # window_partition, # SwinTransformerBlock 内部使用
    # window_reverse, # SwinTransformerBlock 内部使用
    # WindowAttention, # SwinTransformerBlock 内部使用
    # SwinTransformerBlock, # BasicLayer 内部使用
    # BasicLayer, # RSTB 内部使用
    RSTB,
    PatchEmbed,
    PatchUnEmbed,
)


# === CBAM实现 ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        out = x * self.channel_attention(x)
        att_map = self.spatial_attention(out)
        out = out * att_map
        return out, att_map

# === 新GSL分支 ===
class EnhancedGSLBranch(nn.Module):
    """
    GSL分支：编码+CBAM注意力+MLP输出坐标
    """
    def __init__(self, embed_dim, mid_chans=None):
        super().__init__()
        mid_chans = embed_dim if mid_chans is None else mid_chans

        self.encoder = nn.Sequential(
            nn.Conv2d(embed_dim, mid_chans, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chans, mid_chans, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.attention = CBAM(mid_chans)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mid_chans, mid_chans // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mid_chans // 2, 2)
        )

    def forward(self, x):
        feat = self.encoder(x)
        att_feat, gsl_attention_map = self.attention(feat)
        pooled = self.global_pool(att_feat)
        coords = self.mlp(pooled)
        return coords, gsl_attention_map

class AttentionFusionModule(nn.Module):
    """
    注意力融合模块，负责指导GDM分支（保持不变）。
    """
    def __init__(self, dim):
        super(AttentionFusionModule, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, gdm_features, gsl_attention_map):
        fused_features = gdm_features * gsl_attention_map
        refined_features = self.fusion_conv(fused_features)
        return refined_features + gdm_features

class SwinIRFuse(nn.Module): # [核心修改] 模型重命名
    """
    纯Swin Transformer骨干网络，带有任务融合机制的模型。
    """
    """ SwinIR
        基于 Swin Transformer 的图像恢复网络.
    输入:
        img_size (int | tuple(int)): 输入图像的大小，默认为 64*64.
        patch_size (int | tuple(int)): patch 的大小，默认为 1.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): Patch embedding 的维度，默认为 96.
        depths (tuple(int)): Swin Transformer 层的深度.
        num_heads (tuple(int)): 在不同层注意力头的个数.
        window_size (int): 窗口大小，默认为 7.
        mlp_ratio (float): MLP隐藏层特征图通道与嵌入层特征图通道的比，默认为 4.
        qkv_bias (bool): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float): 重写默认的缩放因子，默认为 None.
        drop_rate (float): 随机丢弃神经元，丢弃率默认为 0.
        attn_drop_rate (float): 注意力权重的丢弃率，默认为 0.
        drop_path_rate (float): 深度随机丢弃率，默认为 0.1.
        norm_layer (nn.Module): 归一化操作，默认为 nn.LayerNorm.
        ape (bool): patch embedding 添加绝对位置 embedding，默认为 False.
        patch_norm (bool): 在 patch embedding 后添加归一化操作，默认为 True.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
        upscale: 放大因子， 2/3/4/8 适合图像超分, 1 适合图像去噪和 JPEG 压缩去伪影
        img_range: 灰度值范围， 1 或者 255.
        upsampler: 图像重建方法的选择模块，可选择 pixelshuffle, pixelshuffledirect, nearest+conv 或 None.
        resi_connection: 残差连接之前的卷积块， 可选择 1conv 或 3conv.
    """
    def __init__(self, img_size=16, patch_size=1, in_chans=1,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=6, img_range=1., upsampler='pixelshuffle', 
                 resi_connection='1conv'):
        super(SwinIRFuse, self).__init__()

        # --- 基础参数设置 ---
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.img_range = img_range
        
        # --- 浅层特征提取 ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # --- 深层特征提取骨干 ---
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # [核心修改] 构建骨干网络时，只添加RSTB，不再插入ConvBlock
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size, resi_connection=resi_connection
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        # --- GDM 和 GSL 分支的交互 ---
        self.gsl_branch = EnhancedGSLBranch(embed_dim)
        self.attention_fusion = AttentionFusionModule(embed_dim)

        # 这行提前到这里，所有上采样方式都定义
        self.gdm_entry = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if self.upsampler == 'pixelshuffle':
            # 2x 上采样
            self.conv_up1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            # 3x 上采样
            self.conv_up2 = nn.Conv2d(embed_dim, embed_dim * 9, 3, 1, 1)
            self.pixel_shuffle2 = nn.PixelShuffle(3)
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        elif self.upsampler == 'nearest+conv':
            self.conv_up1 = nn.Sequential(nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)
        else:
            raise NotImplementedError(f'Upsampler [{self.upsampler}] is not supported')
            
        # 为动态损失加权准备的可学习参数
        self.log_var_gdm = nn.Parameter(torch.zeros(1))
        self.log_var_gsl = nn.Parameter(torch.zeros(1))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # [核心修改] 简化了特征提取循环
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x_pad = self.check_image_size(x)
        
        x_first = self.conv_first(x_pad)
        
        shared_features = self.forward_features(x_first)
        shared_features = self.conv_after_body(shared_features)
        shared_features = shared_features + x_first
        
        gsl_position, gsl_attention_map = self.gsl_branch(shared_features)
        
        gdm_features = self.gdm_entry(shared_features)
        gdm_features_guided = self.attention_fusion(gdm_features, gsl_attention_map)
        
        if self.upsampler == 'pixelshuffle':
            # 2x pixelshuffle
            up_feat = self.conv_up1(gdm_features_guided)
            up_feat = self.pixel_shuffle1(up_feat)
            # 3x pixelshuffle
            up_feat = self.conv_up2(up_feat)
            up_feat = self.pixel_shuffle2(up_feat)
            gdm_output = self.conv_last(up_feat)
        elif self.upsampler == 'nearest+conv':
            up_feat = F.interpolate(gdm_features_guided, scale_factor=2, mode='nearest')
            up_feat = self.conv_up1(up_feat)
            
            up_feat = F.interpolate(up_feat, scale_factor=1.5, mode='nearest')
            up_feat = self.conv_up2(up_feat)
            
            up_feat = F.interpolate(up_feat, scale_factor=2, mode='nearest')
            up_feat = self.conv_up3(up_feat)
            
            gdm_output = self.conv_last(up_feat)
        else:
            gdm_output = None 

        if gdm_output is not None:
            gdm_output = gdm_output[..., :H*self.upscale, :W*self.upscale]
        
        return gdm_output, gsl_position