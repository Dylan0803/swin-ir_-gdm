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

class ConvBlock(nn.Module):
    """在RSTB之间插入的卷积模块，用于增强局部特征提取"""
    def __init__(self, dim, expansion_ratio=2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim * expansion_ratio, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, 3, 1, 1, groups=dim * expansion_ratio),
            nn.Conv2d(dim * expansion_ratio, dim, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out


class EnhancedGSLBranch(nn.Module):
    """
    [核心修改] 增强的GSL分支，现在可以同时输出坐标和空间注意力图
    """
    def __init__(self, embed_dim, hidden_dim=64):
        super(EnhancedGSLBranch, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 7, 1, 3),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.position_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x):
        features = self.feature_extraction(x)
        spatial_att_map = self.spatial_attention(features)
        features_attended = features * spatial_att_map
        channel_att = self.channel_attention(features_attended)
        features_fused = features_attended * channel_att
        features_final = self.fusion(features_fused)
        position = self.position_predictor(features_final)
        return position, spatial_att_map

class AttentionFusionModule(nn.Module):
    """
    [新增模块] 注意力融合模块
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

class HybridFuse(nn.Module):
    """
    [核心修改] 新的模型，實現GSL指導GDM的互補學習
    """
    def __init__(self, img_size=16, patch_size=1, in_chans=1,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=6, img_range=1., upsampler='nearest+conv', 
                 resi_connection='1conv'):
        super(HybridFuse, self).__init__()
        
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
            if i_layer < self.num_layers - 1:
                conv_block = ConvBlock(embed_dim)
                self.layers.append(conv_block)
        
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

        if self.upsampler == 'nearest+conv':
            self.gdm_entry = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_up1 = nn.Sequential(nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)
        else:
            raise NotImplementedError(f'Upsampler [{self.upsampler}] is not supported')
        
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

        for layer in self.layers:
            if isinstance(layer, RSTB):
                x = layer(x, x_size)
            else:
                B, L, C = x.shape
                H_feat, W_feat = x_size
                x_conv = x.permute(0, 2, 1).contiguous().view(B, C, H_feat, W_feat)
                x_conv = layer(x_conv)
                x = x_conv.flatten(2).permute(0, 2, 1).contiguous()

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
        
        if self.upsampler == 'nearest+conv':
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