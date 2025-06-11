#原模型使用的上采样器earest+conv，先修改成原SwinIR的上采样器为PixelShuffle子像素卷积
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 导入原有的基础模块
from .network_swinir_multi import (
    Mlp, window_partition, window_reverse, WindowAttention,
    SwinTransformerBlock, PatchMerging, BasicLayer, RSTB,
    PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep
)

class EnhancedGSLBranch(nn.Module):
    """增强的GSL分支，学习气体分布整体特征与泄漏源位置的关系"""
    def __init__(self, embed_dim, hidden_dim=64):
        super(EnhancedGSLBranch, self).__init__()
        
        # 特征提取和转换
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 7, 1, 3),
            nn.Sigmoid()
        )
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        
        # 位置预测
        self.position_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2)  # 输出x,y坐标
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.feature_extraction(x)
        
        # 空间注意力
        spatial_att = self.spatial_attention(features)
        features = features * spatial_att
        
        # 通道注意力
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        # 特征融合
        features = self.fusion(features)
        
        # 位置预测
        position = self.position_predictor(features)
        
        return position

class SwinIRMultiEnhanced(nn.Module):
    """增强版的多任务SwinIR模型，改进GSL分支"""
    def __init__(self, img_size=16, patch_size=1, in_chans=1,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=6, img_range=1., upsampler='nearest+conv', 
                 resi_connection='1conv'):
        super(SwinIRMultiEnhanced, self).__init__()
        
        # 基础参数设置
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.img_range = img_range
        
        # 浅层特征提取
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # 深层特征提取
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        
        # Patch UnEmbedding
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        
        # 位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建RSTB层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(self.patch_embed.patches_resolution[0],
                                self.patch_embed.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        
        # 深层特征提取后的卷积层
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
        
        # GDM分支 - 上采样重建
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            # 第一次上采样 (2x)
            self.conv_up1 = nn.Sequential(
                nn.Conv2d(64, 64 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(inplace=True)
            )
            # 第二次上采样 (1.5x)
            self.conv_up2 = nn.Sequential(
                nn.Conv2d(64, 64 * 9, 3, 1, 1),
                nn.PixelShuffle(3),
                nn.LeakyReLU(inplace=True)
            )
            # 第三次上采样 (2x)
            self.conv_up3 = nn.Sequential(
                nn.Conv2d(64, 64 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 增强的GSL分支
        self.gsl_branch = EnhancedGSLBranch(embed_dim)
        
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
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        # 浅层特征提取
        x = self.conv_first(x)
        
        # 共享特征提取
        shared_features = self.conv_after_body(self.forward_features(x)) + x
        
        # GDM分支 - 超分辨率重建
        gdm_out = shared_features
        gdm_out = self.conv_before_upsample(gdm_out)
        gdm_out = self.conv_up1(gdm_out)
        if self.upscale == 6:
            gdm_out = self.conv_up2(gdm_out)
            gdm_out = self.conv_up3(gdm_out)
        gdm_out = self.conv_last(self.lrelu(self.conv_hr(gdm_out)))
        
        # 增强的GSL分支 - 泄漏源定位
        gsl_out = self.gsl_branch(shared_features)
        
        # 裁剪GDM输出到正确大小
        gdm_out = gdm_out[:, :, :H*self.upscale, :W*self.upscale]
        
        return gdm_out, gsl_out 