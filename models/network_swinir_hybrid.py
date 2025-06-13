import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 导入原有的基础模块
from .network_swinir_multi_enhanced import (
    Mlp, window_partition, window_reverse, WindowAttention,
    SwinTransformerBlock, PatchMerging, BasicLayer, RSTB,
    PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep,
    EnhancedGSLBranch
)

class ConvBlock(nn.Module):
    """在RSTB之间插入的卷积模块，用于增强局部特征提取"""
    def __init__(self, dim, expansion_ratio=2):
        super(ConvBlock, self).__init__()
        
        # 使用深度可分离卷积来减少参数量
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # 深度卷积
            nn.Conv2d(dim, dim * expansion_ratio, 1),  # 逐点卷积
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, 3, 1, 1, groups=dim * expansion_ratio),  # 深度卷积
            nn.Conv2d(dim * expansion_ratio, dim, 1),  # 逐点卷积
            nn.LeakyReLU(inplace=True)
        )
        
        # 残差连接
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        # 确保输入张量的通道数与模型维度匹配
        if x.shape[1] != x.shape[1]:
            x = x.permute(0, 3, 1, 2)  # 调整通道维度位置
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out

class SwinIRHybrid(nn.Module):
    """混合架构的SwinIR模型，在RSTB之间插入卷积模块"""
    def __init__(self, img_size=16, patch_size=1, in_chans=1,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=6, img_range=1., upsampler='nearest+conv', 
                 resi_connection='1conv'):
        super(SwinIRHybrid, self).__init__()
        
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
        
        # 构建RSTB层和卷积模块
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # RSTB层
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
            
            # 在每个RSTB层后添加卷积模块
            if i_layer < self.num_layers - 1:  # 除了最后一层
                conv_block = ConvBlock(embed_dim)
                self.layers.append(conv_block)
        
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
        if self.upsampler == 'nearest+conv':
            # 第一次上采样 (2x)
            self.conv_up1 = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            # 第二次上采样 (1.5x)
            self.conv_up2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            # 第三次上采样 (2x)
            self.conv_up3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            # 最终输出层
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)
        else:
            raise NotImplementedError(f'Upsampler [{self.upsampler}] is not supported')
        
        # GSL分支
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
            if isinstance(layer, RSTB):
                x = layer(x, x_size)
            else:  # ConvBlock
                # 确保输入张量的形状正确
                if len(x.shape) == 3:  # [B, L, C]
                    x = x.permute(0, 2, 1).view(x.shape[0], x.shape[2], x_size[0], x_size[1])
                x = layer(x)
                # 恢复形状
                x = x.flatten(2).permute(0, 2, 1)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        # 浅层特征提取
        x = self.conv_first(x)
        
        # 深层特征提取
        res = self.forward_features(x)
        res = self.conv_after_body(res)
        res = res + x
        
        # GDM分支 - 上采样重建
        if self.upsampler == 'nearest+conv':
            # 第一次上采样 (2x)
            x = F.interpolate(res, scale_factor=2, mode='nearest')
            x = self.conv_up1(x)
            
            # 第二次上采样 (1.5x)
            x = F.interpolate(x, scale_factor=1.5, mode='nearest')
            x = self.conv_up2(x)
            
            # 第三次上采样 (2x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.conv_up3(x)
            
            # 最终输出
            x = self.conv_last(x)
        else:
            raise NotImplementedError(f'Upsampler [{self.upsampler}] is not supported')
        
        # GSL分支 - 泄漏源定位
        gsl_out = self.gsl_branch(res)
        
        # 裁剪到原始大小
        x = x[:, :, :H*self.upscale, :W*self.upscale]
        
        return x, gsl_out 