# -----------------------------------------------------------------------------------
# SwinIR-Multi: 多任务学习网络
# 功能：同时完成气体分布图超分辨率(GDM)和泄漏源定位(GSL)两个任务
# 基于SwinIR: Image Restoration Using Swin Transformer
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# -----------------------------------------------------------------------------------
# 基础模块
# -----------------------------------------------------------------------------------

class Mlp(nn.Module):
    """多层感知机模块
    功能：实现特征的非线性变换
    参数：
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度
        out_features: 输出特征维度
        act_layer: 激活函数
        drop: dropout比率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """将特征图分割成不重叠的窗口
    功能：将输入特征图分割成固定大小的窗口，用于局部自注意力计算
    参数：
        x: 输入特征图 (B, H, W, C)
        window_size: 窗口大小
    返回：
        windows: 分割后的窗口 (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """将窗口重组为特征图
    功能：将分割的窗口重新组合成完整的特征图
    参数：
        windows: 窗口特征 (num_windows*B, window_size, window_size, C)
        window_size: 窗口大小
        H, W: 原始特征图的高度和宽度
    返回：
        x: 重组后的特征图 (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """窗口自注意力模块
    功能：在局部窗口内计算自注意力
    参数：
        dim: 特征维度
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        attn_drop: 注意力dropout率
        proj_drop: 投影dropout率
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块
    功能：实现Swin Transformer的基本构建块
    参数：
        dim: 特征维度
        input_resolution: 输入分辨率
        num_heads: 注意力头数
        window_size: 窗口大小
        shift_size: 移位大小
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        drop: dropout率
        attn_drop: 注意力dropout率
        drop_path: 路径dropout率
        act_layer: 激活函数
        norm_layer: 归一化层
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """计算注意力掩码
        功能：为移位窗口自注意力计算掩码
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """图像块合并模块
    功能：将相邻的图像块合并，实现下采样
    参数：
        input_resolution: 输入分辨率
        dim: 特征维度
        norm_layer: 归一化层
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """基础层模块
    功能：构建Swin Transformer的一个阶段
    参数：
        dim: 特征维度
        input_resolution: 输入分辨率
        depth: 深度
        num_heads: 注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        drop: dropout率
        attn_drop: 注意力dropout率
        drop_path: 路径dropout率
        norm_layer: 归一化层
        downsample: 下采样层
        use_checkpoint: 是否使用checkpoint
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class RSTB(nn.Module):
    """残差Swin Transformer块
    功能：实现带有残差连接的Swin Transformer块
    参数：
        dim: 特征维度
        input_resolution: 输入分辨率
        depth: 深度
        num_heads: 注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        drop: dropout率
        attn_drop: 注意力dropout率
        drop_path: 路径dropout率
        norm_layer: 归一化层
        downsample: 下采样层
        use_checkpoint: 是否使用checkpoint
        img_size: 图像大小
        patch_size: 图像块大小
        resi_connection: 残差连接类型
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

class PatchEmbed(nn.Module):
    """图像块嵌入模块
    功能：将图像分割成块并进行嵌入
    参数：
        img_size: 图像大小
        patch_size: 图像块大小
        in_chans: 输入通道数
        embed_dim: 嵌入维度
        norm_layer: 归一化层
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    """图像块解嵌入模块
    功能：将嵌入的特征转换回图像块
    参数：
        img_size: 图像大小
        patch_size: 图像块大小
        in_chans: 输入通道数
        embed_dim: 嵌入维度
        norm_layer: 归一化层
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

class SwinIRMulti(nn.Module):
    """SwinIR多任务网络
    功能：同时完成气体分布图超分辨率(GDM)和泄漏源定位(GSL)两个任务
    参数：
        img_size: 图像大小
        patch_size: 图像块大小
        in_chans: 输入通道数
        embed_dim: 嵌入维度
        depths: 各层深度
        num_heads: 各层注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        drop_rate: dropout率
        attn_drop_rate: 注意力dropout率
        drop_path_rate: 路径dropout率
        norm_layer: 归一化层
        ape: 是否使用绝对位置编码
        patch_norm: 是否使用patch归一化
        use_checkpoint: 是否使用checkpoint
        upscale: 上采样倍数
        img_range: 图像范围
        upsampler: 上采样器类型
        resi_connection: 残差连接类型
    """
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=6, img_range=1., upsampler='nearest+conv', 
                 resi_connection='1conv'):
        super(SwinIRMulti, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale

        # 1. 浅层特征提取
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. 深层特征提取
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # 图像分块
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 图像块合并
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建RSTB
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
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
                         resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # 深层特征提取后的卷积层
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # GDM任务分支（超分辨率）
        self.gdm_branch = nn.ModuleDict({
            'conv_before_upsample': nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            ),
            'conv_up1': nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            'conv_up2': nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            'conv_hr': nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            'conv_last': nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        })

        # GSL任务分支（泄漏源定位）
        self.gsl_branch = nn.ModuleDict({
            'conv1': nn.Conv2d(embed_dim, 64, 3, 1, 1),
            'conv2': nn.Conv2d(64, 32, 3, 1, 1),
            'pool': nn.AdaptiveAvgPool2d(1),
            'fc': nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 2)  # 输出泄漏源坐标(x,y)
            )
        })

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 修改上采样部分
        if upsampler == 'nearest+conv':
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, in_chans * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        elif upsampler == 'pixelshuffle':
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, in_chans * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        elif upsampler == 'pixelshuffledirect':
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, in_chans * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        else:
            raise NotImplementedError(f'Upsampler [{upsampler}] is not implemented')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化权重
        功能：使用截断正态分布初始化线性层和LayerNorm层的权重
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        """检查图像大小
        功能：确保图像大小是窗口大小的整数倍
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """前向特征提取
        功能：提取深层特征
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        """前向传播
        功能：同时完成超分辨率和泄漏源定位两个任务
        返回：
            gdm_x: 超分辨率结果
            gsl_x: 泄漏源定位结果
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # 共享特征提取
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        # GDM分支（超分辨率）
        gdm_x = self.gdm_branch['conv_before_upsample'](x)
        gdm_x = self.lrelu(self.gdm_branch['conv_up1'](
            F.interpolate(gdm_x, scale_factor=2, mode='nearest')))
        gdm_x = self.lrelu(self.gdm_branch['conv_up2'](
            F.interpolate(gdm_x, scale_factor=2, mode='nearest')))
        gdm_x = self.gdm_branch['conv_last'](
            self.lrelu(self.gdm_branch['conv_hr'](gdm_x)))

        # GSL分支（泄漏源定位）
        gsl_x = self.lrelu(self.gsl_branch['conv1'](x))
        gsl_x = self.lrelu(self.gsl_branch['conv2'](gsl_x))
        gsl_x = self.gsl_branch['pool'](gsl_x)
        gsl_x = gsl_x.view(gsl_x.size(0), -1)
        gsl_x = self.gsl_branch['fc'](gsl_x)

        gdm_x = gdm_x / self.img_range + self.mean

        # 确保输出尺寸与目标一致
        if gdm_x.shape[2:] != (96, 96):
            gdm_x = F.interpolate(gdm_x, size=(96, 96), mode='bilinear', align_corners=False)

        x = self.upsampler(gdm_x)

        return x, gsl_x

if __name__ == '__main__':
    # 测试代码
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIRMulti(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='nearest+conv')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 1, height, width))
    gdm_out, gsl_out = model(x)
    print(f"GDM output shape: {gdm_out.shape}")
    print(f"GSL output shape: {gsl_out.shape}")