import torch
from models.network_swinir import SwinIR
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

# === 命令行参数设置 ===


def parse_args():
    parser = argparse.ArgumentParser(description="SwinIR Inference")
    parser.add_argument('--model_path', type=str,
                        default='./experiments/my_exp_20250418-153000/best_model.pth', help='模型路径')
    parser.add_argument('--data_path', type=str,
                        default='/content/drive/MyDrive/1.h5', help='h5 文件路径')
    parser.add_argument('--sample_index', type=int,
                        default=0, help='测试第几个样本（比如0）')
    parser.add_argument('--scale', type=int, default=6, help='放大倍数，和训练时一致')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'cpu', help='设备类型: cuda 或 cpu')
    parser.add_argument('--interactive', action='store_true', help='是否启用交互模式')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    return parser.parse_args()


# === 加载命令行参数 ===
args = parse_args()

# === 参数设置 ===
model_path = args.model_path  # 模型路径
data_path = args.data_path    # h5 文件路径
sample_index = args.sample_index  # 测试第几个样本（比如0）
scale = args.scale            # 放大倍数，和训练时一致
device = torch.device(args.device)  # 使用指定的设备
interactive = args.interactive  # 是否启用交互模式
output_dir = args.output_dir   # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# === 加载模型 ===
model = SwinIR(
    upscale=scale,
    in_chans=1,
    img_size=16,
    window_size=4,
    img_range=1.,
    depths=[6, 6, 6, 6],
    embed_dim=60,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffledirect',
    resi_connection='1conv'
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 加载一个测试样本 ===
with h5py.File(data_path, 'r') as f:
    lr_data = f['LR'][sample_index]  # shape: (b, b)
    hr_data = f['HR'][sample_index]  # shape: (a, a)

# 转为张量，添加 batch 和 channel 维度
lr_tensor = torch.from_numpy(lr_data).unsqueeze(
    0).unsqueeze(0).float().to(device)  # shape: [1,1,b,b]

# === 推理 ===
with torch.no_grad():
    sr_tensor = model(lr_tensor)  # shape: [1,1,a,a]
    sr_image = sr_tensor.squeeze().cpu().numpy()  # shape: [a, a]

if interactive:
    # === 创建交互式显示 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示图像
    im1 = axes[0].imshow(hr_data, cmap='viridis')
    axes[0].set_title("Ground Truth HR")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')
    
    im2 = axes[1].imshow(lr_data, cmap='viridis')
    axes[1].set_title("LR input")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')
    
    im3 = axes[2].imshow(sr_image, cmap='viridis')
    axes[2].set_title("SR output")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')
    
    # 添加像素值显示
    pixel_info = plt.figtext(0.5, 0.01, "鼠标悬停以查看像素值", 
                            ha="center", fontsize=12, 
                            bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    def format_coord(x, y, ax_idx):
        if ax_idx == 0 and 0 <= int(x + 0.5) < hr_data.shape[1] and 0 <= int(y + 0.5) < hr_data.shape[0]:
            z = hr_data[int(y + 0.5), int(x + 0.5)]
            return f'Ground Truth HR: x={int(x + 0.5)}, y={int(y + 0.5)}, 值={z:.6f}'
        elif ax_idx == 1 and 0 <= int(x + 0.5) < lr_data.shape[1] and 0 <= int(y + 0.5) < lr_data.shape[0]:
            z = lr_data[int(y + 0.5), int(x + 0.5)]
            return f'LR input: x={int(x + 0.5)}, y={int(y + 0.5)}, 值={z:.6f}'
        elif ax_idx == 2 and 0 <= int(x + 0.5) < sr_image.shape[1] and 0 <= int(y + 0.5) < sr_image.shape[0]:
            z = sr_image[int(y + 0.5), int(x + 0.5)]
            return f'SR output: x={int(x + 0.5)}, y={int(y + 0.5)}, 值={z:.6f}'
        return ""
    
    # 自定义坐标格式化函数
    for i, ax in enumerate(axes):
        ax.format_coord = lambda x, y, i=i: format_coord(x, y, i)
    
    def hover(event):
        if event.inaxes in axes:
            ax_idx = axes.index(event.inaxes)
            if event.xdata is not None and event.ydata is not None:
                text = format_coord(event.xdata, event.ydata, ax_idx)
                pixel_info.set_text(text)
    
    fig.canvas.mpl_connect('motion_notify_event', hover)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为底部的文本腾出空间
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    print(f"对比图像已保存到 {os.path.join(output_dir, 'comparison.png')}")
    
    # 显示交互式图像
    plt.show()
else:
    # === 非交互式可视化对比并保存 ===
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth HR")
    plt.imshow(hr_data, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("LR input")
    plt.imshow(lr_data, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("SR output")
    plt.imshow(sr_image, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 保存为图像文件
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    print(f"对比图像已保存到 {os.path.join(output_dir, 'comparison.png')}")
