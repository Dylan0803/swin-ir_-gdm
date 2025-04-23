import torch
from models.network_swinir import SwinIR
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from matplotlib.figure import Figure

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
    return parser.parse_args()


# === 加载命令行参数 ===
args = parse_args()

# === 参数设置 ===
model_path = args.model_path  # 模型路径
data_path = args.data_path  # h5 文件路径
sample_index = args.sample_index  # 测试第几个样本（比如0）
scale = args.scale  # 放大倍数，和训练时一致
device = torch.device(args.device)  # 使用指定的设备
interactive = args.interactive  # 是否启用交互模式

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
    root = tk.Tk()
    root.title("SwinIR 超分辨率结果 - 像素值查看器")
    root.geometry("1200x800")
    
    # 创建Figure
    fig = Figure(figsize=(15, 5), dpi=100)
    
    # 添加子图
    ax1 = fig.add_subplot(131)
    ax1.set_title("Ground Truth HR")
    im1 = ax1.imshow(hr_data, cmap='viridis')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(132)
    ax2.set_title("LR input")
    im2 = ax2.imshow(lr_data, cmap='viridis')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(133)
    ax3.set_title("SR output")
    im3 = ax3.imshow(sr_image, cmap='viridis')
    ax3.axis('off')
    
    # 添加颜色条
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 创建Canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # 创建文本显示区
    pixel_info = tk.Label(root, text="鼠标悬停以查看像素值", font=("Arial", 12))
    pixel_info.pack(side=tk.BOTTOM, fill=tk.X)
    
    # 添加鼠标移动事件
    def on_motion(event):
        if event.inaxes == ax1:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < hr_data.shape[1] and 0 <= y < hr_data.shape[0]:
                pixel_value = hr_data[y, x]
                pixel_info.config(text=f"Ground Truth HR: 位置=({x}, {y}), 值={pixel_value:.6f}")
        elif event.inaxes == ax2:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < lr_data.shape[1] and 0 <= y < lr_data.shape[0]:
                pixel_value = lr_data[y, x]
                pixel_info.config(text=f"LR input: 位置=({x}, {y}), 值={pixel_value:.6f}")
        elif event.inaxes == ax3:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < sr_image.shape[1] and 0 <= y < sr_image.shape[0]:
                pixel_value = sr_image[y, x]
                pixel_info.config(text=f"SR output: 位置=({x}, {y}), 值={pixel_value:.6f}")
    
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    # 运行Tkinter事件循环
    tk.mainloop()
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
    plt.savefig('output_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison image saved as 'output_comparison.png'")
    
    # 可选：显示图像
    plt.show()
