import torch
from models.network_swinir import SwinIR
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
    parser.add_argument('--output_file', type=str, default='comparison_data.npz', 
                        help='保存比较数据的文件名')
    return parser.parse_args()


# === 加载命令行参数 ===
args = parse_args()

# === 参数设置 ===
model_path = args.model_path  # 模型路径
data_path = args.data_path  # h5 文件路径
sample_index = args.sample_index  # 测试第几个样本（比如0）
scale = args.scale  # 放大倍数，和训练时一致
device = torch.device(args.device)  # 使用指定的设备
output_file = args.output_file  # 输出文件名

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

# === 分析SR和HR数据 ===
# 计算差异
diff = hr_data - sr_image
mse = np.mean(np.square(diff))
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
mae = np.mean(np.abs(diff))

# 打印数据统计信息
print("\n=== 数据分析结果 ===")
print(f"HR 形状: {hr_data.shape}, 最小值: {hr_data.min():.6f}, 最大值: {hr_data.max():.6f}, 均值: {hr_data.mean():.6f}")
print(f"SR 形状: {sr_image.shape}, 最小值: {sr_image.min():.6f}, 最大值: {sr_image.max():.6f}, 均值: {sr_image.mean():.6f}")
print(f"差异 - 最小值: {diff.min():.6f}, 最大值: {diff.max():.6f}, 均值: {diff.mean():.6f}")
print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, MAE: {mae:.6f}")

# 保存数据以便进一步分析
np.savez(output_file, hr=hr_data, sr=sr_image, diff=diff)
print(f"数据已保存到 {output_file} 文件中")

# === 可视化对比并保存 ===
plt.figure(figsize=(15, 12))

# 原始图像对比
plt.subplot(2, 3, 1)
plt.title("Ground Truth HR")
plt.imshow(hr_data, cmap='viridis')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("LR input")
plt.imshow(lr_data, cmap='viridis')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("SR output")
plt.imshow(sr_image, cmap='viridis')
plt.colorbar()
plt.axis('off')

# 添加差异图及直方图
plt.subplot(2, 3, 4)
plt.title("Difference (HR - SR)")
diff_plot = plt.imshow(diff, cmap='coolwarm')
plt.colorbar(diff_plot)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("HR 直方图")
plt.hist(hr_data.flatten(), bins=50, alpha=0.7)
plt.grid(True)

plt.subplot(2, 3, 6)
plt.title("SR 直方图")
plt.hist(sr_image.flatten(), bins=50, alpha=0.7)
plt.grid(True)

# 保存为图像文件，而不是直接展示
plt.tight_layout()
plt.savefig('output_comparison.png', dpi=300)
print("分析图像已保存为 'output_comparison.png'")
