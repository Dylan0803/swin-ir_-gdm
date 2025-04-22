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
    parser.add_argument('--preserve_zeros', action='store_true', 
                        help='是否保持原图中的0值区域')
    parser.add_argument('--zero_threshold', type=float, default=1e-5,
                        help='判断为0值的阈值，小于此值被视为0')
    return parser.parse_args()


# === 加载命令行参数 ===
args = parse_args()

# === 参数设置 ===
model_path = args.model_path  # 模型路径
data_path = args.data_path  # h5 文件路径
sample_index = args.sample_index  # 测试第几个样本（比如0）
scale = args.scale  # 放大倍数，和训练时一致
device = torch.device(args.device)  # 使用指定的设备
preserve_zeros = args.preserve_zeros  # 是否保持原图中的0值区域
zero_threshold = args.zero_threshold  # 判断为0值的阈值

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

    # 如果启用了保持0值区域的功能
    if preserve_zeros:
        # 创建HR中0值区域的掩码(将小于阈值的视为0)
        hr_mask = (hr_data < zero_threshold)
        
        # 应用掩码到SR结果，将掩码为True的位置设为0
        sr_image[hr_mask] = 0
        
        print(f"应用了零值保持处理，阈值: {zero_threshold}")
        # 计算0值区域所占比例
        zero_percentage = np.mean(hr_mask) * 100
        print(f"原始图像中0值区域占比: {zero_percentage:.2f}%")

# === 可视化对比并保存 ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Ground Truth HR")
plt.imshow(hr_data, cmap='viridis')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("LR input")
plt.imshow(lr_data, cmap='viridis')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("SR output" + (" (with zero preservation)" if preserve_zeros else ""))
plt.imshow(sr_image, cmap='viridis')
plt.axis('off')

# 保存为图像文件，而不是直接展示
output_filename = 'output_comparison' + ('_zero_preserved' if preserve_zeros else '') + '.png'
plt.savefig(output_filename)
print(f"Comparison image saved as '{output_filename}'")

# 保存处理后的图像数据
np.save('sr_output.npy', sr_image)
print("SR result data saved as 'sr_output.npy'")
