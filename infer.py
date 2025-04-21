import torch
from models.network_swinir import SwinIR
import h5py
import numpy as np
import matplotlib.pyplot as plt

# === 参数设置 ===
model_path = './experiments/my_exp_20250418-153000/best_model.pth'  # 模型路径
data_path = '/content/drive/MyDrive/1.h5'  # h5 文件路径
sample_index = 0  # 测试第几个样本（比如0）
scale = 6  # 放大倍数，和训练时一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# === 可选：可视化对比 ===
print(
    f"LR Data Shape: {lr_data.shape}, SR Image Shape: {sr_image.shape}, HR Data Shape: {hr_data.shape}")

plt.subplot(1, 3, 1)
plt.title("LR input")
plt.imshow(lr_data, cmap='hot')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("SR output")
plt.imshow(sr_image, cmap='hot')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Ground Truth HR")
plt.imshow(hr_data, cmap='hot')
plt.axis('off')

plt.show()
