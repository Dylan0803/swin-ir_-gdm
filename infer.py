import torch
from models.network_swinir import SwinIR
import h5py
import numpy as np
import argparse
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 命令行参数设置 ===


def parse_args():
    parser = argparse.ArgumentParser(description="SwinIR Inference with Plotly Visualization")
    parser.add_argument('--model_path', type=str,
                        default='./experiments/my_exp_20250418-153000/best_model.pth', help='模型路径')
    parser.add_argument('--data_path', type=str,
                        default='/content/drive/MyDrive/1.h5', help='h5 文件路径')
    parser.add_argument('--sample_index', type=int,
                        default=0, help='测试第几个样本（比如0）')
    parser.add_argument('--scale', type=int, default=6, help='放大倍数，和训练时一致')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'cpu', help='设备类型: cuda 或 cpu')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--save_html', action='store_true', help='是否保存HTML文件而不是显示')
    return parser.parse_args()


# === 加载命令行参数 ===
args = parse_args()

# === 参数设置 ===
model_path = args.model_path  # 模型路径
data_path = args.data_path    # h5 文件路径
sample_index = args.sample_index  # 测试第几个样本（比如0）
scale = args.scale            # 放大倍数，和训练时一致
device = torch.device(args.device)  # 使用指定的设备
output_dir = args.output_dir   # 输出目录
save_html = args.save_html     # 是否保存HTML而不是显示

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

# === 使用Plotly创建交互式可视化 ===
# 创建包含3个子图的figure
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Ground Truth HR", "LR Input", "SR Output"),
    horizontal_spacing=0.05
)

# 添加热力图
fig.add_trace(
    go.Heatmap(
        z=hr_data,
        colorscale='Viridis',
        colorbar=dict(
            title="像素值",
            x=0.3,  # 调整颜色条位置
        ),
        hovertemplate='位置: (%{x}, %{y})<br>值: %{z:.6f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(
        z=lr_data,
        colorscale='Viridis',
        colorbar=dict(
            title="像素值",
            x=0.63,  # 调整颜色条位置
        ),
        hovertemplate='位置: (%{x}, %{y})<br>值: %{z:.6f}<extra></extra>'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Heatmap(
        z=sr_image,
        colorscale='Viridis',
        colorbar=dict(
            title="像素值",
            x=0.96,  # 调整颜色条位置
        ),
        hovertemplate='位置: (%{x}, %{y})<br>值: %{z:.6f}<extra></extra>'
    ),
    row=1, col=3
)

# 更新布局
fig.update_layout(
    title="SwinIR 超分辨率结果 - 交互式像素值可视化",
    height=600,
    width=1200,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    ),
    margin=dict(l=50, r=50, t=80, b=50),
)

# 添加数据统计信息
hr_stats = f"HR统计: 最小值={hr_data.min():.6f}, 最大值={hr_data.max():.6f}, 均值={hr_data.mean():.6f}, 标准差={hr_data.std():.6f}"
lr_stats = f"LR统计: 最小值={lr_data.min():.6f}, 最大值={lr_data.max():.6f}, 均值={lr_data.mean():.6f}, 标准差={lr_data.std():.6f}"
sr_stats = f"SR统计: 最小值={sr_image.min():.6f}, 最大值={sr_image.max():.6f}, 均值={sr_image.mean():.6f}, 标准差={sr_image.std():.6f}"

fig.add_annotation(
    text=hr_stats,
    x=0,
    y=1.1,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=10)
)

fig.add_annotation(
    text=lr_stats,
    x=0.33,
    y=1.1,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=10)
)

fig.add_annotation(
    text=sr_stats,
    x=0.66,
    y=1.1,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=10)
)

# 输出统计信息到控制台
print("数据统计信息:")
print(hr_stats)
print(lr_stats)
print(sr_stats)

# 保存HTML文件或显示
if save_html:
    html_path = os.path.join(output_dir, 'swinir_interactive.html')
    fig.write_html(html_path)
    print(f"交互式可视化已保存到: {html_path}")
else:
    # 尝试导入plotly所需的离线显示模块
    try:
        import plotly.io as pio
        pio.renderers.default = "browser"  # 在浏览器中打开
        fig.show()
    except ImportError:
        # 如果导入失败，则保存为HTML
        html_path = os.path.join(output_dir, 'swinir_interactive.html')
        fig.write_html(html_path)
        print(f"无法显示，已保存交互式可视化到: {html_path}")
