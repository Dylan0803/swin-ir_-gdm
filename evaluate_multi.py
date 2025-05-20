import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_dataset
import pandas as pd
import h5py
import argparse

def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
    
    返回:
        dict: 包含各项评估指标的字典
    """
    model.eval()
    metrics = {
        'gdm_psnr': [],
        'gdm_ssim': [],
        'gsl_position_error': [],
        'gsl_max_pos_error': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 数据预处理
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            source_pos = batch['source_pos'].to(device)
            hr_max_pos = batch['hr_max_pos'].to(device)
            
            # 模型推理
            gdm_out, gsl_out = model(lr)
            
            # 计算GDM指标
            for i in range(gdm_out.size(0)):
                # PSNR
                mse = torch.mean((gdm_out[i] - hr[i]) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                metrics['gdm_psnr'].append(psnr.item())
                
                # SSIM (简化版)
                c1 = (0.01 * 1.0) ** 2
                c2 = (0.03 * 1.0) ** 2
                mu1 = torch.mean(gdm_out[i])
                mu2 = torch.mean(hr[i])
                sigma1 = torch.var(gdm_out[i])
                sigma2 = torch.var(hr[i])
                sigma12 = torch.mean((gdm_out[i] - mu1) * (hr[i] - mu2))
                ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
                metrics['gdm_ssim'].append(ssim.item())
            
            # 计算GSL指标
            position_error = torch.sqrt(torch.sum((gsl_out - source_pos) ** 2, dim=1))
            max_pos_error = torch.sqrt(torch.sum((gsl_out - hr_max_pos) ** 2, dim=1))
            
            metrics['gsl_position_error'].extend(position_error.cpu().numpy())
            metrics['gsl_max_pos_error'].extend(max_pos_error.cpu().numpy())
    
    # 计算平均指标
    results = {
        'GDM_PSNR': np.mean(metrics['gdm_psnr']),
        'GDM_SSIM': np.mean(metrics['gdm_ssim']),
        'GSL_Position_Error': np.mean(metrics['gsl_position_error']),
        'GSL_MaxPos_Error': np.mean(metrics['gsl_max_pos_error'])
    }
    
    return results, metrics

def plot_metrics(metrics, save_dir):
    """绘制评估指标分布图"""
    plt.figure(figsize=(15, 10))
    
    # GDM PSNR分布
    plt.subplot(221)
    plt.hist(metrics['gdm_psnr'], bins=50)
    plt.title('GDM PSNR Distribution')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    
    # GDM SSIM分布
    plt.subplot(222)
    plt.hist(metrics['gdm_ssim'], bins=50)
    plt.title('GDM SSIM Distribution')
    plt.xlabel('SSIM')
    plt.ylabel('Count')
    
    # GSL位置误差分布
    plt.subplot(223)
    plt.hist(metrics['gsl_position_error'], bins=50)
    plt.title('GSL Position Error Distribution')
    plt.xlabel('Error (normalized)')
    plt.ylabel('Count')
    
    # GSL最大浓度位置误差分布
    plt.subplot(224)
    plt.hist(metrics['gsl_max_pos_error'], bins=50)
    plt.title('GSL Max Position Error Distribution')
    plt.xlabel('Error (normalized)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'))
    plt.close()

def visualize_results(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_path):
    """
    可视化推理结果
    
    参数:
        lr: 低分辨率输入 [1, H, W]
        hr: 高分辨率真值 [1, H, W]
        gdm_out: 模型GDM输出 [1, H, W]
        gsl_out: 模型GSL输出 [2]
        source_pos: 真实泄漏源位置 [2]
        hr_max_pos: HR中最大浓度位置 [2]
        save_path: 保存路径
    """
    # 转换为numpy数组
    lr = lr.squeeze().cpu().numpy()
    hr = hr.squeeze().cpu().numpy()
    gdm_out = gdm_out.squeeze().cpu().numpy()
    gsl_out = gsl_out.cpu().numpy()
    source_pos = source_pos.cpu().numpy()
    hr_max_pos = hr_max_pos.cpu().numpy()
    
    # 计算差异图
    diff = hr - gdm_out
    
    # 创建图形
    fig = plt.figure(figsize=(20, 5))
    
    # 1. 低分辨率输入
    plt.subplot(141)
    plt.imshow(lr, cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.title('Low Resolution Input')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 2. 高分辨率真值
    plt.subplot(142)
    plt.imshow(hr, cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.title('High Resolution Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 3. 模型输出
    plt.subplot(143)
    plt.imshow(gdm_out, cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.title('Model Output (SR)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 4. 差异图
    plt.subplot(144)
    plt.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(label='Difference (HR-SR)')
    plt.title('Difference Map (HR-SR)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 在第二个和第三个子图上标记位置
    # 将归一化坐标转换回原始坐标
    gsl_out = gsl_out * 95.0
    source_pos = source_pos * 95.0
    hr_max_pos = hr_max_pos * 95.0
    
    # 在HR图上标记真实位置
    plt.subplot(142)
    plt.plot(source_pos[0], source_pos[1], 'r*', markersize=10, label='True Source')
    plt.plot(hr_max_pos[0], hr_max_pos[1], 'b*', markersize=10, label='Max Concentration')
    plt.legend()
    
    # 在SR图上标记预测位置
    plt.subplot(143)
    plt.plot(gsl_out[0], gsl_out[1], 'g*', markersize=10, label='Predicted Source')
    plt.plot(hr_max_pos[0], hr_max_pos[1], 'b*', markersize=10, label='Max Concentration')
    plt.legend()
    
    # 添加位置信息文本
    info_text = f'True Source: ({source_pos[0]:.1f}, {source_pos[1]:.1f})\n'
    info_text += f'Predicted: ({gsl_out[0]:.1f}, {gsl_out[1]:.1f})\n'
    info_text += f'Max Conc: ({hr_max_pos[0]:.1f}, {hr_max_pos[1]:.1f})'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def infer_model(model, data_path, save_dir, num_samples=5, use_valid=True):
    """
    使用模型进行推理并可视化结果
    
    参数:
        model: 训练好的模型
        data_path: 数据文件路径
        save_dir: 结果保存目录
        num_samples: 要推理的样本数量
        use_valid: 是否使用验证集
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    if use_valid:
        # 使用验证集
        _, valid_dataset = generate_train_valid_dataset(data_path, train_ratio=0.8, shuffle=True)
        dataset = valid_dataset
    else:
        # 使用整个数据集
        dataset = MultiTaskDataset(data_path, shuffle=True)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 选择前num_samples个样本进行推理
    for i, batch in enumerate(tqdm(dataloader, desc="Inferring")):
        if i >= num_samples:
            break
            
        # 数据预处理
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        source_pos = batch['source_pos'].to(device)
        hr_max_pos = batch['hr_max_pos'].to(device)
        
        # 模型推理
        with torch.no_grad():
            gdm_out, gsl_out = model(lr)
        
        # 可视化结果
        save_path = os.path.join(save_dir, f'inference_result_{i+1}.png')
        visualize_results(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_path)
        
        # 计算评估指标
        mse = torch.mean((gdm_out - hr) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        position_error = torch.sqrt(torch.sum((gsl_out - source_pos) ** 2))
        
        print(f"\n样本 {i+1} 的评估结果:")
        print(f"PSNR: {psnr.item():.2f} dB")
        print(f"位置误差: {position_error.item():.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='模型推理和可视化')
    parser.add_argument('--model_path', type=str, required=True,
                      help='训练好的模型路径')
    parser.add_argument('--data_path', type=str, required=True,
                      help='数据集路径')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='结果保存目录')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='要推理的样本数量')
    parser.add_argument('--use_valid', action='store_true',
                      help='是否使用验证集进行推理')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载模型
    model = SwinIRMulti(
        img_size=16,
        patch_size=1,
        in_chans=1,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        upscale=6,
        img_range=1.,
        upsampler='nearest+conv',
        resi_connection='1conv'
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # 进行推理和可视化
    infer_model(model, args.data_path, args.save_dir, 
               num_samples=args.num_samples, use_valid=args.use_valid)
    
    print(f"\n推理结果已保存至: {args.save_dir}")

if __name__ == '__main__':
    main()