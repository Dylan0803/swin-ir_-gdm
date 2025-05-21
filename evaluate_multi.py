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
import torch.nn as nn
import torch.nn.functional as F

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
    # 转换为numpy数组并移除batch维度
    lr = lr.squeeze().cpu().numpy()
    hr = hr.squeeze().cpu().numpy()
    gdm_out = gdm_out.squeeze().cpu().numpy()
    gsl_out = gsl_out.squeeze().cpu().numpy()
    source_pos = source_pos.squeeze().cpu().numpy()
    hr_max_pos = hr_max_pos.squeeze().cpu().numpy()
    
    # 计算差值图
    diff = hr - gdm_out
    
    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Gas Concentration Distribution', fontsize=16)
    
    # 低分辨率输入
    im0 = axes[0, 0].imshow(lr, cmap='viridis')
    axes[0, 0].set_title('Low Resolution Input')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 高分辨率真实值
    im1 = axes[0, 1].imshow(hr, cmap='viridis')
    axes[0, 1].set_title('High Resolution Ground Truth')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 模型输出（超分辨率）
    im2 = axes[1, 0].imshow(gdm_out, cmap='viridis')
    axes[1, 0].set_title('Super Resolution Output')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 差值图
    im3 = axes[1, 1].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Difference (HR-SR)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def infer_model(model, data_path, save_dir, num_samples=5, use_valid=True):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    dataset = MultiTaskDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 设置设备
    device = next(model.parameters()).device
    
    # 评估指标
    total_psnr = 0
    total_position_error = 0
    
    # 处理指定数量的样本
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        # 解包数据
        if isinstance(batch, (list, tuple)):
            lr, hr, source_pos = batch
        else:
            # 如果batch是字典，则按键获取数据
            lr = batch['lr']
            hr = batch['hr']
            source_pos = batch['source_pos']
        
        # 将数据移到设备上
        lr = lr.to(device)
        hr = hr.to(device)
        source_pos = source_pos.to(device)
        
        # 模型推理
        with torch.no_grad():
            gdm_out, gsl_out = model(lr)
        
        # 计算HR图像的最大值位置
        hr_max_pos = torch.tensor([torch.argmax(hr[0, 0]) % hr.shape[3], 
                                 torch.argmax(hr[0, 0]) // hr.shape[3]], 
                                dtype=torch.float32).to(device)
        hr_max_pos = hr_max_pos / torch.tensor([hr.shape[3], hr.shape[2]], 
                                             dtype=torch.float32).to(device)
        
        # 计算评估指标
        mse = F.mse_loss(gdm_out, hr)
        psnr = 10 * torch.log10(1.0 / mse)
        position_error = torch.norm(gsl_out - source_pos)
        
        total_psnr += psnr.item()
        total_position_error += position_error.item()
        
        # 保存结果
        save_path = os.path.join(save_dir, f'sample_{i+1}.png')
        visualize_results(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_path)
    
    # 计算平均指标
    avg_psnr = total_psnr / num_samples
    avg_position_error = total_position_error / num_samples
    
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average Position Error: {avg_position_error:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SwinIR Multi-task Model')
    
    # 添加缺失的参数
    parser.add_argument('--img_size', type=int, default=16,
                      help='input image size')
    parser.add_argument('--in_chans', type=int, default=1,
                      help='number of input channels')
    parser.add_argument('--upscale', type=int, default=6,
                      help='upscale factor')
    parser.add_argument('--window_size', type=int, default=8,
                      help='window size')
    parser.add_argument('--img_range', type=float, default=1.,
                      help='image range')
    parser.add_argument('--depths', type=list, default=[6, 6, 6, 6],
                      help='depths of each Swin Transformer layer')
    parser.add_argument('--embed_dim', type=int, default=60,
                      help='embedding dimension')
    parser.add_argument('--num_heads', type=list, default=[6, 6, 6, 6],
                      help='number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=2.,
                      help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--upsampler', type=str, default='nearest+conv',
                      help='upsampler type')
    
    # 原有的参数
    parser.add_argument('--model_path', type=str, required=True,
                      help='path to the model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='path to the dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='directory to save results')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='number of samples to evaluate')
    parser.add_argument('--use_valid', action='store_true',
                      help='use validation set')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SwinIRMulti(
        img_size=args.img_size,
        in_chans=args.in_chans,
        upscale=args.upscale,
        window_size=args.window_size,
        img_range=args.img_range,
        depths=args.depths,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        upsampler=args.upsampler
    )
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())
    
    # 根据实际的键加载模型
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 进行推理
    infer_model(model, args.data_path, args.save_dir, args.num_samples, args.use_valid)

if __name__ == '__main__':
    main()