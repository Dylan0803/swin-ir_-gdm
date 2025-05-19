import os
import sys
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import argparse
import datetime
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti as net
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='swinir_multi', help='模型名称')
    parser.add_argument('--exp_name', type=str, default='nearest+conv_test', help='实验名称')
    parser.add_argument('--upsampler', type=str, default='nearest+conv', 
                       choices=['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'],
                       help='上采样器类型')
    parser.add_argument('--scale', type=int, default=6, help='上采样倍数')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                       default='C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\normalized_augmented_dataset.h5',
                       help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--patch_size', type=int, default=64, help='图像块大小')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--gdm_weight', type=float, default=1.0, help='GDM任务损失权重')
    parser.add_argument('--gsl_weight', type=float, default=0.5, help='GSL任务损失权重')
    
    # 保存相关参数
    parser.add_argument('--save_dir', type=str, default='./experiments', help='保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def calculate_position_error(pred_pos, gt_pos):
    return torch.sqrt(torch.sum((pred_pos - gt_pos) ** 2, dim=1))

def train_one_epoch(model, train_loader, optimizer, criterion_gdm, criterion_gsl, 
                   gdm_weight, gsl_weight, device):
    model.train()
    total_gdm_loss = 0
    total_gsl_loss = 0
    total_psnr = 0
    total_pos_error = 0
    n_samples = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        source_pos = batch['source_pos'].to(device)

        optimizer.zero_grad()
        gdm_out, gsl_out = model(lr)
        
        gdm_loss = criterion_gdm(gdm_out, hr)
        gsl_loss = criterion_gsl(gsl_out, source_pos)
        loss = gdm_weight * gdm_loss + gsl_weight * gsl_loss
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            psnr = calculate_psnr(gdm_out, hr)
            pos_error = calculate_position_error(gsl_out, source_pos)

        total_gdm_loss += gdm_loss.item()
        total_gsl_loss += gsl_loss.item()
        total_psnr += psnr.item()
        total_pos_error += pos_error.mean().item()
        n_samples += lr.size(0)

        pbar.set_postfix({
            'GDM Loss': f'{gdm_loss.item():.4f}',
            'GSL Loss': f'{gsl_loss.item():.4f}',
            'PSNR': f'{psnr.item():.2f}',
            'Pos Error': f'{pos_error.mean().item():.2f}'
        })

    return {
        'gdm_loss': total_gdm_loss / n_samples,
        'gsl_loss': total_gsl_loss / n_samples,
        'psnr': total_psnr / n_samples,
        'pos_error': total_pos_error / n_samples
    }

def validate(model, val_loader, criterion_gdm, criterion_gsl, device):
    model.eval()
    total_gdm_loss = 0
    total_gsl_loss = 0
    total_psnr = 0
    total_pos_error = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            source_pos = batch['source_pos'].to(device)

            gdm_out, gsl_out = model(lr)
            
            gdm_loss = criterion_gdm(gdm_out, hr)
            gsl_loss = criterion_gsl(gsl_out, source_pos)
            
            psnr = calculate_psnr(gdm_out, hr)
            pos_error = calculate_position_error(gsl_out, source_pos)

            total_gdm_loss += gdm_loss.item()
            total_gsl_loss += gsl_loss.item()
            total_psnr += psnr.item()
            total_pos_error += pos_error.mean().item()
            n_samples += lr.size(0)

    return {
        'gdm_loss': total_gdm_loss / n_samples,
        'gsl_loss': total_gsl_loss / n_samples,
        'psnr': total_psnr / n_samples,
        'pos_error': total_pos_error / n_samples
    }

def plot_metrics(train_metrics, val_metrics, save_dir):
    metrics = ['gdm_loss', 'gsl_loss', 'psnr', 'pos_error']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(train_metrics[metric], label='Train')
        ax.plot(val_metrics[metric], label='Validation')
        ax.set_title(metric.upper())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def main():
    args = parse_args()
    set_seed(42)

    # 创建保存目录
    save_dir = os.path.join(args.save_dir, f'{args.model_name}_{args.exp_name}')
    os.makedirs(save_dir, exist_ok=True)

    # 保存训练参数
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建数据集和数据加载器
    train_dataset, val_dataset = generate_train_valid_dataset(args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = net(
        img_size=args.patch_size,
        patch_size=1,
        in_chans=1,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        upscale=args.scale,
        img_range=1.,
        upsampler=args.upsampler,
        resi_connection='1conv'
    ).to(device)

    # 定义损失函数
    criterion_gdm = nn.L1Loss()
    criterion_gsl = nn.MSELoss()

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)

    # 训练循环
    best_val_psnr = 0
    train_metrics = {metric: [] for metric in ['gdm_loss', 'gsl_loss', 'psnr', 'pos_error']}
    val_metrics = {metric: [] for metric in ['gdm_loss', 'gsl_loss', 'psnr', 'pos_error']}

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_results = train_one_epoch(
            model, train_loader, optimizer, criterion_gdm, criterion_gsl,
            args.gdm_weight, args.gsl_weight, device
        )
        
        # 验证
        val_results = validate(model, val_loader, criterion_gdm, criterion_gsl, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        for metric in train_metrics:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])
        
        # 打印结果
        print(f'Train - GDM Loss: {train_results["gdm_loss"]:.4f}, GSL Loss: {train_results["gsl_loss"]:.4f}, '
              f'PSNR: {train_results["psnr"]:.2f}, Pos Error: {train_results["pos_error"]:.2f}')
        print(f'Val - GDM Loss: {val_results["gdm_loss"]:.4f}, GSL Loss: {val_results["gsl_loss"]:.4f}, '
              f'PSNR: {val_results["psnr"]:.2f}, Pos Error: {val_results["pos_error"]:.2f}')
        
        # 保存最佳模型
        if val_results['psnr'] > best_val_psnr:
            best_val_psnr = val_results['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_psnr': best_val_psnr,
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_psnr': best_val_psnr,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 绘制训练指标
    plot_metrics(train_metrics, val_metrics, save_dir)

if __name__ == '__main__':
    main()