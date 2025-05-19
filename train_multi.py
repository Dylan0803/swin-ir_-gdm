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
import logging
import pandas as pd
import time
import shutil

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

def train_one_epoch(model, dataloader, gdm_criterion, gsl_criterion, optimizer, device, epoch, total_epochs):
    model.train()
    epoch_gdm_loss = 0.0
    epoch_gsl_loss = 0.0
    epoch_total_loss = 0.0
    
    tbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for batch in tbar:
        try:
            # 数据预处理
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            source_pos = batch['source_pos'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            gdm_out, gsl_out = model(lr)
            
            # 计算损失
            gdm_loss = gdm_criterion(gdm_out, hr)
            gsl_loss = gsl_criterion(gsl_out, source_pos)
            total_loss = args.gdm_weight * gdm_loss + args.gsl_weight * gsl_loss
            
            # 检查损失值
            if torch.isnan(total_loss):
                print(f"Warning: NaN loss detected! Skipping batch...")
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新统计信息
            epoch_gdm_loss += gdm_loss.item()
            epoch_gsl_loss += gsl_loss.item()
            epoch_total_loss += total_loss.item()
            
            # 更新进度条
            tbar.set_postfix({
                'GDM Loss': f"{gdm_loss.item():.4f}",
                'GSL Loss': f"{gsl_loss.item():.4f}",
                'Total Loss': f"{total_loss.item():.4f}"
            })
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Warning: GPU OOM! Clearing cache and skipping batch...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return epoch_gdm_loss / len(dataloader), epoch_gsl_loss / len(dataloader), epoch_total_loss / len(dataloader)

@torch.no_grad()
def valid_one_epoch(model, dataloader, gdm_criterion, gsl_criterion, device):
    model.eval()
    epoch_gdm_loss = 0.0
    epoch_gsl_loss = 0.0
    epoch_total_loss = 0.0
    
    for batch in dataloader:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        source_pos = batch['source_pos'].to(device)
        
        gdm_out, gsl_out = model(lr)
        
        gdm_loss = gdm_criterion(gdm_out, hr)
        gsl_loss = gsl_criterion(gsl_out, source_pos)
        total_loss = args.gdm_weight * gdm_loss + args.gsl_weight * gsl_loss
        
        epoch_gdm_loss += gdm_loss.item()
        epoch_gsl_loss += gsl_loss.item()
        epoch_total_loss += total_loss.item()
    
    return epoch_gdm_loss / len(dataloader), epoch_gsl_loss / len(dataloader), epoch_total_loss / len(dataloader)

def plot_metrics(args, train_metrics, valid_metrics, save_dir):
    """绘制训练指标曲线"""
    plt.figure(figsize=(15, 5))
    
    # GDM损失
    plt.subplot(131)
    plt.plot(train_metrics['gdm_loss'], label='Train GDM Loss')
    plt.plot(valid_metrics['gdm_loss'], label='Valid GDM Loss')
    plt.title('GDM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # GSL损失
    plt.subplot(132)
    plt.plot(train_metrics['gsl_loss'], label='Train GSL Loss')
    plt.plot(valid_metrics['gsl_loss'], label='Valid GSL Loss')
    plt.title('GSL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 总损失
    plt.subplot(133)
    plt.plot(train_metrics['total_loss'], label='Train Total Loss')
    plt.plot(valid_metrics['total_loss'], label='Valid Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def save_training_history(train_metrics, valid_metrics, save_dir):
    """保存训练历史到CSV文件"""
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_metrics['gdm_loss']) + 1),
        'train_gdm_loss': train_metrics['gdm_loss'],
        'train_gsl_loss': train_metrics['gsl_loss'],
        'train_total_loss': train_metrics['total_loss'],
        'valid_gdm_loss': valid_metrics['gdm_loss'],
        'valid_gsl_loss': valid_metrics['gsl_loss'],
        'valid_total_loss': valid_metrics['total_loss']
    })
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

def save_args(args, model_dir):
    # 实现保存参数的逻辑
    pass

def train(args):
    logging.basicConfig(level=logging.INFO)
    print(f"Starting training with arguments: {args}")
    print(f"Using multi-task learning with GDM and GSL")
    print(f"Upsampler mode: {args.upsampler}")

    # CUDA设置和内存管理
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    print(f"Using device: {device}")

    # 数据集加载和打印信息
    train_set, valid_set = generate_train_valid_dataset(
        args.data_path, train_ratio=0.8, shuffle=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # 打印数据集信息
    sample = next(iter(train_loader))
    print(f"Data shapes:")
    print(f"LR: {sample['lr'].shape}")
    print(f"HR: {sample['hr'].shape}")
    print(f"Source position: {sample['source_pos'].shape}")
    print(f"HR max position: {sample['hr_max_pos'].shape}")
    print(f"Data ranges:")
    print(f"LR: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
    print(f"HR: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")
    print(f"Source position: [{sample['source_pos'].min():.4f}, {sample['source_pos'].max():.4f}]")
    print(f"HR max position: [{sample['hr_max_pos'].min():.4f}, {sample['hr_max_pos'].max():.4f}]")

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

    # 创建保存目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join("experiments", f"{args.exp_name}_{timestamp}")
    model_dir = os.path.join(experiment_dir, args.model_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存训练参数
    save_args(args, model_dir)
    
    # 初始化训练历史
    train_metrics = {'gdm_loss': [], 'gsl_loss': [], 'total_loss': []}
    valid_metrics = {'gdm_loss': [], 'gsl_loss': [], 'total_loss': []}
    best_loss = float('inf')
    start_epoch = 0
    
    # 检查是否有checkpoint
    checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_metrics = checkpoint['train_metrics']
        valid_metrics = checkpoint['valid_metrics']
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_gdm_loss, train_gsl_loss, train_total_loss = train_one_epoch(
            model, train_loader, criterion_gdm, criterion_gsl, optimizer, device, epoch, args.epochs)
        
        # 验证
        valid_gdm_loss, valid_gsl_loss, valid_total_loss = valid_one_epoch(
            model, valid_loader, criterion_gdm, criterion_gsl, device)
        
        # 更新训练历史
        train_metrics['gdm_loss'].append(train_gdm_loss)
        train_metrics['gsl_loss'].append(train_gsl_loss)
        train_metrics['total_loss'].append(train_total_loss)
        valid_metrics['gdm_loss'].append(valid_gdm_loss)
        valid_metrics['gsl_loss'].append(valid_gsl_loss)
        valid_metrics['total_loss'].append(valid_total_loss)
        
        # 打印训练信息
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train - GDM Loss: {train_gdm_loss:.4f}, GSL Loss: {train_gsl_loss:.4f}, Total Loss: {train_total_loss:.4f}")
        print(f"Valid - GDM Loss: {valid_gdm_loss:.4f}, GSL Loss: {valid_gsl_loss:.4f}, Total Loss: {valid_total_loss:.4f}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 保存最佳模型
        if valid_total_loss < best_loss:
            best_loss = valid_total_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(f"Best model saved at epoch {epoch+1} with valid loss {best_loss:.4f}")
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'best_loss': best_loss,
            'args': args
        }
        torch.save(checkpoint, os.path.join(model_dir, "latest_checkpoint.pth"))
        
        # 每10个epoch保存一次历史checkpoint
        if (epoch + 1) % 10 == 0:
            history_checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, history_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # 绘制和保存训练曲线
        plot_metrics(args, train_metrics, valid_metrics, model_dir)
        save_training_history(train_metrics, valid_metrics, model_dir)
    
    # 保存最佳模型副本
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(experiment_dir, f"{args.model_name}_best_model.pth"))
        print(f"Best model copy saved to: {os.path.join(experiment_dir, f'{args.model_name}_best_model.pth')}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model and checkpoints saved in {model_dir}")
    
    return model, train_metrics, valid_metrics, best_loss

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    
    # 模型相关参数
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    parser.add_argument('--exp_name', type=str, default='default_experiment', help='实验名称')
    parser.add_argument('--upsampler', type=str, default='nearest+conv', 
                       choices=['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'],
                       help='上采样器类型')
    parser.add_argument('--scale', type=int, default=6, help='上采样倍数')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=True, help='h5 数据路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--patch_size', type=int, default=16, help='LR图像的patch大小')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--gdm_weight', type=float, default=1.0, help='GDM任务损失权重')
    parser.add_argument('--gsl_weight', type=float, default=0.5, help='GSL任务损失权重')
    
    # 解析参数
    args = parser.parse_args()
    
    # 开始训练
    train(args)

if __name__ == "__main__":
    main()
    