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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_multi_enhanced_wind import SwinIRMultiEnhancedWind
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_dataset
import logging
import pandas as pd
import time
import shutil
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train SwinIR Multi-task Model')
    
    # 模型选择参数
    parser.add_argument('--model_type', type=str, default='enhanced',
                      choices=['original', 'enhanced', 'wind'],
                      help='选择模型类型: original, enhanced 或 wind')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                      help='path to the dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='number of epochs')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0002,
                      help='learning rate')
    parser.add_argument('--gdm_weight', type=float, default=1.0,
                      help='weight for GDM (super-resolution) task')
    parser.add_argument('--gsl_weight', type=float, default=0.5,
                      help='weight for GSL (source localization) task')
    
    args = parser.parse_args()
    return args

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

def create_model(args):
    """根据参数创建模型"""
    # 基础模型参数
    base_params = {
        'img_size': 16,  # LR图像大小
        'in_chans': 1,   # 输入通道数
        'upscale': 6,    # 上采样倍数
        'img_range': 1.,  # 图像范围
        'upsampler': 'nearest+conv'  # 上采样器类型
    }
    
    # 原始模型参数
    original_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer窗口大小
        'depths': [6, 6, 6, 6],  # Swin Transformer深度
        'embed_dim': 60,  # 嵌入维度
        'num_heads': [6, 6, 6, 6],  # 注意力头数
        'mlp_ratio': 2.,  # MLP比率
    }
    
    # 增强版模型参数
    enhanced_params = {
        **base_params,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'resi_connection': '1conv'
    }
    
    # 风场引导模型参数
    wind_params = {
        **base_params,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'resi_connection': '1conv'
    }
    
    if args.model_type == 'original':
        model = SwinIRMulti(**original_params)
    elif args.model_type == 'enhanced':
        model = SwinIRMultiEnhanced(**enhanced_params)
    else:  # wind
        model = SwinIRMultiEnhancedWind(**wind_params)
    
    return model

def plot_loss_lines(args, train_losses, valid_losses, train_gdm_losses, train_gsl_losses, 
                    valid_gdm_losses, valid_gsl_losses, save_dir):
    """绘制训练损失曲线"""
    plt.figure(figsize=(15, 10))
    
    # 总损失
    plt.subplot(221)
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(valid_losses, label='Valid Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # GDM损失
    plt.subplot(222)
    plt.plot(train_gdm_losses, label='Train GDM Loss')
    plt.plot(valid_gdm_losses, label='Valid GDM Loss')
    plt.title('GDM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # GSL损失
    plt.subplot(223)
    plt.plot(train_gsl_losses, label='Train GSL Loss')
    plt.plot(valid_gsl_losses, label='Valid GSL Loss')
    plt.title('GSL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def save_training_history(train_losses, valid_losses, train_gdm_losses, train_gsl_losses,
                         valid_gdm_losses, valid_gsl_losses, save_dir):
    """保存训练历史到CSV文件"""
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_total_loss': train_losses,
        'valid_total_loss': valid_losses,
        'train_gdm_loss': train_gdm_losses,
        'train_gsl_loss': train_gsl_losses,
        'valid_gdm_loss': valid_gdm_losses,
        'valid_gsl_loss': valid_gsl_losses
    })
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

def save_args(args, save_dir):
    """保存训练参数"""
    args_dict = vars(args)
    with open(os.path.join(save_dir, 'training_args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

def train_epoch(model, train_loader, criterion_gdm, criterion_gsl, optimizer, device, args):
    model.train()
    total_loss = 0
    gdm_loss_sum = 0
    gsl_loss_sum = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        position = batch['source_pos'].to(device)
        wind_vector = batch.get('wind_vector', None)
        if wind_vector is not None:
            wind_vector = wind_vector.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        gdm_out, gsl_pos, gsl_conf = model(lr, wind_vector)
        
        # 计算损失
        gdm_loss = criterion_gdm(gdm_out, hr)
        gsl_loss, gsl_conf_loss = criterion_gsl(gsl_pos, gsl_conf, position)
        
        # 总损失
        loss = args.gdm_weight * gdm_loss + args.gsl_weight * gsl_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新统计信息
        total_loss += loss.item()
        gdm_loss_sum += gdm_loss.item()
        gsl_loss_sum += gsl_loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'total_loss': f'{loss.item():.4f}',
            'gdm_loss': f'{gdm_loss.item():.4f}',
            'gsl_loss': f'{gsl_loss.item():.4f}'
        })
    
    return total_loss / len(train_loader), gdm_loss_sum / len(train_loader), gsl_loss_sum / len(train_loader)

def validate(model, valid_loader, criterion_gdm, criterion_gsl, device, args):
    model.eval()
    total_loss = 0
    gdm_loss_sum = 0
    gsl_loss_sum = 0
    
    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc='Validation')
        for batch in progress_bar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            position = batch['source_pos'].to(device)
            wind_vector = batch.get('wind_vector', None)
            if wind_vector is not None:
                wind_vector = wind_vector.to(device)
            
            # 前向传播
            gdm_out, gsl_pos, gsl_conf = model(lr, wind_vector)
            
            # 计算损失
            gdm_loss = criterion_gdm(gdm_out, hr)
            gsl_loss, gsl_conf_loss = criterion_gsl(gsl_pos, gsl_conf, position)
            
            # 总损失
            loss = args.gdm_weight * gdm_loss + args.gsl_weight * gsl_loss
            
            # 更新统计信息
            total_loss += loss.item()
            gdm_loss_sum += gdm_loss.item()
            gsl_loss_sum += gsl_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'total_loss': f'{loss.item():.4f}',
                'gdm_loss': f'{gdm_loss.item():.4f}',
                'gsl_loss': f'{gsl_loss.item():.4f}'
            })
    
    return total_loss / len(valid_loader), gdm_loss_sum / len(valid_loader), gsl_loss_sum / len(valid_loader)

def train_model(model, train_loader, valid_loader, args):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 定义损失函数
    gdm_criterion = nn.MSELoss()  # 超分辨率重建损失
    gsl_criterion = nn.MSELoss()  # 泄漏源定位损失
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 创建保存目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(args.save_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存训练参数
    save_args(args, experiment_dir)
    
    # 初始化训练历史记录
    train_losses = []
    valid_losses = []
    train_gdm_losses = []
    train_gsl_losses = []
    valid_gdm_losses = []
    valid_gsl_losses = []
    
    # 训练循环
    best_valid_loss = float('inf')
    start_epoch = 0
    
    # 检查是否有checkpoint可以恢复
    checkpoint_path = os.path.join(experiment_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        train_gdm_losses = checkpoint['train_gdm_losses']
        train_gsl_losses = checkpoint['train_gsl_losses']
        valid_gdm_losses = checkpoint['valid_gdm_losses']
        valid_gsl_losses = checkpoint['valid_gsl_losses']
        best_valid_loss = checkpoint['best_valid_loss']
        print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.num_epochs):
        # 训练阶段
        train_loss, train_gdm_loss, train_gsl_loss = train_epoch(model, train_loader, gdm_criterion, gsl_criterion, optimizer, device, args)
        
        # 验证阶段
        valid_loss, valid_gdm_loss, valid_gsl_loss = validate(model, valid_loader, gdm_criterion, gsl_criterion, device, args)
        
        # 更新学习率
        scheduler.step(valid_loss)
        
        # 记录损失
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_gdm_losses.append(train_gdm_loss)
        train_gsl_losses.append(train_gsl_loss)
        valid_gdm_losses.append(valid_gdm_loss)
        valid_gsl_losses.append(valid_gsl_loss)
        
        # 打印训练信息
        print(f'\nEpoch {epoch+1}/{args.num_epochs} Summary:')
        print(f'Train Loss: {train_loss:.4f} (GDM: {train_gdm_loss:.4f}, GSL: {train_gsl_loss:.4f})')
        print(f'Valid Loss: {valid_loss:.4f} (GDM: {valid_gdm_loss:.4f}, GSL: {valid_gsl_loss:.4f})')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Epoch {epoch+1} finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = os.path.join(experiment_dir, f'best_model_{args.model_type}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'valid_loss': valid_losses,
                'train_gdm_losses': train_gdm_losses,
                'train_gsl_losses': train_gsl_losses,
                'valid_gdm_losses': valid_gdm_losses,
                'valid_gsl_losses': valid_gsl_losses,
                'best_valid_loss': best_valid_loss,
                'args': args
            }, save_path)
            print(f'Best model saved to {save_path}')
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_gdm_losses': train_gdm_losses,
            'train_gsl_losses': train_gsl_losses,
            'valid_gdm_losses': valid_gdm_losses,
            'valid_gsl_losses': valid_gsl_losses,
            'best_valid_loss': best_valid_loss,
            'args': args
        }
        torch.save(checkpoint, os.path.join(experiment_dir, "latest_checkpoint.pth"))
        
        # 每10个epoch保存一次历史checkpoint
        if (epoch + 1) % 10 == 0:
            history_checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, history_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # 绘制损失曲线
        plot_loss_lines(args, train_losses, valid_losses, 
                       train_gdm_losses, train_gsl_losses,
                       valid_gdm_losses, valid_gsl_losses, 
                       experiment_dir)
        
        # 保存训练历史
        save_training_history(train_losses, valid_losses,
                            train_gdm_losses, train_gsl_losses,
                            valid_gdm_losses, valid_gsl_losses,
                            experiment_dir)
    
    # 训练完成后，保存一份最佳模型的副本
    best_model_path = os.path.join(experiment_dir, f'best_model_{args.model_type}.pth')
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(args.save_dir, f"{args.model_type}_best_model.pth"))
        print(f"最佳模型副本已保存至: {os.path.join(args.save_dir, f'{args.model_type}_best_model.pth')}")
    
    print("Training completed!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Model and checkpoints saved in {experiment_dir}")
    
    return model, train_losses, valid_losses, best_valid_loss

def train(args):
    logging.basicConfig(level=logging.INFO)
    print(f"Starting training with arguments: {args}")
    print(f"Using multi-task learning with GDM and GSL")

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
    model = create_model(args)

    # 训练模型
    model, train_losses, valid_losses, best_valid_loss = train_model(model, train_loader, valid_loader, args)

def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset, valid_dataset = generate_train_valid_dataset(args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = create_model(args)
    
    # 训练模型
    train(args)

if __name__ == "__main__":
    main()
    