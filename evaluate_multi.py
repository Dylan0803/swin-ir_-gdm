import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_multi_enhanced_v2 import SwinIRMultiEnhancedV2
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_dataset
import pandas as pd
import h5py
import argparse
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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
            try:
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
                # 反归一化坐标（乘以95.0）
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0
                
                # 计算反归一化后的距离并除以10
                position_error = torch.sqrt(torch.sum((pred_pos - true_pos) ** 2, dim=1)) / 10.0
                max_pos_error = torch.sqrt(torch.sum((pred_pos - hr_max_pos * 95.0) ** 2, dim=1)) / 10.0
                
                metrics['gsl_position_error'].extend(position_error.cpu().numpy())
                metrics['gsl_max_pos_error'].extend(max_pos_error.cpu().numpy())
                
            except KeyError as e:
                print(f"跳过无效数据: {e}")
                continue
    
    # 计算平均指标
    results = {
        'GDM_PSNR': np.mean(metrics['gdm_psnr']) if metrics['gdm_psnr'] else 0,
        'GDM_SSIM': np.mean(metrics['gdm_ssim']) if metrics['gdm_ssim'] else 0,
        'GSL_Position_Error': np.mean(metrics['gsl_position_error']) if metrics['gsl_position_error'] else 0,
        'GSL_MaxPos_Error': np.mean(metrics['gsl_max_pos_error']) if metrics['gsl_max_pos_error'] else 0
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
    """可视化单个样本的结果"""
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
    
    # 在SR图像上标记泄漏源位置
    # 反归一化坐标（乘以95.0）
    true_pos = source_pos * 95.0
    pred_pos = gsl_out * 95.0
    
    # 标记真实泄漏源位置（红色星形）
    axes[1, 0].plot(true_pos[0], true_pos[1], 'r*', markersize=15, label='True Source')
    
    # 标记预测泄漏源位置（绿色星形）
    axes[1, 0].plot(pred_pos[0], pred_pos[1], 'g*', markersize=15, label='Predicted Source')
    
    # 添加图例
    axes[1, 0].legend(loc='upper right')
    
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
    
    # 计算反归一化后的距离并除以10
    distance = np.sqrt(np.sum((pred_pos - true_pos) ** 2)) / 10.0
    return distance

def infer_model(model, data_path, save_dir, num_samples=5, use_valid=True):
    """
    模型推理函数
    
    参数:
        model: 训练好的模型
        data_path: 数据路径
        save_dir: 保存结果的目录
        num_samples: 要评估的样本数量
        use_valid: 是否使用验证集
    """
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
    total_mse = 0
    total_ssim = 0
    valid_samples = 0
    
    # 处理指定数量的样本
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        try:
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
            # MSE损失
            mse = F.mse_loss(gdm_out, hr)
            
            # PSNR
            psnr = 10 * torch.log10(1.0 / mse)
            
            # SSIM计算
            c1 = (0.01 * 1.0) ** 2
            c2 = (0.03 * 1.0) ** 2
            mu1 = torch.mean(gdm_out)
            mu2 = torch.mean(hr)
            sigma1 = torch.var(gdm_out)
            sigma2 = torch.var(hr)
            sigma12 = torch.mean((gdm_out - mu1) * (hr - mu2))
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            # 反归一化坐标（乘以95.0）
            true_pos = source_pos * 95.0
            pred_pos = gsl_out * 95.0
            
            # 计算反归一化后的直线距离并除以10（转换为米）
            position_error = torch.sqrt(torch.sum((pred_pos - true_pos) ** 2)) / 10.0
            
            total_psnr += psnr.item()
            total_position_error += position_error.item()
            total_mse += mse.item()
            total_ssim += ssim.item()
            valid_samples += 1
            
            # 保存结果
            save_path = os.path.join(save_dir, f'sample_{i+1}.png')
            visualize_results(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_path)
            
        except KeyError as e:
            print(f"跳过无效数据: {e}")
            continue
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue
    
    # 计算平均指标
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_position_error = total_position_error / valid_samples
        avg_mse = total_mse / valid_samples
        avg_ssim = total_ssim / valid_samples
        
        print(f"\n评估结果:")
        print(f"有效样本数: {valid_samples}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average Position Error: {avg_position_error:.4f} m")
    else:
        print("没有有效的样本可供评估")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SwinIR Multi-task Model')
    
    # 添加模型类型选择参数
    parser.add_argument('--model_type', type=str, default='original',
                      choices=['original', 'enhanced', 'enhanced_v2'],
                      help='选择模型类型: original, enhanced, or enhanced_v2')
    
    # 添加缺失的参数
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型权重路径')
    parser.add_argument('--data_path', type=str, required=True,
                      help='数据集路径')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                      help='评估结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                      help='使用的设备')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='要评估的样本数量')
    
    args = parser.parse_args()
    return args

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
        'mlp_ratio': 2.,  # MLP比率,
    }
    
    # 增强版模型参数
    enhanced_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer窗口大小
        'depths': [6, 6, 6, 6],  # Swin Transformer深度
        'embed_dim': 60,  # 嵌入维度
        'num_heads': [6, 6, 6, 6],  # 注意力头数
        'mlp_ratio': 2.,  # MLP比率
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
    
    # V2模型特定参数
    enhanced_v2_params = {
        **enhanced_params,
        'upsampler': 'pixelshuffle',  # 使用pixelshuffle上采样
    }
    
    if args.model_type == 'original':
        model = SwinIRMulti(**original_params)
    elif args.model_type == 'enhanced':
        model = SwinIRMultiEnhanced(**enhanced_params)
    else:  # enhanced_v2
        model = SwinIRMultiEnhancedV2(**enhanced_v2_params)
    
    return model

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 创建模型
    model = create_model(args)
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    try:
        # 首先尝试使用 weights_only=True 加载
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=True: {e}")
        print("Attempting to load with weights_only=False...")
        # 如果失败，则使用 weights_only=False 加载
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
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
    infer_model(model, args.data_path, args.save_dir, args.num_samples)

if __name__ == '__main__':
    main()