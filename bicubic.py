import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.h5_dataset import MultiTaskDataset
import pandas as pd
import h5py
import argparse
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import sys
import json
from argparse import Namespace
from scipy.interpolate import griddata
from scipy.ndimage import zoom

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class BicubicInterpolation:
    """
    双三次插值算法实现气体分布重建
    """

    def __init__(self, upscale_factor=6):
        """
        初始化双三次插值器
        
        参数:
            upscale_factor: 上采样倍数，默认6
        """
        self.upscale_factor = upscale_factor

    def interpolate(self, lr_data):
        """
        使用双三次插值重建高分辨率气体分布
        
        参数:
            lr_data: 低分辨率数据 [batch_size, 1, H, W] 或 [1, H, W]
            
        返回:
            hr_data: 高分辨率数据 [batch_size, 1, H*upscale, W*upscale] 或 [1, H*upscale, W*upscale]
        """
        if len(lr_data.shape) == 4:
            # 批处理模式
            batch_size = lr_data.shape[0]
            hr_results = []

            for i in range(batch_size):
                hr_result = self._single_interpolate(lr_data[i])
                hr_results.append(hr_result)

            return torch.stack(hr_results, dim=0)
        else:
            # 单样本模式
            return self._single_interpolate(lr_data)

    def _single_interpolate(self, lr_data):
        """
        对单个样本进行双三次插值
        
        参数:
            lr_data: 低分辨率数据 [1, H, W]
            
        返回:
            hr_data: 高分辨率数据 [1, H*upscale, W*upscale]
        """
        # 转换为numpy数组并移除通道维度
        lr_np = lr_data.squeeze().cpu().numpy()

        # 获取原始尺寸
        h, w = lr_np.shape

        # 计算目标尺寸
        target_h = h * self.upscale_factor
        target_w = w * self.upscale_factor

        # 创建目标网格
        x_old = np.linspace(0, 1, w)
        y_old = np.linspace(0, 1, h)
        x_new = np.linspace(0, 1, target_w)
        y_new = np.linspace(0, 1, target_h)

        # 创建网格点
        X_old, Y_old = np.meshgrid(x_old, y_old)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        # 使用scipy的griddata进行双三次插值
        points = np.column_stack((X_old.flatten(), Y_old.flatten()))
        values = lr_np.flatten()

        # 双三次插值
        hr_np = griddata(points, values, (X_new, Y_new),
                         method='cubic', fill_value=0)

        # 处理边界和异常值
        hr_np = np.clip(hr_np, 0, 1)

        # 转换回tensor并添加通道维度
        hr_tensor = torch.from_numpy(hr_np).float().unsqueeze(0)

        return hr_tensor


def test_dataset_loading(data_path, num_samples=3):
    """
    测试数据集加载功能
    
    参数:
        data_path: 数据集路径
        num_samples: 测试样本数量
    """
    print(f"=== 测试数据集加载 ===")
    print(f"数据集路径: {data_path}")

    try:
        # 创建数据集
        dataset = MultiTaskDataset(data_path)
        print(f"数据集大小: {len(dataset)} 个样本")

        # 测试加载几个样本
        for i in range(min(num_samples, len(dataset))):
            print(f"\n--- 样本 {i+1} ---")
            sample = dataset[i]

            # 打印数据形状和基本信息
            print(f"LR形状: {sample['lr'].shape}")
            print(f"HR形状: {sample['hr'].shape}")
            print(
                f"LR值范围: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
            print(
                f"HR值范围: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")

            # 如果有源位置信息
            if 'source_pos' in sample:
                print(f"源位置: {sample['source_pos']}")
            if 'hr_max_pos' in sample:
                print(f"HR最大值位置: {sample['hr_max_pos']}")

            # 可视化第一个样本
            if i == 0:
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(sample['lr'].squeeze().numpy(), cmap='viridis')
                plt.title('Low Resolution Input')
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.imshow(sample['hr'].squeeze().numpy(), cmap='viridis')
                plt.title('High Resolution Ground Truth')
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.hist(sample['lr'].squeeze().numpy().flatten(),
                         bins=50, alpha=0.7, label='LR')
                plt.hist(sample['hr'].squeeze().numpy().flatten(),
                         bins=50, alpha=0.7, label='HR')
                plt.title('Value Distribution')
                plt.legend()

                plt.tight_layout()
                plt.savefig('dataset_test_visualization.png',
                            dpi=150, bbox_inches='tight')
                plt.show()
                print("可视化结果已保存为 'dataset_test_visualization.png'")

        return True

    except Exception as e:
        print(f"数据集加载失败: {e}")
        return False


def test_bicubic_interpolation(data_path, num_samples=2):
    """
    测试双三次插值算法
    
    参数:
        data_path: 数据集路径
        num_samples: 测试样本数量
    """
    print(f"\n=== 测试双三次插值算法 ===")

    try:
        # 创建数据集和数据加载器
        dataset = MultiTaskDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # 创建双三次插值器
        bicubic_interpolator = BicubicInterpolation(upscale_factor=6)

        # 获取一个批次的数据
        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        print(f"输入LR形状: {lr.shape}")
        print(f"真实HR形状: {hr_true.shape}")

        # 进行双三次插值
        hr_pred = bicubic_interpolator.interpolate(lr)
        print(f"预测HR形状: {hr_pred.shape}")

        # 计算评估指标
        for i in range(lr.size(0)):
            print(f"\n--- 样本 {i+1} 评估结果 ---")

            # 转换为numpy进行评估
            pred_np = hr_pred[i].squeeze().numpy()
            true_np = hr_true[i].squeeze().numpy()

            # PSNR
            mse = np.mean((pred_np - true_np) ** 2)
            if mse > 0:
                psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
            else:
                psnr_val = float('inf')

            # SSIM
            ssim_val = ssim(pred_np, true_np, data_range=1.0)

            print(f"PSNR: {psnr_val:.2f} dB")
            print(f"SSIM: {ssim_val:.4f}")
            print(f"MSE: {mse:.6f}")

            # 可视化结果
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 4, 1)
            plt.imshow(lr[i].squeeze().numpy(), cmap='viridis')
            plt.title('Low Resolution Input')
            plt.colorbar()

            plt.subplot(1, 4, 2)
            plt.imshow(hr_true[i].squeeze().numpy(), cmap='viridis')
            plt.title('Ground Truth HR')
            plt.colorbar()

            plt.subplot(1, 4, 3)
            plt.imshow(pred_np, cmap='viridis')
            plt.title(f'Bicubic Prediction\nPSNR: {psnr_val:.2f}')
            plt.colorbar()

            plt.subplot(1, 4, 4)
            diff = np.abs(pred_np - true_np)
            plt.imshow(diff, cmap='hot')
            plt.title('Absolute Difference')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(
                f'bicubic_test_sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.show()
            print(f"可视化结果已保存为 'bicubic_test_sample_{i+1}.png'")

        return True

    except Exception as e:
        print(f"双三次插值测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Test Dataset Loading and Bicubic Interpolation')

    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径 (.h5文件)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='测试样本数量')

    args = parser.parse_args()
    return args


def main():
    """主函数 - 测试数据集加载和双三次插值"""
    args = parse_args()

    print("开始测试数据集加载和双三次插值算法...")
    print("=" * 50)

    # 测试数据集加载
    dataset_ok = test_dataset_loading(args.data_path, args.num_samples)

    if dataset_ok:
        # 测试双三次插值
        bicubic_ok = test_bicubic_interpolation(
            args.data_path, min(args.num_samples, 2))

        if bicubic_ok:
            print("\n" + "=" * 50)
            print("? 所有测试通过！")
            print("数据集可以正确加载，双三次插值算法工作正常。")
        else:
            print("\n? 双三次插值测试失败")
    else:
        print("\n? 数据集加载测试失败")


if __name__ == '__main__':
    main()
