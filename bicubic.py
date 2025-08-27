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

# �����Ŀ��Ŀ¼��Python·��
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class BicubicInterpolation:
    """
    ˫���β�ֵ�㷨ʵ������ֲ��ؽ�
    """

    def __init__(self, upscale_factor=6):
        """
        ��ʼ��˫���β�ֵ��
        
        ����:
            upscale_factor: �ϲ���������Ĭ��6
        """
        self.upscale_factor = upscale_factor

    def interpolate(self, lr_data):
        """
        ʹ��˫���β�ֵ�ؽ��߷ֱ�������ֲ�
        
        ����:
            lr_data: �ͷֱ������� [batch_size, 1, H, W] �� [1, H, W]
            
        ����:
            hr_data: �߷ֱ������� [batch_size, 1, H*upscale, W*upscale] �� [1, H*upscale, W*upscale]
        """
        if len(lr_data.shape) == 4:
            # ������ģʽ
            batch_size = lr_data.shape[0]
            hr_results = []

            for i in range(batch_size):
                hr_result = self._single_interpolate(lr_data[i])
                hr_results.append(hr_result)

            return torch.stack(hr_results, dim=0)
        else:
            # ������ģʽ
            return self._single_interpolate(lr_data)

    def _single_interpolate(self, lr_data):
        """
        �Ե�����������˫���β�ֵ
        
        ����:
            lr_data: �ͷֱ������� [1, H, W]
            
        ����:
            hr_data: �߷ֱ������� [1, H*upscale, W*upscale]
        """
        # ת��Ϊnumpy���鲢�Ƴ�ͨ��ά��
        lr_np = lr_data.squeeze().cpu().numpy()

        # ��ȡԭʼ�ߴ�
        h, w = lr_np.shape

        # ����Ŀ��ߴ�
        target_h = h * self.upscale_factor
        target_w = w * self.upscale_factor

        # ����Ŀ������
        x_old = np.linspace(0, 1, w)
        y_old = np.linspace(0, 1, h)
        x_new = np.linspace(0, 1, target_w)
        y_new = np.linspace(0, 1, target_h)

        # ���������
        X_old, Y_old = np.meshgrid(x_old, y_old)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        # ʹ��scipy��griddata����˫���β�ֵ
        points = np.column_stack((X_old.flatten(), Y_old.flatten()))
        values = lr_np.flatten()

        # ˫���β�ֵ
        hr_np = griddata(points, values, (X_new, Y_new),
                         method='cubic', fill_value=0)

        # ����߽���쳣ֵ
        hr_np = np.clip(hr_np, 0, 1)

        # ת����tensor�����ͨ��ά��
        hr_tensor = torch.from_numpy(hr_np).float().unsqueeze(0)

        return hr_tensor


def test_dataset_loading(data_path, num_samples=3):
    """
    �������ݼ����ع���
    
    ����:
        data_path: ���ݼ�·��
        num_samples: ������������
    """
    print(f"=== �������ݼ����� ===")
    print(f"���ݼ�·��: {data_path}")

    try:
        # �������ݼ�
        dataset = MultiTaskDataset(data_path)
        print(f"���ݼ���С: {len(dataset)} ������")

        # ���Լ��ؼ�������
        for i in range(min(num_samples, len(dataset))):
            print(f"\n--- ���� {i+1} ---")
            sample = dataset[i]

            # ��ӡ������״�ͻ�����Ϣ
            print(f"LR��״: {sample['lr'].shape}")
            print(f"HR��״: {sample['hr'].shape}")
            print(
                f"LRֵ��Χ: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
            print(
                f"HRֵ��Χ: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")

            # �����Դλ����Ϣ
            if 'source_pos' in sample:
                print(f"Դλ��: {sample['source_pos']}")
            if 'hr_max_pos' in sample:
                print(f"HR���ֵλ��: {sample['hr_max_pos']}")

            # ���ӻ���һ������
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
                print("���ӻ�����ѱ���Ϊ 'dataset_test_visualization.png'")

        return True

    except Exception as e:
        print(f"���ݼ�����ʧ��: {e}")
        return False


def test_bicubic_interpolation(data_path, num_samples=2):
    """
    ����˫���β�ֵ�㷨
    
    ����:
        data_path: ���ݼ�·��
        num_samples: ������������
    """
    print(f"\n=== ����˫���β�ֵ�㷨 ===")

    try:
        # �������ݼ������ݼ�����
        dataset = MultiTaskDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # ����˫���β�ֵ��
        bicubic_interpolator = BicubicInterpolation(upscale_factor=6)

        # ��ȡһ�����ε�����
        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        print(f"����LR��״: {lr.shape}")
        print(f"��ʵHR��״: {hr_true.shape}")

        # ����˫���β�ֵ
        hr_pred = bicubic_interpolator.interpolate(lr)
        print(f"Ԥ��HR��״: {hr_pred.shape}")

        # ��������ָ��
        for i in range(lr.size(0)):
            print(f"\n--- ���� {i+1} ������� ---")

            # ת��Ϊnumpy��������
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

            # ���ӻ����
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
            print(f"���ӻ�����ѱ���Ϊ 'bicubic_test_sample_{i+1}.png'")

        return True

    except Exception as e:
        print(f"˫���β�ֵ����ʧ��: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """���������в���"""
    parser = argparse.ArgumentParser(
        description='Test Dataset Loading and Bicubic Interpolation')

    parser.add_argument('--data_path', type=str, required=True,
                        help='���ݼ�·�� (.h5�ļ�)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='������������')

    args = parser.parse_args()
    return args


def main():
    """������ - �������ݼ����غ�˫���β�ֵ"""
    args = parse_args()

    print("��ʼ�������ݼ����غ�˫���β�ֵ�㷨...")
    print("=" * 50)

    # �������ݼ�����
    dataset_ok = test_dataset_loading(args.data_path, args.num_samples)

    if dataset_ok:
        # ����˫���β�ֵ
        bicubic_ok = test_bicubic_interpolation(
            args.data_path, min(args.num_samples, 2))

        if bicubic_ok:
            print("\n" + "=" * 50)
            print("? ���в���ͨ����")
            print("���ݼ�������ȷ���أ�˫���β�ֵ�㷨����������")
        else:
            print("\n? ˫���β�ֵ����ʧ��")
    else:
        print("\n? ���ݼ����ز���ʧ��")


if __name__ == '__main__':
    main()
