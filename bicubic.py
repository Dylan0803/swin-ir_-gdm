# -*- coding: utf-8 -*-
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
from scipy.ndimage import zoom, gaussian_filter
from scipy.signal import savgol_filter

# �����Ŀ��Ŀ¼��Python·��
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class AdvancedBicubicInterpolation:
    """
    �߼�˫���β�ֵ�㷨��֧�ֶ��ֿɵ������Ż�
    """

    def __init__(self, upscale_factor=6, **kwargs):
        """
        ��ʼ���߼�˫���β�ֵ��

        ����:
            upscale_factor: �ϲ���������Ĭ��6
            **kwargs: ������Ż�����
        """
        self.upscale_factor = upscale_factor

        # ��ֵ����
        # 'linear', 'nearest', 'cubic', 'quintic'
        self.method = kwargs.get('method', 'cubic')

        # �߽紦��
        self.fill_value = kwargs.get('fill_value', 0)
        # 'constant', 'reflect', 'wrap'
        self.boundary_mode = kwargs.get('boundary_mode', 'constant')

        # �������
        self.grid_range = kwargs.get('grid_range', [0, 1])  # [��Сֵ, ���ֵ]
        self.grid_density = kwargs.get(
            'grid_density', 'uniform')  # 'uniform', 'log', 'exp'

        # �������
        self.clip_range = kwargs.get('clip_range', [0, 1])  # [��Сֵ, ���ֵ]
        self.smooth_sigma = kwargs.get('smooth_sigma', 0)  # ��˹ƽ����sigmaֵ
        self.savgol_window = kwargs.get(
            'savgol_window', None)  # Savitzky-Golay�˲������ڴ�С
        self.savgol_polyorder = kwargs.get(
            'savgol_polyorder', 2)  # Savitzky-Golay����ʽ����

        # ��Ե��ǿ
        self.edge_enhance = kwargs.get('edge_enhance', False)
        self.edge_strength = kwargs.get('edge_strength', 0.1)

        print(f"��ʼ���߼�˫���β�ֵ������������:")
        print(f"  ��ֵ����: {self.method}")
        print(f"  ���ֵ: {self.fill_value}")
        print(f"  ����Χ: {self.grid_range}")
        print(f"  ƽ��sigma: {self.smooth_sigma}")
        print(f"  ��Ե��ǿ: {self.edge_enhance}")

    def interpolate(self, lr_data):
        """
        ʹ�ø߼���ֵ�㷨�ؽ��߷ֱ�������ֲ�

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
        �Ե����������и߼���ֵ

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

        # �����ܶ����ô�������
        if self.grid_density == 'uniform':
            # ��������
            x_old = np.linspace(self.grid_range[0], self.grid_range[1], w)
            y_old = np.linspace(self.grid_range[0], self.grid_range[1], h)
            x_new = np.linspace(
                self.grid_range[0], self.grid_range[1], target_w)
            y_new = np.linspace(
                self.grid_range[0], self.grid_range[1], target_h)
        elif self.grid_density == 'log':
            # ���������ʺ�ָ���ֲ�����
            x_old = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), w)
            y_old = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), h)
            x_new = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), target_w)
            y_new = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), target_h)
        else:  # exponential
            # ָ������
            x_old = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), w))
            y_old = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), h))
            x_new = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), target_w))
            y_new = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), target_h))

        # ���������
        X_old, Y_old = np.meshgrid(x_old, y_old)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        # ʹ��scipy griddata���в�ֵ
        points = np.column_stack((X_old.flatten(), Y_old.flatten()))
        values = lr_np.flatten()

        # ��ֵ
        hr_np = griddata(points, values, (X_new, Y_new),
                         method=self.method, fill_value=self.fill_value)

        # ����NaNֵ
        if np.any(np.isnan(hr_np)):
            hr_np = np.nan_to_num(hr_np, nan=self.fill_value)

        # ����
        hr_np = self._post_process(hr_np)

        # ת����tensor�����ͨ��ά��
        hr_tensor = torch.from_numpy(hr_np).float().unsqueeze(0)

        return hr_tensor

    def _post_process(self, hr_np):
        """
        �Բ�ֵ������к���

        ����:
            hr_np: ��ֵ���numpy����

        ����:
            ������numpy����
        """
        # ֵ������
        hr_np = np.clip(hr_np, self.clip_range[0], self.clip_range[1])

        # ��˹ƽ��
        if self.smooth_sigma > 0:
            hr_np = gaussian_filter(hr_np, sigma=self.smooth_sigma)

        # Savitzky-Golay�˲�����1D���ݣ�Ӧ����ÿ��/�У�
        if self.savgol_window is not None:
            try:
                # ����Ӧ���˲�
                for i in range(hr_np.shape[0]):
                    hr_np[i, :] = savgol_filter(
                        hr_np[i, :], self.savgol_window, self.savgol_polyorder)
                # ����Ӧ���˲�
                for j in range(hr_np.shape[1]):
                    hr_np[:, j] = savgol_filter(
                        hr_np[:, j], self.savgol_window, self.savgol_polyorder)
            except:
                pass  # ������ڴ�С̫��������

        # ��Ե��ǿ
        if self.edge_enhance:
            # ʹ���ݶȽ��м򵥵ı�Ե��ǿ
            grad_x = np.gradient(hr_np, axis=1)
            grad_y = np.gradient(hr_np, axis=0)
            edge_strength_map = np.sqrt(grad_x**2 + grad_y**2)
            hr_np = hr_np + self.edge_strength * edge_strength_map
            hr_np = np.clip(hr_np, self.clip_range[0], self.clip_range[1])

        return hr_np


def test_parameter_optimization(data_path, num_samples=2):
    """
    ���Բ�ͬ������ϵ��Ż�Ч��

    ����:
        data_path: ���ݼ�·��
        num_samples: ������������
    """
    print(f"\n=== ���Բ����Ż� ===")

    try:
        # �������ݼ������ݼ�����
        dataset = MultiTaskDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # ��ȡһ�����ε�����
        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        # ����Ҫ���ԵĲ������
        param_combinations = [
            {
                'name': '����˫���β�ֵ',
                'params': {'method': 'cubic', 'fill_value': 0}
            },
            {
                'name': '���Բ�ֵ',
                'params': {'method': 'linear', 'fill_value': 0}
            },
            {
                'name': '����ڲ�ֵ',
                'params': {'method': 'nearest', 'fill_value': 0}
            },
            {
                'name': '˫���β�ֵ+ƽ��',
                'params': {'method': 'cubic', 'fill_value': 0, 'smooth_sigma': 0.5}
            },
            {
                'name': '˫���β�ֵ+��Ե��ǿ',
                'params': {'method': 'cubic', 'fill_value': 0, 'edge_enhance': True, 'edge_strength': 0.1}
            },
            {
                'name': '˫���β�ֵ+Savitzky-Golay�˲�',
                'params': {'method': 'cubic', 'fill_value': 0, 'savgol_window': 5, 'savgol_polyorder': 2}
            },
            {
                'name': '˫���β�ֵ+��������',
                'params': {'method': 'cubic', 'fill_value': 0, 'grid_density': 'log'}
            }
        ]

        results = []

        for combo in param_combinations:
            print(f"\n--- ���� {combo['name']} ---")

            # ʹ�õ�ǰ����������ֵ��
            interpolator = AdvancedBicubicInterpolation(
                upscale_factor=6, **combo['params'])

            # ִ�в�ֵ
            hr_pred = interpolator.interpolate(lr)

            # ����ÿ��������ָ��
            sample_metrics = []
            for i in range(lr.size(0)):
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

                sample_metrics.append({
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'mse': mse
                })

            # ����ƽ��ָ��
            avg_psnr = np.mean([m['psnr'] for m in sample_metrics])
            avg_ssim = np.mean([m['ssim'] for m in sample_metrics])
            avg_mse = np.mean([m['mse'] for m in sample_metrics])

            print(f"ƽ��PSNR: {avg_psnr:.2f} dB")
            print(f"ƽ��SSIM: {avg_ssim:.4f}")
            print(f"ƽ��MSE: {avg_mse:.6f}")

            results.append({
                'name': combo['name'],
                'params': combo['params'],
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'avg_mse': avg_mse
            })

        # �ҵ���Ѳ���
        best_psnr = max(results, key=lambda x: x['avg_psnr'])
        best_ssim = max(results, key=lambda x: x['avg_ssim'])

        print(f"\n=== �Ż���� ===")
        print(f"���PSNR: {best_psnr['name']} ({best_psnr['avg_psnr']:.2f} dB)")
        print(f"���SSIM: {best_ssim['name']} ({best_ssim['avg_ssim']:.4f})")

        # ������
        results_df = pd.DataFrame(results)
        results_df.to_csv('parameter_optimization_results.csv', index=False)
        print("��ϸ����ѱ��浽 'parameter_optimization_results.csv'")

        return results

    except Exception as e:
        print(f"�����Ż�����ʧ��: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    """���������в���"""
    parser = argparse.ArgumentParser(description='�������ݼ����غ͸߼�˫���β�ֵ')

    parser.add_argument('--data_path', type=str,
                        required=True, help='���ݼ�·�� (.h5�ļ�)')
    parser.add_argument('--num_samples', type=int, default=3, help='������������')
    parser.add_argument('--no_plots', action='store_true', help='���ñ�����ӻ�ͼƬ')
    parser.add_argument('--optimize', action='store_true', help='���в����Ż�����')

    # ����ģʽ������ѡ�񣨶��� evaluate_multi.py��
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='����ģʽ��generalization���������ԣ���test_set�����Լ�����all_generalization������ȫ������all_test_set�����Լ�ȫ����')
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='�������Ե���������÷ֺŷָ������磺wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='���Լ��������ö��ŷָ������磺1,2,3,4,5')

    # �߼���ֵ����
    parser.add_argument('--method', type=str, default='cubic',
                        choices=['linear', 'nearest', 'cubic', 'quintic'], help='��ֵ����')
    parser.add_argument('--smooth_sigma', type=float,
                        default=0, help='��˹ƽ��sigmaֵ')
    parser.add_argument('--edge_enhance', action='store_true', help='���ñ�Ե��ǿ')
    parser.add_argument('--edge_strength', type=float,
                        default=0.1, help='��Ե��ǿǿ��')
    parser.add_argument('--savgol_window', type=int,
                        default=None, help='Savitzky-Golay�˲������ڴ�С')

    args = parser.parse_args()
    return args


def get_dataset_indices(sample_specs, dataset):
    """
    Parse sample specs to dataset indices.
    Format: 'wind_group,source_group,time_steps' separated by ';'
    e.g. 'wind1_0,s1,50;wind2_0,s2,30'
    """
    if not sample_specs:
        return []
    specs = [spec.strip() for spec in sample_specs.split(';') if spec.strip()]
    indices = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(',')]
        if len(parts) < 3:
            continue
        wind_name, source_name, time_steps = parts[0], parts[1], int(parts[2])
        for i, data_info in enumerate(dataset.data_indices):
            if (data_info['wind_group'] == wind_name and
                data_info['source_group'] == source_name and
                    data_info['time_step'] <= time_steps):
                indices.append(i)
    return indices


def get_test_set_indices(test_indices, dataset_len):
    """
    Parse comma-separated indices string into a valid index list.
    e.g. '1,2,3,4,5'
    """
    if not test_indices:
        return []
    out = []
    for s in test_indices.split(','):
        s = s.strip()
        if not s:
            continue
        try:
            v = int(s)
            if 0 <= v < dataset_len:
                out.append(v)
        except:
            continue
    return out


def main():
    args = parse_args()
    print("Starting advanced bicubic interpolation test...")
    print("=" * 50)
    save_plots = not args.no_plots

    # Build dataset and subset indices by mode
    from datasets.h5_dataset import MultiTaskDataset
    dataset = MultiTaskDataset(args.data_path)

    if args.test_mode == 'generalization':
        if args.sample_specs is None:
            print("Error: --sample_specs required for generalization mode")
            return
        indices_to_evaluate = get_dataset_indices(args.sample_specs, dataset)
        print(f"Generalization mode, sample_specs: {args.sample_specs}")
    elif args.test_mode == 'all_generalization':
        indices_to_evaluate = list(range(len(dataset)))
        print("All-generalization mode, evaluating all samples")
    elif args.test_mode == 'all_test_set':
        # If you have a real split, replace with that; here we use all as a placeholder.
        indices_to_evaluate = list(range(len(dataset)))
        print("All-test-set mode (treated as full dataset here)")
    else:  # 'test_set'
        if args.test_indices is None:
            print("Error: --test_indices required for test_set mode")
            return
        indices_to_evaluate = get_test_set_indices(
            args.test_indices, len(dataset))
        print(f"Test-set mode, indices: {args.test_indices}")

    if not indices_to_evaluate:
        print("No samples found to evaluate!")
        return

    print(f"Found {len(indices_to_evaluate)} samples to evaluate")

    # Build subset dataset and evaluate on it
    subset_dataset = MultiTaskDataset(
        args.data_path, index_list=indices_to_evaluate, shuffle=False)

    if args.optimize:
        # Optional: adapt test_parameter_optimization to take 'dataset=subset_dataset' if needed.
        test_parameter_optimization(
            args.data_path, num_samples=min(args.num_samples, 2))
    else:
        # Evaluate strictly on the subset in a single pass
        test_advanced_bicubic_interpolation(
            args.data_path,
            num_samples=len(indices_to_evaluate),
            save_plots=save_plots,
            args=args,
            dataset=subset_dataset
        )


def test_advanced_bicubic_interpolation(data_path, num_samples=2, save_plots=True, args=None, dataset=None):
    """
    If 'dataset' is provided, use it directly (subset) for evaluation.
    All figure titles are in English to avoid font warnings.
    """
    print(f"\n=== Testing Advanced Bicubic Interpolation ===")

    try:
        if dataset is None:
            from datasets.h5_dataset import MultiTaskDataset
            dataset = MultiTaskDataset(data_path)

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # Build interpolator from args
        params = {}
        if args:
            params.update({
                'method': getattr(args, 'method', 'cubic'),
                'smooth_sigma': getattr(args, 'smooth_sigma', 0.0),
                'edge_enhance': getattr(args, 'edge_enhance', False),
                'edge_strength': getattr(args, 'edge_strength', 0.1),
                'savgol_window': getattr(args, 'savgol_window', None),
            })
        interpolator = AdvancedBicubicInterpolation(upscale_factor=6, **params)

        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        print(f"Input LR shape: {lr.shape}")
        print(f"True HR shape: {hr_true.shape}")

        hr_pred = interpolator.interpolate(lr)
        print(f"Predicted HR shape: {hr_pred.shape}")

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.metrics import structural_similarity as ssim

        # �ۼ�ָ���Լ���ƽ��ֵ
        psnr_list = []
        ssim_list = []
        mse_list = []

        for i in range(lr.size(0)):
            print(f"\n--- Sample {i+1} evaluation ---")
            pred_np = hr_pred[i].squeeze().numpy()
            true_np = hr_true[i].squeeze().numpy()

            mse = np.mean((pred_np - true_np) ** 2)
            psnr_val = 20 * np.log10(1.0 / np.sqrt(mse)
                                     ) if mse > 0 else float('inf')
            ssim_val = ssim(pred_np, true_np, data_range=1.0)

            print(f"PSNR: {psnr_val:.2f} dB")
            print(f"SSIM: {ssim_val:.4f}")
            print(f"MSE: {mse:.6f}")

            # ��¼ָ��
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse)

            if save_plots:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(lr[i].squeeze().numpy(), cmap='viridis')
                plt.title('Low-resolution input')
                plt.colorbar()

                plt.subplot(1, 4, 2)
                plt.imshow(hr_true[i].squeeze().numpy(), cmap='viridis')
                plt.title('Ground truth HR')
                plt.colorbar()

                plt.subplot(1, 4, 3)
                plt.imshow(pred_np, cmap='viridis')
                plt.title(f'Bicubic prediction\nPSNR: {psnr_val:.2f}')
                plt.colorbar()

                plt.subplot(1, 4, 4)
                diff = np.abs(pred_np - true_np)
                plt.imshow(diff, cmap='hot')
                plt.title('Absolute difference')
                plt.colorbar()

                plt.tight_layout()
                plt.savefig(
                    f'advanced_bicubic_test_sample_{i+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(
                    f"Visualization saved as 'advanced_bicubic_test_sample_{i+1}.png'")

        # �ڶ������������������ƽ��ָ��
        if len(psnr_list) > 0:
            avg_psnr = float(np.mean(psnr_list))
            avg_ssim = float(np.mean(ssim_list))
            avg_mse = float(np.mean(mse_list))

            print("\n=== Average metrics over samples ===")
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average MSE: {avg_mse:.6f}")

        return True

    except Exception as e:
        print(f"Advanced bicubic interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    main()
