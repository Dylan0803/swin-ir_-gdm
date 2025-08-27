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
from scipy.ndimage import zoom

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class BicubicInterpolation:
    """
    Bicubic interpolation algorithm for gas distribution reconstruction
    """

    def __init__(self, upscale_factor=6):
        """
        Initialize bicubic interpolator

        Args:
            upscale_factor: upsampling factor, default 6
        """
        self.upscale_factor = upscale_factor

    def interpolate(self, lr_data):
        """
        Reconstruct high-resolution gas distribution using bicubic interpolation

        Args:
            lr_data: low-resolution data [batch_size, 1, H, W] or [1, H, W]

        Returns:
            hr_data: high-resolution data [batch_size, 1, H*upscale, W*upscale] or [1, H*upscale, W*upscale]
        """
        if len(lr_data.shape) == 4:
            # Batch mode
            batch_size = lr_data.shape[0]
            hr_results = []

            for i in range(batch_size):
                hr_result = self._single_interpolate(lr_data[i])
                hr_results.append(hr_result)

            return torch.stack(hr_results, dim=0)
        else:
            # Single sample mode
            return self._single_interpolate(lr_data)

    def _single_interpolate(self, lr_data):
        """
        Perform bicubic interpolation on a single sample

        Args:
            lr_data: low-resolution data [1, H, W]

        Returns:
            hr_data: high-resolution data [1, H*upscale, W*upscale]
        """
        # Convert to numpy array and remove channel dimension
        lr_np = lr_data.squeeze().cpu().numpy()

        # Get original dimensions
        h, w = lr_np.shape

        # Calculate target dimensions
        target_h = h * self.upscale_factor
        target_w = w * self.upscale_factor

        # Create target grid
        x_old = np.linspace(0, 1, w)
        y_old = np.linspace(0, 1, h)
        x_new = np.linspace(0, 1, target_w)
        y_new = np.linspace(0, 1, target_h)

        # Create grid points
        X_old, Y_old = np.meshgrid(x_old, y_old)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        # Use scipy griddata for bicubic interpolation
        points = np.column_stack((X_old.flatten(), Y_old.flatten()))
        values = lr_np.flatten()

        # Bicubic interpolation
        hr_np = griddata(points, values, (X_new, Y_new),
                         method='cubic', fill_value=0)

        # Handle boundaries and outliers
        hr_np = np.clip(hr_np, 0, 1)

        # Convert back to tensor and add channel dimension
        hr_tensor = torch.from_numpy(hr_np).float().unsqueeze(0)

        return hr_tensor


def test_dataset_loading(data_path, num_samples=3):
    """
    Test dataset loading functionality

    Args:
        data_path: dataset path
        num_samples: number of test samples
    """
    print("=== Testing Dataset Loading ===")
    print(f"Dataset path: {data_path}")

    try:
        # Create dataset
        dataset = MultiTaskDataset(data_path)
        print(f"Dataset size: {len(dataset)} samples")

        # Test loading several samples
        for i in range(min(num_samples, len(dataset))):
            print(f"\n--- Sample {i+1} ---")
            sample = dataset[i]

            # Print data shape and basic information
            print(f"LR shape: {sample['lr'].shape}")
            print(f"HR shape: {sample['hr'].shape}")
            print(
                f"LR value range: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
            print(
                f"HR value range: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")

            # If source position information exists
            if 'source_pos' in sample:
                print(f"Source position: {sample['source_pos']}")
            if 'hr_max_pos' in sample:
                print(f"HR max position: {sample['hr_max_pos']}")

            # Visualize first sample
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
                print("Visualization saved as 'dataset_test_visualization.png'")

        return True

    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return False


def test_bicubic_interpolation(data_path, num_samples=2):
    """
    Test bicubic interpolation algorithm

    Args:
        data_path: dataset path
        num_samples: number of test samples
    """
    print(f"\n=== Testing Bicubic Interpolation Algorithm ===")

    try:
        # Create dataset and dataloader
        dataset = MultiTaskDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # Create bicubic interpolator
        bicubic_interpolator = BicubicInterpolation(upscale_factor=6)

        # Get one batch of data
        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        print(f"Input LR shape: {lr.shape}")
        print(f"True HR shape: {hr_true.shape}")

        # Perform bicubic interpolation
        hr_pred = bicubic_interpolator.interpolate(lr)
        print(f"Predicted HR shape: {hr_pred.shape}")

        # Calculate evaluation metrics
        for i in range(lr.size(0)):
            print(f"\n--- Sample {i+1} Evaluation Results ---")

            # Convert to numpy for evaluation
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

            # Visualize results
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
            print(f"Visualization saved as 'bicubic_test_sample_{i+1}.png'")

        return True

    except Exception as e:
        print(f"Bicubic interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test Dataset Loading and Bicubic Interpolation')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Dataset path (.h5 file)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of test samples')

    args = parser.parse_args()
    return args


def main():
    """Main function - test dataset loading and bicubic interpolation"""
    args = parse_args()

    print("Starting dataset loading and bicubic interpolation algorithm test...")
    print("=" * 50)

    # Test dataset loading
    dataset_ok = test_dataset_loading(args.data_path, args.num_samples)

    if dataset_ok:
        # Test bicubic interpolation
        bicubic_ok = test_bicubic_interpolation(
            args.data_path, min(args.num_samples, 2))

        if bicubic_ok:
            print("\n" + "=" * 50)
            print("? All tests passed!")
            print(
                "Dataset can be loaded correctly, bicubic interpolation algorithm works normally.")
        else:
            print("\n? Bicubic interpolation test failed")
    else:
        print("\n? Dataset loading test failed")


if __name__ == '__main__':
    main()
