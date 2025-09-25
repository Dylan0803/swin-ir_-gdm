from models.network_swinir_multi_gdm import SwinIRMulti as SwinIRMultiGDM
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_hybrid import SwinIRHybrid
from models.network_swinir_fuse import SwinIRFuse
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
import torch.serialization
from argparse import Namespace
# �����Ŀ��Ŀ¼��Python·��
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def evaluate_model(model, dataloader, device):
    """
    ����ģ������

    ����:
        model: ѵ���õ�ģ��
        dataloader: ���ݼ�����
        device: �豸

    ����:
        dict: ������������ָ����ֵ�
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
                # ����Ԥ����
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                source_pos = batch['source_pos'].to(device)
                hr_max_pos = batch['hr_max_pos'].to(device)

                # ģ������
                gdm_out, gsl_out = model(lr)

                # ����GDMָ��
                for i in range(gdm_out.size(0)):
                    # PSNR
                    mse = torch.mean((gdm_out[i] - hr[i]) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    metrics['gdm_psnr'].append(psnr.item())

                    # SSIM (�򻯰�)
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

                # ����GSLָ��
                # ����һ�����꣨����95.0��
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0

                # ���㷴��һ����ľ��벢����10
                position_error = torch.sqrt(
                    torch.sum((pred_pos - true_pos) ** 2, dim=1)) / 10.0
                max_pos_error = torch.sqrt(
                    torch.sum((pred_pos - hr_max_pos * 95.0) ** 2, dim=1)) / 10.0

                metrics['gsl_position_error'].extend(
                    position_error.cpu().numpy())
                metrics['gsl_max_pos_error'].extend(
                    max_pos_error.cpu().numpy())

            except KeyError as e:
                print(f"������Ч����: {e}")
                continue

    # ����ƽ��ָ��
    results = {
        'GDM_PSNR': np.mean(metrics['gdm_psnr']) if metrics['gdm_psnr'] else 0,
        'GDM_SSIM': np.mean(metrics['gdm_ssim']) if metrics['gdm_ssim'] else 0,
        'GSL_Position_Error': np.mean(metrics['gsl_position_error']) if metrics['gsl_position_error'] else 0,
        'GSL_MaxPos_Error': np.mean(metrics['gsl_max_pos_error']) if metrics['gsl_max_pos_error'] else 0
    }

    return results, metrics


def plot_metrics(metrics, save_dir):
    """��������ָ��ֲ�ͼ"""
    plt.figure(figsize=(15, 10))

    # GDM PSNR�ֲ�
    plt.subplot(221)
    plt.hist(metrics['gdm_psnr'], bins=50)
    plt.title('GDM PSNR Distribution')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')

    # GDM SSIM�ֲ�
    plt.subplot(222)
    plt.hist(metrics['gdm_ssim'], bins=50)
    plt.title('GDM SSIM Distribution')
    plt.xlabel('SSIM')
    plt.ylabel('Count')

    # GSLλ�����ֲ�
    plt.subplot(223)
    plt.hist(metrics['gsl_position_error'], bins=50)
    plt.title('GSL Position Error Distribution')
    plt.xlabel('Error (normalized)')
    plt.ylabel('Count')

    # GSL���Ũ��λ�����ֲ�
    plt.subplot(224)
    plt.hist(metrics['gsl_max_pos_error'], bins=50)
    plt.title('GSL Max Position Error Distribution')
    plt.xlabel('Error (normalized)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'), dpi=120)
    plt.close()


def visualize_results(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_path):
    """���ӻ����������Ľ��"""
    # ת��Ϊnumpy���鲢�Ƴ�batchά��
    lr = lr.squeeze().cpu().numpy()
    hr = hr.squeeze().cpu().numpy()
    gdm_out = gdm_out.squeeze().cpu().numpy()

    # �����ֵͼ - ���طǾ���ֵ
    diff = hr - gdm_out

    # ����ͼ�񣬵�����ͼ��������ɵײ�ͼע
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # �����ܱ����ڵײ�
    fig.suptitle('Gas Concentration Distribution', fontsize=16, y=0.02)

    # �ͷֱ�������
    im0 = axes[0, 0].imshow(lr, cmap='viridis')
    axes[0, 0].set_title('Low Resolution Input', pad=20, y=-0.15)
    axes[0, 0].axis('off')
    # �Ƴ���ɫ��

    # �߷ֱ�����ʵֵ
    im1 = axes[0, 1].imshow(hr, cmap='viridis')
    axes[0, 1].set_title('High Resolution Ground Truth', pad=20, y=-0.15)
    axes[0, 1].axis('off')
    # �Ƴ���ɫ��

    # ģ����������ֱ��ʣ�
    im2 = axes[1, 0].imshow(gdm_out, cmap='viridis')
    axes[1, 0].set_title('Super Resolution Output', pad=20, y=-0.15)
    axes[1, 0].axis('off')
    # �Ƴ���ɫ��

    # ֻ�ڷ�gdmģ��ʱ��й©Դ��
    if (gsl_out is not None) and (source_pos is not None) and (hr_max_pos is not None):
        gsl_out = gsl_out.squeeze().cpu().numpy()
        source_pos = source_pos.squeeze().cpu().numpy()
        hr_max_pos = hr_max_pos.squeeze().cpu().numpy()
        # ����һ�����꣨����95.0��
        true_pos = source_pos * 95.0
        pred_pos = gsl_out * 95.0
        # �����ʵй©Դλ�ã���ɫ���Σ�
        axes[1, 0].plot(true_pos[0], true_pos[1], 'r*',
                        markersize=15, label='True Source')
        # ���Ԥ��й©Դλ�ã���ɫ���Σ�
        axes[1, 0].plot(pred_pos[0], pred_pos[1], 'g*',
                        markersize=15, label='Predicted Source')
        # ���ͼ��
        axes[1, 0].legend(loc='upper right')

    # ��ֵͼ - ���طǾ���ֵ��ʹ��RdBu_r��ɫӳ��
    im3 = axes[1, 1].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('SR-HR', pad=20, y=-0.15)
    axes[1, 1].axis('off')
    # �Ƴ���ɫ��

    # �������֣�Ϊ�ײ����������ռ�
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Ϊ�ײ�������������ռ�

    # ����ͼ��
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()

    # ֻ��pred_pos��true_pos���Ѷ���ʱ���ؾ���
    if ('pred_pos' in locals()) and ('true_pos' in locals()):
        distance = np.sqrt(np.sum((pred_pos - true_pos) ** 2)) / 10.0
        return distance
    else:
        return None


def save_sample_data(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_dir, sample_idx, model_type):
    """����ÿ�����������ݵ�ָ��Ŀ¼����Ϊ������������Ŀ¼��"""
    import numpy as np

    # save_dir ���Ǹ�������Ŀ¼
    os.makedirs(save_dir, exist_ok=True)

    # ת��Ϊnumpy���鲢�Ƴ�batchά��
    lr_np = lr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()
    gdm_out_np = gdm_out.squeeze().cpu().numpy()

    # ����LR��HR��GDM�������ΪCSV
    np.savetxt(os.path.join(save_dir, 'lr.csv'),
               lr_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(save_dir, 'hr.csv'),
               hr_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(save_dir, 'gdm_out.csv'),
               gdm_out_np, delimiter=',', fmt='%.6f')

    # �����ֵ����ΪCSV
    diff = hr_np - gdm_out_np
    np.savetxt(os.path.join(save_dir, 'difference.csv'),
               diff, delimiter=',', fmt='%.6f')

    # ����GSL������ݣ�������ڣ�
    if model_type != 'swinir_gdm' and gsl_out is not None and source_pos is not None:
        gsl_out_np = gsl_out.squeeze().cpu().numpy()
        source_pos_np = source_pos.squeeze().cpu().numpy()
        hr_max_pos_np = hr_max_pos.squeeze().cpu(
        ).numpy() if hr_max_pos is not None else None

        # �� original ģ�ͣ�ȫ����Ϊ CSV ����
        if model_type == 'original':
            np.savetxt(os.path.join(save_dir, 'gsl_out.csv'),
                       gsl_out_np, delimiter=',', fmt='%.6f')
            np.savetxt(os.path.join(save_dir, 'source_pos.csv'),
                       source_pos_np, delimiter=',', fmt='%.6f')
            if hr_max_pos_np is not None:
                np.savetxt(os.path.join(save_dir, 'hr_max_pos.csv'),
                           hr_max_pos_np, delimiter=',', fmt='%.6f')
        else:
            np.save(os.path.join(save_dir, 'gsl_out.npy'), gsl_out_np)
            np.save(os.path.join(save_dir, 'source_pos.npy'), source_pos_np)
            if hr_max_pos_np is not None:
                np.save(os.path.join(save_dir,
                        'hr_max_pos.npy'), hr_max_pos_np)

    # ����Ԫ������Ϣ
    metadata = {
        'sample_idx': sample_idx,
        'model_type': model_type,
        'lr_shape': lr_np.shape,
        'hr_shape': hr_np.shape,
        'gdm_out_shape': gdm_out_np.shape,
        'diff_shape': diff.shape,
        'lr_range': [float(lr_np.min()), float(lr_np.max())],
        'hr_range': [float(hr_np.min()), float(hr_np.max())],
        'gdm_out_range': [float(gdm_out_np.min()), float(gdm_out_np.max())],
        'diff_range': [float(diff.min()), float(diff.max())]
    }

    # ����Ԫ����ΪJSON�ļ�
    import json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"���� {sample_idx} �������ѱ��浽: {save_dir}")


def infer_model(model, data_path, save_dir, num_samples=5, sample_indices=None, model_type='original'):
    """
    ģ��������

    ����:
        model: ѵ���õ�ģ��
        data_path: ����·��
        save_dir: ��������Ŀ¼
        num_samples: Ҫ��������������
        sample_indices: ָ��Ҫ���������������б����ΪNone�����ѡ��
    """
    # ����ģ��ר�������Ŀ¼
    base_save_dir = os.path.abspath(save_dir)
    if model_type == 'swinir_gdm':
        model_root = os.path.join(base_save_dir, 'swinir_gdm_results')
    else:  # original ������
        model_root = os.path.join(base_save_dir, 'swinir_multi_results')
    os.makedirs(model_root, exist_ok=True)

    # �������ݼ�������shuffle=Falseȷ�����ݲ�����
    dataset = MultiTaskDataset(data_path, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # �����豸
    device = next(model.parameters()).device

    # ����ָ��
    total_psnr = 0
    total_position_error = 0
    total_mse = 0
    total_ssim = 0
    valid_samples = 0

    # ����ṩ��������������ֻ����ָ��������
    if sample_indices is not None and len(sample_indices) > 0:
        indices_to_evaluate = sample_indices
    else:
        indices_to_evaluate = list(range(len(dataset)))
        if num_samples < len(indices_to_evaluate):
            indices_to_evaluate = indices_to_evaluate[:num_samples]

    print(f"������������: {indices_to_evaluate}")

    # ����ָ������������
    for idx in indices_to_evaluate:
        try:
            # ��ȡָ������������
            batch = dataset[idx]

            # �������Ƶ��豸��
            lr = batch['lr'].unsqueeze(0).to(device)  # ���batchά��
            hr = batch['hr'].unsqueeze(0).to(device)
            source_pos = batch['source_pos'].unsqueeze(0).to(
                device) if 'source_pos' in batch else None

            # ģ������
            with torch.no_grad():
                if model_type == 'swinir_gdm':
                    gdm_out = model(lr)
                    gsl_out = None
                else:
                    gdm_out, gsl_out = model(lr)

            # ����HRͼ������ֵλ��
            hr_max_pos = torch.tensor([torch.argmax(hr[0, 0]) % hr.shape[3],
                                       torch.argmax(hr[0, 0]) // hr.shape[3]],
                                      dtype=torch.float32).to(device)
            hr_max_pos = hr_max_pos / torch.tensor([hr.shape[3], hr.shape[2]],
                                                   dtype=torch.float32).to(device)

            # ��������ָ��
            # MSE��ʧ
            mse = F.mse_loss(gdm_out, hr)

            # PSNR
            psnr = 10 * torch.log10(1.0 / mse)

            # SSIM����
            c1 = (0.01 * 1.0) ** 2
            c2 = (0.03 * 1.0) ** 2
            mu1 = torch.mean(gdm_out)
            mu2 = torch.mean(hr)
            sigma1 = torch.var(gdm_out)
            sigma2 = torch.var(hr)
            sigma12 = torch.mean((gdm_out - mu1) * (hr - mu2))
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))

            total_psnr += psnr.item()
            total_mse += mse.item()
            total_ssim += ssim.item()
            valid_samples += 1

            # ֻ�ڷ�gdmģ��ʱ����GSL���
            if model_type != 'swinir_gdm':
                # ����һ�����꣨����95.0��
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0
                # ���㷴��һ�����ֱ�߾��벢����10��ת��Ϊ�ף�
                position_error = torch.sqrt(
                    torch.sum((pred_pos - true_pos) ** 2)) / 10.0
                total_position_error += position_error.item()

            # Ϊ��������������Ŀ¼
            sample_dir = os.path.join(model_root, f'sample_{idx}')
            os.makedirs(sample_dir, exist_ok=True)

            # �����������ݣ���npy��metadata��
            save_sample_data(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos,
                             sample_dir, idx, model_type)

            # �������ͼ������Ŀ¼
            save_path = os.path.join(sample_dir, f'sample_{idx}_composite.png')
            if model_type == 'swinir_gdm':
                visualize_results(lr, hr, gdm_out, None, None, None, save_path)
            else:
                visualize_results(lr, hr, gdm_out, gsl_out,
                                  source_pos, hr_max_pos, save_path)

            # ���ⵥ������ LR / HR / GDM ����ͼ����������/��ɫ����
            lr_np = lr.squeeze().detach().cpu().numpy()
            hr_np = hr.squeeze().detach().cpu().numpy()
            gdm_np = gdm_out.squeeze().detach().cpu().numpy()

            plt.figure(figsize=(5, 5))
            plt.imshow(lr_np, cmap='viridis')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(
                sample_dir, f'sample_{idx}_LR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.imshow(hr_np, cmap='viridis')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(
                sample_dir, f'sample_{idx}_HR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.imshow(gdm_np, cmap='viridis')
            # �� original ģ�ͣ��ڶ���GDMͼ�ϱ�ע��ֵ��Ԥ��й©Դλ��
            if model_type != 'swinir_gdm' and gsl_out is not None and source_pos is not None:
                gsl_out_np = gsl_out.squeeze().detach().cpu().numpy()
                source_pos_np = source_pos.squeeze().detach().cpu().numpy()
                true_pos = source_pos_np * 95.0
                pred_pos = gsl_out_np * 95.0
                plt.plot(true_pos[0], true_pos[1], 'r*',
                         markersize=15, label='True Source')
                plt.plot(pred_pos[0], pred_pos[1], 'g*',
                         markersize=15, label='Predicted Source')
                plt.legend(loc='upper right')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(
                sample_dir, f'sample_{idx}_GDM.png'), dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        except KeyError as e:
            print(f"������Ч����: {e}")
            continue
        except Exception as e:
            print(f"��������ʱ����: {e}")
            continue

    # ����ƽ��ָ��
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_mse = total_mse / valid_samples
        avg_ssim = total_ssim / valid_samples
        print(f"\n�������:")
        print(f"��Ч������: {valid_samples}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        if model_type != 'swinir_gdm':
            avg_position_error = total_position_error / valid_samples
            print(f"Average Position Error: {avg_position_error:.4f} m")
    else:
        print("û����Ч�������ɹ�����")


def get_dataset_indices(sample_specs, dataset):
    """
    ������������ȡ���ݼ��е�ʵ���������ϸ� sample_specs ˳�򷵻�
    ֻ������ڵ����ݣ����������ڵ���
    """
    actual_indices = []
    if not sample_specs:
        return actual_indices

    # �ȹ���һ�����ұ�ֻ�������ڵ�����
    index_map = {}
    for idx in range(len(dataset)):
        try:
            data_info = dataset.data_indices[idx]
            key = f"{data_info['wind_group']},{data_info['source_group']},{data_info['time_step']}"
            index_map[key] = idx
        except Exception:
            # ��Ĭ���������ڵ�����
            continue

    # �� sample_specs ˳����ң�ֻ������ڵ�����
    for spec in sample_specs:
        try:
            idx = index_map.get(spec)
            if idx is not None:
                actual_indices.append(idx)
                print(f"�ҵ�ƥ������: {spec}, ����={idx}")
            else:
                print(f"δ�ҵ�ƥ������: {spec}")
        except Exception:
            # ��Ĭ���������ڵ�����
            continue

    return actual_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SwinIR Multi-task Model')

    # ���ģ������ѡ�����
    parser.add_argument('--model_type', type=str, default='original',
                        choices=['original', 'enhanced',
                                 'hybrid', 'fuse', 'swinir_gdm'],
                        help='ѡ��ģ������: original, enhanced, hybrid, fuse, swinir_gdm')

    # ���ȱʧ�Ĳ���
    parser.add_argument('--model_path', type=str, required=True,
                        help='ģ��Ȩ��·��')
    parser.add_argument('--data_path', type=str, required=True,
                        help='���ݼ�·��')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='�����������Ŀ¼')
    parser.add_argument('--device', type=str, default='cuda',
                        help='ʹ�õ��豸')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Ҫ��������������')

    # ����ģʽѡ������all_generalization��all_test_set
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='����ģʽ��generalization���������ԣ���test_set�����Լ�����all_generalization������ȫ������all_test_set�����Լ�ȫ����')

    # ����ѡ�����
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='�������Ե���������÷ֺŷָ������磺wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='���Լ��������ö��ŷָ������磺1,2,3,4,5')

    # ��� config ����
    parser.add_argument('--config', type=str, default=None,
                        help='ѵ������json·������training_args.json��')

    parser.add_argument('--upsampler', type=str, default='nearest+conv',
                        choices=['nearest+conv', 'pixelshuffle'],
                        help='�ϲ�����ʽ: nearest+conv��Ĭ�ϣ��� pixelshuffle')

    # �������� original(SwinIRMulti) ��Ч�� RSTB ��������
    parser.add_argument('--multi_rstb', type=int, default=None,
                        help='���� original(SwinIRMulti) ��Ч������ RSTB �������� 2/4/6/8/...��')

    args = parser.parse_args()
    return args


def create_model(args):
    """���ݲ�������ģ��"""
    # ���ȶ�ȡ config��training_args.json��
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        # ֻ����ģ����ز���
        ignore_keys = [
            'model_type', 'model_path', 'data_path', 'save_dir', 'device', 'num_samples',
            'test_mode', 'sample_specs', 'test_indices', 'seed', 'batch_size', 'num_epochs',
            'lr', 'weight_decay', 'use_test_set', 'upsampler'
        ]
        model_params = {k: v for k,
                        v in config.items() if k not in ignore_keys}
        print("��������ģ�Ͳ������£���У�ԡ�")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
        # ѡ��ģ������
        if args.model_type == 'original':
            model = SwinIRMulti(**model_params)
        elif args.model_type == 'enhanced':
            model = SwinIRMultiEnhanced(**model_params)
        elif args.model_type == 'hybrid':
            model = SwinIRHybrid(**model_params)
        elif args.model_type == 'swinir_gdm':
            model = SwinIRMultiGDM(**model_params)  # �ɸ�����Ҫ�����趨����
        else:
            model = SwinIRFuse(**model_params)
        return model

    # ����ģ�Ͳ���
    base_params = {
        'img_size': 16,  # LRͼ���С
        'in_chans': 1,   # ����ͨ����
        'upscale': 6,    # �ϲ�������
        'img_range': 1.,  # ͼ��Χ
        'upsampler': args.upsampler  # ���������в�������
    }

    # ԭʼģ�Ͳ������ɰ� multi_rstb ���� RSTB ������
    default_depth_per_group = 6
    default_heads_per_group = 6
    default_groups = 4
    groups = args.multi_rstb if (
        hasattr(args, 'multi_rstb') and args.multi_rstb) else default_groups
    depths = [default_depth_per_group for _ in range(groups)]
    num_heads = [default_heads_per_group for _ in range(groups)]
    original_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer���ڴ�С
        'depths': depths,  # RSTB �������б��Ⱦ���
        'embed_dim': 60,  # Ƕ��ά��
        'num_heads': num_heads,  # ע����ͷ���б����� depths ͬ����
        'mlp_ratio': 2.,  # MLP����,
    }

    # ��ǿ��ģ�Ͳ���
    enhanced_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer���ڴ�С
        'depths': [6, 6, 6, 6],  # Swin Transformer���
        'embed_dim': 60,  # Ƕ��ά��
        'num_heads': [6, 6, 6, 6],  # ע����ͷ��
        'mlp_ratio': 2.,  # MLP����
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

    # ��ϼܹ�ģ�Ͳ���
    hybrid_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer���ڴ�С
        'depths': [6, 6, 6, 6],  # Swin Transformer���
        'embed_dim': 60,  # Ƕ��ά��
        'num_heads': [6, 6, 6, 6],  # ע����ͷ��
        'mlp_ratio': 2.,  # MLP����
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

    # Fuseģ�Ͳ��� (�Ƴ���resi_connection)
    fuse_params = {
        'img_size': 16,
        'in_chans': 1,
        'upscale': 6,
        'img_range': 1.,
        'upsampler': args.upsampler,
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
    }

    if args.model_type == 'original':
        model = SwinIRMulti(**original_params)
    elif args.model_type == 'enhanced':
        model = SwinIRMultiEnhanced(**enhanced_params)
    elif args.model_type == 'hybrid':
        model = SwinIRHybrid(**hybrid_params)
    elif args.model_type == 'swinir_gdm':
        model = SwinIRMultiGDM(**original_params)  # �ɸ�����Ҫ�����趨����
    else:  # fuse
        model = SwinIRFuse(**fuse_params)

    return model


def get_test_set_indices(test_indices_str, dataset):
    """
    ���ݲ��Լ������ַ�����ȡҪ��������������

    ����:
        test_indices_str: ���ŷָ��������ַ��������磺"1,2,3,4,5"
        dataset: ���ݼ�����

    ����:
        list: Ҫ���������������б�
    """
    if not test_indices_str:
        return []

    try:
        # ���������ַ���
        indices = [int(idx.strip()) for idx in test_indices_str.split(',')]
        # ��֤�����Ƿ���Ч
        valid_indices = [idx for idx in indices if 0 <= idx < len(dataset)]
        if len(valid_indices) != len(indices):
            print(f"���棺��������������Χ����������Ч����")
            print(f"����������Χ��0 �� {len(dataset)-1}")
            print(
                f"��Ч��������{[idx for idx in indices if idx < 0 or idx >= len(dataset)]}")
        return valid_indices
    except ValueError as e:
        print(f"������Ч��������ʽ - {e}")
        print(f"����������Χ��0 �� {len(dataset)-1}")
        return []


def batch_infer_model(model, dataset, save_dir, model_type, device='cuda', batch_size=16, num_workers=4, max_visualize=10):
    """
    ������������������all_generalization��all_test_set
    """
    import torch
    from torch.utils.data import DataLoader
    import os

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    model = model.to(device)
    model.eval()

    total_psnr = 0
    total_position_error = 0
    total_mse = 0
    total_ssim = 0
    valid_samples = 0
    visualized = 0  # ���ӻ�������
    total_processed = 0  # ȫ���������������������� sample_{idx}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Batch Evaluating")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            source_pos = batch.get('source_pos', None)
            if source_pos is not None:
                source_pos = source_pos.to(device)

            if model_type == 'swinir_gdm':
                gdm_out = model(lr)
                gsl_out = None
            else:
                gdm_out, gsl_out = model(lr)

            # ����ָ�꣨������
            mse = F.mse_loss(gdm_out, hr, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            psnr = 10 * torch.log10(1.0 / mse)
            c1 = (0.01 * 1.0) ** 2
            c2 = (0.03 * 1.0) ** 2
            mu1 = gdm_out.mean(dim=[1, 2, 3])
            mu2 = hr.mean(dim=[1, 2, 3])
            sigma1 = gdm_out.var(dim=[1, 2, 3])
            sigma2 = hr.var(dim=[1, 2, 3])
            sigma12 = ((gdm_out - mu1[:, None, None, None]) *
                       (hr - mu2[:, None, None, None])).mean(dim=[1, 2, 3])
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))

            total_psnr += psnr.sum().item()
            total_mse += mse.sum().item()
            total_ssim += ssim.sum().item()
            valid_samples += lr.size(0)

            # GSL���
            if model_type != 'swinir_gdm' and source_pos is not None and gsl_out is not None:
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0
                position_error = torch.sqrt(
                    torch.sum((pred_pos - true_pos) ** 2, dim=1)) / 10.0
                total_position_error += position_error.sum().item()

            # Ϊ�������е�ÿ��������������Ŀ¼������������ͼƬ������max_visualize���ƣ�
            batch_size_now = lr.size(0)
            for j in range(batch_size_now):
                sample_idx_global = total_processed
                sample_dir = os.path.join(
                    save_dir, f'sample_{sample_idx_global}')
                os.makedirs(sample_dir, exist_ok=True)

                # �����������ݣ�CSV/NPY����ģ�����͵ļ����߼���
                save_sample_data(
                    lr[j:j+1], hr[j:j+1], gdm_out[j:j+1],
                    (gsl_out[j:j+1] if (gsl_out is not None and model_type !=
                     'swinir_gdm') else None),
                    (source_pos[j:j+1] if (source_pos is not None and model_type !=
                     'swinir_gdm') else None),
                    None,
                    sample_dir, sample_idx_global, model_type
                )

                # �������ͼ
                composite_path = os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_composite.png')
                if model_type == 'swinir_gdm':
                    visualize_results(
                        lr[j:j+1], hr[j:j+1], gdm_out[j:j+1], None, None, None, composite_path)
                else:
                    visualize_results(
                        lr[j:j+1], hr[j:j+1], gdm_out[j:j+1],
                        (gsl_out[j:j+1] if gsl_out is not None else None),
                        (source_pos[j:j+1]
                         if source_pos is not None else None),
                        None, composite_path
                    )

                # �������� LR / HR / GDM ͼ
                lr_np = lr[j].squeeze().detach().cpu().numpy()
                hr_np = hr[j].squeeze().detach().cpu().numpy()
                gdm_np = gdm_out[j].squeeze().detach().cpu().numpy()

                plt.figure(figsize=(5, 5))
                plt.imshow(lr_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_LR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(hr_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_HR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(gdm_np, cmap='viridis')
                # original ģ���ڶ���GDMͼ�ϱ�עй©Դ��ֵ/Ԥ��
                if model_type != 'swinir_gdm' and gsl_out is not None and source_pos is not None:
                    gsl_out_np = gsl_out[j].squeeze().detach().cpu().numpy()
                    source_pos_np = source_pos[j].squeeze(
                    ).detach().cpu().numpy()
                    true_pos = source_pos_np * 95.0
                    pred_pos = gsl_out_np * 95.0
                    plt.plot(true_pos[0], true_pos[1], 'r*',
                             markersize=15, label='True Source')
                    plt.plot(pred_pos[0], pred_pos[1], 'g*',
                             markersize=15, label='Predicted Source')
                    plt.legend(loc='upper right')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_GDM.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                total_processed += 1

            # ���⣬�������������μ��������ͼ����Ӱ���������棩
            if visualized < max_visualize:
                preview_path = os.path.join(
                    save_dir, f'batch_{i}_sample_preview.png')
                visualize_results(lr[0:1], hr[0:1], gdm_out[0:1], None if model_type == 'swinir_gdm' else (
                    gsl_out[0:1] if gsl_out is not None else None), None, None, preview_path)
                visualized += 1

    # ���ƽ��ָ��
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_mse = total_mse / valid_samples
        avg_ssim = total_ssim / valid_samples
        print(f"\n�����������:")
        print(f"��Ч������: {valid_samples}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        if model_type != 'swinir_gdm':
            avg_position_error = total_position_error / valid_samples
            print(f"Average Position Error: {avg_position_error:.4f} m")
    else:
        print("û����Ч�������ɹ�����")


def main():
    args = parse_args()

    # �����豸
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ����ģ��
    model = create_model(args)

    # ����ģ��
    print(f"Loading model from {args.model_path}")
    try:
        # ��� original/enhanced/hybrid ������׳��Ȩ�ؼ���
        if args.model_type in ['original', 'enhanced', 'hybrid']:
            torch.serialization.add_safe_globals([Namespace])
            checkpoint = torch.load(
                args.model_path, map_location='cpu', weights_only=False)
            # �ж�������checkpoint����state_dict
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            # fuseģ��ֱ�Ӽ���
            torch.serialization.add_safe_globals([Namespace])
            checkpoint = torch.load(
                args.model_path, map_location='cpu', weights_only=False)
            # ����Ƿ�������checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"����ģ��Ȩ��ʧ��: {e}")
        return

    model = model.to(device)
    model.eval()

    use_batch = False  # <--- ������һ�У�Ĭ�ϲ�����������

    # �������ݼ�
    dataset = None
    # ���ݲ���ģʽѡ��Ҫ����������
    indices_to_evaluate = []
    if args.test_mode == 'generalization':
        # ��������ģʽ
        if args.sample_specs is not None:
            sample_specs = [spec.strip()
                            for spec in args.sample_specs.split(';')]
            dataset = MultiTaskDataset(args.data_path)
            indices_to_evaluate = get_dataset_indices(sample_specs, dataset)
            print(f"ʹ�÷�������ģʽ���������{args.sample_specs}")
        else:
            print("���󣺷�������ģʽ��Ҫ�ṩ sample_specs ����")
            return
    elif args.test_mode == 'all_generalization':
        dataset = MultiTaskDataset(args.data_path)
        indices_to_evaluate = list(range(len(dataset)))
        print("ʹ�÷�������ȫ��ģʽ��������������")
        use_batch = True
    elif args.test_mode == 'all_test_set':
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        indices_to_evaluate = list(range(len(dataset)))
        print("ʹ�ò��Լ�ȫ��ģʽ��������������")
        use_batch = True
    else:
        # ���Լ�ģʽ
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        if args.test_indices is not None:
            indices_to_evaluate = get_test_set_indices(
                args.test_indices, dataset)
            print(f"ʹ�ò��Լ�ģʽ������������{args.test_indices}")
        else:
            print("���󣺲��Լ�ģʽ��Ҫ�ṩ test_indices ����")
            return

    if not indices_to_evaluate:
        print("û���ҵ�Ҫ������������")
        return
    print(f"�ҵ� {len(indices_to_evaluate)} ��Ҫ����������")
    # ��������
    if use_batch:
        # ����ģ��ר�������Ŀ¼
        base_save_dir = os.path.abspath(args.save_dir)
        if args.model_type == 'swinir_gdm':
            model_root = os.path.join(base_save_dir, 'swinir_gdm_results')
        else:
            model_root = os.path.join(base_save_dir, 'swinir_multi_results')
        os.makedirs(model_root, exist_ok=True)
        batch_infer_model(model, dataset, model_root,
                          args.model_type, device=args.device)
    else:
        infer_model(model, args.data_path, args.save_dir,
                    args.num_samples, indices_to_evaluate, args.model_type)


if __name__ == '__main__':
    main()
