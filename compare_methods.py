# compare_methods.py
# ͳһ�Աȣ�LR��HR��Bicubic(Yb)��GKDM(Yk)��SwinIR_GDM(Ys)��SwinIR_Multi(Ym)
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace

# ����������ģ��
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_test_dataset
from models.network_swinir_multi_gdm import SwinIRMulti as SwinIRMultiGDM
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_hybrid import SwinIRHybrid
from models.network_swinir_fuse import SwinIRFuse

# ���� KDM �� Bicubic �е�ʵ��
from bicubic import AdvancedBicubicInterpolation
from eval_gkdm import load_data_from_swinir_h5, get_gaussian_kdm_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='ͳһ�Աȣ�Bicubic/KDM/SwinIR(GDM & Multi)')
    # ͨ��
    parser.add_argument('--data_path', type=str, required=True, help='H5 ����·��')
    parser.add_argument('--save_dir', type=str,
                        default='compare_outputs', help='�������Ŀ¼')
    parser.add_argument('--device', type=str,
                        default='cuda', help='�豸��cuda �� cpu')
    parser.add_argument('--upsampler', type=str, default='nearest+conv',
                        choices=['nearest+conv', 'pixelshuffle'], help='�ϲ�����ʽ��ģ�ͣ�')
    parser.add_argument('--config', type=str, default=None,
                        help='��ѡ��training_args.json������ģ�ͽṹ����')
    parser.add_argument('--scale_factor', type=int,
                        default=6, help='�ϲ������� / �²������ʣ�һ��=6')

    # ����ѡ���߼����� evaluate_multi / bicubic / eval_gkdm ���룩
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='����ģʽ')
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='�����������wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='���Լ�������1,2,3')
    parser.add_argument('--pick', type=int, default=0,
                        help='��ƥ�䵽�������ʱ��ȡ�ڼ�����0-based����Ĭ��ȡ��һ��')

    # ����������Ȩ��
    parser.add_argument('--model_path_gdm', type=str,
                        required=True, help='SwinIR_GDM Ȩ��·��')
    parser.add_argument('--model_path_multi', type=str,
                        required=True, help='SwinIR_Multi Ȩ��·��')
    # SwinIR_Multi �ı���ѡ�񣨱����� evaluate_multi һ�£�
    parser.add_argument('--multi_variant', type=str, default='original',
                        choices=['original', 'enhanced', 'hybrid', 'fuse'],
                        help='SwinIR_Multi ����')

    # KDM ����������� eval_gkdm ���룬Rco �Զ���һ������ֵ��Ҳ�ɰ�����չ������������
    parser.add_argument('--mat_width', type=float, default=9.6, help='������(m)')
    parser.add_argument('--mat_height', type=float,
                        default=9.6, help='����߶�(m)')
    parser.add_argument('--kdm_rco', type=float,
                        default=1.0, help='KDM �� Rco��Gama=Rco/3')

    # Bicubic �������� bicubic.py ����Ĺؼ��
    parser.add_argument('--bicubic_method', type=str, default='cubic',
                        choices=['linear', 'nearest', 'cubic', 'quintic'], help='��ֵ����')
    parser.add_argument('--bicubic_smooth_sigma', type=float,
                        default=0.0, help='��˹ƽ�� sigma')
    parser.add_argument('--bicubic_edge_enhance',
                        action='store_true', help='��Ե��ǿ')
    parser.add_argument('--bicubic_edge_strength',
                        type=float, default=0.1, help='��Ե��ǿǿ��')
    parser.add_argument('--bicubic_savgol_window', type=int,
                        default=None, help='Savitzky-Golay ����')

    return parser.parse_args()


def select_indices_by_mode(args):
    dataset = None
    indices = []
    use_batch = False

    if args.test_mode == 'generalization':
        if args.sample_specs is None:
            raise ValueError('generalization ģʽ��Ҫ --sample_specs')
        dataset = MultiTaskDataset(args.data_path, shuffle=False)
        # �� evaluate_multi.get_dataset_indices һ�µĲ��ң��ϸ� specs ˳�򣬴��ڲż��룩
        index_map = {}
        for i in range(len(dataset)):
            try:
                info = dataset.data_indices[i]
                key = f"{info['wind_group']},{info['source_group']},{info['time_step']}"
                index_map[key] = i
            except Exception:
                continue
        specs = [s.strip() for s in args.sample_specs.split(';') if s.strip()]
        for s in specs:
            idx = index_map.get(s)
            if idx is not None:
                indices.append(idx)
        if not indices:
            raise ValueError('δ�ҵ�ƥ������������ sample_specs')
    elif args.test_mode == 'all_generalization':
        dataset = MultiTaskDataset(args.data_path, shuffle=False)
        indices = list(range(len(dataset)))
        use_batch = True
    elif args.test_mode == 'all_test_set':
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        indices = list(range(len(dataset)))
        use_batch = True
    else:  # test_set
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        if args.test_indices is None:
            raise ValueError('test_set ģʽ��Ҫ --test_indices')
        raw = [int(v.strip())
               for v in args.test_indices.split(',') if v.strip()]
        indices = [v for v in raw if 0 <= v < len(dataset)]
        if not indices:
            raise ValueError('test_indices Ϊ�ջ�Խ��')

    return dataset, indices, use_batch


def create_models(args):
    # ���� evaluate_multi.py ��Ĭ�ϲ���
    base_params = {
        'img_size': 16,
        'in_chans': 1,
        'upscale': args.scale_factor,
        'img_range': 1.,
        'upsampler': args.upsampler
    }
    original_params = {
        **base_params,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
    }
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
    hybrid_params = dict(enhanced_params)
    fuse_params = {
        'img_size': 16,
        'in_chans': 1,
        'upscale': args.scale_factor,
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

    # ��ѡ���� config ����ģ�ͽṹ
    def build_params(base, override_cfg):
        if override_cfg is None:
            return base
        ignore_keys = {
            'model_type', 'model_path', 'data_path', 'save_dir', 'device', 'num_samples',
            'test_mode', 'sample_specs', 'test_indices', 'seed', 'batch_size', 'num_epochs',
            'lr', 'weight_decay', 'use_test_set', 'upsampler'
        }
        p = dict(base)
        for k, v in override_cfg.items():
            if k not in ignore_keys:
                p[k] = v
        return p

    cfg = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)

    gdm_params = build_params(original_params, cfg)
    multi_map = {
        'original': original_params,
        'enhanced': enhanced_params,
        'hybrid': hybrid_params,
        'fuse': fuse_params
    }
    multi_params = build_params(multi_map[args.multi_variant], cfg)

    model_gdm = SwinIRMultiGDM(**gdm_params)
    if args.multi_variant == 'original':
        model_multi = SwinIRMulti(**multi_params)
    elif args.multi_variant == 'enhanced':
        model_multi = SwinIRMultiEnhanced(**multi_params)
    elif args.multi_variant == 'hybrid':
        model_multi = SwinIRHybrid(**multi_params)
    else:
        model_multi = SwinIRFuse(**multi_params)

    return model_gdm, model_multi


def load_weights(model, ckpt_path):
    torch.serialization.add_safe_globals([Namespace])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)


def tensor_to_np(img):
    return img.squeeze().detach().cpu().numpy()


def save_img(np_img, save_path, title=None, cmap='viridis'):
    plt.figure(figsize=(5, 5))
    plt.imshow(np_img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ѡ������
    dataset, indices, _ = select_indices_by_mode(args)
    if not indices:
        raise ValueError('û�п�������')
    pick_idx = indices[min(args.pick, len(indices)-1)]
    sample = dataset[pick_idx]
    lr = sample['lr'].unsqueeze(0).to(device)   # [1,1,H,W]
    hr = sample['hr'].unsqueeze(0).to(device)   # [1,1,H*,W*]
    info = dataset.data_indices[pick_idx]
    wind_group = info['wind_group']
    source_group = info['source_group']
    time_step = info['time_step']

    # 1) ���� LR/HR ԭͼ
    lr_np = tensor_to_np(lr)
    hr_np = tensor_to_np(hr)
    save_img(lr_np, os.path.join(args.save_dir, f's{pick_idx}_LR.png'), 'LR')
    save_img(hr_np, os.path.join(args.save_dir, f's{pick_idx}_HR.png'), 'HR')

    # 2) Bicubic(Yb)��ʹ�� AdvancedBicubicInterpolation
    bicubic = AdvancedBicubicInterpolation(
        upscale_factor=args.scale_factor,
        method=args.bicubic_method,
        smooth_sigma=args.bicubic_smooth_sigma,
        edge_enhance=args.bicubic_edge_enhance,
        edge_strength=args.bicubic_edge_strength,
        savgol_window=args.bicubic_savgol_window,
    )
    yb = bicubic.interpolate(lr)  # [1,1,H*,W*]
    yb_np = tensor_to_np(yb)
    save_img(yb_np, os.path.join(args.save_dir,
             f's{pick_idx}_Yb_bicubic.png'), 'Bicubic')

    # 3) GKDM(Yk)�����ļ�����ȡ HR/LR/���꣬�������ܶ��ؽ�
    gt_mat, lr_mat_file, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
        filename=args.data_path,
        wind_group=wind_group,
        source_group=source_group,
        time_step=time_step,
        scale_factor=args.scale_factor
    )
    if gt_mat is None:
        print('GKDM ���ݼ���ʧ�ܣ����� GKDM��')
        yk_np = None
    else:
        rco = args.kdm_rco
        gama = rco / 3.0
        yk_np = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=(args.mat_width, args.mat_height),
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_mat_file,
            Rco=rco,
            Gama=gama
        )
        # ���뵽 GT ��̬��Χ�����ڿ��ӶԱȣ�
        yk_np = np.clip(yk_np, gt_mat.min(), gt_mat.max())
        save_img(yk_np, os.path.join(args.save_dir,
                 f's{pick_idx}_Yk_kdm.png'), f'KDM (Rco={rco})')

    # 4) SwinIR_GDM(Ys)
    model_gdm, model_multi = create_models(args)
    load_weights(model_gdm, args.model_path_gdm)
    model_gdm = model_gdm.to(device).eval()
    with torch.no_grad():
        ys = model_gdm(lr)  # [1,1,H*,W*]
    ys_np = tensor_to_np(ys)
    save_img(ys_np, os.path.join(args.save_dir,
             f's{pick_idx}_Ys_swinir_gdm.png'), 'SwinIR_GDM')

    # 5) SwinIR_Multi(Ym)
    load_weights(model_multi, args.model_path_multi)
    model_multi = model_multi.to(device).eval()
    with torch.no_grad():
        ym, _ = model_multi(lr)  # [1,1,H*,W*], [1,2] (gsl)
    ym_np = tensor_to_np(ym)
    save_img(ym_np, os.path.join(args.save_dir, f's{pick_idx}_Ym_swinir_multi_{args.multi_variant}.png'),
             f'SwinIR_Multi({args.multi_variant})')

    # ��ѡ�������ָ��
    def psnr_np(a, b):
        mse = np.mean((a - b) ** 2)
        return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    print(f'[Sample idx={pick_idx}]')
    print(f'  Bicubic  PSNR: {psnr_np(yb_np, hr_np):.2f} dB')
    if yk_np is not None:
        print(f'  KDM      PSNR: {psnr_np(yk_np, hr_np):.2f} dB')
    print(f'  SwinIR_GDM  PSNR: {psnr_np(ys_np, hr_np):.2f} dB')
    print(f'  SwinIR_Multi PSNR: {psnr_np(ym_np, hr_np):.2f} dB')

    print('��ɣ������ 6 ��ͼƬ���� GKDM ���ݿ��ã���')


if __name__ == '__main__':
    main()
