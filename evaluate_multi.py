import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_dataset
import pandas as pd

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
            position_error = torch.sqrt(torch.sum((gsl_out - source_pos) ** 2, dim=1))
            max_pos_error = torch.sqrt(torch.sum((gsl_out - hr_max_pos) ** 2, dim=1))
            
            metrics['gsl_position_error'].extend(position_error.cpu().numpy())
            metrics['gsl_max_pos_error'].extend(max_pos_error.cpu().numpy())
    
    # 计算平均指标
    results = {
        'GDM_PSNR': np.mean(metrics['gdm_psnr']),
        'GDM_SSIM': np.mean(metrics['gdm_ssim']),
        'GSL_Position_Error': np.mean(metrics['gsl_position_error']),
        'GSL_MaxPos_Error': np.mean(metrics['gsl_max_pos_error'])
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

def main():
    # 设置参数
    model_path = "experiments/your_experiment/model_name/best_model.pth"  # 修改为你的模型路径
    data_path = "path/to/your/dataset.h5"  # 修改为你的数据集路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = SwinIRMulti(
        img_size=16,
        patch_size=1,
        in_chans=1,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        upscale=6,
        img_range=1.,
        upsampler='nearest+conv',
        resi_connection='1conv'
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载测试数据
    _, test_set = generate_train_valid_dataset(data_path, train_ratio=0.8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
    
    # 评估模型
    results, metrics = evaluate_model(model, test_loader, device)
    
    # 打印结果
    print("\n=== 模型评估结果 ===")
    print(f"GDM PSNR: {results['GDM_PSNR']:.2f} dB")
    print(f"GDM SSIM: {results['GDM_SSIM']:.4f}")
    print(f"GSL Position Error: {results['GSL_Position_Error']:.4f}")
    print(f"GSL Max Position Error: {results['GSL_MaxPos_Error']:.4f}")
    
    # 保存结果
    save_dir = os.path.dirname(model_path)
    pd.DataFrame([results]).to_csv(os.path.join(save_dir, 'evaluation_results.csv'), index=False)
    plot_metrics(metrics, save_dir)
    
    print(f"\n评估结果已保存至: {save_dir}")

if __name__ == '__main__':
    main()