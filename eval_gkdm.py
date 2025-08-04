# 导入操作系统相关模块
import os
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入时间模块
import time
# 导入数学函数模块
import math
# 导入HDF5文件读取模块
import h5py
# 导入数值计算库
import numpy as np
# 导入命令行参数解析模块
import argparse


# 这个函数保持不变，我们仍需要它来计算采样点坐标
def down_sample_2D_verbose(mat, scale_factor):
    """
    对2D矩阵进行下采样，生成低分辨率矩阵、稀疏矩阵和索引矩阵
    """
    scale_factor = math.floor(scale_factor)
    height, width = mat.shape
    output_height = height // scale_factor
    output_width = width // scale_factor
    res_height = height - output_height * scale_factor
    res_width = width - output_width * scale_factor
    start_heigh = res_height // 2
    start_width = res_width // 2
    output_lr_mat = np.zeros((output_height, output_width))
    output_lr_index_mat = np.zeros((output_height, output_width, 2))
    output_sparse_mat = np.zeros(mat.shape)
    for height_idx in range(output_height):
        for width_idx in range(output_width):
            hr_x_idx = start_heigh + height_idx * scale_factor
            hr_y_idx = start_width + width_idx * scale_factor
            output_lr_mat[height_idx, width_idx] = mat[hr_x_idx, hr_y_idx]
            output_sparse_mat[hr_x_idx, hr_y_idx] = mat[hr_x_idx, hr_y_idx]
            output_lr_index_mat[height_idx, width_idx, :] = np.array([hr_x_idx, hr_y_idx])
    return output_lr_mat, output_sparse_mat, output_lr_index_mat


#全新的数据加载函数，以匹配h5_dataset.py 逻辑
def load_data_from_swinir_h5(filename, wind_group, source_group, time_step, scale_factor):
    """
    根据 h5_dataset.py 的逻辑从H5文件中加载数据，并为KDM准备输入。

    :param filename: .h5 文件路径
    :param wind_group: 风场组名称 (e.g., 'wind1_0')
    :param source_group: 泄漏源组名称 (e.g., 's1')
    :param time_step: 时间步索引 (e.g., 1)
    :param scale_factor: 下采样比例，必须与生成LR数据时使用的一致
    :return: (gt_mat, lr_mat_from_file, sparse_mat_corrected, lr_index_mat)
    """
    print(f"Loading data: Wind={wind_group}, Source={source_group}, Time={time_step}")
    try:
        with h5py.File(filename, 'r') as f:
            # 检查路径是否存在
            if wind_group not in f:
                raise KeyError(f"Wind group '{wind_group}' not found in H5 file.")
            if source_group not in f[wind_group]:
                raise KeyError(f"Source group '{source_group}' not found in wind group '{wind_group}'.")

            # 1. 直接从文件读取 HR 和 LR 数据
            hr_dataset_name = f'HR_{time_step}'
            lr_dataset_name = f'LR_{time_step}'
            if hr_dataset_name not in f[wind_group][source_group]:
                 raise KeyError(f"Dataset '{hr_dataset_name}' not found.")
            if lr_dataset_name not in f[wind_group][source_group]:
                 raise KeyError(f"Dataset '{lr_dataset_name}' not found.")

            gt_mat = f[wind_group][source_group][hr_dataset_name][:]
            lr_mat_from_file = f[wind_group][source_group][lr_dataset_name][:]

            # 2. 调用 down_sample_2D_verbose，主要目的是为了获取采样坐标 `lr_index_mat`
            _, _, lr_index_mat = down_sample_2D_verbose(gt_mat, scale_factor)

            # 3. (关键步骤) 使用文件中的LR数据值 和 down_sample生成的坐标 来重建一个正确的稀疏矩阵
            sparse_mat_corrected = np.zeros_like(gt_mat)
            for h_idx in range(lr_index_mat.shape[0]):
                for w_idx in range(lr_index_mat.shape[1]):
                    # 获取在高分辨率图中的坐标
                    hr_coord = lr_index_mat[h_idx, w_idx, :].astype(int)
                    # 将文件中的低分辨率值，填充到稀疏矩阵的对应位置
                    sparse_mat_corrected[hr_coord[0], hr_coord[1]] = lr_mat_from_file[h_idx, w_idx]

            return gt_mat, lr_mat_from_file, sparse_mat_corrected, lr_index_mat

    except KeyError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None


# KDM核心算法函数保持不变
def get_gaussian_kdm_matrix(measure_mat,
                            mat_real_size,
                            sample_index_mat,
                            sample_conc_mat,
                            Rco, Gama):
    '''
    使用矩阵运算进行高斯核密度映射重建
    '''
    # 函数其余部分保持不变...
    assert len(measure_mat.shape) == 2, f'[get_gaussian_kdm_matrix] measure_mat should be a 2D matrix'
    assert len(sample_index_mat.shape) == 3, f'[get_gaussian_kdm_matrix] index_mat should be a 3D matrix'
    assert len(mat_real_size) == 2, f'[get_gaussian_kdm_matrix] mat_real_size should be a tuple with two elements'
    rows, columns = measure_mat.shape
    point_real_size = mat_real_size[0] / rows, mat_real_size[1] / columns
    samples_number = sample_index_mat.shape[0] * sample_index_mat.shape[1]
    x, y = np.mgrid[0:rows, 0:columns]
    mat_index = np.array(list(map(lambda xe, ye: [(ex, ey) for ex, ey in zip(xe, ye)], x, y)))
    sample_index_list = np.reshape(sample_index_mat, (-1, 2))
    sample_conc_list = np.reshape(sample_conc_mat, (-1,))
    sample_index_mat_extend = np.zeros((rows, columns, samples_number, 2))
    sample_index_mat_extend[:, :, :, 0] = np.tile(sample_index_list[:,0], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_index_mat_extend[:, :, :, 1] = np.tile(sample_index_list[:,1], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_conc_mat_extend = np.tile(sample_conc_list, (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    mat_index_extend = np.tile(mat_index, (samples_number, )).reshape(
        (rows, columns, 2, samples_number), order='F').transpose((0, 1, 3, 2))
    distance_matrix_index = mat_index_extend - sample_index_mat_extend
    distance_matrix_index[:, :, :, 0] = distance_matrix_index[:, :, :, 0] * point_real_size[0]
    distance_matrix_index[:, :, :, 1] = distance_matrix_index[:, :, :, 1] * point_real_size[1]
    distance_matrix = distance_matrix_index[:, :, :, 0] ** 2 + distance_matrix_index[:, :, :, 1] ** 2
    gama_sq = Gama ** 2
    if gama_sq == 0:
        gama_sq = 1e-12
    sample_conc_mat_extend = np.where(distance_matrix < Rco ** 2, sample_conc_mat_extend, 0)
    distance_matrix = np.where(distance_matrix < Rco ** 2, distance_matrix, 0)
    w_mat_extend = (1 / (2 * np. pi* gama_sq)) * np.exp(-(distance_matrix) / 2 / gama_sq)
    conc = w_mat_extend * sample_conc_mat_extend
    w_sum = w_mat_extend.sum(axis=2)
    conc_sum = conc.sum(axis=2)
    reconstruct_mat = np.divide(conc_sum, w_sum, out=np.zeros_like(conc_sum), where=w_sum!=0)
    return reconstruct_mat


# 可视化流程函数，更新了一下标题使其更精确
def gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=None):
    """
    可视化KDM的输入和输出（去除sparse_mat_corrected，只保留3个子图）
    """
    # 检查数据范围
    print(f"Visualization data ranges:")
    print(f"  GT: [{gt_mat.min():.6f}, {gt_mat.max():.6f}]")
    print(f"  LR: [{lr_mat.min():.6f}, {lr_mat.max():.6f}]")
    print(f"  Reconstruct: [{reconstruct_mat.min():.6f}, {reconstruct_mat.max():.6f}]")
    reconstruct_mat_clipped = np.clip(reconstruct_mat, gt_mat.min(), gt_mat.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 真实值
    im1 = axes[0].imshow(gt_mat, cmap='viridis')
    axes[0].set_title('Ground Truth (HR)')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 低分辨率图 (从文件加载)
    im2 = axes[1].imshow(lr_mat, cmap='viridis')
    axes[1].set_title('LR')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    # 第三个子图标题增加Rco值
    if rco_value is not None:
        axes[2].set_title(f'KDM (Rco={rco_value})')
    else:
        axes[2].set_title('KDM')
    im3 = axes[2].imshow(reconstruct_mat_clipped, cmap='viridis')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('gkdm_result.png', dpi=300, bbox_inches='tight')
    print("Image saved as gkdm_result.png")
    plt.show()


def calculate_normalized_mse(gt_mat, reconstruct_mat):
    """
    计算归一化MSE，将重建结果归一化到GT范围后再计算MSE
    """
    gt_range = gt_mat.max() - gt_mat.min()
    recon_range = reconstruct_mat.max() - reconstruct_mat.min()
    if recon_range > 0:
        reconstruct_normalized = (reconstruct_mat - reconstruct_mat.min()) / recon_range * gt_range + gt_mat.min()
    else:
        reconstruct_normalized = np.zeros_like(reconstruct_mat)
    mse_normalized = np.mean((gt_mat - reconstruct_normalized) ** 2)
    return mse_normalized





def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GKDM (Gaussian Kernel Density Mapping) Reconstruction Algorithm')
    
    # 必需参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='H5 data file path (required)')
    
    # 新增：直接指定Rco值
    parser.add_argument('--rco', type=float, required=True,
                       help='Rco value to use for reconstruction (required)')
    
    # 保存目录参数
    parser.add_argument('--save_dir', type=str, default='gkdm_results',
                       help='Results save directory')
    
    # 测试模式选择
    parser.add_argument('--test_mode', type=str, default='generalization',
                      choices=['generalization', 'test_set', 'all_generalization', 'all_test_set'],
                      help='Test mode: generalization, test_set, all_generalization, all_test_set')
    
    # 样本选择参数
    parser.add_argument('--sample_specs', type=str, default=None,
                      help='Generalization test sample specs, separated by semicolon, e.g.: wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                      help='Test set indices, separated by comma, e.g.: 1,2,3,4,5')
    
    # 物理参数
    parser.add_argument('--mat_width', type=float, default=9.6,
                       help='Physical width (meters)')
    parser.add_argument('--mat_height', type=float, default=9.6,
                       help='Physical height (meters)')
    parser.add_argument('--scale_factor', type=int, default=6,
                       help='Downsampling scale factor')
    
    args = parser.parse_args()
    return args


def get_sample_indices_from_specs(sample_specs_str, h5_file_path):
    """从样本规格字符串中获取样本索引"""
    if not sample_specs_str:
        return []
    
    sample_specs = [spec.strip() for spec in sample_specs_str.split(';')]
    indices = []
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            for spec in sample_specs:
                wind_group, source_group, time_step = spec.split(',')
                wind_group = wind_group.strip()
                source_group = source_group.strip()
                time_step = int(time_step.strip())
                
                # 检查数据是否存在
                if wind_group in f and source_group in f[wind_group]:
                    hr_key = f'HR_{time_step}'
                    lr_key = f'LR_{time_step}'
                    if hr_key in f[wind_group][source_group] and lr_key in f[wind_group][source_group]:
                        indices.append({
                            'wind_group': wind_group,
                            'source_group': source_group,
                            'time_step': time_step
                        })
                        print(f"Found sample: {wind_group}, {source_group}, {time_step}")
                    else:
                        print(f"Warning: Data not found for {wind_group}, {source_group}, {time_step}")
                else:
                    print(f"Warning: Group not found for {wind_group}, {source_group}")
    
    except Exception as e:
        print(f"Error reading H5 file: {e}")
        return []
    
    return indices


def get_test_set_indices(test_indices_str, h5_file_path):
    """从测试集索引字符串中获取样本索引"""
    if not test_indices_str:
        return []
    
    try:
        # 解析测试集索引
        indices = [int(idx.strip()) for idx in test_indices_str.split(',')]
        
        # 加载数据集获取测试集
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(h5_file_path, seed=42)
        
        # 获取测试集的样本信息
        sample_indices = []
        for idx in indices:
            if idx < len(test_dataset.data_indices):
                sample_info = test_dataset.data_indices[idx]
                sample_indices.append(sample_info)
                print(f"Found test set sample {idx}: {sample_info}")
            else:
                print(f"Warning: Test index {idx} out of range")
        
        return sample_indices
    
    except Exception as e:
        print(f"Error processing test set indices: {e}")
        return []


def evaluate_samples_with_fixed_rco(h5_file_path, sample_indices, rco_value, mat_real_size, scale_factor, save_dir):
    """使用固定的Rco值评估多个样本，计算平均MSE"""
    print(f"Evaluating {len(sample_indices)} samples with fixed Rco={rco_value}...")
    
    mse_values = []
    gama = rco_value / 3.0
    
    for i, sample_info in enumerate(sample_indices):
        print(f"\nEvaluating sample {i+1}/{len(sample_indices)}: {sample_info}")
        
        # 加载数据
        gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
            h5_file_path,
            sample_info['wind_group'],
            sample_info['source_group'],
            sample_info['time_step'],
            scale_factor
        )
        
        if gt_mat is None:
            print(f"Failed to load data for sample {i+1}")
            continue
        
        # 使用指定的Rco值进行重建
        reconstruct_mat = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=mat_real_size,
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_mat,
            Rco=rco_value,
            Gama=gama
        )
        
        mse_normalized = calculate_normalized_mse(gt_mat, reconstruct_mat)
        mse_values.append(mse_normalized)
        
        print(f"  Sample {i+1} MSE: {mse_normalized:.6f}")
    
    if mse_values:
        avg_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        print(f"\n=== Results Summary ===")
        print(f"Rco value: {rco_value}")
        print(f"Number of samples: {len(mse_values)}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Standard deviation MSE: {std_mse:.6f}")
        print(f"Min MSE: {min(mse_values):.6f}")
        print(f"Max MSE: {max(mse_values):.6f}")
        
        return avg_mse, std_mse, mse_values
    else:
        print("No valid samples evaluated!")
        return None, None, []








if __name__ == '__main__':
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置参数
    h5_file_path = args.data_path
    mat_real_size = (args.mat_width, args.mat_height)
    scale_factor = args.scale_factor
    rco_value = args.rco
    
    print("--- Starting GKDM Evaluation with Fixed Rco ---")
    print(f"Data file path: {h5_file_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"Test mode: {args.test_mode}")
    print(f"Fixed Rco value: {rco_value}")
    
    # 根据测试模式选择样本
    sample_indices = []
    if args.test_mode == 'generalization':
        if args.sample_specs:
            sample_indices = get_sample_indices_from_specs(args.sample_specs, h5_file_path)
            print(f"Using generalization test mode with {len(sample_indices)} samples")
        else:
            print("Error: generalization test mode requires sample_specs parameter")
            exit(1)
    elif args.test_mode == 'test_set':
        if args.test_indices:
            sample_indices = get_test_set_indices(args.test_indices, h5_file_path)
            print(f"Using test set mode with {len(sample_indices)} samples")
        else:
            print("Error: test_set mode requires test_indices parameter")
            exit(1)
    elif args.test_mode == 'all_generalization':
        # 获取所有可用的样本
        sample_indices = get_sample_indices_from_specs("wind1_0,s1,50;wind1_0,s2,50;wind2_0,s1,50;wind2_0,s2,50;wind3_0,s1,50;wind3_0,s2,50", h5_file_path)
        print(f"Using all generalization test mode with {len(sample_indices)} samples")
    elif args.test_mode == 'all_test_set':
        # 获取所有测试集样本
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(h5_file_path, seed=42)
        sample_indices = test_dataset.data_indices
        print(f"Using all test set mode with {len(sample_indices)} samples")
    else:
        print("Using single sample test mode")
        # 使用默认样本
        sample_indices = [{
            'wind_group': 'wind1_0',
            'source_group': 's1',
            'time_step': 50
        }]
    
    if not sample_indices:
        print("No samples to evaluate!")
        exit(1)
    
    # 使用固定Rco值评估样本
    avg_mse, std_mse, mse_values = evaluate_samples_with_fixed_rco(
        h5_file_path, sample_indices, rco_value, mat_real_size, scale_factor, args.save_dir
    )
    
    if avg_mse is not None:
        print(f"\n=== Final Results ===")
        print(f"Rco: {rco_value}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Standard deviation: {std_mse:.6f}")
        
        # 使用第一个样本进行可视化
        print(f"\nGenerating visualization for first sample...")
        sample_info = sample_indices[0]
        gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
            h5_file_path, sample_info['wind_group'], sample_info['source_group'], sample_info['time_step'], scale_factor
        )
        
        if gt_mat is not None:
            gama = rco_value / 3.0
            reconstruct_mat = get_gaussian_kdm_matrix(
                measure_mat=sparse_mat,
                mat_real_size=mat_real_size,
                sample_index_mat=lr_index_mat,
                sample_conc_mat=lr_mat,
                Rco=rco_value,
                Gama=gama
            )
            gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=rco_value)
    else:
        print("Evaluation failed!")