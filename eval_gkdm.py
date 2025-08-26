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
            output_lr_index_mat[height_idx, width_idx,
                                :] = np.array([hr_x_idx, hr_y_idx])
    return output_lr_mat, output_sparse_mat, output_lr_index_mat


# 全新的数据加载函数，以匹配h5_dataset.py 逻辑
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
    try:
        with h5py.File(filename, 'r') as f:
            # 检查路径是否存在
            if wind_group not in f:
                raise KeyError(
                    f"Wind group '{wind_group}' not found in H5 file.")
            if source_group not in f[wind_group]:
                raise KeyError(
                    f"Source group '{source_group}' not found in wind group '{wind_group}'.")

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
                    sparse_mat_corrected[hr_coord[0], hr_coord[1]
                                         ] = lr_mat_from_file[h_idx, w_idx]

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
    assert len(
        measure_mat.shape) == 2, f'[get_gaussian_kdm_matrix] measure_mat should be a 2D matrix'
    assert len(
        sample_index_mat.shape) == 3, f'[get_gaussian_kdm_matrix] index_mat should be a 3D matrix'
    assert len(
        mat_real_size) == 2, f'[get_gaussian_kdm_matrix] mat_real_size should be a tuple with two elements'
    rows, columns = measure_mat.shape
    point_real_size = mat_real_size[0] / rows, mat_real_size[1] / columns
    samples_number = sample_index_mat.shape[0] * sample_index_mat.shape[1]
    x, y = np.mgrid[0:rows, 0:columns]
    mat_index = np.array(
        list(map(lambda xe, ye: [(ex, ey) for ex, ey in zip(xe, ye)], x, y)))
    sample_index_list = np.reshape(sample_index_mat, (-1, 2))
    sample_conc_list = np.reshape(sample_conc_mat, (-1,))
    sample_index_mat_extend = np.zeros((rows, columns, samples_number, 2))
    sample_index_mat_extend[:, :, :, 0] = np.tile(sample_index_list[:, 0], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_index_mat_extend[:, :, :, 1] = np.tile(sample_index_list[:, 1], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_conc_mat_extend = np.tile(sample_conc_list, (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    mat_index_extend = np.tile(mat_index, (samples_number, )).reshape(
        (rows, columns, 2, samples_number), order='F').transpose((0, 1, 3, 2))
    distance_matrix_index = mat_index_extend - sample_index_mat_extend
    distance_matrix_index[:, :, :, 0] = distance_matrix_index[:,
                                                              :, :, 0] * point_real_size[0]
    distance_matrix_index[:, :, :, 1] = distance_matrix_index[:,
                                                              :, :, 1] * point_real_size[1]
    distance_matrix = distance_matrix_index[:, :, :,
                                            0] ** 2 + distance_matrix_index[:, :, :, 1] ** 2
    gama_sq = Gama ** 2
    if gama_sq == 0:
        gama_sq = 1e-12
    sample_conc_mat_extend = np.where(
        distance_matrix < Rco ** 2, sample_conc_mat_extend, 0)
    distance_matrix = np.where(distance_matrix < Rco ** 2, distance_matrix, 0)
    w_mat_extend = (1 / (2 * np. pi * gama_sq)) * \
        np.exp(-(distance_matrix) / 2 / gama_sq)
    conc = w_mat_extend * sample_conc_mat_extend
    w_sum = w_mat_extend.sum(axis=2)
    conc_sum = conc.sum(axis=2)
    reconstruct_mat = np.divide(
        conc_sum, w_sum, out=np.zeros_like(conc_sum), where=w_sum != 0)
    return reconstruct_mat


def gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=None, save_path='gkdm_result.png'):
    """
    可视化KDM的输入和输出（去除sparse_mat_corrected，只保留3个子图）
    不在每个子图右侧显示图例（颜色条）
    """
    # 检查数据范围
    print(f"Visualization data ranges:")
    print(f"  GT: [{gt_mat.min():.6f}, {gt_mat.max():.6f}]")
    print(f"  LR: [{lr_mat.min():.6f}, {lr_mat.max():.6f}]")
    print(
        f"  Reconstruct: [{reconstruct_mat.min():.6f}, {reconstruct_mat.max():.6f}]")
    reconstruct_mat_clipped = np.clip(
        reconstruct_mat, gt_mat.min(), gt_mat.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 真实值
    im1 = axes[0].imshow(gt_mat, cmap='viridis')
    axes[0].text(0.5, -0.08, '(a) Ground Truth (HR)', transform=axes[0].transAxes,
                 ha='center', va='top', fontsize=14)

    # 低分辨率图 (从文件加载)
    im2 = axes[1].imshow(lr_mat, cmap='viridis')
    axes[1].text(0.5, -0.08, '(b) LR', transform=axes[1].transAxes,
                 ha='center', va='top', fontsize=14)

    # 第三个子图标题增加Rco值
    if rco_value is not None:
        axes[2].text(0.5, -0.08, f'(c) KDM (Rco={rco_value})', transform=axes[2].transAxes,
                     ha='center', va='top', fontsize=14)
    else:
        axes[2].text(0.5, -0.08, '(c) KDM', transform=axes[2].transAxes,
                     ha='center', va='top', fontsize=14)
    im3 = axes[2].imshow(reconstruct_mat_clipped, cmap='viridis')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    # 确保底部题注不被裁剪
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved as {save_path}")

    # 保存用于后期处理的数据（CSV），便于在 Origin 等软件中进一步调整
    try:
        save_dir = os.path.dirname(
            save_path) if os.path.dirname(save_path) else '.'
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        gt_csv_path = os.path.join(save_dir, f"{base_name}_GT.csv")
        lr_csv_path = os.path.join(save_dir, f"{base_name}_LR.csv")
        recon_csv_path = os.path.join(save_dir, f"{base_name}_Reconstruct.csv")
        np.savetxt(gt_csv_path, gt_mat, delimiter=',', fmt='%.10g')
        np.savetxt(lr_csv_path, lr_mat, delimiter=',', fmt='%.10g')
        np.savetxt(recon_csv_path, reconstruct_mat_clipped,
                   delimiter=',', fmt='%.10g')
        print(f"Data saved: {gt_csv_path}, {lr_csv_path}, {recon_csv_path}")
    except Exception as e:
        print(f"Warning: failed to save CSV data for gkdm_result: {e}")

    plt.show()


def calculate_normalized_mse(gt_mat, reconstruct_mat):
    """
    计算归一化MSE，将重建结果归一化到GT范围后再计算MSE
    """
    gt_range = gt_mat.max() - gt_mat.min()
    recon_range = reconstruct_mat.max() - reconstruct_mat.min()
    if recon_range > 0:
        reconstruct_normalized = (
            reconstruct_mat - reconstruct_mat.min()) / recon_range * gt_range + gt_mat.min()
    else:
        reconstruct_normalized = np.zeros_like(reconstruct_mat)
    mse_normalized = np.mean((gt_mat - reconstruct_normalized) ** 2)
    return mse_normalized


def test_rco_parameters(h5_file_path, wind_group, source_group, time_step,
                        scale_factor, mat_real_size, rco_values):
    """
    测试不同Rco值对重建质量的影响，只返回归一化MSE
    """
    print("Starting Rco parameter test...")
    gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
        h5_file_path, wind_group, source_group, time_step, scale_factor
    )
    if gt_mat is None:
        print("Data loading failed")
        return [], []
    rco_list = []
    mse_normalized_list = []
    for rco in rco_values:
        gama = rco / 3.0
        reconstruct_mat = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=mat_real_size,
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_mat,
            Rco=rco,
            Gama=gama
        )
        mse_normalized = calculate_normalized_mse(gt_mat, reconstruct_mat)
        rco_list.append(rco)
        mse_normalized_list.append(mse_normalized)
        print(f"  Rco={rco}, Gama={gama:.4f}, MSE={mse_normalized:.6f}")
    return rco_list, mse_normalized_list


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GKDM (Gaussian Kernel Density Mapping) Reconstruction Algorithm')

    # 必需参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='H5 data file path (required)')

    # 保存目录参数
    parser.add_argument('--save_dir', type=str, default='gkdm_results',
                        help='Results save directory')

    # 测试模式选择
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='Test mode: generalization, test_set, all_generalization, all_test_set')

    # 样本选择参数
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='Generalization test sample specs, separated by semicolon, e.g.: wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='Test set indices, separated by comma, e.g.: 1,2,3,4,5')

    # Rco参数范围
    parser.add_argument('--rco_start', type=float, default=0.1,
                        help='Start value for Rco range')
    parser.add_argument('--rco_end', type=float, default=3.0,
                        help='End value for Rco range')
    parser.add_argument('--rco_step', type=float, default=0.1,
                        help='Step size for Rco range')

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
                        print(
                            f"Found sample: {wind_group}, {source_group}, {time_step}")
                    else:
                        print(
                            f"Warning: Data not found for {wind_group}, {source_group}, {time_step}")
                else:
                    print(
                        f"Warning: Group not found for {wind_group}, {source_group}")

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
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            h5_file_path, seed=42)

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


def evaluate_multiple_samples(h5_file_path, sample_indices, rco_values, mat_real_size, scale_factor, save_dir):
    """评估多个样本，计算每个Rco的平均MSE，选择平均MSE最小的Rco"""
    print(f"Evaluating {len(sample_indices)} samples...")

    all_results = {}
    for rco in rco_values:
        all_results[rco] = []

    # 记录每个样本的最佳Rco值（用于对比分析）
    sample_best_rcos = []

    for i, sample_info in enumerate(sample_indices):
        print(f"\n--- Processing Sample {i+1}/{len(sample_indices)} ---")
        print(f"Sample info: {sample_info}")

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

        # 测试不同Rco值，记录这个样本的所有MSE
        sample_mse_results = {}
        print(f"Testing Rco values for sample {i+1}:")
        for rco in rco_values:
            gama = rco / 3.0
            reconstruct_mat = get_gaussian_kdm_matrix(
                measure_mat=sparse_mat,
                mat_real_size=mat_real_size,
                sample_index_mat=lr_index_mat,
                sample_conc_mat=lr_mat,
                Rco=rco,
                Gama=gama
            )

            mse_normalized = calculate_normalized_mse(gt_mat, reconstruct_mat)
            all_results[rco].append(mse_normalized)
            sample_mse_results[rco] = mse_normalized
            print(
                f"  Rco={rco:4.2f}, Gama={gama:6.4f}, MSE={mse_normalized:.6f}")

        # 找到这个样本的最佳Rco值（用于对比分析）
        best_rco_for_sample = min(
            sample_mse_results, key=sample_mse_results.get)
        best_mse_for_sample = sample_mse_results[best_rco_for_sample]
        sample_best_rcos.append(best_rco_for_sample)
        print(
            f"  => Best Rco for this sample: {best_rco_for_sample} (MSE={best_mse_for_sample:.6f})")

    # 计算每个Rco的平均MSE
    avg_mse_per_rco = {}
    print("\n--- Calculating Average MSE for Each Rco ---")
    for rco in rco_values:
        if all_results[rco]:  # 确保有数据
            avg_mse_per_rco[rco] = np.mean(all_results[rco])
            print(
                f"Rco={rco:4.2f}: Average MSE = {avg_mse_per_rco[rco]:.6f} (based on {len(all_results[rco])} samples)")
        else:
            avg_mse_per_rco[rco] = float('inf')  # 如果没有数据，设为无穷大
            print(f"Rco={rco:4.2f}: No data available")

    # 找到平均MSE最小的Rco值
    best_rco = min(avg_mse_per_rco, key=avg_mse_per_rco.get)
    best_avg_mse = avg_mse_per_rco[best_rco]

    # 统计每个Rco值被选为最佳的次数（用于对比分析）
    rco_counts = {}
    for rco in rco_values:
        rco_counts[rco] = sample_best_rcos.count(rco)

    print(f"\n=== Average MSE Analysis ===")
    for rco in sorted(rco_values):
        if avg_mse_per_rco[rco] != float('inf'):
            count = rco_counts[rco]
            print(
                f"Rco={rco:4.2f}: Average MSE = {avg_mse_per_rco[rco]:.6f} (best for {count} samples)")

    print(f"\n=== Final Selection ===")
    print(
        f"Best Rco by average MSE: {best_rco} (Average MSE={best_avg_mse:.6f})")

    return best_rco, best_avg_mse, rco_counts, sample_best_rcos, avg_mse_per_rco


def plot_rco_vs_mse(rco_list, mse_normalized_list, save_path='rco_vs_mse.png'):
    """绘制Rco值与归一化MSE的关系图"""
    plt.figure(figsize=(10, 6))
    plt.plot(rco_list, mse_normalized_list, 'bo-',
             linewidth=2, markersize=8, label='MSE')
    best_mse_idx = np.argmin(mse_normalized_list)
    best_rco = rco_list[best_mse_idx]
    best_mse = mse_normalized_list[best_mse_idx]
    plt.plot(best_rco, best_mse, 'ro', markersize=12,
             label=f'Best MSE (Rco={best_rco}, MSE={best_mse:.6f})')
    plt.xlabel('Rco Value', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Effect of Rco Value on MSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Rco vs MSE plot saved as {save_path}")

    # 保存曲线数据为 CSV，便于在 Origin 等软件中重绘
    try:
        save_dir = os.path.dirname(
            save_path) if os.path.dirname(save_path) else '.'
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        csv_path = os.path.join(save_dir, f"{base_name}_data.csv")
        data_mat = np.column_stack(
            (np.asarray(rco_list), np.asarray(mse_normalized_list)))
        np.savetxt(csv_path, data_mat, delimiter=',',
                   fmt='%.10g', header='Rco,MSE', comments='')
        print(f"Data saved: {csv_path}")
    except Exception as e:
        print(f"Warning: failed to save CSV data for rco_vs_mse: {e}")

    plt.show()
    return best_rco, best_mse


def plot_average_mse_vs_rco(rco_values, avg_mse_per_rco, best_rco, save_path='average_mse_vs_rco.png'):
    """绘制平均MSE vs Rco的关系图"""
    plt.figure(figsize=(10, 6))

    # 准备数据
    rco_list = sorted(rco_values)
    mse_list = [avg_mse_per_rco[rco] for rco in rco_list]

    plt.plot(rco_list, mse_list, 'bo-', linewidth=2,
             markersize=8, label='Average MSE')

    # 标记最佳Rco值
    best_mse = avg_mse_per_rco[best_rco]
    plt.plot(best_rco, best_mse, 'ro', markersize=12,
             label=f'Best Average MSE (Rco={best_rco}, MSE={best_mse:.6f})')

    plt.xlabel('Rco Value', fontsize=12)
    plt.ylabel('Average MSE', fontsize=12)
    plt.title('Average MSE vs Rco Value Across All Samples', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Average MSE vs Rco plot saved as {save_path}")
    plt.show()

    return best_rco, best_mse


if __name__ == '__main__':
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置参数
    h5_file_path = args.data_path
    mat_real_size = (args.mat_width, args.mat_height)
    scale_factor = args.scale_factor

    # 生成Rco值范围
    rco_values = [round(x, 2) for x in np.arange(
        args.rco_start, args.rco_end + args.rco_step, args.rco_step)]

    print("--- Starting GKDM Parameter Test ---")
    print(f"Data file path: {h5_file_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"Test mode: {args.test_mode}")
    print(
        f"Rco range: {args.rco_start} to {args.rco_end}, step: {args.rco_step}")
    print(f"Testing Rco values: {rco_values}")

    # 根据测试模式选择样本
    sample_indices = []
    if args.test_mode == 'generalization':
        if args.sample_specs:
            sample_indices = get_sample_indices_from_specs(
                args.sample_specs, h5_file_path)
            print(
                f"Using generalization test mode with {len(sample_indices)} samples")
        else:
            print("Error: generalization test mode requires sample_specs parameter")
            exit(1)
    elif args.test_mode == 'test_set':
        if args.test_indices:
            sample_indices = get_test_set_indices(
                args.test_indices, h5_file_path)
            print(f"Using test set mode with {len(sample_indices)} samples")
        else:
            print("Error: test_set mode requires test_indices parameter")
            exit(1)
    elif args.test_mode == 'all_generalization':
        # 获取所有可用的样本
        sample_indices = get_sample_indices_from_specs(
            "wind1_0,s1,50;wind1_0,s2,50;wind2_0,s1,50;wind2_0,s2,50;wind3_0,s1,50;wind3_0,s2,50", h5_file_path)
        print(
            f"Using all generalization test mode with {len(sample_indices)} samples")
    elif args.test_mode == 'all_test_set':
        # 获取所有测试集样本
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            h5_file_path, seed=42)
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

    # 评估多个样本
    if len(sample_indices) > 1:
        best_rco, avg_mse, rco_counts, sample_best_rcos, avg_mse_per_rco = evaluate_multiple_samples(
            h5_file_path, sample_indices, rco_values, mat_real_size, scale_factor, args.save_dir)

        # 绘制平均MSE vs Rco图
        plot_average_mse_vs_rco(rco_values, avg_mse_per_rco, best_rco, os.path.join(
            args.save_dir, 'average_mse_vs_rco.png'))

        print(f"\n=== Test Results Summary ===")
        print(
            f"Best Rco by average MSE: {best_rco}, Average MSE: {avg_mse:.6f}")

        # 使用最佳参数进行最终重建
        print(f"\nReconstructing with best parameters...")
        sample_info = sample_indices[0]  # 使用第一个样本进行可视化
        gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
            h5_file_path, sample_info['wind_group'], sample_info['source_group'], sample_info['time_step'], scale_factor
        )

        if gt_mat is not None:
            # 手动指定Rco
            custom_rco = best_rco  # 用于测试Rco值
            reconstruct_mat = get_gaussian_kdm_matrix(
                measure_mat=sparse_mat,
                mat_real_size=mat_real_size,
                sample_index_mat=lr_index_mat,
                sample_conc_mat=lr_mat,
                Rco=custom_rco,
                Gama=custom_rco/3.0
            )
            gkdm_flow(gt_mat, lr_mat, sparse_mat,
                      reconstruct_mat, rco_value=custom_rco)
    else:
        # 单样本测试（原有逻辑保持不变）
        sample_info = sample_indices[0]
        rco_list, mse_normalized_list = test_rco_parameters(
            h5_file_path,
            sample_info['wind_group'],
            sample_info['source_group'],
            sample_info['time_step'],
            scale_factor,
            mat_real_size,
            rco_values
        )

        if rco_list:
            best_rco, best_mse = plot_rco_vs_mse(
                rco_list, mse_normalized_list, os.path.join(args.save_dir, 'rco_vs_mse.png'))

            print(f"\n=== Test Results Summary ===")
            print(f"Best MSE Rco value: {best_rco}, MSE: {best_mse:.6f}")
            print(f"All results:")
            for i, rco in enumerate(rco_list):
                print(f"  Rco={rco}: MSE={mse_normalized_list[i]:.6f}")

            print(f"\nReconstructing with best parameters...")
            gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
                h5_file_path, sample_info['wind_group'], sample_info['source_group'], sample_info['time_step'], scale_factor
            )

            if gt_mat is not None:
                reconstruct_mat = get_gaussian_kdm_matrix(
                    measure_mat=sparse_mat,
                    mat_real_size=mat_real_size,
                    sample_index_mat=lr_index_mat,
                    sample_conc_mat=lr_mat,
                    Rco=best_rco,
                    Gama=best_rco/3.0
                )
                gkdm_flow(gt_mat, lr_mat, sparse_mat,
                          reconstruct_mat, rco_value=best_rco)
        else:
            print("Parameter test failed")
