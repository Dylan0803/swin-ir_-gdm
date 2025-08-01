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


# ADDED: 全新的数据加载函数，以匹配你的 h5_dataset.py 逻辑
def load_data_from_swinir_h5(filename, wind_group, source_group, time_step, scale_factor):
    """
    根据 h5_dataset.py 的逻辑从H5文件中加载数据，并为KDM准备输入。

    :param filename: .h5 文件路径
    :param wind_group: 风场组名称 (e.g., 'wind_0_5')
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
    w_mat_extend = (1 / (2 * np.pi * gama_sq)) * np.exp(-(distance_matrix) / 2 / gama_sq)
    conc = w_mat_extend * sample_conc_mat_extend
    w_sum = w_mat_extend.sum(axis=2)
    conc_sum = conc.sum(axis=2)
    reconstruct_mat = np.divide(conc_sum, w_sum, out=np.zeros_like(conc_sum), where=w_sum!=0)
    return reconstruct_mat


# 可视化流程函数，更新了一下标题使其更精确
def gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat):
    """
    可视化KDM的输入和输出
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 真实值
    im1 = axes[0].imshow(gt_mat, cmap='viridis')
    axes[0].set_title('Ground Truth (HR)')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 低分辨率图 (从文件加载)
    im2 = axes[1].imshow(lr_mat, cmap='viridis')
    axes[1].set_title('Low-Resolution (from file)')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 稀疏采样点 (重建后)
    im3 = axes[2].imshow(sparse_mat, cmap='viridis')
    axes[2].set_title(f'Corrected Sparse Matrix')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 重建结果
    im4 = axes[3].imshow(reconstruct_mat, cmap='viridis')
    axes[3].set_title('GKDM Reconstruction')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # =======================================================================
    #  命令行参数解析
    # =======================================================================
    parser = argparse.ArgumentParser(description='GKDM (Gaussian Kernel Density Mapping) 重建算法')
    
    # 必需参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='H5数据文件路径 (必需)')
    
    args = parser.parse_args()
    
    # =======================================================================
    #  TODO: 请在这里根据你的数据修改以下参数
    # =======================================================================

    # --- 1. 数据集参数 ---
    # 使用命令行传入的数据路径
    h5_file_path = args.data_path
    
    # 想测试的样本 (风场, 泄漏源, 时间步)
    wind_group_to_test = 'wind_0_5'
    source_group_to_test = 's1'
    time_step_to_test = 10  # e.g., for HR_10, LR_10

    # --- 2. 物理和实验参数 ---
    # 数据的真实物理尺寸 (单位: 米)
    mat_real_size = (9.6, 9.6) 
    
    # TODO: (非常重要) 必须与生成LR数据时用的下采样因子一致
    scale_factor = 6 

    # --- 3. KDM算法超参数 (建议调整以获得最佳效果) ---
    # TODO: 尝试调整 Rco (截断半径) 和 Gama (高斯核标准差)
    Rco_value = 2.5
    Gama_value = Rco_value / 3.0

    # =======================================================================
    #  执行流程 (通常无需修改以下部分)
    # =======================================================================
    
    print("--- Starting GKDM Test with Command Line Data Path ---")
    print(f"数据文件路径: {h5_file_path}")
    
    # 1. 使用新的加载函数加载所有需要的数据
    gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
        h5_file_path,
        wind_group_to_test,
        source_group_to_test,
        time_step_to_test,
        scale_factor
    )
    
    # 2. 检查数据是否加载成功
    if gt_mat is not None:
        print(f"Data loaded successfully. GT shape: {gt_mat.shape}, LR shape: {lr_mat.shape}")

        # 3. 执行KDM重建
        print(f"Running GKDM with Rco={Rco_value}, Gama={Gama_value:.4f}")
        time_start = time.time()
        # KDM算法的输入:
        # - measure_mat: 使用我们修正后的稀疏矩阵
        # - sample_index_mat: 使用down_sample生成的坐标矩阵
        # - sample_conc_mat: 使用从文件中加载的低分辨率矩阵
        reconstruct_mat = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat, 
            mat_real_size=mat_real_size, 
            sample_index_mat=lr_index_mat, 
            sample_conc_mat=lr_mat, 
            Rco=Rco_value, 
            Gama=Gama_value
        )
        time_end = time.time()
        print(f'Reconstruction time: {time_end - time_start:.4f}s')

        # 4. 可视化结果
        gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat)

    else:
        print("\n--- GKDM Test Failed: Data could not be loaded. Please check parameters and file paths. ---")
        print("使用示例:")
        print("python gkdm.py --data_path your_data.h5")