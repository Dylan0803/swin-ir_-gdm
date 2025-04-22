import h5py
import numpy as np


def average_downsample(data, scale):
    """将二维矩阵平均下采样，目标是从96x96到16x16（scale=6）"""
    h, w = data.shape
    new_h, new_w = h // scale, w // scale
    
    # 确保输入尺寸能被scale整除
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"Input dimensions ({h}, {w}) must be divisible by scale {scale}")
    
    return data.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))


def check_dataset_dimensions(h5_path):
    """检查h5文件中的数据维度"""
    with h5py.File(h5_path, 'r') as f:
        if 'HR' in f and 'LR' in f:
            hr_shape = f['HR'].shape
            lr_shape = f['LR'].shape
            print(f"数据集维度检查:")
            print(f"HR数据形状: {hr_shape}")
            print(f"LR数据形状: {lr_shape}")
            print(f"缩放比例: {hr_shape[1] // lr_shape[1]}")
            
            # 检查第一个样本
            hr_sample = f['HR'][0]
            lr_sample = f['LR'][0]
            print(f"\n第一个样本检查:")
            print(f"HR样本形状: {hr_sample.shape}")
            print(f"LR样本形状: {lr_sample.shape}")
            print(f"HR数据类型: {hr_sample.dtype}")
            print(f"LR数据类型: {lr_sample.dtype}")
            print(f"HR数值范围: [{hr_sample.min():.4f}, {hr_sample.max():.4f}]")
            print(f"LR数值范围: [{lr_sample.min():.4f}, {lr_sample.max():.4f}]")
        else:
            print("数据集中没有找到'HR'或'LR'键")


def process_and_save_new_h5(original_path, new_path):
    with h5py.File(original_path, 'r') as f_in:
        if 'HR' not in f_in:
            raise ValueError("HR 数据集在原始文件中不存在")

        hr_data = f_in['HR'][:]  # shape: (2400, 100, 100)
        if hr_data.shape[1:] != (100, 100):
            raise ValueError("HR 数据形状应为 (2400, 100, 100)")

        # 裁剪为 96x96
        hr_cropped = hr_data[:, 2:98, 2:98]  # 确保尺寸是96x96

        # 检查HR数据尺寸
        if hr_cropped.shape[1:] != (96, 96):
            raise ValueError(f"HR数据裁剪后尺寸错误: {hr_cropped.shape[1:]}, 应该是(96, 96)")

        # 平均下采样为 16x16
        lr_data = np.array([average_downsample(img, scale=6)
                           for img in hr_cropped])

        # 检查LR数据尺寸
        if lr_data.shape[1:] != (16, 16):
            raise ValueError(f"LR数据下采样后尺寸错误: {lr_data.shape[1:]}, 应该是(16, 16)")

    # 将裁剪后的 HR 和 LR 写入新的文件
    with h5py.File(new_path, 'w') as f_out:
        f_out.create_dataset('HR', data=hr_cropped, dtype=np.float32)
        f_out.create_dataset('LR', data=lr_data, dtype=np.float32)

    print(f"新文件已保存：{new_path}")
    print(f"HR数据形状：{hr_cropped.shape}")
    print(f"LR数据形状：{lr_data.shape}")
    print("数据类型：float32")
    print(f"缩放比例：6 (96x96 -> 16x16)")


if __name__ == '__main__':
    # 使用示例
    original_path = 'C:\\Users\\yy143\\Desktop\\dataset\\test_normalized.h5'
    new_path = 'C:\\Users\\yy143\\Desktop\\dataset\\HRLRpairs_test_normalized.h5'
    
    # 处理数据集
    process_and_save_new_h5(original_path, new_path)
    
    # 检查处理后的数据集
    print("\n检查处理后的数据集:")
    check_dataset_dimensions(new_path)
