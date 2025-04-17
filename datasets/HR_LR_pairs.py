import h5py
import numpy as np


def average_downsample(data, scale):
    """将二维矩阵平均下采样，目标是从96x96到16x16（scale=6）"""
    h, w = data.shape
    new_h, new_w = h // scale, w // scale
    return data.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))


def process_and_save_new_h5(original_path, new_path):
    with h5py.File(original_path, 'r') as f_in:
        if 'HR' not in f_in:
            raise ValueError("HR 数据集在原始文件中不存在")

        hr_data = f_in['HR'][:]  # shape: (2400, 100, 100)
        if hr_data.shape[1:] != (100, 100):
            raise ValueError("HR 数据形状应为 (2400, 100, 100)")

        # 裁剪为 96x96
        hr_cropped = hr_data[:, 2:98, 2:98]

        # 平均下采样为 16x16
        lr_data = np.array([average_downsample(img, scale=6)
                           for img in hr_cropped])

    # 将裁剪后的 HR 和 LR 写入新的文件
    with h5py.File(new_path, 'w') as f_out:
        f_out.create_dataset('HR', data=hr_cropped, dtype=np.float32)
        f_out.create_dataset('LR', data=lr_data, dtype=np.float32)

    print(f"新文件已保存：{new_path}，包含 HR（96x96）和 LR（16x16）数据。")


# 设置路径
original_file = 'C:\\Users\\yy143\\Desktop\\dataset\\dataset_normalized.h5'       # 原始文件名
# 新保存的文件名
new_file = 'C:\\Users\\yy143\\Desktop\\dataset\\HRLRpairs_normalized.h5'

process_and_save_new_h5(original_file, new_file)
