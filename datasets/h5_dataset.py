"""
最开始进行超分辨率所用的数据集读取程序，读取至神经网络的输入
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import os

# 自定义数据集类，继承自 PyTorch 的 Dataset


class ConcDatasetTorch(Dataset):
    def __init__(self, dataset_file_name, index_list=None, shuffle=True):
        """
        参数：
        dataset_file_name：.h5 文件路径
        index_list：用于指定使用哪些索引，默认使用全部
        shuffle：是否在 index_list 内部进行打乱
        """
        super(ConcDatasetTorch, self).__init__()
        self.dataset_file = h5py.File(dataset_file_name, 'r')

        # 加载 'HR' 和 'LR' 数据组
        self.hr_data = self.dataset_file['HR']
        self.lr_data = self.dataset_file['LR']

        # 确保 HR 和 LR 数量一致
        assert len(self.lr_data) == len(self.hr_data), \
            f'Mismatch in HR and LR dataset lengths: {len(self.lr_data)} vs {len(self.hr_data)}'

        self.dataset_length = len(self.lr_data)

        # 如果未指定索引列表，就使用全部数据
        if index_list is None:
            index_list = list(range(self.dataset_length))

        # 是否打乱索引
        if shuffle:
            random.shuffle(index_list)

        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        返回数据时，自动读取 HR 和 LR 并加上 channel 维度： [H, W] → [1, H, W]
        """
        idx_in_file = self.index_list[idx]
        lr = self.lr_data[idx_in_file]
        hr = self.hr_data[idx_in_file]

        # 增加通道维度
        if len(lr.shape) == 2:
            lr = lr[np.newaxis, :, :]
        if len(hr.shape) == 2:
            hr = hr[np.newaxis, :, :]

        # 转换为 tensor
        lr_tensor = torch.tensor(lr, dtype=torch.float32)
        hr_tensor = torch.tensor(hr, dtype=torch.float32)

        return lr_tensor, hr_tensor

# 划分训练集和验证集（8:2）


def generate_train_valid_dataset(data_file, train_ratio=0.8, shuffle=True):
    """
    data_file: .h5 数据文件路径
    train_ratio: 训练集比例，默认 0.8
    shuffle: 是否打乱数据
    返回：训练集对象，验证集对象
    """
    with h5py.File(data_file, 'r') as f:
        total_len = len(f['HR'])
        index_list = list(range(total_len))

        if shuffle:
            random.shuffle(index_list)

        split_idx = int(train_ratio * total_len)
        train_list = index_list[:split_idx]
        valid_list = index_list[split_idx:]

    # 创建数据集实例
    train_dataset = ConcDatasetTorch(data_file, train_list, shuffle=False)
    valid_dataset = ConcDatasetTorch(data_file, valid_list, shuffle=False)
    return train_dataset, valid_dataset


def augment_dataset(h5_path, output_path):
    """
    对数据集进行增强，生成5种增强情况
    
    Args:
        h5_path: 原始h5文件路径
        output_path: 增强后的h5文件路径
    """
    with h5py.File(h5_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # 处理每种风场情况
        for wind_idx in range(1, 4):
            wind_group = f_in[f'wind{wind_idx}']
            
            # 创建原始数据组（添加_0后缀）
            wind_out_group = f_out.create_group(f'wind{wind_idx}_0')
            wind_out_group.create_dataset('points', data=wind_group['points'][:])
            wind_out_group.create_dataset('velocity', data=wind_group['velocity'][:])
            
            # 处理每种源位置
            for source_idx in range(1, 9):
                source_group = wind_group[f's{source_idx}']
                source_out_group = wind_out_group.create_group(f's{source_idx}')
                
                # 复制原始数据
                for i in range(1, len([k for k in source_group.keys() if k.startswith('HR_')]) + 1):
                    source_out_group.create_dataset(f'HR_{i}', data=source_group[f'HR_{i}'][:], compression='gzip')
                    source_out_group.create_dataset(f'LR_{i}', data=source_group[f'LR_{i}'][:], compression='gzip')
                    # 复制元数据
                    for key, value in source_group[f'HR_{i}'].attrs.items():
                        source_out_group[f'HR_{i}'].attrs[key] = value
                        source_out_group[f'LR_{i}'].attrs[key] = value
                
                # 复制源位置信息
                source_out_group.create_dataset('source_info', data=source_group['source_info'][:])
            
            # 定义增强操作
            augmentations = {
                'rot90': lambda x: np.rot90(x, k=1),
                'rot180': lambda x: np.rot90(x, k=2),
                'rot270': lambda x: np.rot90(x, k=3),
                'flip_h': lambda x: np.fliplr(x),
                'flip_v': lambda x: np.flipud(x)
            }
            
            # 对每种增强操作进行处理
            for aug_name, aug_func in augmentations.items():
                # 创建增强后的风场组
                wind_aug_group = f_out.create_group(f'wind{wind_idx}_{aug_name}')
                
                # 处理风场数据
                points = wind_group['points'][:]
                velocity = wind_group['velocity'][:]
                
                # 根据不同的增强操作处理风场数据
                if 'rot' in aug_name:
                    # 旋转操作
                    center = np.array([48, 48])
                    relative_points = points - center
                    if aug_name == 'rot90':
                        rotation_matrix = np.array([[0, -1], [1, 0]])
                    elif aug_name == 'rot180':
                        rotation_matrix = np.array([[-1, 0], [0, -1]])
                    else:  # rot270
                        rotation_matrix = np.array([[0, 1], [-1, 0]])
                    
                    relative_points = np.dot(relative_points, rotation_matrix.T)
                    velocity = np.dot(velocity, rotation_matrix.T)
                    points = relative_points + center
                else:
                    # 翻转操作
                    if aug_name == 'flip_h':
                        points[:, 0] = 96 - points[:, 0]
                        velocity[:, 0] = -velocity[:, 0]
                    else:  # flip_v
                        points[:, 1] = 96 - points[:, 1]
                        velocity[:, 1] = -velocity[:, 1]
                
                wind_aug_group.create_dataset('points', data=points)
                wind_aug_group.create_dataset('velocity', data=velocity)
                
                # 处理每种源位置
                for source_idx in range(1, 9):
                    source_group = wind_group[f's{source_idx}']
                    source_aug_group = wind_aug_group.create_group(f's{source_idx}')
                    
                    # 处理每个时间步
                    for i in range(1, len([k for k in source_group.keys() if k.startswith('HR_')]) + 1):
                        # 增强浓度数据
                        hr_data = aug_func(source_group[f'HR_{i}'][:])
                        lr_data = aug_func(source_group[f'LR_{i}'][:])
                        
                        source_aug_group.create_dataset(f'HR_{i}', data=hr_data, compression='gzip')
                        source_aug_group.create_dataset(f'LR_{i}', data=lr_data, compression='gzip')
                        
                        # 复制元数据
                        for key, value in source_group[f'HR_{i}'].attrs.items():
                            source_aug_group[f'HR_{i}'].attrs[key] = value
                            source_aug_group[f'LR_{i}'].attrs[key] = value
                    
                    # 处理源位置信息
                    source_info = source_group['source_info'][:]
                    if 'rot' in aug_name:
                        # 旋转源位置
                        center = np.array([48, 48])
                        relative_pos = source_info[:2] - center
                        relative_pos = np.dot(relative_pos, rotation_matrix.T)
                        source_info[:2] = relative_pos + center
                    else:
                        # 翻转源位置
                        if aug_name == 'flip_h':
                            source_info[0] = 96 - source_info[0]
                        else:  # flip_v
                            source_info[1] = 96 - source_info[1]
                    
                    source_aug_group.create_dataset('source_info', data=source_info)

def main():
    # 设置路径
    data_root = 'C:\\Users\\yy143\\Desktop\\dataset\\data'  # 原始数据根目录
    output_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\dataset.h5'  # 输出h5文件路径
    aug_output_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\augmented_dataset.h5'  # 增强后的h5文件路径
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换数据
    txt_to_h5(data_root, output_path)
    
    # 进行数据增强
    augment_dataset(output_path, aug_output_path)

if __name__ == '__main__':
    main()
