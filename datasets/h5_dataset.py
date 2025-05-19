"""
多任务学习数据集加载器，用于超分辨率重建和泄漏源位置预测
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import os

class MultiTaskDataset(Dataset):
    def __init__(self, dataset_file_name, index_list=None, shuffle=True):
        """
        参数：
        dataset_file_name：.h5 文件路径
        index_list：用于指定使用哪些索引，默认使用全部
        shuffle：是否在 index_list 内部进行打乱
        """
        super(MultiTaskDataset, self).__init__()
        self.dataset_file = h5py.File(dataset_file_name, 'r')
        
        # 构建数据索引列表
        self.data_indices = []
        
        # 遍历所有风场组
        for wind_group_name in self.dataset_file.keys():
            wind_group = self.dataset_file[wind_group_name]
            # 遍历所有源位置
            for source_idx in range(1, 9):
                source_group = wind_group[f's{source_idx}']
                # 获取时间步数量
                time_steps = len([k for k in source_group.keys() if k.startswith('HR_')])
                # 为每个时间步创建索引
                for time_step in range(1, time_steps + 1):
                    self.data_indices.append({
                        'wind_group': wind_group_name,
                        'source_idx': source_idx,
                        'time_step': time_step
                    })
        
        # 如果未指定索引列表，就使用全部数据
        if index_list is None:
            index_list = list(range(len(self.data_indices)))
        
        # 是否打乱索引
        if shuffle:
            random.shuffle(index_list)
        
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        返回数据时，自动读取 LR、HR 和泄漏源位置信息
        返回：
        - lr_tensor: 低分辨率数据 [1, H, W]
        - hr_tensor: 高分辨率数据 [1, H, W]
        - source_pos: 泄漏源位置真值 [2] (x, y坐标)
        - hr_max_pos: HR中浓度最高值的位置 [2] (x, y坐标)
        """
        idx_in_file = self.index_list[idx]
        data_info = self.data_indices[idx_in_file]
        
        # 获取数据组
        wind_group = self.dataset_file[data_info['wind_group']]
        source_group = wind_group[f's{data_info["source_idx"]}']
        
        # 读取LR和HR数据
        lr = source_group[f'LR_{data_info["time_step"]}'][:]
        hr = source_group[f'HR_{data_info["time_step"]}'][:]
        
        # 读取泄漏源位置信息（前两个值是x, y坐标）
        source_info = source_group['source_info'][:]
        source_pos = source_info[:2]  # 只取位置信息
        
        # 计算HR中浓度最高值的位置
        # 注意：numpy的unravel_index返回的是(y, x)顺序，需要转换为(x, y)
        hr_max_pos = np.unravel_index(hr.argmax(), hr.shape)
        hr_max_pos = np.array([hr_max_pos[1], hr_max_pos[0]], dtype=np.float32)  # 转换为(x, y)顺序
        
        # 增加通道维度
        if len(lr.shape) == 2:
            lr = lr[np.newaxis, :, :]
        if len(hr.shape) == 2:
            hr = hr[np.newaxis, :, :]
        
        # 转换为tensor
        lr_tensor = torch.tensor(lr, dtype=torch.float32)
        hr_tensor = torch.tensor(hr, dtype=torch.float32)
        source_pos_tensor = torch.tensor(source_pos, dtype=torch.float32)
        hr_max_pos_tensor = torch.tensor(hr_max_pos, dtype=torch.float32)
        
        return {
            'lr': lr_tensor,          # 输入数据
            'hr': hr_tensor,          # 超分辨率任务的目标
            'source_pos': source_pos_tensor,  # 泄漏源位置真值
            'hr_max_pos': hr_max_pos_tensor  # HR中浓度最高位置，用于学习位置关系
        }

def generate_train_valid_dataset(data_file, train_ratio=0.8, shuffle=True):
    """
    data_file: .h5 数据文件路径
    train_ratio: 训练集比例，默认 0.8
    shuffle: 是否打乱数据
    返回：训练集对象，验证集对象
    """
    with h5py.File(data_file, 'r') as f:
        # 计算总数据量
        total_len = 0
        for wind_group_name in f.keys():
            wind_group = f[wind_group_name]
            for source_idx in range(1, 9):
                source_group = wind_group[f's{source_idx}']
                total_len += len([k for k in source_group.keys() if k.startswith('HR_')])
        
        index_list = list(range(total_len))
        
        if shuffle:
            random.shuffle(index_list)
        
        split_idx = int(train_ratio * total_len)
        train_list = index_list[:split_idx]
        valid_list = index_list[split_idx:]
    
    # 创建数据集实例
    train_dataset = MultiTaskDataset(data_file, train_list, shuffle=False)
    valid_dataset = MultiTaskDataset(data_file, valid_list, shuffle=False)
    return train_dataset, valid_dataset

if __name__ == '__main__':
    # 测试数据集加载
    h5_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\normalized_augmented_dataset.h5'
    
    # 创建数据集实例
    dataset = MultiTaskDataset(h5_path)
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print("\n样本信息:")
    print(f"LR数据形状: {sample['lr'].shape}")
    print(f"HR数据形状: {sample['hr'].shape}")
    print(f"泄漏源位置真值: {sample['source_pos']}")
    print(f"HR最大浓度位置: {sample['hr_max_pos']}")
    
    # 验证数据
    print("\n数据验证:")
    print(f"泄漏源位置真值类型: {sample['source_pos'].dtype}")
    print(f"HR最大浓度位置类型: {sample['hr_max_pos'].dtype}")
    print(f"LR数据范围: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
    print(f"HR数据范围: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")
