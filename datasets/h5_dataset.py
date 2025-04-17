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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 使用自己的 h5 路径
    h5_path = '/your/path/to/dataset.h5'

    train_ds, valid_ds = generate_train_valid_dataset(h5_path)

    # 可视化第一个样本
    lr, hr = train_ds[0]
    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)

    # 可视化
    plt.subplot(1, 2, 1)
    plt.imshow(lr.squeeze().numpy(), cmap='viridis')
    plt.title('Low Resolution')

    plt.subplot(1, 2, 2)
    plt.imshow(hr.squeeze().numpy(), cmap='viridis')
    plt.title('High Resolution')

    plt.show()
