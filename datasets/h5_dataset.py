import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import os

# �Զ������ݼ��࣬�̳��� PyTorch �� Dataset


class ConcDatasetTorch(Dataset):
    def __init__(self, dataset_file_name, index_list=None, shuffle=True):
        """
        ������
        dataset_file_name��.h5 �ļ�·��
        index_list������ָ��ʹ����Щ������Ĭ��ʹ��ȫ��
        shuffle���Ƿ��� index_list �ڲ����д���
        """
        super(ConcDatasetTorch, self).__init__()
        self.dataset_file = h5py.File(dataset_file_name, 'r')

        # ���� 'HR' �� 'LR' ������
        self.hr_data = self.dataset_file['HR']
        self.lr_data = self.dataset_file['LR']

        # ȷ�� HR �� LR ����һ��
        assert len(self.lr_data) == len(self.hr_data), \
            f'Mismatch in HR and LR dataset lengths: {len(self.lr_data)} vs {len(self.hr_data)}'

        self.dataset_length = len(self.lr_data)

        # ���δָ�������б���ʹ��ȫ������
        if index_list is None:
            index_list = list(range(self.dataset_length))

        # �Ƿ��������
        if shuffle:
            random.shuffle(index_list)

        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        ��������ʱ���Զ���ȡ HR �� LR ������ channel ά�ȣ� [H, W] �� [1, H, W]
        """
        idx_in_file = self.index_list[idx]
        lr = self.lr_data[idx_in_file]
        hr = self.hr_data[idx_in_file]

        # ����ͨ��ά��
        if len(lr.shape) == 2:
            lr = lr[np.newaxis, :, :]
        if len(hr.shape) == 2:
            hr = hr[np.newaxis, :, :]

        # ת��Ϊ tensor
        lr_tensor = torch.tensor(lr, dtype=torch.float32)
        hr_tensor = torch.tensor(hr, dtype=torch.float32)

        return lr_tensor, hr_tensor

# ����ѵ��������֤����8:2��


def generate_train_valid_dataset(data_file, train_ratio=0.8, shuffle=True):
    """
    data_file: .h5 �����ļ�·��
    train_ratio: ѵ����������Ĭ�� 0.8
    shuffle: �Ƿ��������
    ���أ�ѵ����������֤������
    """
    with h5py.File(data_file, 'r') as f:
        total_len = len(f['HR'])
        index_list = list(range(total_len))

        if shuffle:
            random.shuffle(index_list)

        split_idx = int(train_ratio * total_len)
        train_list = index_list[:split_idx]
        valid_list = index_list[split_idx:]

    # �������ݼ�ʵ��
    train_dataset = ConcDatasetTorch(data_file, train_list, shuffle=False)
    valid_dataset = ConcDatasetTorch(data_file, valid_list, shuffle=False)
    return train_dataset, valid_dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ʹ���Լ��� h5 ·��
    h5_path = '/your/path/to/dataset.h5'

    train_ds, valid_ds = generate_train_valid_dataset(h5_path)

    # ���ӻ���һ������
    lr, hr = train_ds[0]
    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)

    # ���ӻ�
    plt.subplot(1, 2, 1)
    plt.imshow(lr.squeeze().numpy(), cmap='viridis')
    plt.title('Low Resolution')

    plt.subplot(1, 2, 2)
    plt.imshow(hr.squeeze().numpy(), cmap='viridis')
    plt.title('High Resolution')

    plt.show()
