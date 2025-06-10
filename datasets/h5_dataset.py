"""
多任务学习数据集加载器，用于超分辨率重建和泄漏源位置预测
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random

class MultiTaskDataset(Dataset):
    def __init__(self, dataset_file_name, index_list=None, shuffle=True):
        """
        参数：
        dataset_file_name：.h5 文件路径
        index_list：用于指定使用哪些索引，可以是整数列表或(wind_group, source_group)元组列表
        shuffle：是否在 index_list 内部进行打乱
        """
        super(MultiTaskDataset, self).__init__()
        self.dataset_file = h5py.File(dataset_file_name, 'r')
        
        # 构建数据索引列表
        self.data_indices = []
        
        # 遍历所有风场组
        for wind_group_name in self.dataset_file.keys():
            wind_group = self.dataset_file[wind_group_name]
            # 获取所有源位置组（s1-s8）
            source_groups = [f's{i}' for i in range(1, 9)]
            
            # 遍历所有源位置
            for source_group_name in source_groups:
                try:
                    source_group = wind_group[source_group_name]
                    # 获取时间步数量
                    time_steps = len([k for k in source_group.keys() if k.startswith('HR_')])
                    # 为每个时间步创建索引
                    for time_step in range(1, time_steps + 1):
                        self.data_indices.append({
                            'wind_group': wind_group_name,
                            'source_group': source_group_name,
                            'time_step': time_step
                        })
                except Exception as e:
                    print(f"警告：无法处理组 {wind_group_name}/{source_group_name}: {str(e)}")
                    continue
        
        # 如果未指定索引列表，就使用全部数据
        if index_list is None:
            self.index_list = list(range(len(self.data_indices)))
        else:
            # 如果index_list是元组列表，需要转换为对应的索引
            if isinstance(index_list[0], tuple):
                self.index_list = []
                for wind_group, source_group in index_list:
                    # 找到所有匹配的索引
                    for i, data_info in enumerate(self.data_indices):
                        if data_info['wind_group'] == wind_group and data_info['source_group'] == source_group:
                            self.index_list.append(i)
            else:
                self.index_list = index_list
        
        # 是否打乱索引
        if shuffle:
            random.shuffle(self.index_list)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        返回数据时，自动读取 LR、HR 和泄漏源位置信息，并进行归一化处理
        返回：
        - lr_tensor: 低分辨率数据 [1, H, W]，值范围[0,1]
        - hr_tensor: 高分辨率数据 [1, H, W]，值范围[0,1]
        - source_pos: 泄漏源位置真值 [2]，值范围[0,1]
        - hr_max_pos: HR中浓度最高值的位置 [2]，值范围[0,1]
        - wind_vector: 风场向量 [2]，值范围[-1,1]
        """
        try:
            idx_in_file = self.index_list[idx]
            data_info = self.data_indices[idx_in_file]
            
            # 获取数据组
            wind_group = self.dataset_file[data_info['wind_group']]
            source_group = wind_group[data_info['source_group']]
            
            # 读取LR和HR数据
            lr = source_group[f'LR_{data_info["time_step"]}'][:]
            hr = source_group[f'HR_{data_info["time_step"]}'][:]
            
            # 读取泄漏源位置信息（前两个值是x, y坐标）
            source_info = source_group['source_info'][:]
            source_pos = source_info[:2]  # 只取位置信息
            
            # 计算HR中浓度最高值的位置
            hr_max_pos = np.unravel_index(hr.argmax(), hr.shape)
            hr_max_pos = np.array([hr_max_pos[1], hr_max_pos[0]], dtype=np.float32)  # 转换为(x, y)顺序
            
            # 获取图像尺寸
            height, width = hr.shape
            # 归一化坐标到[0,1]范围
            source_pos = source_pos / (width - 1)
            hr_max_pos = hr_max_pos / (width - 1)
            
            # 读取风场信息
            wind_velocity = wind_group['velocity'][:]  # 读取风场速度
            # 归一化风场向量到[-1,1]范围
            wind_vector = wind_velocity / np.max(np.abs(wind_velocity))
            
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
            wind_vector_tensor = torch.tensor(wind_vector, dtype=torch.float32)
            
            return {
                'lr': lr_tensor,          # 输入数据，已归一化到[0,1]
                'hr': hr_tensor,          # 超分辨率任务的目标，已归一化到[0,1]
                'source_pos': source_pos_tensor,  # 泄漏源位置真值，已归一化到[0,1]
                'hr_max_pos': hr_max_pos_tensor,  # HR中浓度最高位置，已归一化到[0,1]
                'wind_vector': wind_vector_tensor  # 风场向量，已归一化到[-1,1]
            }
        except Exception as e:
            print(f"错误：处理索引 {idx} 时发生错误: {str(e)}")
            raise

def generate_train_valid_dataset(data_file, train_ratio=0.8, shuffle=True):
    """
    按风场和源位置划分数据集，确保数据独立性
    使用train_ratio=0.8时，将得到19组训练数据和5组验证数据
    
    参数：
    data_file: .h5 数据文件路径
    train_ratio: 训练集比例，默认0.8
    shuffle: 是否打乱数据
    返回：训练集对象，验证集对象
    """
    # 基础风场组（不包括增强）
    base_wind_groups = ['wind1_0', 'wind2_0', 'wind3_0']
    source_groups = [f's{i}' for i in range(1, 9)]
    
    # 创建基础组
    base_groups = []
    for wind in base_wind_groups:
        for source in source_groups:
            base_groups.append((wind, source))
    
    # 随机打乱基础组
    if shuffle:
        random.shuffle(base_groups)
    
    # 计算训练集组数，确保得到19组
    total_groups = len(base_groups)  # 24组
    if abs(train_ratio - 0.8) < 1e-6:  # 如果比例接近0.8
        train_group_count = 19  # 直接使用19
    else:
        # 使用向上取整，确保训练集不会太小
        train_group_count = int(np.ceil(total_groups * train_ratio))
    
    # 划分训练集和验证集
    train_base_groups = base_groups[:train_group_count]
    valid_base_groups = base_groups[train_group_count:]
    
    # 为训练集添加增强数据
    train_groups = []
    for wind, source in train_base_groups:
        # 添加原始数据
        train_groups.append((wind, source))
        # 添加增强数据
        wind_idx = wind.split('_')[0]  # 提取风场编号
        train_groups.extend([
            (f'{wind_idx}_rot90', source),
            (f'{wind_idx}_rot180', source),
            (f'{wind_idx}_rot270', source),
            (f'{wind_idx}_flip_h', source),
            (f'{wind_idx}_flip_v', source)
        ])
    
    # 验证集只使用原始数据
    valid_groups = valid_base_groups
    
    # 创建数据集
    train_dataset = MultiTaskDataset(data_file, train_groups, shuffle=True)
    valid_dataset = MultiTaskDataset(data_file, valid_groups, shuffle=False)
    
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
    print(f"风场向量形状: {sample['wind_vector'].shape}")
    
    # 验证数据
    print("\n数据验证:")
    print(f"泄漏源位置真值类型: {sample['source_pos'].dtype}")
    print(f"HR最大浓度位置类型: {sample['hr_max_pos'].dtype}")
    print(f"风场向量类型: {sample['wind_vector'].dtype}")
    print(f"LR数据范围: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
    print(f"HR数据范围: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")
    print(f"风场向量范围: [{sample['wind_vector'].min():.4f}, {sample['wind_vector'].max():.4f}]")
