"""
多任务学习数据集加载器，用于超分辨率重建和泄漏源位置预测
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random

# 设置随机种子
def set_seed(seed=42):
    """
    设置随机种子，确保结果可复现
    
    参数：
    seed: 随机种子值，默认42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
                except Exception:
                    continue
        
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

def generate_train_valid_test_dataset(data_file, train_ratio=0.8, valid_ratio=0.1, shuffle=True, seed=42):
    """
    生成训练集、验证集和测试集 (按泄漏模拟组划分，防止数据泄露)
    """
    set_seed(seed)
    
    # 1. 识别所有独立的模拟组
    simulation_groups = []
    with h5py.File(data_file, 'r') as f:
        for wind_group_name in f.keys():
            wind_group = f[wind_group_name]
            source_groups_in_file = [f's{i}' for i in range(1, 9)]
            for source_group_name in source_groups_in_file:
                if source_group_name in wind_group:
                    simulation_groups.append((wind_group_name, source_group_name))

    # 2. 打乱模拟组
    if shuffle:
        random.shuffle(simulation_groups)
    
    # 3. 按比例划分
    num_groups = len(simulation_groups)
    train_split_idx = int(train_ratio * num_groups)
    valid_split_idx = int((train_ratio + valid_ratio) * num_groups)
    train_groups = simulation_groups[:train_split_idx]
    valid_groups = simulation_groups[train_split_idx:valid_split_idx]
    test_groups = simulation_groups[valid_split_idx:]
    print(f"总共有 {num_groups} 个独立的模拟组。")
    print(f"划分: {len(train_groups)} 组用于训练, {len(valid_groups)} 组用于验证, {len(test_groups)} 组用于测试。")

    # 4. 构建全局索引映射
    all_data_indices = []
    global_idx_map = {}
    with h5py.File(data_file, 'r') as f:
        current_global_idx = 0
        for wind_group_name in f.keys():
            wind_group = f[wind_group_name]
            source_groups_in_file = [f's{i}' for i in range(1, 9)]
            for source_group_name in source_groups_in_file:
                if source_group_name in wind_group:
                    source_group = wind_group[source_group_name]
                    time_steps_count = len([k for k in source_group.keys() if k.startswith('HR_')])
                    for time_step in range(1, time_steps_count + 1):
                        all_data_indices.append({
                            'wind_group': wind_group_name,
                            'source_group': source_group_name,
                            'time_step': time_step
                        })
                        key = (wind_group_name, source_group_name, time_step)
                        global_idx_map[key] = current_global_idx
                        current_global_idx += 1

    # 5. 根据组划分生成索引列表
    train_list, valid_list, test_list = [], [], []
    with h5py.File(data_file, 'r') as f:
        for group_set, index_list in [(train_groups, train_list), (valid_groups, valid_list), (test_groups, test_list)]:
            for wind_group_name, source_group_name in group_set:
                source_group = f[wind_group_name][source_group_name]
                time_steps_count = len([k for k in source_group.keys() if k.startswith('HR_')])
                for time_step in range(1, time_steps_count + 1):
                    key = (wind_group_name, source_group_name, time_step)
                    index_list.append(global_idx_map[key])

    # 6. 创建数据集实例
    train_dataset = MultiTaskDataset(data_file, train_list, shuffle=True)
    valid_dataset = MultiTaskDataset(data_file, valid_list, shuffle=False)
    test_dataset = MultiTaskDataset(data_file, test_list, shuffle=False)
    
    return train_dataset, valid_dataset, test_dataset

# ================== 泄漏检查测试模块 ==================
if __name__ == "__main__":
    # 假设你有 MultiTaskDataset 类和一个 h5 文件路径
    data_file = "your_data.h5"
    train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(data_file)

    # 提取每个数据集用到的 simulation_groups
    def get_sim_groups(dataset):
        sim_groups = set()
        for idx in dataset.indices:
            info = dataset.all_data_indices[idx]
            sim_groups.add((info['wind_group'], info['source_group']))
        return sim_groups

    train_groups = get_sim_groups(train_dataset)
    test_groups = get_sim_groups(test_dataset)
    valid_groups = get_sim_groups(valid_dataset)

    print("训练集与测试集的组交集：", train_groups & test_groups)
    print("训练集与验证集的组交集：", train_groups & valid_groups)
    print("验证集与测试集的组交集：", valid_groups & test_groups)
    if not (train_groups & test_groups):
        print("✅ 训练集与测试集无泄漏！")
    else:
        print("❌ 训练集与测试集有泄漏！")
