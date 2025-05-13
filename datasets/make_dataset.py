import h5py
import numpy as np
import os
from tqdm.auto import tqdm

def read_wind_data(wind_file):
    """
    读取风场数据
    
    Args:
        wind_file: 风场CSV文件路径
    
    Returns:
        wind_data: 包含位置和速度矢量的字典
    """
    # 读取CSV文件
    data = np.loadtxt(wind_file, delimiter=',', skiprows=1)  # 跳过标题行
    
    # 提取数据
    points = data[:, :2]  # Points:0, Points:1
    velocity = data[:, 2:]  # U:0, U:1
    
    return {
        'points': points,
        'velocity': velocity
    }

def read_concentration_file(file_path):
    """
    读取浓度数据文件
    
    Args:
        file_path: 浓度文件路径
    
    Returns:
        metadata: 文件元数据（z值、迭代次数等）
        data: 浓度数据数组
    """
    metadata = {}
    data = []
    
    with open(file_path, 'r') as f:
        # 读取元数据
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value = line.strip('# ').split(':')
                    metadata[key.strip()] = value.strip()
            else:
                # 读取数据行
                if line.strip():  # 跳过空行
                    x, y, conc = map(float, line.strip().split())
                    data.append([x, y, conc])
    
    # 转换为numpy数组
    data = np.array(data)
    
    # 获取宽度和高度
    width = int(metadata['Width'])
    height = int(metadata['Height'])
    
    # 重塑数据为2D数组
    conc_data = np.zeros((height, width))
    for x, y, conc in data:
        conc_data[int(y), int(x)] = conc
    
    # 垂直翻转数据
    conc_data = np.flipud(conc_data)
    
    return metadata, conc_data

def get_source_positions():
    """
    获取所有泄漏源位置信息
    
    Returns:
        source_positions: 包含所有泄漏源位置信息的字典
    """
    source_positions = {}
    source_file = 'C:\\Users\\yy143\\Desktop\\dataset\\data\\source.txt'
    
    with open(source_file, 'r') as f:
        for line in f:
            if line.strip():
                # 解析行数据
                parts = line.strip().split('-------')
                if len(parts) == 2:
                    key = parts[0].strip()
                    coords = parts[1].strip().split()
                    
                    # 提取x和y坐标
                    x = float(coords[0].split(':')[1])
                    y = float(coords[1].split(':')[1])
                    
                    # 转换坐标（乘以10并取整）
                    x = int(x * 10)
                    y = int(y * 10)
                    
                    source_positions[key] = np.array([x, y])
    
    return source_positions

def txt_to_h5(data_root, output_path):
    """
    将txt浓度数据转换为h5文件
    
    Args:
        data_root: 原始数据根目录，包含w1s1-w1s8等文件夹
        output_path: 输出h5文件路径
    """
    # 获取泄漏源位置信息
    source_positions = get_source_positions()
    
    # 创建h5文件
    with h5py.File(output_path, 'w') as f:
        # 首先处理风场数据
        wind_files = {
            'wind1': 'C:\\Users\\yy143\\Desktop\\dataset\\data\\inlet11_05.csv ',
            'wind2': 'C:\\Users\\yy143\\Desktop\\dataset\\data\\inlet22_05.csv',
            'wind3': 'C:\\Users\\yy143\\Desktop\\dataset\\data\\inlet33_05.csv'
        }
        
        for wind_idx, wind_file in wind_files.items():
            # 读取风场数据
            wind_data = read_wind_data(os.path.join(data_root, wind_file))
            
            # 创建风场组
            wind_group = f.create_group(wind_idx)
            
            # 存储风场数据
            wind_group.create_dataset('points', data=wind_data['points'])
            wind_group.create_dataset('velocity', data=wind_data['velocity'])
        
        # 处理三种风场情况下的浓度数据
        for wind_idx in range(1, 4):
            wind_group = f[f'wind{wind_idx}']
            
            # 处理8种泄漏源位置
            for source_idx in range(1, 9):
                source_group = wind_group.create_group(f's{source_idx}')
                
                # 构建源数据路径
                source_dir = os.path.join(data_root, f'w{wind_idx}s{source_idx}')
                
                # 获取所有concentration文件
                conc_files = sorted([f for f in os.listdir(source_dir) 
                                  if f.startswith('concentration_') and f.endswith('.txt')])
                
                # 读取并存储每个浓度文件
                for i, conc_file in enumerate(tqdm(conc_files, 
                    desc=f'Processing wind{wind_idx} source{source_idx}')):
                    
                    # 读取txt文件
                    metadata, conc_data = read_concentration_file(
                        os.path.join(source_dir, conc_file)
                    )
                    
                    # 存储到h5文件
                    dataset = source_group.create_dataset(
                        f'HR_{i+1}', 
                        data=conc_data,
                        compression='gzip'  # 使用gzip压缩
                    )
                    
                    # 存储元数据
                    for key, value in metadata.items():
                        dataset.attrs[key] = value
                
                # 存储源位置信息
                source_key = f'w{wind_idx}s{source_idx}'
                if source_key in source_positions:
                    # 获取源位置
                    source_pos = source_positions[source_key]
                    # 获取数据高度
                    height = source_group['HR_1'].shape[0]
                    # 垂直翻转y坐标
                    source_pos[1] = height - 1 - source_pos[1]
                    # 存储翻转后的源位置
                    source_group.create_dataset('source_position', data=source_pos)

def main():
    # 设置路径
    data_root = 'C:\\Users\\yy143\\Desktop\\dataset\\data'  # 原始数据根目录
    output_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\dataset.h5'  # 输出h5文件路径
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换数据
    txt_to_h5(data_root, output_path)

if __name__ == '__main__':
    main()
