import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
from tqdm.auto import tqdm

def visualize_concentration(h5_path):
    """
    可视化h5文件中的浓度数据
    
    Args:
        h5_path: h5文件路径
    """
    with h5py.File(h5_path, 'r') as f:
        # 获取所有风场和源位置
        wind_groups = list(f.keys())
        source_groups = list(f[wind_groups[0]].keys())
        
        # 获取第一个数据集的形状
        first_data = f[wind_groups[0]][source_groups[0]]['HR_1']
        num_iterations = len(f[wind_groups[0]][source_groups[0]].keys()) - 1  # 减去source_position
        
        # 创建图形和子图
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.3)  # 为滑块留出空间
        
        # 创建滑块
        wind_ax = plt.axes([0.2, 0.2, 0.6, 0.03])
        source_ax = plt.axes([0.2, 0.15, 0.6, 0.03])
        iter_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        wind_slider = Slider(wind_ax, 'Wind', 1, len(wind_groups), valinit=1, valstep=1)
        source_slider = Slider(source_ax, 'Source', 1, len(source_groups), valinit=1, valstep=1)
        iter_slider = Slider(iter_ax, 'Iteration', 1, num_iterations, valinit=1, valstep=1)
        
        # 创建颜色条
        im = ax.imshow(np.zeros(first_data.shape), cmap='viridis')
        plt.colorbar(im, ax=ax, label='Concentration')
        
        def update(val):
            # 获取当前选择
            wind_idx = int(wind_slider.val)
            source_idx = int(source_slider.val)
            iter_idx = int(iter_slider.val)
            
            # 读取数据
            wind_group = f[f'wind{wind_idx}']
            source_group = wind_group[f's{source_idx}']
            data = source_group[f'HR_{iter_idx}'][:]
            
            # 更新图像
            im.set_data(data)
            im.set_clim(vmin=data.min(), vmax=data.max())
            
            # 更新标题
            ax.set_title(f'Wind {wind_idx}, Source {source_idx}, Iteration {iter_idx}')
            
            # 如果有源位置信息，显示源位置
            if 'source_position' in source_group:
                source_pos = source_group['source_position'][:]
                ax.plot(source_pos[0], source_pos[1], 'r*', markersize=10, label='Source')
                ax.legend()
            
            fig.canvas.draw_idle()
        
        # 注册更新函数
        wind_slider.on_changed(update)
        source_slider.on_changed(update)
        iter_slider.on_changed(update)
        
        # 添加重置按钮
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        
        def reset(event):
            wind_slider.reset()
            source_slider.reset()
            iter_slider.reset()
        
        reset_button.on_clicked(reset)
        
        # 显示第一个数据
        update(1)
        
        plt.show()

def display_dataset_info(h5_path):
    """
    显示数据集的基本信息
    """
    with h5py.File(h5_path, 'r') as f:
        print("\n数据集信息:")
        for wind_idx in range(1, 4):
            wind_group = f[f'wind{wind_idx}']
            print(f"\n风场 {wind_idx}:")
            for source_idx in range(1, 9):
                source_group = wind_group[f's{source_idx}']
                print(f"  源位置 {source_idx}:")
                print(f"    数据集数量: {len(source_group.keys()) - 1}")  # 减去source_position
                if 'source_position' in source_group:
                    print(f"    源位置信息: {source_group['source_position'][:]}")
                print(f"    数据形状: {source_group['HR_1'].shape}")
                print(f"    数值范围: [{source_group['HR_1'][:].min():.4f}, {source_group['HR_1'][:].max():.4f}]")

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

def txt_to_h5(data_root, output_path):
    """
    将txt浓度数据转换为h5文件
    
    Args:
        data_root: 原始数据根目录，包含w1s1-w1s8等文件夹
        output_path: 输出h5文件路径
    """
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
                
                # 存储源位置信息（如果需要）
                source_pos_file = os.path.join(source_dir, 'source_position.txt')
                if os.path.exists(source_pos_file):
                    source_pos = np.loadtxt(source_pos_file)
                    source_group.create_dataset('source_position', data=source_pos)

def main():
    # 设置路径
    data_root = 'C:\\Users\\yy143\\Desktop\\dataset\\data'  # 原始数据根目录
    output_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\dataset.h5'  # 输出h5文件路径
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换数据
    txt_to_h5(data_root, output_path)
    
    # 验证数据
    with h5py.File(output_path, 'r') as f:
        print("\n数据验证:")
        for wind_idx in range(1, 4):
            wind_group = f[f'wind{wind_idx}']
            print(f"\n风场 {wind_idx}:")
            # 打印风场数据信息
            print(f"  风场数据:")
            print(f"    位置点数量: {len(wind_group['points'])}")
            print(f"    速度矢量数量: {len(wind_group['velocity'])}")
            
            for source_idx in range(1, 9):
                source_group = wind_group[f's{source_idx}']
                print(f"  源位置 {source_idx}:")
                print(f"    数据集数量: {len(source_group.keys())}")
                if 'source_position' in source_group:
                    print(f"    源位置信息: {source_group['source_position'][:]}")
                print(f"    第一个数据集形状: {source_group['HR_1'].shape}")
                print(f"    第一个数据集元数据: {dict(source_group['HR_1'].attrs)}")

if __name__ == '__main__':
    main()
