import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def visualize_concentration(h5_path):
    """
    可视化h5文件中的浓度数据

    Args:
        h5_path: h5文件路径
    """
    with h5py.File(h5_path, 'r') as f:
        # 获取所有风场和源位置
        wind_groups = list(f.keys())
        source_groups = [k for k in f[wind_groups[0]].keys() if k.startswith('s')]  # 排除 points/velocity 等
        
        # 获取第一个数据集的形状
        first_data = f[wind_groups[0]][source_groups[0]]['HR_1'][:]
        num_iterations = len(f[wind_groups[0]][source_groups[0]].keys()) - (1 if 'source_position' in f[wind_groups[0]][source_groups[0]] else 0)

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
        im = ax.imshow(first_data, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Concentration')

        def update(val):
            wind_idx = int(wind_slider.val)
            source_idx = int(source_slider.val)
            iter_idx = int(iter_slider.val)

            wind_group_name = f'wind{wind_idx}'
            source_group_name = f's{source_idx}'

            try:
                wind_group = f[wind_group_name]
                source_group = wind_group[source_group_name]
                data = source_group[f'HR_{iter_idx}'][:]
            except KeyError:
                print(f"无法找到 {wind_group_name}/{source_group_name}/HR_{iter_idx}")
                return

            im.set_data(data)
            im.set_clim(vmin=data.min(), vmax=data.max())
            ax.set_title(f'Wind {wind_idx}, Source {source_idx}, Iteration {iter_idx}')

            # 清除之前画的源位置点
            for line in ax.lines:
                line.remove()

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
                source_key = f's{source_idx}'
                if source_key not in wind_group:
                    continue
                source_group = wind_group[source_key]
                print(f"  源位置 {source_idx}:")
                num_keys = [k for k in source_group.keys() if k.startswith('HR_')]
                print(f"    数据集数量: {len(num_keys)}")
                if 'source_position' in source_group:
                    print(f"    源位置信息: {source_group['source_position'][:]}")
                print(f"    数据形状: {source_group['HR_1'].shape}")
                print(f"    数值范围: [{source_group['HR_1'][:].min():.4f}, {source_group['HR_1'][:].max():.4f}]")

def main():
    h5_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\dataset.h5'
    display_dataset_info(h5_path)
    visualize_concentration(h5_path)

if __name__ == '__main__':
    main()
