# GKDM 评估程序使用说明

## 程序功能
修改后的 `eval_gkdm.py` 程序现在支持通过命令行直接指定一个 Rco 值，并计算多个样本的平均 MSE。

## 主要变化
1. **新增必需参数 `--rco`**: 直接指定要使用的 Rco 值
2. **简化逻辑**: 移除了 Rco 范围测试功能，专注于使用固定 Rco 值进行评估
3. **平均 MSE 计算**: 对多个样本计算平均 MSE 和标准差

## 使用方法

### 基本用法
```bash
python eval_gkdm.py --data_path your_data.h5 --rco 1.5
```

### 完整参数示例
```bash
python eval_gkdm.py \
    --data_path datasets/your_dataset.h5 \
    --rco 1.5 \
    --test_mode all_generalization \
    --save_dir results \
    --mat_width 9.6 \
    --mat_height 9.6 \
    --scale_factor 6
```

### 参数说明

#### 必需参数
- `--data_path`: H5 数据文件路径
- `--rco`: 要使用的 Rco 值（必需）

#### 可选参数
- `--save_dir`: 结果保存目录（默认: gkdm_results）
- `--test_mode`: 测试模式选择
  - `generalization`: 使用指定的样本规格
  - `test_set`: 使用指定的测试集索引
  - `all_generalization`: 使用所有泛化测试样本
  - `all_test_set`: 使用所有测试集样本
- `--sample_specs`: 样本规格字符串（用于 generalization 模式）
- `--test_indices`: 测试集索引（用于 test_set 模式）
- `--mat_width`: 物理宽度（米，默认: 9.6）
- `--mat_height`: 物理高度（米，默认: 9.6）
- `--scale_factor`: 下采样比例因子（默认: 6）

### 使用示例

#### 1. 使用固定 Rco 值评估单个样本
```bash
python eval_gkdm.py --data_path data.h5 --rco 1.2
```

#### 2. 使用固定 Rco 值评估所有泛化测试样本
```bash
python eval_gkdm.py --data_path data.h5 --rco 1.5 --test_mode all_generalization
```

#### 3. 使用固定 Rco 值评估指定样本
```bash
python eval_gkdm.py --data_path data.h5 --rco 1.8 --test_mode generalization --sample_specs "wind1_0,s1,50;wind2_0,s2,30"
```

#### 4. 使用固定 Rco 值评估测试集样本
```bash
python eval_gkdm.py --data_path data.h5 --rco 1.0 --test_mode test_set --test_indices "1,2,3,4,5"
```

## 输出结果

程序会输出以下信息：
1. **每个样本的 MSE**: 显示每个样本的归一化 MSE 值
2. **统计摘要**: 
   - 平均 MSE
   - 标准差
   - 最小/最大 MSE
3. **可视化**: 使用第一个样本生成重建结果的可视化图像

### 示例输出
```
--- Starting GKDM Evaluation with Fixed Rco ---
Data file path: data.h5
Save directory: gkdm_results
Test mode: all_generalization
Fixed Rco value: 1.5

Evaluating 6 samples with fixed Rco=1.5...

Evaluating sample 1/6: {'wind_group': 'wind1_0', 'source_group': 's1', 'time_step': 50}
  Sample 1 MSE: 0.123456

Evaluating sample 2/6: {'wind_group': 'wind1_0', 'source_group': 's2', 'time_step': 50}
  Sample 2 MSE: 0.234567

...

=== Results Summary ===
Rco value: 1.5
Number of samples: 6
Average MSE: 0.178901
Standard deviation MSE: 0.045678
Min MSE: 0.123456
Max MSE: 0.345678

=== Final Results ===
Rco: 1.5
Average MSE: 0.178901
Standard deviation: 0.045678
```

## 注意事项

1. **Rco 值**: 程序会自动计算 Gama = Rco / 3.0
2. **数据加载**: 确保 H5 文件包含正确的数据结构和路径
3. **内存使用**: 对于大量样本，注意内存使用情况
4. **可视化**: 程序会生成 `gkdm_result.png` 文件显示重建结果 