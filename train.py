import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 使用自定义数据集类
from datasets.h5_dataset import ConcDatasetTorch, generate_train_valid_dataset
from models.network_swinir import SwinIR
from utils import plot_loss_lines, save_args


# 单轮训练过程
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()  # 切换为训练模式
    epoch_loss = 0.0
    tbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")  # 显示进度条

    for lr, hr in tbar:
        # 确保数据维度正确
        if len(lr.shape) == 3:
            lr = lr.unsqueeze(0)  # 添加batch维度
        if len(hr.shape) == 3:
            hr = hr.unsqueeze(0)  # 添加batch维度

        lr, hr = lr.to(device), hr.to(device)  # 将数据移到 GPU/CPU
        
        optimizer.zero_grad()  # 清空梯度
        sr = model(lr)         # 前向传播得到预测图像
        loss = criterion(sr, hr)  # 计算 MSE 损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        batch_loss = loss.item()
        epoch_loss += batch_loss
        tbar.set_postfix(loss=f"{batch_loss:.4f}")  # 实时显示损失

    return epoch_loss / len(dataloader)  # 返回平均训练损失


# 验证过程，不做反向传播，关闭 Dropout 等训练专用操作
@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device):
    model.eval()  # 切换为评估模式
    epoch_loss = 0.0

    for lr, hr in dataloader:
        # 确保数据维度正确
        if len(lr.shape) == 3:
            lr = lr.unsqueeze(0)  # 添加batch维度
        if len(hr.shape) == 3:
            hr = hr.unsqueeze(0)  # 添加batch维度

        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        loss = criterion(sr, hr)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)  # 返回平均验证损失


# 主训练流程
def train(args):
    logging.basicConfig(level=logging.INFO)
    print(f"Starting training with arguments: {args}")

    # 设置设备为 GPU（如可用），否则为 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 使用 generate_train_valid_dataset 函数来划分训练集和验证集
    train_set, valid_set = generate_train_valid_dataset(
        args.data_path, train_ratio=0.8, shuffle=True)

    # 封装为 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False)

    # 打印数据集信息
    sample_lr, sample_hr = next(iter(train_loader))
    print(f"Data shapes - LR: {sample_lr.shape}, HR: {sample_hr.shape}")

    # 初始化 SwinIR 模型
    model = SwinIR(
        upscale=args.scale,
        in_chans=1,  # 设置为1通道
        img_size=16,  # LR图像尺寸
        window_size=4,  # 更小的窗口大小，更适合16x16的输入
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',  # 使用直接像素重排上采样
        resi_connection='1conv'
    ).to(device)

    print(f"Model created with upscale factor: {args.scale}")
    print(f"Using window_size={model.window_size} for {model.img_size}x{model.img_size} input images")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建实验保存路径
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # 当前时间戳
    save_dir = os.path.join("experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存训练参数
    save_args(args, save_dir)

    # 初始化最佳验证损失
    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    # 正式开始训练循环
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        valid_loss = valid_one_epoch(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(
            f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        # ✅ 每个 epoch 结束后，打印当前时间
        print(
            f"Epoch {epoch+1} finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 保存最佳模型（验证损失最低）
        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"✅ Best model saved at epoch {epoch+1} with valid loss {best_loss:.4f}")

    # 画出训练/验证损失变化曲线
    plot_loss_lines(args, train_losses, valid_losses)


# 启动训练的命令行入口
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    parser.add_argument('--exp_name', type=str,
                        default='default_experiment', help='实验名称')
    parser.add_argument('--data_path', type=str, required=True, help='h5 数据路径')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--scale', type=int, default=6, help='放大倍数')
    parser.add_argument('--patch_size', type=int, default=16, help='LR图像的patch大小')
    args = parser.parse_args()

    args.output_dir = os.path.join(
        './experiments', args.model_name, args.exp_name)

    train(args)  # 开始训练
