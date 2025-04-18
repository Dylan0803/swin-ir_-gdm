import os
import time
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入自定义的数据集类
from datasets.h5_dataset import H5Dataset

# 导入 SwinIR 超分辨率模型
from models.swinir import SwinIR

# 工具函数：画图、保存参数
from utils import plot_loss_curve, save_args


# 单个 epoch 的训练流程
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0.0
    tbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for lr, hr in tbar:
        lr, hr = lr.to(device), hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        tbar.set_postfix(loss=f"{batch_loss:.4f}")

    return epoch_loss / len(dataloader)


# 单个 epoch 的验证流程
@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    for lr, hr in dataloader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        loss = criterion(sr, hr)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# 主训练逻辑
def train(args):
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_set = H5Dataset(args.data_path, split='train')
    valid_set = H5Dataset(args.data_path, split='val')

    # 自动获取 patch size（假设 HR 图像尺寸是 [C, H, W]）
    sample_hr = train_set[0][1]
    patch_size = sample_hr.shape[-1]
    print(f"Auto-detected patch size from HR image: {patch_size}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = SwinIR(upscale=args.scale, img_size=patch_size).to(device)

    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建保存模型和图像的文件夹
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("experiments", f"{args.exp_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存训练超参数
    save_args(args, save_dir)

    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        valid_loss = valid_one_epoch(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(
            f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(
            f"Epoch {epoch+1} finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 如果当前验证损失更小，则保存模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"✅ Best model saved at epoch {epoch+1} with valid loss {best_loss:.4f}")

    # 绘制损失变化图
    plot_loss_curve(train_losses, valid_losses,
                    save_path=os.path.join(save_dir, "loss.png"))


# CLI 入口（命令行执行入口）
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        default='swinir_gdm', help='模型名称')
    parser.add_argument('--exp_name', type=str, default='exp', help='实验保存名')
    parser.add_argument('--data_path', type=str,
                        default='./data/dataset.h5', help='h5 数据路径')
    parser.add_argument('--batch_size', type=int, default=16, help='每个批次的样本数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--scale', type=int, default=6,
                        help='放大倍数（由 HR/LR 尺寸决定）')

    args = parser.parse_args()
    train(args)
