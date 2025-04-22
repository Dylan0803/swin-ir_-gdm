'''
给接近 0 的 Ground Truth 区域加大惩罚（权重），引导模型在这些区域更“干净”

zero_threshold = 0.05：把小于 0.05 的区域视为“应该压为零”区域；
high_weight = 5.0：这些区域的损失被加权 5 倍；
default_weight = 1.0：其他区域保持正常权重。
'''

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, zero_threshold=0.05, high_weight=5.0, default_weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.zero_threshold = zero_threshold
        self.high_weight = high_weight
        self.default_weight = default_weight

    def forward(self, prediction, target):
        # 构造权重矩阵：target接近0的区域给予更高权重
        weight = torch.where(
            target < self.zero_threshold,
            torch.full_like(target, self.high_weight),
            torch.full_like(target, self.default_weight)
        )
        loss = weight * (prediction - target) ** 2
        return loss.mean()
