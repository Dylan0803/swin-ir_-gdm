'''
给接近 0 的 Ground Truth 区域加大惩罚（权重），引导模型在这些区域更“干净”

zero_threshold = 0.05：把小于 0.05 的区域视为“应该压为零”区域；
high_weight = 5.0：这些区域的损失被加权 5 倍；
default_weight = 1.0：其他区域保持正常权重。
'''

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.3, threshold=0.05):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, input, target):
        # 计算权重：目标值大于阈值时增加权重
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.alpha
        weights[target <= self.threshold] = self.beta
        
        loss = weights * (input - target) ** 2
        return loss.mean()

