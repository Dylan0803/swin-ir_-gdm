import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network_swinir import SwinIR
from models.network_swinir_multi import SwinIRMulti

class WindGuidedAttention(nn.Module):
    """风场引导的注意力模块"""
    def __init__(self, in_channels, wind_channels=2):
        super(WindGuidedAttention, self).__init__()
        self.wind_processor = nn.Sequential(
            nn.Conv2d(wind_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, wind_vector):
        # 处理风场信息
        wind_features = self.wind_processor(wind_vector)
        # 生成注意力图
        attention = self.attention(torch.cat([x, wind_features], dim=1))
        # 应用注意力
        return x * attention

class EnhancedGSLBranchWithWind(nn.Module):
    """增强的GSL分支，支持风场引导"""
    def __init__(self, in_channels, num_sources=1):
        super(EnhancedGSLBranchWithWind, self).__init__()
        self.num_sources = num_sources
        
        # 多尺度特征提取
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        
        # 风场引导的注意力模块
        self.wind_attention = WindGuidedAttention(64)
        
        # 泄漏源位置预测器
        self.position_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, kernel_size=1)
            ) for _ in range(num_sources)
        ])
        
        # 置信度预测器
        self.confidence_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_sources)
        ])
        
    def forward(self, x, wind_vector=None):
        # 多尺度特征提取
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # 特征融合
        fused = self.fusion(torch.cat([feat1, feat2, feat3], dim=1))
        
        # 如果有风场信息，使用风场引导
        if wind_vector is not None:
            fused = self.wind_attention(fused, wind_vector)
        
        # 预测多个泄漏源位置和置信度
        positions = []
        confidences = []
        for i in range(self.num_sources):
            pos = self.position_predictors[i](fused)
            conf = self.confidence_predictors[i](fused)
            positions.append(pos)
            confidences.append(conf)
        
        # 在训练时返回所有预测
        if self.training:
            return torch.cat(positions, dim=1), torch.cat(confidences, dim=1)
        
        # 在推理时返回置信度最高的预测
        positions = torch.stack(positions, dim=1)  # [B, num_sources, 2, H, W]
        confidences = torch.stack(confidences, dim=1)  # [B, num_sources, 1, H, W]
        best_idx = torch.argmax(confidences, dim=1)  # [B, 1, H, W]
        best_positions = torch.gather(positions, 1, best_idx.unsqueeze(2).expand(-1, -1, 2, -1, -1))
        best_confidences = torch.gather(confidences, 1, best_idx.unsqueeze(2))
        
        return best_positions.squeeze(1), best_confidences.squeeze(1)

class SwinIRMultiEnhancedWind(SwinIRMulti):
    """增强的SwinIR多任务模型，支持风场引导"""
    def __init__(self, *args, **kwargs):
        super(SwinIRMultiEnhancedWind, self).__init__(*args, **kwargs)
        # 替换GSL分支
        self.gsl_branch = EnhancedGSLBranchWithWind(
            in_channels=kwargs.get('embed_dim', 60),
            num_sources=1
        )
        
    def forward(self, x, wind_vector=None):
        # 使用父类的特征提取和上采样
        x = super().forward(x)
        
        # 如果输入是元组（来自父类的forward），取第一个元素
        if isinstance(x, tuple):
            x = x[0]
            
        # GSL任务
        gsl_pos, gsl_conf = self.gsl_branch(x, wind_vector)
        
        return x, gsl_pos, gsl_conf 