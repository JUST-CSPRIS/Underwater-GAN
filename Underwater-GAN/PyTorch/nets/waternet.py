import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from PyTorch.nets.antialias import Downsample


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def inv(x , t):
    fx = []
    a = []
    t = 1.0 / t
    for j in range(x.size(0)):
        for i in range(3):
            fx.append(x[j][i] * t[j][0])
        a.append(torch.cat([fx[0].unsqueeze(0), fx[1].unsqueeze(0), fx[2].unsqueeze(0)], 0).unsqueeze(0))
    J = torch.cat(a, 0)
    return J
class DarkChannelProcessor(nn.Module):
    """共享模块，处理暗通道提取、特征增强和散射噪声抑制"""
    def __init__(self, in_channels):
        super().__init__()
        # 暗通道融合层
        self.dark_conv = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1)
        )

        # 散射抑制层
        self.scatter_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 暗通道提取与融合
        dark = torch.min(x, dim=1, keepdim=True)[0]
        dark_feat = self.dark_conv(dark)

        # 特征增强
        x = x + dark_feat

        # 散射噪声抑制
        att_mask = self.scatter_att(x)
        x = x * att_mask

        return x


class SimpleDown(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 使用共享模块处理暗通道等
        self.processor = DarkChannelProcessor(in_channels)

        # 带特征保持的下采样
        self.down = nn.Sequential(
            Downsample(channels=in_channels, filt_size=3, stride=2),
            # 保持输出通道与输入一致
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.processor(x)
        return self.down(x)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 使用共享模块处理暗通道等
        self.processor = DarkChannelProcessor(in_channels)
        # 添加通道对齐卷积
        self.align_conv = nn.Conv2d(in_channels, in_channels, 1) if in_channels != 64 else nn.Identity()

    def forward(self, x):
        return self.align_conv(self.processor(x))


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down1 = SimpleDown(in_channels)
        self.down2 = SimpleDown(in_channels)  # 统一使用in_channels

        self.layer1 = ResidualLayer(in_channels)
        self.layer2 = ResidualLayer(in_channels)
        self.layer3 = ResidualLayer(in_channels)

        self.channel_align = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        y1 = self.layer1(x1)
        y2 = self.layer2(x2)
        y3 = self.layer3(x3)

        # 空间对齐
        y2 = F.interpolate(y2, size=y1.shape[2:], mode='bilinear', align_corners=False)
        y3 = F.interpolate(y3, size=y1.shape[2:], mode='bilinear', align_corners=False)

        # 通道整合
        out = torch.cat([y1, y2, y3], dim=1)
        out = self.channel_align(out)

        return out + x


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MSFN3(nn.Module):
    def __init__(self, n_colors, base_ch, n_blocks, residual=False):
        super(MSFN3, self).__init__()
        self.residual = residual
        self.head = nn.Conv2d(n_colors, base_ch, kernel_size=3, stride=1, padding=1)

        self.regularizer1 = MultiScaleBlock(base_ch)
        self.alpha1 = SEAttention(base_ch)

        self.regularizer2 = MultiScaleBlock(base_ch)
        self.alpha2 = SEAttention(base_ch)
        self.beta2 = SEAttention(base_ch)

        self.regularizer3 = MultiScaleBlock(base_ch)
        self.alpha3 = SEAttention(base_ch)
        self.beta3 = SEAttention(base_ch)

        self.end = nn.Conv2d(base_ch, n_colors, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 使用inv函数代替head卷积层
        x0 = self.head(x)

        reg = self.regularizer1(x0)
        adj = 0
        grad = adj - reg
        x1 = x0 + self.alpha1(grad)

        reg = self.regularizer2(x1)
        adj = x1 - x0
        grad = adj - reg
        x2 = x1 + self.alpha2(grad) + self.beta2(x1 - x0)

        reg = self.regularizer3(x2)
        adj = x2 - x0
        grad = (adj - reg)
        x3 = x2 + self.alpha3(grad) + self.beta3(x2 - x1)

        res = self.end(x3)
        return res + x if self.residual else res


