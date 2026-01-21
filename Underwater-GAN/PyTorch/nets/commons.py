"""
 > Common/standard network archutectures and modules
 > Credit for some functions
    * github.com/eriklindernoren/PyTorch-GAN
    * pluralsight.com/guides/artistic-neural-style-transfer-with-pytorch
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np


def Weights_Normal(m):
    # 同时使用类型检查和类名查找
    if isinstance(m, (nn.Conv2d, nn.Linear)):  # 卷积层和全连接层
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):  # 各种批归一化层
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

    # 额外的类名检查作为备用
    classname = m.__class__.__name__
    if 'Conv' in classname and hasattr(m, 'weight') and not isinstance(m, nn.Conv2d):
        # 处理其他名称包含Conv的自定义模块
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ Standard UNet up-sampling block
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


# class Gradient_Difference_Loss(nn.Module):
#     def __init__(self, alpha=1, chans=3, cuda=True):
#         super(Gradient_Difference_Loss, self).__init__()
#         self.alpha = alpha
#         self.chans = chans
#         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#         SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#         SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
#         self.Kx = Tensor(SobelX).expand(self.chans, 1, 3, 3)
#         self.Ky = Tensor(SobelY).expand(self.chans, 1, 3, 3)
#
#     def get_gradients(self, im):
#         gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
#         gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
#         return gx, gy
#
#     def forward(self, pred, true):
#         # get graduent of pred and true
#         gradX_true, gradY_true = self.get_gradients(true)
#         grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
#         gradX_pred, gradY_pred = self.get_gradients(pred)
#         grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
#         # compute and return GDL
#         return 0.5 * torch.mean(grad_true - grad_pred_a)

class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # 修正Sobel核定义
        SobelX = [[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]]  # 正确的水平梯度核
        SobelY = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]  # 正确的垂直梯度核

        # 创建适合RGB输入的卷积核
        self.Kx = Tensor(SobelX).repeat(chans, 1, 1, 1)  # 形状 [3, 1, 3, 3]
        self.Ky = Tensor(SobelY).repeat(chans, 1, 1, 1)  # 形状 [3, 1, 3, 3]

        # 注册为缓冲区确保设备兼容性
        self.register_buffer('Kx_buffer', self.Kx)
        self.register_buffer('Ky_buffer', self.Ky)

    def get_gradients(self, im):
        # 使用 groups=self.chans 实现通道独立滤波
        gx = F.conv2d(im, self.Kx_buffer, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky_buffer, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        gradX_true, gradY_true = self.get_gradients(true)
        gradX_pred, gradY_pred = self.get_gradients(pred)

        # 计算绝对值差异的损失
        loss_x = torch.mean(torch.abs(torch.abs(gradX_true) - torch.abs(gradX_pred)))
        loss_y = torch.mean(torch.abs(torch.abs(gradY_true) - torch.abs(gradY_pred)))

        return (loss_x + loss_y) * 0.5

class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features[:16].eval()
        for param in self.parameters():
            param.requires_grad = False
        self.loss = nn.L1Loss()

    def forward(self, gen, gt):
        vgg_gen = self.vgg(gen)
        vgg_gt = self.vgg(gt)
        return self.loss(vgg_gen, vgg_gt)


class MultiScaleVGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True).features
        self.layer_ids = [3, 8, 17, 26, 35]  # Conv1_2, Conv2_2, Conv3_4, Conv4_4, Conv5_4
        self.weights = [1.0, 0.8, 0.5, 0.3, 0.1]  # 多尺度权重

    def forward(self, gen, gt):
        loss = 0
        x_gen, x_gt = gen, gt
        for i in range(max(self.layer_ids) + 1):
            x_gen = self.vgg[i](x_gen)
            x_gt = self.vgg[i](x_gt)
            if i in self.layer_ids:
                layer_idx = self.layer_ids.index(i)
                loss += self.weights[layer_idx] * F.l1_loss(x_gen, x_gt)
        return loss