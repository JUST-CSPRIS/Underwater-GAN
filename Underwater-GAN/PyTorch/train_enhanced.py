"""
 > Training pipeline for Improved FUnIE-GAN
 > Based on Composite Loss (Dark Channel + Gradient Direction + L1)
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from matplotlib import pyplot as plt
from PyTorch.nets.enhanced import EnhancedGenerator,Discriminator
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from nets.commons import Weights_Normal, Gradient_Difference_Loss, VGGLoss, MultiScaleVGGLoss
from nets.funiegan import DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="/home/xie/xcl/paper/code/Underwater-GAN/PyTorch/configs/train_ufo.yaml")
parser.add_argument("--epoch", type=int, default=0, help="start epoch")
parser.add_argument("--num_epochs", type=int, default=201, help="total epochs")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: beta1")
parser.add_argument("--b2", type=float, default=0.99, help="adam: beta2")

args = parser.parse_args()

# Training parameters
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2

# Load dataset config
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
img_width, img_height = cfg["im_width"], cfg["im_height"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

# Output directories
samples_dir = os.path.join("samples测试/测试GAM/", dataset_name)
checkpoint_dir = os.path.join("checkpoints测试/测试GAM/", dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize models and losses
generator = EnhancedGenerator()
discriminator = Discriminator()
L1_loss = torch.nn.L1Loss()  # L1 loss term
GDL_loss = Gradient_Difference_Loss()  # Gradient Difference Loss
vgg_loss = MultiScaleVGGLoss()

# CUDA setup
if torch.cuda.is_available():
    generator = generator.cuda()
    vgg_loss = vgg_loss.cuda()
    discriminator = discriminator.cuda()
    L1_loss = L1_loss.cuda()
    GDL_loss = GDL_loss.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Load pretrained if needed
if args.epoch != 0:
    gen_path = f"checkpoints测试/测试GAM/{dataset_name}/generator_{args.epoch}.pth"
    dis_path = f"checkpoints测试/测试GAM/{dataset_name}/discriminator_{args.epoch}.pth"
    generator.load_state_dict(torch.load(gen_path))
    discriminator.load_state_dict(torch.load(dis_path))
    print(f"Loaded models from epoch {args.epoch}")
else:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)

# Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate * 0.5, betas=(lr_b1, lr_b2))


# 使用初始的优化器配置
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate * 0.5, betas=(lr_b1, lr_b2))
# 在优化器定义后添加学习率调度器
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)
# Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_loader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

val_loader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

# ... 之前的代码保持不变 ...

# Training history
train_history = {
    'D': [],
    'G_total': [],
    'G_GAN': [],
    'L1': [],
    'GDL': [],
    'VGG': [],  # 添加 VGG loss 记录
}

# ... 之前的代码保持不变 ...

if __name__ == '__main__':
    # Training loop
    patch = (1, img_height // 16, img_width // 16)  # PatchGAN scale
    # 创建一个文件来保存损失日志
    log_file = os.path.join(samples_dir, 'loss_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Discriminator Loss,Generator Total Loss,Generator GAN Loss,Generator L1 Loss,Generator GDL Loss,Generator VGG Loss\n")

    for epoch in range(epoch, num_epochs):
        epoch_losses = {
            'D': 0.0,
            'G_total': 0.0,
            'G_GAN': 0.0,
            'L1': 0.0,
            'GDL': 0.0,
            'VGG': 0.0,  # 添加 VGG loss 记录
        }

        for i, batch in enumerate(train_loader):
            # Prepare data
            imgs_distorted = Variable(batch["A"].type(Tensor))
            imgs_gt = Variable(batch["B"].type(Tensor))
            valid = Variable(Tensor(np.random.uniform(0.9, 1.0, (imgs_distorted.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(imgs_gt, imgs_distorted)
            loss_real = torch.mean((pred_real - 1.0) ** 2)  # 真实样本目标为1

            # Fake loss
            imgs_fake = generator(imgs_distorted).detach()
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_fake = torch.mean(pred_fake ** 2)          # 生成样本目标为0

            # Total D loss
            # Total D loss (原始部分)
            loss_D = 0.5 * (loss_real + loss_fake)

            # --- 梯度惩罚修正代码 ---
            # 生成插值样本
            alpha = Tensor(np.random.random((imgs_gt.size(0), 1, 1, 1)))
            interpolates = (alpha * imgs_gt + (1 - alpha) * imgs_fake.detach()).requires_grad_(True)

            # 判别器对插值的输出
            d_interpolates = discriminator(interpolates, imgs_distorted)

            # 计算梯度
            grad_outputs = torch.ones(d_interpolates.size(), device=imgs_gt.device)
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            # 梯度惩罚项
            gradient_penalty = ((gradients.norm(2, dim=(1, 2, 3)) - 1)** 2).mean()
            lambda_gp = 10  # 初始值
            if epoch > 30:
                lambda_gp = max(5, 10 - (epoch - 30) // 10 * 2)  # 每10epoch降低2
            loss_D += lambda_gp * gradient_penalty

            # 反向传播
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            # Generate images
            imgs_fake = generator(imgs_distorted)

            # Calculate losses
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_gan = torch.mean((pred_fake - 1.0) ** 2)   # 欺骗判别器目标为1
            loss_l1 = L1_loss(imgs_fake, imgs_gt)
            loss_gdl = GDL_loss(imgs_fake, imgs_gt)
            loss_vgg = vgg_loss(imgs_fake, imgs_gt)

            # Total G loss
            # 修改生成器总损失权重
            # loss_G = 0.4 * loss_gan + 0.5 * loss_l1 + 0.1 * loss_gdl  # 原始比例
            # 动态调整权重（示例：前50epoch侧重内容，后期侧重纹理）
            if epoch < 50:
                w_gan, w_l1, w_gdl, w_vgg = 0.3, 0.4, 0.1, 0.2
            else:
                w_gan, w_l1, w_gdl, w_vgg = 0.2, 0.3, 0.1, 0.4

            loss_G = w_gan * loss_gan + w_l1 * loss_l1 + w_gdl * loss_gdl + w_vgg * loss_vgg
            loss_G.backward()
            optimizer_G.step()

            # Record losses
            epoch_losses['D'] += loss_D.item()
            epoch_losses['G_total'] += loss_G.item()
            epoch_losses['G_GAN'] += loss_gan.item()
            epoch_losses['L1'] += loss_l1.item()
            epoch_losses['GDL'] += loss_gdl.item()
            epoch_losses['VGG'] += loss_vgg.item()  # 记录 VGG loss

            # Log progress
            if i % 50 == 0:
                sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [D: %.3f] [G: %.3f (GAN: %.3f, L1: %.3f, GDL: %.3f, VGG: %.3f)]\n" % (
                    epoch, num_epochs, i, len(train_loader),
                    loss_D.item(), loss_G.item(), loss_gan.item(), loss_l1.item(), loss_gdl.item(), loss_vgg.item()
                ))

            # Save samples
            if i % val_interval == 0:
                val_batch = next(iter(val_loader))
                imgs_val = Variable(val_batch["val"].type(Tensor))
                imgs_gen = generator(imgs_val)
                img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
                save_path = os.path.join(samples_dir, f"{epoch}_{i}.png")
                save_image(img_sample, save_path, nrow=4, normalize=True)

        # Epoch statistics
        num_batches = len(train_loader)
        for key in epoch_losses:
            train_history[key].append(epoch_losses[key] / num_batches)

        # Print epoch losses
        print("\nEpoch %d/%d completed!" % (epoch, num_epochs))
        print("Discriminator Loss: %.3f" % (train_history['D'][-1]))
        print("Generator Total Loss: %.3f" % (train_history['G_total'][-1]))
        print("Generator GAN Loss: %.3f" % (train_history['G_GAN'][-1]))
        print("Generator L1 Loss: %.3f" % (train_history['L1'][-1]))
        print("Generator GDL Loss: %.3f" % (train_history['GDL'][-1]))
        print("Generator VGG Loss: %.3f" % (train_history['VGG'][-1]))  # 打印 VGG loss

        # 将每 epoch 的损失写入文件
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_history['D'][-1]},{train_history['G_total'][-1]},{train_history['G_GAN'][-1]},{train_history['L1'][-1]},{train_history['GDL'][-1]},{train_history['VGG'][-1]}\n")

        # Save checkpoints
        if epoch % ckpt_interval == 0:
            torch.save(generator.state_dict(),
                       os.path.join(checkpoint_dir, f"generator_{epoch}.pth"))
            torch.save(discriminator.state_dict(),
                       os.path.join(checkpoint_dir, f"discriminator_{epoch}.pth"))
    # 在每个epoch结束后调用
    scheduler_G.step()
    scheduler_D.step()
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train_history['D'], 'c--', label='Discriminator Loss')
    plt.plot(train_history['G_total'], 'b-', linewidth=2, label='Generator Total Loss')
    plt.plot(train_history['G_GAN'], 'm--', label='Generator GAN Loss')
    plt.plot(train_history['L1'], 'y--', label='Generator L1 Loss')
    plt.plot(train_history['GDL'], 'g--', label='Generator GDL Loss')
    plt.plot(train_history['VGG'], 'k--', label='Generator VGG Loss')  # 添加 VGG loss 曲线
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(samples_dir, 'loss_components.png'))
    plt.close()