"""
# Modified Script for 256x256 Evaluation
# Updates:
# 1. Add explicit resizing methods
# 2. Add path validation
# 3. Add progress tracking
"""
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, basename, exists
from tqdm import tqdm  # 添加进度条

# Local quality metrics
from imqual_utils import getSSIM, getPSNR  # 确保这些函数能处理numpy数组


def aligned_resize(img_path, target_size=(256, 256), resample_method=Image.LANCZOS):
    """统一的图像加载和调整尺寸方法"""
    try:
        img = Image.open(img_path).convert('RGB')
        return img.resize(target_size, resample=resample_method)
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None


def SSIMs_PSNRs(gtr_dir, gen_dir,
                target_size=(256, 256),
                gtr_resample=Image.LANCZOS,
                gen_resample=Image.BICUBIC):
    """
    参数说明：
    - gtr_dir: 原始高清图像目录
    - gen_dir: 生成图像目录
    - target_size: 统一调整的目标尺寸
    - gtr_resample: 原始图像下采样方法
    - gen_resample: 生成图像调整方法（当尺寸不符时使用）
    """
    # 路径验证
    if not exists(gtr_dir) or not exists(gen_dir):
        raise FileNotFoundError("检查输入目录是否存在")

    # 获取配对路径
    pairs = []
    for gtr_path in glob(join(gtr_dir, "*.*")):
        base = basename(gtr_path).split('.')[0]
        gen_path = join(gen_dir, f"{base}.*")
        match = glob(gen_path)
        if match:
            pairs.append((gtr_path, match[0]))

    # 指标计算
    ssims, psnrs = [], []
    for gtr_path, gen_path in tqdm(pairs, desc="处理样本"):
        # 调整尺寸
        gtr_img = aligned_resize(gtr_path, target_size, gtr_resample)
        gen_img = aligned_resize(gen_path, target_size, gen_resample)

        if gtr_img and gen_img:
            # 转换numpy数组
            gtr_np = np.array(gtr_img)
            gen_np = np.array(gen_img)

            # 计算SSIM（RGB空间）
            ssims.append(getSSIM(gtr_np, gen_np))

            # 计算PSNR（亮度通道）
            gtr_gray = np.array(gtr_img.convert('L'))
            gen_gray = np.array(gen_img.convert('L'))
            psnrs.append(getPSNR(gtr_gray, gen_gray))

    return np.array(ssims), np.array(psnrs)


if __name__ == "__main__":
    # 数据集路径
    gtr_dir = "/home/xie/xcl/paper/data/UFOtest/TEST/hr/"  # 原始高清图像640x480
    # gtr_dir = "/home/xie/xcl/paper/GAN/test_samples/test_samples/GTr/"
    # gtr_dir = "/home/xie/xcl/paper/data/LSUI/GT/"


    gen_dir = "/home/xie/xcl/paper/code/Underwater-GAN/PyTorch/data/GAM-ufo"  # 生成图像256x256
    # 执行评估
    ssim_scores, psnr_scores = SSIMs_PSNRs(
        gtr_dir, gen_dir,
        target_size=(256, 256),
        gtr_resample=Image.LANCZOS,  # 高质量下采样
        gen_resample=Image.NEAREST  # 生成图像保持原生像素
    )

    # 结果展示
    print(f"\n评估结果（{len(ssim_scores)}个有效样本）")
    print("SSIM | 均值: {:.4f} ± {:.4f}".format(
        np.mean(ssim_scores), np.std(ssim_scores)))
    print("PSNR | 均值: {:.2f} dB ± {:.2f}".format(
        np.mean(psnr_scores), np.std(psnr_scores)))