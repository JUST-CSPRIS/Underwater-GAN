import piq
import torch
from PIL import Image
import numpy as np
import os
import glob


def calculate_single_fsim(original_path, enhanced_path):
    """计算单对图像的FSIM分数"""
    try:
        # 读取图像 - 保持为RGB格式
        original_img = np.array(Image.open(original_path).convert('RGB'))
        enhanced_img = np.array(Image.open(enhanced_path).convert('RGB'))

        # 调整尺寸使其一致
        if original_img.shape != enhanced_img.shape:
            enhanced_img = np.array(
                Image.open(enhanced_path).convert('RGB').resize(
                    (original_img.shape[1], original_img.shape[0])
                )
            )

        # 转换为tensor并调整维度顺序 (H, W, C) -> (C, H, W)
        original_tensor = torch.tensor(original_img).permute(2, 0, 1).unsqueeze(0).float()
        enhanced_tensor = torch.tensor(enhanced_img).permute(2, 0, 1).unsqueeze(0).float()

        # 计算FSIM - 使用彩色图像
        fsim_score = piq.fsim(original_tensor, enhanced_tensor, data_range=255.0, chromatic=True).item()

        return fsim_score

    except Exception as e:
        print(f"计算 {os.path.basename(original_path)} 和 {os.path.basename(enhanced_path)} 时发生错误: {e}")
        return None


def find_image_pairs(original_dir, enhanced_dir):
    """自动查找原始图像和增强图像的对应关系"""
    pairs = []

    # 支持的图像格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    # 获取原始图像列表
    original_images = []
    for ext in extensions:
        original_images.extend(glob.glob(os.path.join(original_dir, ext)))
        original_images.extend(glob.glob(os.path.join(original_dir, ext.upper())))

    for original_path in original_images:
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(original_path))[0]

        # 尝试在增强目录中查找对应文件
        enhanced_path = None
        for ext in extensions:
            potential_path = os.path.join(enhanced_dir, f"{filename}{ext[1:]}")
            if os.path.exists(potential_path):
                enhanced_path = potential_path
                break

        # 如果找到对应图像，则添加到对列表中
        if enhanced_path:
            pairs.append((original_path, enhanced_path))
        else:
            print(f"警告: 找不到与 {filename} 对应的增强图像")

    return pairs


def main():
    """主函数"""
    # 设置文件夹路径
    original_dir = "/home/xie/xcl/paper/data/UFOtest/TEST/hr"  # 原始图像文件夹
    #enhanced_dir = "/home/xie/xcl/paper/code/FUnIE-GAN-master/PyTorch/data/ELA-ufo"  # 增强图像文件夹
    enhanced_dir = "/home/xie/wln/1.20/Water-GAN-master/PyTorch/data/ELA-ufo"
    # 检查文件夹是否存在
    if not os.path.exists(original_dir):
        print(f"错误: 原始图像文件夹 '{original_dir}' 不存在")
        return

    if not os.path.exists(enhanced_dir):
        print(f"错误: 增强图像文件夹 '{enhanced_dir}' 不存在")
        return

    # 查找图像对
    print("正在查找图像对...")
    image_pairs = find_image_pairs(original_dir, enhanced_dir)

    if not image_pairs:
        print("没有找到匹配的图像对")
        return

    print(f"找到 {len(image_pairs)} 对图像")

    # 计算FSIM分数
    print("\n开始计算FSIM分数...")
    scores = []

    for original_path, enhanced_path in image_pairs:
        original_name = os.path.basename(original_path)
        enhanced_name = os.path.basename(enhanced_path)

        score = calculate_single_fsim(original_path, enhanced_path)

        if score is not None:
            scores.append(score)
            print(f"{original_name} vs {enhanced_name}: {score:.4f}")
        else:
            print(f"{original_name} vs {enhanced_name}: 计算失败")

    # 计算平均值
    if scores:
        average_score = sum(scores) / len(scores)

        # 质量评价
        if average_score >= 0.95:
            quality = "优秀"
        elif average_score >= 0.90:
            quality = "很好"
        elif average_score >= 0.85:
            quality = "良好"
        elif average_score >= 0.80:
            quality = "一般"
        elif average_score >= 0.75:
            quality = "较差"
        else:
            quality = "很差"

        # 输出结果
        print(f"\n=== 最终结果 ===")
        print(f"总图像对数量: {len(image_pairs)}")
        print(f"成功计算数量: {len(scores)}")
        print(f"平均FSIM分数: {average_score:.4f}")
        print(f"质量评价: {quality}")

        # 保存结果到指定目录
        result_dir = "/home/xie/wln/1.20/Water-GAN-master/PyTorch/result"
        result_path = os.path.join(result_dir, "fsim_results.txt")

        # 确保目标目录存在
        os.makedirs(result_dir, exist_ok=True)

        with open(result_path, "w") as f:
            f.write("FSIM计算结果\n")
            f.write("============\n")
            f.write(f"原始图像文件夹: {original_dir}\n")
            f.write(f"增强图像文件夹: {enhanced_dir}\n")
            f.write(f"总图像对数量: {len(image_pairs)}\n")
            f.write(f"成功计算数量: {len(scores)}\n")
            f.write(f"平均FSIM分数: {average_score:.4f}\n")
            f.write(f"质量评价: {quality}\n\n")
            f.write("各图像对分数:\n")
            for i, (score, (orig, enh)) in enumerate(zip(scores, image_pairs)):
                f.write(f"{i + 1}. {os.path.basename(orig)} vs {os.path.basename(enh)}: {score:.4f}\n")

        print(f"详细结果已保存到 {result_path}")
    else:
        print("所有图像对计算失败")


if __name__ == "__main__":
    main()