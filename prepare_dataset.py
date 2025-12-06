"""
数据集准备脚本
用于下载和预处理人脸数据集
"""
import os
import requests
import zipfile
import tarfile
from PIL import Image
import numpy as np
import argparse
import shutil


def download_file(url, filename):
    """下载文件"""
    print(f"正在下载 {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"下载完成: {filename}")


def extract_archive(archive_path, extract_to):
    """解压归档文件"""
    print(f"正在解压 {archive_path} 到 {extract_to}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print("解压完成")


def validate_and_resize_images(input_dir, output_dir, target_size=(224, 224)):
    """验证图像并调整大小"""
    print(f"验证和调整图像大小: {input_dir} -> {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                input_path = os.path.join(root, file)
                
                try:
                    # 打开并验证图像
                    img = Image.open(input_path)
                    
                    # 转换为RGB（如果需要）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 调整大小
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 保存处理后的图像
                    output_path = os.path.join(output_dir, file)
                    img.save(output_path, 'JPEG', quality=95)
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"已处理 {count} 张图像")
                        
                except Exception as e:
                    print(f"跳过无法处理的图像: {input_path}, 错误: {e}")
                    continue
    
    print(f"图像处理完成，共处理 {count} 张图像")


def prepare_celeba_sample(output_dir):
    """准备CelebA数据集的示例 """
    print("CelebA数据集准备指南:")
    print("1. 访问 https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. 点击 'Download' -> 'Align&Cropped Images' -> 'img_align_celeba.zip'")
    print("3. 下载完成后，解压到指定目录")
    print("4. 使用 validate_and_resize_images 函数调整图像大小")
    
    print("\n或者使用Keras/TensorFlow内置的CelebA数据集:")
    print("pip install tensorflow-datasets")
    print("然后在Python中使用:")
    print("import tensorflow_datasets as tfds")
    print("ds = tfds.load('celeb_a', split='train', shuffle_files=True)")


def prepare_lapa_dataset(output_dir):
    """准备LaPa数据集的示例"""
    print("LaPa数据集准备指南:")
    print("1. 访问 https://github.com/JDAI-CV/lapa-dataset")
    print("2. 按照文档下载数据集")
    print("3. 解压后使用 validate_and_resize_images 函数处理")


def prepare_custom_dataset(images_dir, output_dir):
    """准备自定义人脸数据集"""
    print(f"准备自定义数据集: {images_dir} -> {output_dir}")
    validate_and_resize_images(images_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description='准备人脸数据集用于训练')
    parser.add_argument('--method', type=str, choices=['celeba', 'custom'], 
                       default='custom', help='数据集准备方法')
    parser.add_argument('--input_dir', type=str, help='输入图像目录（用于custom方法）')
    parser.add_argument('--output_dir', type=str, default='./data/faces', 
                       help='输出目录')
    parser.add_argument('--target_size', type=str, default='224,224',
                       help='目标图像尺寸，格式: width,height')
    
    args = parser.parse_args()
    
    # 解析目标尺寸
    target_size = tuple(map(int, args.target_size.split(',')))
    
    print(f"开始准备数据集...")
    print(f"目标尺寸: {target_size}")
    
    if args.method == 'celeba':
        prepare_celeba_sample(args.output_dir)
    elif args.method == 'custom':
        if not args.input_dir:
            print("错误: --input_dir 参数是必须的")
            return
        validate_and_resize_images(args.input_dir, args.output_dir, target_size)
    
    print(f"数据集已准备完成，存储在: {args.output_dir}")


if __name__ == "__main__":
    main()