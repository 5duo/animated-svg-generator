import os
import numpy as np
import cv2
import cairosvg
import xml.etree.ElementTree as ET
from PIL import Image
import random
import argparse

def generate_random_svg_params():
    """
    随机生成SVG参数，包括脸型、眼睛位置、嘴型等
    """
    # 脸型参数
    face_rx = random.uniform(70, 90)  # 脸型椭圆的rx
    face_ry = random.uniform(80, 100)  # 脸型椭圆的ry
    
    # 眼睛位置
    eye_left_x = 128 - random.uniform(30, 50)  # 左眼x坐标
    eye_right_x = 128 + random.uniform(30, 50)  # 右眼x坐标
    
    # 选择嘴型 (A/O/E 三类)
    mouth_type = random.choice(['A', 'O', 'E'])
    
    return {
        'face_rx': face_rx,
        'face_ry': face_ry,
        'eye_left_x': eye_left_x,
        'eye_right_x': eye_right_x,
        'mouth_type': mouth_type
    }

def create_svg_from_params(params):
    """
    从参数创建SVG字符串
    """
    face_rx = params['face_rx']
    face_ry = params['face_ry']
    eye_left_x = params['eye_left_x']
    eye_right_x = params['eye_right_x']
    mouth_type = params['mouth_type']
    
    svg_template = f'''<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <!-- 脸型 -->
  <ellipse id="face" cx="128" cy="128" rx="{face_rx}" ry="{face_ry}" fill="#f8d9e9" stroke="#d9a8c8" stroke-width="2"/>

  <!-- 左眼 -->
  <g id="eye-left">
    <ellipse cx="{eye_left_x}" cy="100" rx="12" ry="15" fill="white" stroke="#333" stroke-width="1"/>
    <ellipse cx="{eye_left_x}" cy="100" rx="6" ry="8" fill="#333"/>
    <ellipse cx="{eye_left_x + 2}" cy="98" rx="2" ry="2" fill="white"/>
  </g>

  <!-- 右眼 -->
  <g id="eye-right">
    <ellipse cx="{eye_right_x}" cy="100" rx="12" ry="15" fill="white" stroke="#333" stroke-width="1"/>
    <ellipse cx="{eye_right_x}" cy="100" rx="6" ry="8" fill="#333"/>
    <ellipse cx="{eye_right_x + 2}" cy="98" rx="2" ry="2" fill="white"/>
  </g>

  <!-- 嘴型A -->
  <path id="mouth-A" d="M{128-18},160 Q128,180 {128+18},160" stroke="#e75480" stroke-width="3" fill="none" {"display='none'" if mouth_type != 'A' else ""}/>

  <!-- 嘴型O -->
  <circle id="mouth-O" cx="128" cy="165" r="8" stroke="#e75480" stroke-width="3" fill="none" {"display='none'" if mouth_type != 'O' else ""}/>

  <!-- 嘴型E -->
  <path id="mouth-E" d="M{128-18},165 L{128+18},165" stroke="#e75480" stroke-width="3" fill="none" {"display='none'" if mouth_type != 'E' else ""}/>
</svg>'''
    
    return svg_template

def generate_dataset(output_dir, num_samples=10000, append_mode=True):
    """
    生成训练数据集
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建目录
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # 如果启用追加模式，找到下一个可用的编号
    start_index = 0
    if append_mode:
        existing_files = [f for f in os.listdir(images_dir) if f.endswith('.png') and f[:-4].isdigit()]
        if existing_files:
            numbers = [int(f[:-4]) for f in existing_files if f[:-4].isdigit()]  # 移除.png扩展名并转换为整数
            if numbers:  # 确保numbers列表非空
                start_index = max(numbers) + 1

    print(f"开始生成 {num_samples} 个训练样本，从索引 {start_index} 开始...")

    for i in range(start_index, start_index + num_samples):
        # 生成随机参数
        params = generate_random_svg_params()

        # 创建SVG
        svg_content = create_svg_from_params(params)

        # 保存SVG
        svg_path = os.path.join(labels_dir, f"{i:05d}.svg")
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        # 转换为PNG
        png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
        # 从PNG数据创建PIL图像并转换为RGB（移除alpha通道）
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(png_data))
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        # 保存RGB图像
        rgb_png_path = os.path.join(images_dir, f"{i:05d}.png")
        img.save(rgb_png_path, 'PNG')

        # 保存参数标签文件
        label_path = os.path.join(labels_dir, f"{i:05d}.txt")
        with open(label_path, 'w') as f:
            f.write(f"face_rx {params['face_rx']}\n")
            f.write(f"face_ry {params['face_ry']}\n")
            f.write(f"eye_left_x {params['eye_left_x']}\n")
            f.write(f"eye_right_x {params['eye_right_x']}\n")
            f.write(f"mouth_type {params['mouth_type']}\n")

        if (i + 1 - start_index) % 1000 == 0:
            print(f"已生成 {i + 1 - start_index} 个样本...")

    print(f"数据集生成完成！共 {num_samples} 个样本，从索引 {start_index} 到 {start_index + num_samples - 1}，保存在 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成训练数据集')
    parser.add_argument('--output_dir', type=str, default='./data', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10000, help='生成样本数量')
    parser.add_argument('--append', action='store_true', default=True, help='是否追加到现有数据（默认为是）')

    args = parser.parse_args()

    generate_dataset(args.output_dir, args.num_samples, append_mode=args.append)