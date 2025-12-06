"""
使用人脸检测方法准备训练数据
此脚本使用OpenCV人脸检测为现有人脸图像生成标签
"""
import os
import cv2
import numpy as np
from PIL import Image
import argparse
import json
from sklearn.preprocessing import LabelEncoder
import glob


def detect_face_features(image_path):
    """
    检测人脸特征并生成SVG参数
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 加载OpenCV预训练分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None  # 没有检测到人脸
    
    # 取第一个检测到的人脸
    (x, y, w, h) = faces[0]
    
    # 在人脸区域内检测眼睛
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    # 检测嘴部
    smiles = smile_cascade.detectMultiScale(roi_gray)
    
    # 计算面部参数
    face_params = {
        'face_rx': float(w / 2),  # 椭圆x半径
        'face_ry': float(h / 2),  # 椭圆y半径
        'center_x': float(x + w // 2),  # 中心x坐标
        'center_y': float(y + h // 2),  # 中心y坐标
        'image_width': float(image.shape[1]),
        'image_height': float(image.shape[0])
    }
    
    # 处理眼睛位置
    if len(eyes) >= 2:
        # 根据x坐标排序，确定左右眼
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        left_eye = eyes_sorted[0]
        right_eye = eyes_sorted[1] if len(eyes_sorted) > 1 else eyes_sorted[0]
        
        # 计算眼睛中心位置（相对于原图）
        face_params['eye_left_x'] = float(x + left_eye[0] + left_eye[2] // 2)
        face_params['eye_left_y'] = float(y + left_eye[1] + left_eye[3] // 2)
        face_params['eye_right_x'] = float(x + right_eye[0] + right_eye[2] // 2)
        face_params['eye_right_y'] = float(y + right_eye[1] + right_eye[3] // 2)
    else:
        # 如果检测不到两个眼睛，使用默认位置
        face_params['eye_left_x'] = float(x + w * 0.3)
        face_params['eye_left_y'] = float(y + h * 0.3)
        face_params['eye_right_x'] = float(x + w * 0.7)
        face_params['eye_right_y'] = float(y + h * 0.3)
    
    # 嘴型检测和分类
    if len(smiles) > 0:
        # 取最大的笑容区域
        largest_smile = max(smiles, key=lambda s: s[2] * s[3])
        sx, sy, sw, sh = largest_smile
        
        # 根据嘴部长宽比判断嘴型
        aspect_ratio = sw / max(sh, 1)  # 防止除零
        if aspect_ratio > 2.0:
            mouth_type = 'A'  # 宽嘴
        elif aspect_ratio < 1.0:
            mouth_type = 'O'  # 圆嘴
        else:
            mouth_type = 'E'  # 普通嘴
    else:
        # 如果检测不到嘴，使用默认嘴型
        mouth_type = 'E'
    
    face_params['mouth_type'] = mouth_type
    
    return face_params


def process_images_to_dataset(input_dir, output_dir, target_size=(224, 224)):
    """
    处理图像目录中的所有图像，生成训练数据集
    """
    print(f"正在处理图像目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # 支持的图像格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    processed_count = 0
    skipped_count = 0
    
    for i, img_path in enumerate(image_paths):
        print(f"处理图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # 检测人脸特征
        face_params = detect_face_features(img_path)
        
        if face_params is None:
            print(f"  跳过 (未检测到人脸): {os.path.basename(img_path)}")
            skipped_count += 1
            continue
        
        # 读取并调整图像大小
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 保存调整大小后的图像
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_img_path = os.path.join(images_output_dir, f"{base_name}.jpg")
        img.save(output_img_path, 'JPEG', quality=95)
        
        # 保存标签文件
        output_label_path = os.path.join(labels_output_dir, f"{base_name}.json")
        with open(output_label_path, 'w', encoding='utf-8') as f:
            json.dump(face_params, f, indent=2, ensure_ascii=False)
        
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"  已处理 {processed_count} 张图像")
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 张")
    print(f"跳过: {skipped_count} 张")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='使用人脸检测为图像生成标签，用于训练人脸到SVG模型')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='包含人脸图像的输入目录')
    parser.add_argument('--output_dir', type=str, default='./data/faces_with_labels',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 检查输入目录是否包含图像
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_count = 0
    for ext in extensions:
        image_count += len(glob.glob(os.path.join(args.input_dir, ext)))
        image_count += len(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    if image_count == 0:
        print(f"警告: 在 {args.input_dir} 中未找到图像文件")
        print("支持的格式: jpg, jpeg, png, bmp, tiff")
        return
    
    print(f"在 {args.input_dir} 中找到 {image_count} 张图像")
    
    process_images_to_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()