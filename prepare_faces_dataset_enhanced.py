"""
增强版人脸数据集准备脚本
此脚本使用多种方法为图像生成标签
"""
import os
import cv2
import numpy as np
from PIL import Image
import argparse
import json
import glob


def detect_face_features_advanced(image_path):
    """
    使用多种方法检测人脸特征并生成SVG参数
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 定义多个Haar分类器
    face_cascades = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_profileface.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml'
    ]
    
    faces = []
    
    # 尝试多种分类器
    for face_cascade_path in face_cascades:
        if os.path.exists(face_cascade_path):
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(detected_faces) > 0:
                faces = detected_faces
                break
    
    if len(faces) == 0:
        print(f"  检测失败: {os.path.basename(image_path)} - 未检测到人脸")
        return None  # 没有检测到人脸
    
    # 取最大的人脸（通常是最近的）
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # 计算面部中心和基本参数
    face_params = {
        'face_rx': float(w / 2),  # 椭圆x半径
        'face_ry': float(h / 2),  # 椭圆y半径
        'center_x': float(x + w // 2),  # 中心x坐标
        'center_y': float(y + h // 2),  # 中心y坐标
        'image_width': float(image.shape[1]),
        'image_height': float(image.shape[0])
    }
    
    # 检测眼睛（在人脸区域内）
    roi_gray = gray[y:y+h, x:x+w]
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
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
        # 如果检测不到两个眼睛，使用基于人脸位置的估算
        eye_y = y + h * 0.3
        left_eye_x = x + w * 0.3
        right_eye_x = x + w * 0.7
        
        face_params['eye_left_x'] = float(left_eye_x)
        face_params['eye_left_y'] = float(eye_y)
        face_params['eye_right_x'] = float(right_eye_x)
        face_params['eye_right_y'] = float(eye_y)
    
    # 检测嘴部
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    smiles = smile_cascade.detectMultiScale(roi_gray)
    
    if len(smiles) > 0:
        # 取最大的笑容区域
        largest_smile = max(smiles, key=lambda s: s[2] * s[3])
        sx, sy, sw, sh = largest_smile
        
        # 根据嘴部长宽比判断嘴型
        aspect_ratio = sw / max(sh, 1)  # 防止除零
        if aspect_ratio > 1.5:
            mouth_type = 'A'  # 宽嘴
        elif aspect_ratio < 1.0:
            mouth_type = 'O'  # 圆嘴
        else:
            mouth_type = 'E'  # 普通嘴
    else:
        # 检查图片内容来判断嘴型（对简单图形的特殊处理）
        # 对于简单图形，我们可以通过位置估算嘴的位置
        mouth_y = y + int(h * 0.65)  # 嘴巴通常在脸下半部分
        
        # 简单启发式判断：如果底部区域有特定颜色或形状，判断嘴型
        mouth_region = gray[int(h * 0.6):int(h * 0.75), x:x+w]
        # 如果是简单图形，根据颜色区域判断
        if np.mean(mouth_region) < 100:  # 比较暗的区域可能是嘴巴
            mouth_type = 'E'  # 默认平嘴
        else:
            mouth_type = 'E'  # 默认
    
    face_params['mouth_type'] = mouth_type
    
    return face_params


def generate_synthetic_face_labels(image_path, output_json_path):
    """
    为简单生成的面部图像生成标签
    这是一个后备方法，用于处理无法通过OpenCV检测的简单图像
    """
    # 简单启发式方法，基于图像名称或内容
    filename = os.path.basename(image_path).lower()
    
    # 根据文件名判断嘴型
    mouth_type = 'E'  # 默认
    if 'happy' in filename or 'a' in filename:
        mouth_type = 'A'
    elif 'surprised' in filename or 'o' in filename:
        mouth_type = 'O'
    elif 'neutral' in filename or 'e' in filename:
        mouth_type = 'E'
    
    # 读取图像尺寸
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算默认面部参数
    face_params = {
        'face_rx': width * 0.3,  # 面部椭圆半径
        'face_ry': height * 0.35,
        'center_x': width * 0.5,  # 面部中心
        'center_y': height * 0.45,
        'eye_left_x': width * 0.35,  # 眼睛位置
        'eye_left_y': height * 0.35,
        'eye_right_x': width * 0.65,
        'eye_right_y': height * 0.35,
        'mouth_type': mouth_type,
        'image_width': float(width),
        'image_height': float(height)
    }
    
    # 保存标签
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(face_params, f, indent=2, ensure_ascii=False)
    
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
        
        # 首先尝试使用OpenCV检测
        face_params = detect_face_features_advanced(img_path)
        
        if face_params is None:
            print(f"  OpenCV检测失败，尝试启发式方法: {os.path.basename(img_path)}")
            # 如果OpenCV检测失败，使用启发式方法
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_label_path = os.path.join(labels_output_dir, f"{base_name}.json")
            face_params = generate_synthetic_face_labels(img_path, output_label_path)
        
        if face_params is not None:
            # 读取并调整图像大小
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 保存调整大小后的图像
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_img_path = os.path.join(images_output_dir, f"{base_name}.jpg")
            img.save(output_img_path, 'JPEG', quality=95)
            
            # 如果是启发式生成的标签，需要单独保存
            if not os.path.exists(os.path.join(labels_output_dir, f"{base_name}.json")):
                output_label_path = os.path.join(labels_output_dir, f"{base_name}.json")
                with open(output_label_path, 'w', encoding='utf-8') as f:
                    json.dump(face_params, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"  已处理 {processed_count} 张图像")
        else:
            print(f"  跳过 (无法生成标签): {os.path.basename(img_path)}")
            skipped_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 张")
    print(f"跳过: {skipped_count} 张")
    print(f"输出目录: {output_dir}")
    
    # 显示生成的标签示例
    if processed_count > 0:
        label_files = glob.glob(os.path.join(labels_output_dir, "*.json"))
        if label_files:
            print(f"\n标签示例 ({os.path.basename(label_files[0])}):")
            with open(label_files[0], 'r', encoding='utf-8') as f:
                import pprint
                example = json.load(f)
                pprint.pprint(example)


def main():
    parser = argparse.ArgumentParser(description='使用多种方法为图像生成标签，用于训练人脸到SVG模型')
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