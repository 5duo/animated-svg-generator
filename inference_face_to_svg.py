"""
使用训练好的人脸到SVG模型进行推理
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
import cv2
from train_face_to_svg import FaceToSVGModel


def load_model(model_path, device=None):
    """加载训练好的模型"""
    # 如果未指定设备，则自动检测
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        # 如果指定了设备，则检查其可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU进行推理")
            device = 'cpu'

    device = torch.device(device)
    model = FaceToSVGModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """预处理输入图像"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # 添加批次维度


def predict_svg_params(model, image_tensor, original_image_shape, device='cpu'):
    """使用模型预测SVG参数"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)

        # 提取预测结果（已归一化到[0,1]）
        face_params = outputs['face_params'][0].cpu().numpy()
        mouth_logits = outputs['mouth_logits'][0].cpu().numpy()

        # 获取原始图像尺寸
        orig_h, orig_w = original_image_shape[:2]

        # 反归一化参数到原始图像尺寸
        face_rx = face_params[0] * orig_w
        face_ry = face_params[1] * orig_h
        center_x = face_params[2] * orig_w
        center_y = face_params[3] * orig_h
        eye_left_x = face_params[4] * orig_w
        eye_left_y = face_params[5] * orig_h
        eye_right_x = face_params[6] * orig_w
        eye_right_y = face_params[7] * orig_h

        # 获取嘴型预测
        mouth_type_idx = np.argmax(mouth_logits)
        mouth_type_map = {0: 'A', 1: 'O', 2: 'E'}
        mouth_type = mouth_type_map[mouth_type_idx]

    return {
        'face_rx': float(face_rx),
        'face_ry': float(face_ry),
        'center_x': float(center_x),
        'center_y': float(center_y),
        'eye_left_x': float(eye_left_x),
        'eye_left_y': float(eye_left_y),
        'eye_right_x': float(eye_right_x),
        'eye_right_y': float(eye_right_y),
        'mouth_type': mouth_type
    }


def create_svg_from_predictions(params, image_shape):
    """根据预测参数创建SVG"""
    orig_h, orig_w = image_shape[:2]
    center_x = params['center_x']
    center_y = params['center_y']
    face_rx = params['face_rx']
    face_ry = params['face_ry']
    eye_left_x = params['eye_left_x']
    eye_left_y = params['eye_left_y']
    eye_right_x = params['eye_right_x']
    eye_right_y = params['eye_right_y']
    mouth_type = params['mouth_type']

    # 根据预测的嘴型决定显示哪个嘴型
    mouth_a_display = "display='none'" if mouth_type != 'A' else ""
    mouth_o_display = "display='none'" if mouth_type != 'O' else ""
    mouth_e_display = "display='none'" if mouth_type != 'E' else ""

    svg_content = f'''<svg width="{orig_w}" height="{orig_h}" viewBox="0 0 {orig_w} {orig_h}" xmlns="http://www.w3.org/2000/svg">
  <!-- 脸型 -->
  <ellipse id="face" cx="{center_x}" cy="{center_y}" rx="{face_rx}" ry="{face_ry}" fill="#f8d9e9" stroke="#d9a8c8" stroke-width="2"/>

  <!-- 左眼 -->
  <g id="eye-left">
    <ellipse cx="{eye_left_x}" cy="{eye_left_y}" rx="12" ry="15" fill="white" stroke="#333" stroke-width="1"/>
    <ellipse cx="{eye_left_x}" cy="{eye_left_y}" rx="6" ry="8" fill="#333"/>
    <ellipse cx="{eye_left_x + 2}" cy="{eye_left_y - 2}" rx="2" ry="2" fill="white"/>
  </g>

  <!-- 右眼 -->
  <g id="eye-right">
    <ellipse cx="{eye_right_x}" cy="{eye_right_y}" rx="12" ry="15" fill="white" stroke="#333" stroke-width="1"/>
    <ellipse cx="{eye_right_x}" cy="{eye_right_y}" rx="6" ry="8" fill="#333"/>
    <ellipse cx="{eye_right_x + 2}" cy="{eye_right_y - 2}" rx="2" ry="2" fill="white"/>
  </g>

  <!-- 嘴型A -->
  <path id="mouth-A" d="M{center_x-18},{center_y+20} Q{center_x},{center_y+40} {center_x+18},{center_y+20}" stroke="#e75480" stroke-width="3" fill="none" {mouth_a_display}/>

  <!-- 嘴型O -->
  <circle id="mouth-O" cx="{center_x}" cy="{center_y+25}" r="8" stroke="#e75480" stroke-width="3" fill="none" {mouth_o_display}/>

  <!-- 嘴型E -->
  <path id="mouth-E" d="M{center_x-18},{center_y+25} L{center_x+18},{center_y+25}" stroke="#e75480" stroke-width="3" fill="none" {mouth_e_display}/>
</svg>'''

    return svg_content


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型从人脸图像生成SVG')
    parser.add_argument('--input_image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--model_path', type=str, default='./models/face2svg_final_model.pth', help='模型路径')
    parser.add_argument('--output_svg', type=str, default='./output.svg', help='输出SVG路径')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cuda, cpu, 或不指定则自动检测)')

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在。请先训练模型。")
        print("请运行: python train_face_to_svg.py --data_dir <人脸图像目录>")
        return

    # 加载模型
    print("加载模型...")
    model = load_model(args.model_path, args.device)

    # 获取原始图像尺寸
    original_image = cv2.imread(args.input_image)
    if original_image is None:
        print(f"错误: 无法读取图像 {args.input_image}")
        return

    # 预处理输入图像
    print("预处理输入图像...")
    image_tensor = preprocess_image(args.input_image)

    # 预测SVG参数
    print("预测SVG参数...")
    params = predict_svg_params(model, image_tensor, original_image.shape, args.device)

    print(f"预测参数: {params}")

    # 创建SVG
    print("生成SVG...")
    svg_content = create_svg_from_predictions(params, original_image.shape)

    # 保存SVG
    with open(args.output_svg, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"SVG已保存到: {args.output_svg}")


if __name__ == "__main__":
    main()