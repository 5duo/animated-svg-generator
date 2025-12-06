"""
可控SVG生成器 - 结合特征检测
实现嘴型控制(A/O/E)和眨眼功能
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import argparse
import os


class ControllableFeatureDetector:
    def __init__(self):
        # 初始化OpenCV人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect_features(self, image_path):
        """检测人脸特征"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # 获取第一个人脸的位置
            (x, y, w, h) = faces[0]
            
            # 在人脸区域内检测眼睛和嘴巴
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # 检测眼睛
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # 检测嘴巴（使用笑脸检测器作为嘴型检测）
            mouths = self.mouth_cascade.detectMultiScale(roi_gray)
            
            features = {
                'face_x': x,
                'face_y': y,
                'face_w': w,
                'face_h': h,
                'eye_count': len(eyes),
                'mouth_detected': len(mouths) > 0,
                'eyes': eyes,  # 眼睛位置信息
            }
            
            # 根据嘴巴大小判断嘴型 (A/O/E)
            if len(mouths) > 0:
                # 可以根据嘴巴的长宽比来判断嘴型
                mx, my, mw, mh = mouths[0]  # 使用最大的嘴巴
                aspect_ratio = mw / mh if mh > 0 else 0
                if aspect_ratio > 2.0:  # 宽嘴巴 - A型
                    features['mouth_type'] = 'A'
                elif aspect_ratio < 1.0:  # 圆嘴巴 - O型
                    features['mouth_type'] = 'O'
                else:  # 中等形状 - E型
                    features['mouth_type'] = 'E'
            else:
                features['mouth_type'] = 'E'  # 默认嘴型
            
            # 判断是否眨眼（通过眼睛数量判断）
            if len(eyes) < 2:  # 少于两只眼睛可能表示眨眼
                features['is_blinking'] = True
            else:
                features['is_blinking'] = False
                
            return features, image.shape[:2]  # 返回图像尺寸
        
        return None, image.shape[:2]

    def generate_controllable_svg(self, features, image_shape, controls=None):
        """根据检测特征和控制参数生成SVG"""
        if controls is None:
            controls = {
                'mouth_type': features.get('mouth_type', 'E'),
                'is_blinking': features.get('is_blinking', False),
                'face_color': '#f8d9e9',
                'stroke_color': '#d9a8c8',
                'stroke_width': '2',
                'eye_color': 'white',
                'pupil_color': '#333',
            }
        
        height, width = image_shape
        # 计算面部中心和尺寸
        face_x = features.get('face_x', width//2 - 100)
        face_y = features.get('face_y', height//2 - 100)
        face_w = features.get('face_w', 200)
        face_h = features.get('face_h', 200)
        
        # 计算眼睛位置（基于面部位置）
        left_eye_x = face_x + face_w//3
        right_eye_x = face_x + 2*face_w//3
        eye_y = face_y + face_h//3
        
        # 计算嘴部位置
        mouth_x = face_x + face_w//2
        mouth_y = face_y + 2*face_h//3
        
        # 根据面部大小调整SVG参数
        face_rx = face_w // 2
        face_ry = face_h // 2
        
        svg_content = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <!-- 脸型 -->
  <ellipse id="face" cx="{face_x + face_w//2}" cy="{face_y + face_h//2}" rx="{face_rx}" ry="{face_ry}" fill="{controls['face_color']}" stroke="{controls['stroke_color']}" stroke-width="{controls['stroke_width']}"/>

  <!-- 左眼 -->
  <g id="eye-left">
    <ellipse cx="{left_eye_x}" cy="{eye_y}" rx="12" ry="15" fill="{controls['eye_color']}" stroke="{controls['pupil_color']}" stroke-width="1"/>
    <ellipse cx="{left_eye_x}" cy="{eye_y}" rx="6" ry="8" fill="{controls['pupil_color']}"/>
    <ellipse cx="{left_eye_x + 2}" cy="{eye_y - 2}" rx="2" ry="2" fill="white"/>
    <!-- 眨眼时的额外效果 -->
    {f'<path d="M {left_eye_x-12} {eye_y} Q {left_eye_x} {eye_y+5} {left_eye_x+12} {eye_y}" stroke="{controls["pupil_color"]}" stroke-width="2" fill="none"/>' if controls['is_blinking'] else ''}
  </g>

  <!-- 右眼 -->
  <g id="eye-right">
    <ellipse cx="{right_eye_x}" cy="{eye_y}" rx="12" ry="15" fill="{controls['eye_color']}" stroke="{controls['pupil_color']}" stroke-width="1"/>
    <ellipse cx="{right_eye_x}" cy="{eye_y}" rx="6" ry="8" fill="{controls['pupil_color']}"/>
    <ellipse cx="{right_eye_x + 2}" cy="{eye_y - 2}" rx="2" ry="2" fill="white"/>
    <!-- 眨眼时的额外效果 -->
    {f'<path d="M {right_eye_x-12} {eye_y} Q {right_eye_x} {eye_y+5} {right_eye_x+12} {eye_y}" stroke="{controls["pupil_color"]}" stroke-width="2" fill="none"/>' if controls['is_blinking'] else ''}
  </g>

  <!-- 嘴型A - 张开的嘴 -->
  <path id="mouth-A" d="M{mouth_x-18},{mouth_y+10} Q{mouth_x},{mouth_y+30} {mouth_x+18},{mouth_y+10}" stroke="#e75480" stroke-width="3" fill="none" {'display="none"' if controls['mouth_type'] != 'A' else ''}/>

  <!-- 嘴型O - 圆形嘴 -->
  <circle id="mouth-O" cx="{mouth_x}" cy="{mouth_y+15}" r="8" stroke="#e75480" stroke-width="3" fill="none" {'display="none"' if controls['mouth_type'] != 'O' else ''}/>

  <!-- 嘴型E - 平嘴 -->
  <path id="mouth-E" d="M{mouth_x-18},{mouth_y+15} L{mouth_x+18},{mouth_y+15}" stroke="#e75480" stroke-width="3" fill="none" {'display="none"' if controls['mouth_type'] != 'E' else ''}/>
</svg>'''

        return svg_content


def main():
    parser = argparse.ArgumentParser(description='从真人照片生成可控SVG（支持嘴型和眨眼控制）')
    parser.add_argument('--input_image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output_svg', type=str, default='./controllable_output.svg', help='输出SVG路径')
    parser.add_argument('--mouth_type', type=str, choices=['A', 'O', 'E'], default=None, help='强制设置嘴型 (A/O/E)')
    parser.add_argument('--blink', action='store_true', help='启用眨眼效果')

    args = parser.parse_args()
    
    detector = ControllableFeatureDetector()
    
    # 检测图像特征
    features, image_shape = detector.detect_features(args.input_image)
    
    if features is None:
        print("未检测到人脸，使用默认参数生成SVG")
        # 使用默认参数
        controls = {
            'mouth_type': args.mouth_type or 'E',
            'is_blinking': args.blink,
        }
        # 使用默认图像尺寸
        temp_image = cv2.imread(args.input_image)
        default_features = {
            'face_x': temp_image.shape[1]//2 - 100,
            'face_y': temp_image.shape[0]//2 - 100,
            'face_w': 200,
            'face_h': 200,
        }
        image_shape = temp_image.shape[:2]
        svg_content = detector.generate_controllable_svg(default_features, image_shape, controls)
    else:
        # 使用检测到的特征生成基本SVG
        # 如果用户指定了控制参数，则覆盖检测结果
        controls = {
            'mouth_type': args.mouth_type or features.get('mouth_type', 'E'),
            'is_blinking': args.blink or features.get('is_blinking', False),
            'face_color': '#f8d9e9',
            'stroke_color': '#d9a8c8',
            'stroke_width': '2',
            'eye_color': 'white',
            'pupil_color': '#333',
        }
        svg_content = detector.generate_controllable_svg(features, image_shape, controls)
    
    # 保存SVG
    with open(args.output_svg, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"可控SVG已生成并保存到: {args.output_svg}")
    print(f"检测到的特征: {features if features else '未检测到人脸'}")
    print(f"使用的控制参数: mouth_type={controls['mouth_type']}, blink={controls['is_blinking']}")


if __name__ == "__main__":
    main()