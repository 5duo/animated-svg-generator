"""
真人照片生成可控SVG
使用OpenCV进行边缘检测和轮廓提取生成SVG
"""

import cv2
import numpy as np
from PIL import Image
import argparse
import os


class ControllableSVGGenerator:
    def __init__(self):
        pass

    def detect_edges_and_contours(self, image_path, low_threshold=50, high_threshold=150, min_contour_area=100):
        """
        使用Canny边缘检测和轮廓查找生成可控制的SVG
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        return filtered_contours, image.shape

    def contour_to_svg_path(self, contour):
        """将轮廓转换为SVG路径字符串"""
        if len(contour) < 2:
            return ""
        
        # 简化轮廓点以减少SVG大小
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 构建SVG路径
        path_data = []
        for i, point in enumerate(approx):
            x, y = point[0]
            if i == 0:
                path_data.append(f"M {x},{y}")
            else:
                path_data.append(f" L {x},{y}")
        
        # 闭合路径
        path_data.append(" Z")
        return "".join(path_data)

    def generate_svg_from_contours(self, contours, image_shape, control_params=None):
        """
        根据轮廓生成SVG
        control_params: 控制参数字典，如颜色、线条粗细等
        """
        if control_params is None:
            control_params = {
                'stroke_color': '#000000',
                'stroke_width': 2,
                'fill_color': 'none',
                'simplify_factor': 0.005
            }
        
        height, width = image_shape[:2]
        
        # 开始构建SVG
        svg_content = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
        
        # 添加每个轮廓作为路径
        for contour in contours:
            path_data = self.contour_to_svg_path(contour)
            if path_data:
                svg_content += f'  <path d="{path_data}" stroke="{control_params["stroke_color"]}" stroke-width="{control_params["stroke_width"]}" fill="{control_params["fill_color"]}"/>\n'
        
        svg_content += '</svg>'
        return svg_content

    def process_image(self, image_path, output_svg_path, control_params=None):
        """
        处理图像并生成可控的SVG
        """
        # 检测边缘和轮廓
        contours, image_shape = self.detect_edges_and_contours(image_path)
        
        # 生成SVG
        svg_content = self.generate_svg_from_contours(contours, image_shape, control_params)
        
        # 保存SVG
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"SVG已生成并保存到: {output_svg_path}")
        return svg_content

    def process_with_face_detection(self, image_path, output_svg_path, control_params=None):
        """
        使用人脸检测增强的处理方法
        """
        if control_params is None:
            control_params = {
                'stroke_color': '#000000',
                'stroke_width': 2,
                'fill_color': 'none'
            }
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 尝试使用OpenCV的人脸检测
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        height, width = image.shape[:2]
        
        # 开始构建SVG
        svg_content = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
        
        # 如果检测到人脸，在人脸区域使用更精细的参数
        for (x, y, w, h) in faces:
            # 为人脸区域创建一个更精细的边缘检测
            face_roi = image[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # 为人脸区域应用边缘检测
            face_blurred = cv2.GaussianBlur(face_gray, (3, 3), 0)
            face_edges = cv2.Canny(face_blurred, 30, 100)  # 人脸使用更敏感的参数
            
            # 查找人脸区域的轮廓
            face_contours, _ = cv2.findContours(face_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 添加人脸轮廓到SVG
            for contour in face_contours:
                # 将轮廓坐标转换回原图坐标系
                contour = contour + np.array([x, y])
                path_data = self.contour_to_svg_path(contour)
                if path_data:
                    svg_content += f'  <path d="{path_data}" stroke="{control_params["stroke_color"]}" stroke-width="{control_params["stroke_width"]}" fill="{control_params["fill_color"]}"/>\n'
        
        # 对整张图片进行边缘检测（除了人脸区域，以避免重复）
        # 这里我们使用较宽松的参数检测身体轮廓
        body_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        body_edges = cv2.Canny(body_blurred, 70, 200)
        
        # 查找身体轮廓
        body_contours, _ = cv2.findContours(body_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤掉已经被人脸检测覆盖的小轮廓
        min_body_contour_area = 500
        filtered_body_contours = [cnt for cnt in body_contours if cv2.contourArea(cnt) > min_body_contour_area]
        
        # 添加身体轮廓到SVG（避免与人脸区域重叠）
        for contour in filtered_body_contours:
            # 检查该轮廓是否与任何人脸区域重叠
            contour_bbox = cv2.boundingRect(contour)
            overlap = False
            
            for (fx, fy, fw, fh) in faces:
                face_rect = (fx, fy, fx+fw, fy+fh)
                contour_rect = (contour_bbox[0], contour_bbox[1], 
                              contour_bbox[0]+contour_bbox[2], contour_bbox[1]+contour_bbox[3])
                
                # 简单的边界框重叠检测
                if not (contour_rect[2] < face_rect[0] or contour_rect[0] > face_rect[2] or 
                        contour_rect[3] < face_rect[1] or contour_rect[1] > face_rect[3]):
                    overlap = True
                    break
            
            if not overlap:
                path_data = self.contour_to_svg_path(contour)
                if path_data:
                    svg_content += f'  <path d="{path_data}" stroke="{control_params["stroke_color"]}" stroke-width="{control_params["stroke_width"]}" fill="{control_params["fill_color"]}"/>\n'
        
        svg_content += '</svg>'
        
        # 保存SVG
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"增强版SVG已生成并保存到: {output_svg_path}")
        return svg_content


def main():
    parser = argparse.ArgumentParser(description='从真人照片生成可控SVG')
    parser.add_argument('--input_image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output_svg', type=str, default='./output.svg', help='输出SVG路径')
    parser.add_argument('--method', type=str, choices=['contour', 'face_aware'], default='face_aware', 
                       help='处理方法: contour(普通轮廓) 或 face_aware(人脸感知)')
    parser.add_argument('--stroke_color', type=str, default='#000000', help='线条颜色')
    parser.add_argument('--stroke_width', type=int, default=2, help='线条宽度')
    parser.add_argument('--fill_color', type=str, default='none', help='填充颜色')
    
    args = parser.parse_args()
    
    generator = ControllableSVGGenerator()
    
    control_params = {
        'stroke_color': args.stroke_color,
        'stroke_width': args.stroke_width,
        'fill_color': args.fill_color
    }
    
    try:
        if args.method == 'face_aware':
            svg_content = generator.process_with_face_detection(
                args.input_image, args.output_svg, control_params
            )
        else:
            svg_content = generator.process_image(
                args.input_image, args.output_svg, control_params
            )
        
        print("SVG生成完成!")
        print("生成的SVG内容预览 (前500个字符):")
        print(svg_content[:500] + "..." if len(svg_content) > 500 else svg_content)
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()