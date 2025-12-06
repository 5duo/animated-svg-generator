"""
真人照片生成可控SVG
使用OpenCV和mediapipe进行人体关键点检测，生成SVG轮廓
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class HumanSVGGenerator:
    def __init__(self):
        # 初始化MediaPipe人体检测
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # 初始化脸部检测
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect_pose_landmarks(self, image_path):
        """检测人体关键点"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])))
            return landmarks
        return None

    def detect_face_landmarks(self, image_path):
        """检测面部关键点"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = []
            for face_landmark in results.multi_face_landmarks[0].landmark:
                face_landmarks.append((int(face_landmark.x * image.shape[1]), int(face_landmark.y * image.shape[0])))
            return face_landmarks
        return None

    def generate_svg_from_landmarks(self, pose_landmarks, face_landmarks, image_width, image_height):
        """根据关键点生成SVG"""
        # 创建SVG根元素
        svg_content = f'<svg width="{image_width}" height="{image_height}" viewBox="0 0 {image_width} {image_height}" xmlns="http://www.w3.org/2000/svg">\n'
        
        # 如果有面部关键点，生成面部轮廓
        if face_landmarks:
            # 简化面部特征
            svg_content += self._draw_face(face_landmarks, image_width, image_height)
        
        # 如果有人体关键点，生成身体轮廓
        if pose_landmarks:
            # 简化身体轮廓
            svg_content += self._draw_body(pose_landmarks, image_width, image_height)
        
        svg_content += '</svg>'
        return svg_content

    def _draw_face(self, face_landmarks, image_width, image_height):
        """绘制面部特征"""
        svg_str = ""
        
        # 获取面部特征点
        face_points = np.array(face_landmarks)
        
        # 绘制轮廓 (使用部分点形成面部轮廓)
        contour_points = [
            face_landmarks[10],   # forehead
            face_landmarks[234],  # right forehead
            face_landmarks[127],  # right eye brow
            face_landmarks[162],  # right cheek
            face_landmarks[21],   # nose tip
            face_landmarks[54],   # left cheek
            face_landmarks[454],  # left eye brow
            face_landmarks[446],  # left forehead
        ]
        
        # 生成面部轮廓路径
        path_data = "M " + " L ".join([f"{x},{y}" for x, y in contour_points])
        svg_str += f'  <path d="{path_data}" stroke="#000" stroke-width="2" fill="none"/>\n'
        
        # 绘制眼睛
        # 右眼
        right_eye_points = [face_landmarks[i] for i in [33, 160, 159, 158, 133, 153, 145, 144]]
        svg_str += f'  <path d="M {" L ".join([f"{x},{y}" for x, y in right_eye_points])} Z" stroke="#000" stroke-width="1" fill="none"/>\n'
        
        # 左眼
        left_eye_points = [face_landmarks[i] for i in [263, 388, 387, 386, 362, 384, 373, 382]]
        svg_str += f'  <path d="M {" L ".join([f"{x},{y}" for x, y in left_eye_points])} Z" stroke="#000" stroke-width="1" fill="none"/>\n'
        
        return svg_str

    def _draw_body(self, pose_landmarks, image_width, image_height):
        """绘制身体轮廓"""
        svg_str = ""
        
        # 定义身体连接关系
        body_connections = [
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value),
            (self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value, self.mp_pose.PoseLandmark.LEFT_EYE.value),
            (self.mp_pose.PoseLandmark.LEFT_EYE.value, self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value),
            (self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value, self.mp_pose.PoseLandmark.NOSE.value),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value),
            (self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, self.mp_pose.PoseLandmark.RIGHT_EYE.value),
            (self.mp_pose.PoseLandmark.RIGHT_EYE.value, self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value),
            (self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value, self.mp_pose.PoseLandmark.NOSE.value),
            
            # 脖子到肩膀
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            
            # 肩膀到手臂
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value),
            
            # 身体
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_HIP.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
            
            # 腿部
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value),
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
        ]
        
        # 绘制连接线
        for connection in body_connections:
            if connection[0] < len(pose_landmarks) and connection[1] < len(pose_landmarks):
                x1, y1 = pose_landmarks[connection[0]]
                x2, y2 = pose_landmarks[connection[1]]
                svg_str += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" stroke-width="2"/>\n'
        
        return svg_str

    def process_image(self, image_path, output_svg_path):
        """处理图像并生成SVG"""
        # 检测关键点
        pose_landmarks = self.detect_pose_landmarks(image_path)
        face_landmarks = self.detect_face_landmarks(image_path)
        
        # 获取图像尺寸
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        
        # 生成SVG
        svg_content = self.generate_svg_from_landmarks(pose_landmarks, face_landmarks, image_width, image_height)
        
        # 保存SVG
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return svg_content


def main():
    # 使用示例
    generator = HumanSVGGenerator()
    
    # 处理图像
    input_image_path = './test_image.jpg'  # 替换为您的图像路径
    output_svg_path = './output.svg'
    
    try:
        svg_content = generator.process_image(input_image_path, output_svg_path)
        print(f"SVG已生成并保存到: {output_svg_path}")
        print("SVG内容预览:")
        print(svg_content[:500] + "..." if len(svg_content) > 500 else svg_content)
    except Exception as e:
        print(f"处理图像时出错: {e}")


if __name__ == "__main__":
    main()