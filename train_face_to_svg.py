"""
人脸特征到SVG参数的深度学习模型训练脚本
用于学习从人脸图像到SVG参数的映射关系
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import argparse
import glob
from sklearn.preprocessing import LabelEncoder
import cv2
import json


class FaceToSVGDataset(Dataset):
    """人脸图像到SVG参数的数据集"""

    def __init__(self, images_dir, labels_dir=None, transform=None, generate_labels=False):
        self.images_dir = images_dir
        self.transform = transform

        # 获取所有图片路径
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

        # 如果没有标签文件，则自动生成
        if generate_labels or not labels_dir:
            print(f"INFO: 检测到 {len(self.image_paths)} 张图像，正在为数据集生成标签...")
            self.labels = self._generate_labels()
            print(f"INFO: 标签生成完成")
        else:
            # 加载预生成的标签 - 支持JSON格式
            print(f"INFO: 检测到 {len(self.image_paths)} 张图像，正在加载预生成的标签...")
            self.labels = self._load_labels(labels_dir)
            print(f"INFO: 标签加载完成")
    
    def _generate_labels(self):
        """为图像自动生成标签"""
        print(f"INFO: 开始为 {len(self.image_paths)} 张图像生成标签...")
        labels = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        total_images = len(self.image_paths)
        for idx, img_path in enumerate(self.image_paths):
            if idx % 100 == 0:  # 每处理100张图像输出一次进度
                print(f"INFO: 正在处理图像 {idx+1}/{total_images}...")

            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]

                # 计算人脸参数
                face_params = {
                    'face_rx': w / 2.0,  # 椭圆x半径
                    'face_ry': h / 2.0,  # 椭圆y半径
                    'center_x': x + w // 2,  # 中心x
                    'center_y': y + h // 2,  # 中心y
                }

                # 在人脸区域检测眼睛
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    # 计算左右眼位置
                    eye_centers = []
                    for (ex, ey, ew, eh) in eyes:
                        eye_centers.append((x + ex + ew//2, y + ey + eh//2))

                    # 确保左眼在左，右眼在右
                    eye_centers.sort(key=lambda p: p[0])
                    if len(eye_centers) >= 2:
                        face_params['eye_left_x'] = float(eye_centers[0][0])
                        face_params['eye_left_y'] = float(eye_centers[0][1])
                        face_params['eye_right_x'] = float(eye_centers[1][0])
                        face_params['eye_right_y'] = float(eye_centers[1][1])
                    else:
                        # 如果无法确定左右眼，使用平均值
                        avg_x = np.mean([c[0] for c in eye_centers])
                        face_params['eye_left_x'] = float(avg_x - 30)
                        face_params['eye_right_x'] = float(avg_x + 30)
                        face_params['eye_left_y'] = face_params['eye_right_y'] = float(y + h//3)
                else:
                    # 默认眼睛位置
                    face_params['eye_left_x'] = float(x + w//3)
                    face_params['eye_right_x'] = float(x + 2*w//3)
                    face_params['eye_left_y'] = face_params['eye_right_y'] = float(y + h//3)

                # 检测嘴型
                mouths = mouth_cascade.detectMultiScale(roi_gray)
                if len(mouths) > 0:
                    mx, my, mw, mh = mouths[0]
                    aspect_ratio = mw / mh if mh > 0 else 0
                    if aspect_ratio > 2.0:  # 宽嘴巴 - A型
                        face_params['mouth_type'] = 0  # A
                    elif aspect_ratio < 1.0:  # 圆嘴巴 - O型
                        face_params['mouth_type'] = 1  # O
                    else:  # 中等形状 - E型
                        face_params['mouth_type'] = 2  # E
                else:
                    face_params['mouth_type'] = 2  # 默认E型

                labels.append(face_params)
            else:
                # 没有检测到人脸，使用默认参数
                height, width = img.shape[:2]
                labels.append({
                    'face_rx': width // 4,
                    'face_ry': height // 4,
                    'center_x': width // 2,
                    'center_y': height // 2,
                    'eye_left_x': width // 3,
                    'eye_left_y': height // 3,
                    'eye_right_x': 2 * width // 3,
                    'eye_right_y': height // 3,
                    'mouth_type': 2  # E
                })

        print(f"INFO: 图像标签生成完成，共处理 {len(labels)} 张图像")
        return labels
    
    def _load_labels(self, labels_dir):
        """加载预生成的标签"""
        labels = []
        label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir) if f.endswith('.json')}

        for img_path in self.image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # 检查对应的标签文件是否存在
            if img_name in label_files:
                label_path = os.path.join(labels_dir, label_files[img_name])

                with open(label_path, 'r') as f:
                    original_label = json.load(f)

                    # 标准化标签格式以匹配模型期望的格式
                    normalized_label = {
                        'face_rx': original_label.get('face_rx', 50.0),
                        'face_ry': original_label.get('face_ry', 50.0),
                        'center_x': original_label.get('center_x', 112.0),
                        'center_y': original_label.get('center_y', 112.0),
                        'eye_left_x': original_label.get('eye_left_x', 78.0),
                        'eye_left_y': original_label.get('eye_left_y', 78.0),
                        'eye_right_x': original_label.get('eye_right_x', 146.0),
                        'eye_right_y': original_label.get('eye_right_y', 78.0),
                        'mouth_type': ['A', 'O', 'E'].index(original_label.get('mouth_type', 'E'))
                    }
                    labels.append(normalized_label)
            else:
                print(f"警告: 标签文件不存在 {img_name}.json")
                # 添加默认标签
                labels.append({
                    'face_rx': 50.0,
                    'face_ry': 50.0,
                    'center_x': 112.0,
                    'center_y': 112.0,
                    'eye_left_x': 78.0,
                    'eye_left_y': 78.0,
                    'eye_right_x': 146.0,
                    'eye_right_y': 78.0,
                    'mouth_type': 2  # 默认'E'
                })
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.labels[idx]
        
        # 标准化参数到[0,1]范围
        img_width, img_height = 224, 224  # 经过transform后的尺寸
        
        sample = {
            'image': image,
            'face_rx': torch.tensor(label['face_rx'] / img_width, dtype=torch.float32),
            'face_ry': torch.tensor(label['face_ry'] / img_height, dtype=torch.float32),
            'center_x': torch.tensor(label['center_x'] / img_width, dtype=torch.float32),
            'center_y': torch.tensor(label['center_y'] / img_height, dtype=torch.float32),
            'eye_left_x': torch.tensor(label['eye_left_x'] / img_width, dtype=torch.float32),
            'eye_left_y': torch.tensor(label['eye_left_y'] / img_height, dtype=torch.float32),
            'eye_right_x': torch.tensor(label['eye_right_x'] / img_width, dtype=torch.float32),
            'eye_right_y': torch.tensor(label['eye_right_y'] / img_height, dtype=torch.float32),
            'mouth_type': torch.tensor(label['mouth_type'], dtype=torch.long)
        }

        return sample


class FaceToSVGModel(nn.Module):
    """人脸图像到SVG参数的预测模型"""
    
    def __init__(self, num_mouth_classes=3):
        super(FaceToSVGModel, self).__init__()
        
        # 使用预训练的ResNet作为特征提取器
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 替换最后的分类层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原始分类层
        
        # 回归头：预测面部几何参数
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),  # face_rx, face_ry, center_x, center_y, eye_left_x, eye_left_y, eye_right_x, eye_right_y
        )
        
        # 分类头：预测嘴型 (A/O/E)
        self.classification_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_mouth_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        
        # 预测几何参数
        regression_output = self.regression_head(features)
        
        # 预测嘴型
        classification_output = self.classification_head(features)
        
        return {
            'face_params': regression_output,
            'mouth_logits': classification_output
        }


def train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.001, device=None, resume_from=None, model_path='./models/face2svg_final_model.pth'):
    """训练模型"""
    # 如果未指定设备，则自动检测
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        # 如果指定了设备，则检查其可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU进行训练")
            device = 'cpu'

    device = torch.device(device)
    print(f"INFO: 使用设备: {device}")
    print(f"INFO: 开始加载数据集从: {data_dir}")

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("INFO: 正在创建数据集...")
    # 创建数据集
    dataset = FaceToSVGDataset(
        images_dir=data_dir,
        transform=transform,
        generate_labels=True  # 自动生成标签
    )
    print(f"INFO: 数据集创建完成，共 {len(dataset)} 个样本")

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"INFO: 数据加载器创建完成，批次大小: {batch_size}, 总批次: {len(dataloader)}")

    print("INFO: 正在创建模型...")
    # 创建模型
    model = FaceToSVGModel().to(device)

    # 定义损失函数
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("INFO: 模型、损失函数和优化器创建完成")

    start_epoch = 0  # 默认从第0个epoch开始

    # 如果需要从检查点恢复训练
    if resume_from and os.path.exists(resume_from):
        print(f"INFO: 正在从检查点 {resume_from} 恢复训练...")
        start_epoch = load_checkpoint(model, optimizer, resume_from, device)
    elif resume_from:
        print(f"警告: 检查点文件 {resume_from} 不存在，将从头开始训练")

    print("INFO: 开始训练...")
    model.train()

    print(f"INFO: 开始训练循环，总轮数: {epochs}，从第 {start_epoch} 轮开始")
    for epoch in range(start_epoch, epochs):
        print(f"INFO: 开始第 {epoch+1}/{epochs} 轮训练...")
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:  # 每个epoch开始时输出日志
                print(f"INFO: 正在处理第 {epoch+1} 轮，当前批次 {batch_idx+1}/{len(dataloader)}...")

            images = batch['image'].to(device)

            # 真实标签
            true_face_params = torch.stack([
                batch['face_rx'],
                batch['face_ry'],
                batch['center_x'],
                batch['center_y'],
                batch['eye_left_x'],
                batch['eye_left_y'],
                batch['eye_right_x'],
                batch['eye_right_y']
            ], dim=1).to(device)

            true_mouth_type = batch['mouth_type'].to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            reg_loss = reg_criterion(outputs['face_params'], true_face_params.float())
            cls_loss = cls_criterion(outputs['mouth_logits'], true_mouth_type)

            # 总损失
            loss = reg_loss + cls_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reg_loss += reg_loss.item() * images.size(0)
            total_cls_loss += cls_loss.item() * images.size(0)
            total_samples += images.size(0)

            # 每100个批次输出一次详细信息
            if batch_idx % 100 == 0:
                print(f'TRAIN: Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}')

        avg_reg_loss = total_reg_loss / total_samples
        avg_cls_loss = total_cls_loss / total_samples

        print(f'DONE: 第 {epoch+1}/{epochs} 轮训练完成 - '
              f'平均回归损失: {avg_reg_loss:.4f}, 平均分类损失: {avg_cls_loss:.4f}')

        # 保存模型检查点
        if (epoch + 1) % 10 == 0:
            # 从数据目录提取数据集名称
            data_dir_parts = data_dir.split('/')
            dataset_name = data_dir_parts[-1] if data_dir_parts[-1] else data_dir_parts[-2] if len(data_dir_parts) > 1 else 'dataset'
            # 清理数据集名称，只保留字母数字
            clean_dataset_name = ''.join(c for c in dataset_name if c.isalnum()) or 'dataset'

            checkpoint_path = f"./models/face2svg_checkpoint_{clean_dataset_name}_e{epochs}_b{batch_size}_lr{learning_rate}_epoch_{epoch+1}.pth"
            os.makedirs("./models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"CHECKPOINT: 模型检查点已保存: {checkpoint_path}")

    # 保存最终模型
    print(f"INFO: 训练完成，正在保存最终模型到: {model_path}")
    torch.save(model.state_dict(), model_path)
    print(f"SUCCESS: 最终模型已保存: {model_path}")

    return model


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    """从检查点加载模型和优化器状态"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"INFO: 从检查点加载模型，开始于第 {epoch + 1} 轮")
    return epoch + 1  # 返回下一个要训练的轮数

def main():
    parser = argparse.ArgumentParser(description='训练人脸到SVG参数的模型')
    parser.add_argument('--data_dir', type=str, required=True, help='人脸图像数据目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu 或 cuda)')
    parser.add_argument('--resume_from', type=str, help='从检查点文件继续训练')
    parser.add_argument('--model_path', type=str, default='./models/face2svg_final_model.pth', help='模型保存路径')

    args = parser.parse_args()

    # 确保模型目录存在
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    trained_model = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        resume_from=args.resume_from,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()