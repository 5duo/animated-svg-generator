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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import re

class LoliDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # 获取所有图片路径
        self.image_paths = glob.glob(os.path.join(images_dir, "*.png"))
        
        # 初始化标签编码器
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['A', 'O', 'E'])  # 嘴型类别
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path).split('.')[0]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 加载标签
        label_path = os.path.join(self.labels_dir, f"{img_name}.txt")
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 解析标签
        labels_dict = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                key, value = parts[0], ' '.join(parts[1:])
                if key == 'mouth_type':
                    labels_dict[key] = self.label_encoder.transform([value])[0]
                else:
                    labels_dict[key] = float(value)
        
        # 返回特征和标签
        sample = {
            'image': image,
            'face_rx': torch.tensor(labels_dict['face_rx'], dtype=torch.float32),
            'face_ry': torch.tensor(labels_dict['face_ry'], dtype=torch.float32),
            'eye_left_x': torch.tensor(labels_dict['eye_left_x'], dtype=torch.float32),
            'eye_right_x': torch.tensor(labels_dict['eye_right_x'], dtype=torch.float32),
            'mouth_type': torch.tensor(labels_dict['mouth_type'], dtype=torch.long)
        }
        
        return sample


class LoliSVGGenerator(nn.Module):
    def __init__(self, num_classes=3):
        super(LoliSVGGenerator, self).__init__()
        
        # 使用预训练的EfficientNet-B0作为特征提取器
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 修改最后的分类层为回归任务
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])  # 移除最后的分类层
        
        # 获取特征维度
        num_features = self.backbone.classifier[1].in_features
        
        # 回归头：预测脸型参数 (rx, ry) 和眼睛位置 (left_x, right_x)
        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 4),  # face_rx, face_ry, eye_left_x, eye_right_x
        )
        
        # 分类头：预测嘴型 (A/O/E)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        # 回归预测
        regression_output = self.regression_head(x)
        face_rx = torch.sigmoid(regression_output[:, 0]) * 150 + 50  # 映射到合理范围 (50, 200)
        face_ry = torch.sigmoid(regression_output[:, 1]) * 150 + 50
        eye_left_x = torch.sigmoid(regression_output[:, 2]) * 100 + 78   # 映射到范围 (78, 178)
        eye_right_x = torch.sigmoid(regression_output[:, 3]) * 100 + 78
        
        # 分类预测
        classification_output = self.classification_head(x)
        
        return {
            'face_rx': face_rx,
            'face_ry': face_ry,
            'eye_left_x': eye_left_x,
            'eye_right_x': eye_right_x,
            'mouth_type_probs': classification_output
        }


def train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.001, device='cpu'):
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet-B0期望输入为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    dataset = LoliDataset(images_dir, labels_dir, transform=transform)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 创建模型
    model = LoliSVGGenerator().to(device)
    
    # 定义损失函数
    # 对于回归任务使用MSE，对于分类任务使用CrossEntropy
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    print("开始训练...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # 真实标签
            true_face_rx = batch['face_rx'].to(device)
            true_face_ry = batch['face_ry'].to(device)
            true_eye_left_x = batch['eye_left_x'].to(device)
            true_eye_right_x = batch['eye_right_x'].to(device)
            true_mouth_type = batch['mouth_type'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算回归损失
            reg_loss = (
                reg_criterion(outputs['face_rx'], true_face_rx) +
                reg_criterion(outputs['face_ry'], true_face_ry) +
                reg_criterion(outputs['eye_left_x'], true_eye_left_x) +
                reg_criterion(outputs['eye_right_x'], true_eye_right_x)
            ) / 4
            
            # 计算分类损失
            cls_loss = cls_criterion(outputs['mouth_type_probs'], true_mouth_type)
            
            # 总损失
            loss = reg_loss + cls_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        
        print(f'Epoch [{epoch+1}/{epochs}] 平均损失: {avg_loss:.4f}, '
              f'回归损失: {avg_reg_loss:.4f}, 分类损失: {avg_cls_loss:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存模型检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"./models/checkpoint_epoch_{epoch+1}.pth"
            os.makedirs("./models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"模型检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = "./models/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存: {final_model_path}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练Loli SVG生成模型')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu 或 cuda)')
    
    args = parser.parse_args()
    
    trained_model = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )