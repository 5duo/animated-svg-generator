# SVG角色生成器 - 人脸到SVG可控转换系统

## 项目概述

这是一个基于深度学习的人脸到SVG转换系统，能够将人脸图像转换为可控的SVG矢量图形。用户可以控制嘴型（A/O/E）和眨眼效果，生成个性化的SVG角色。系统还提供了从CelebA大数据集中生成更多训练数据的功能，以增强模型的泛化能力。

## 主要特性

### 1. 人脸到SVG转换
- 使用深度学习模型将真实人脸图像转换为SVG矢量图形
- 支持嘴型控制：A（张嘴）、O（圆嘴）、E（平嘴）
- 支持眨眼效果控制

### 2. 可控SVG生成
- 基于人脸特征检测的可控SVG生成
- 可选择嘴型和启用/禁用眨眼效果
- 实时预览生成的SVG

### 3. 模型训练
- 支持使用CelebA数据集进行模型训练
- 提供人脸特征检测方法的数据集准备工具
- 训练过程中实时显示进度和日志

### 4. 优化的数据处理
- 按数字顺序处理CelebA数据集文件
- 从已处理文件的最大编号之后继续处理
- 自动跳过已存在的输出文件
- 支持GPU优先处理策略
- 限制处理数量以控制资源使用

### 5. 完整的Web界面
- 直观的用户界面
- 实时训练监控
- 系统资源监控（CPU/内存使用率）
- 任务状态持久化（中断后可恢复）
- 页面刷新后状态保持（刷新后仍显示当前训练状态）
- 从CelebA数据集生成训练数据的功能
- 数据管理功能，可设置处理数量限制

## 系统架构

```
.
├── data/                     # 数据目录
│   ├── faces_with_labels/    # 带标签的人脸数据
│   │   ├── images/           # 人脸图像
│   │   └── labels/           # 标签文件（JSON格式）
│   └── img_align_celeba/     # CelebA原始数据
├── models/                   # 模型保存目录
├── task_data/                # 任务状态数据
├── venv/                     # 虚拟环境
├── web/                      # Web服务目录
│   ├── app.py               # Flask Web应用
│   └── index.html           # 前端界面
├── train_face_to_svg.py     # 模型训练脚本
├── inference_face_to_svg.py # 模型推理脚本
├── controllable_features_svg.py # 特征检测SVG生成
├── prepare_faces_dataset.py # 数据集准备脚本
├── train.py                 # 模型训练主程序
├── inference.py             # 模型推理主程序
└── README.md
```

## 安装与部署

### 1. 环境准备

```bash
# 克隆项目
git clone ...

# 进入项目目录
cd animated-svg-generator

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

**重要提示：在运行训练、推理或Web服务时，必须在激活的虚拟环境中运行Python脚本。**

### 2. 数据集准备

#### 使用已下载的CelebA数据集（推荐）
CelebA数据集已下载并解压到 `./data/img_align_celeba/` 目录（实际图片在 `./data/img_align_celeba/img_align_celeba/`），可以直接使用：
```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate   # Windows

python prepare_faces_dataset.py --input_dir ./data/img_align_celeba/img_align_celeba
```

准备好的数据会存储在 `./data/faces_with_labels/` 目录，其中包含：
- `images/`: 处理后的人脸图像
- `labels/`: 对应的标签文件（JSON格式）

### 3. 模型训练

```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate   # Windows

# 训练新模型
python train_face_to_svg.py --data_dir ./data/faces_with_labels/images --epochs 50 --batch_size 32
```

### 4. 启动Web服务

```bash
# 确保虚拟环境已激活
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate   # Windows

cd web
python app.py
```

访问 http://localhost:5681 查看Web界面

## 使用说明

### 1. SVG生成
1. 点击"生成结果预览"部分的"选择文件"按钮，上传人脸图像
2. 可选择嘴型（A/O/E）或使用自动检测
3. 可选择是否启用眨眼效果
4. 点击"✨ 生成SVG"按钮
5. 生成的SVG将在预览区域显示

### 2. 模型训练
1. 在"训练参数设置"部分配置训练参数
2. 选择数据目录（默认：./data/faces_with_labels/images）
3. 设置训练轮数、批次大小、学习率
4. 点击"🚀 开始训练"启动训练
5. 实时查看训练进度和日志

### 3. 数据管理
- 使用"检查数据状态"按钮查看当前数据集情况
- 使用"🗑️ 清空人脸数据"按钮清理数据
- **新增**：使用"从CelebA生成训练数据"按钮从CelebA数据集扩展训练数据
- 可设置CelebA数据输入目录、输出目录和最大处理数量
- 可在训练配置中修改数据目录路径

### 4. 训练功能特性
- **状态保持**：在训练过程中刷新页面，UI会自动同步到当前训练状态
- 实时监控训练进度、时间、批次信息
- 可随时停止或终止训练任务
- 开始训练后，按钮状态会正确更新（开始按钮禁用，停止按钮启用）
- 刷新页面后，进度条、时间信息和任务详情会同步到当前状态

## 技术详情

### 模型架构
- 使用ResNet50作为骨干网络
- 回归头：预测面部几何参数（脸型、眼睛位置）
- 分类头：预测嘴型类别（A/O/E）

### 数据预处理
- 图像尺寸统一为224x224
- 标准化处理（ImageNet均值和标准差）
- 面部特征标注（基于OpenCV人脸检测）

### 训练策略
- 使用MSELoss进行回归任务
- 使用CrossEntropyLoss进行分类任务
- Adam优化器，学习率可调
- 支持训练进度保存和恢复

### 数据扩展策略
- 从CelebA数据集中提取人脸图像并自动生成标签
- 使用OpenCV的人脸、眼睛和微笑检测器生成面部特征参数
- 按数字顺序处理文件，从最大已处理编号之后开始
- 可控制处理数量以管理训练时间和资源
- 生成的标签格式与现有训练流程兼容

## 项目亮点

1. **可控性**：用户可以主动控制嘴型和眨眼效果
2. **易用性**：提供完整的Web界面，操作简单直观
3. **可扩展性**：支持接入不同的人脸数据集
4. **持久性**：训练状态可保存，断点可恢复
5. **准确性**：基于人脸特征检测，生成的SVG与原图更相似

## 文件说明

- `train_face_to_svg.py`：模型训练主程序
- `inference_face_to_svg.py`：模型推理主程序
- `controllable_features_svg.py`：基于特征检测的SVG生成
- `prepare_faces_dataset.py`：数据集准备工具
- `train.py`：模型训练主程序
- `inference.py`：模型推理主程序
- `web/app.py`：Web服务主程序，包含从CelebA生成数据的API
- `web/index.html`：前端界面，包含数据生成功能

## 注意事项

1. 训练需要大量计算资源，建议使用GPU
2. CelebA数据集已下载到 `./data/img_align_celeba/` 目录（实际图片在 `./data/img_align_celeba/img_align_celeba/`）
3. 模型训练时间取决于数据量和硬件配置
4. 确保有足够的磁盘空间存储模型和中间数据
5. 从CelebA生成数据可能需要较长时间，建议设置合适的最大处理数量
6. 数据生成过程中不能同时进行模型训练，以避免资源冲突
7. 生成的数据将保存到标准数据格式，可用于后续训练
8. **重要**: 运行所有Python脚本前，必须激活虚拟环境（source venv/bin/activate）
9. 数据处理按数字顺序进行，从输出目录中的最大编号之后开始
10. 系统自动跳过已存在的输出文件，避免重复处理

## 致谢

- 使用CelebA数据集进行模型训练
- 基于OpenCV进行人脸特征检测
- TensorFlow Datasets用于数据加载