from flask import Flask, request, jsonify, send_from_directory
from flask import send_file
from flask_cors import CORS
import subprocess
import threading
import os
import json
import sys
import psutil
import atexit
import re

app = Flask(__name__)
CORS(app)  # 启用CORS

import time

# 全局变量用于跟踪任务进程
training_process = None
training_status = {
    'running': False,
    'progress': 0,
    'message': '等待开始训练',
    'start_time': None,
    'elapsed_time': 0,  # 本次训练时长（秒）
    'total_training_time': 0,  # 累计训练时长（秒）
    'log': [],  # 训练日志
    'current_epoch': 0,  # 当前epoch
    'total_epochs': 0,  # 总epochs数
    'resume_from_epoch': 0  # 从哪个epoch恢复训练
}

# 数据生成状态
data_generation_status = {
    'running': False,
    'progress': 0,
    'message': '等待开始数据生成',
    'current_file': '',  # 正在处理的文件
    'total_files': 0,    # 总文件数
    'processed_files': 0  # 已处理文件数
}

# 任务状态文件路径
TASK_STATUS_FILE = '../task_data/training_status.json'
LOG_FILE = '../task_data/training_log.txt'


def load_task_status():
    """加载任务状态"""
    global training_status
    try:
        with open(TASK_STATUS_FILE, 'r', encoding='utf-8') as f:
            saved_status = json.load(f)

        # 仅在训练正在运行时恢复状态
        if saved_status.get('running', False):
            training_status.update(saved_status)
            print(f"恢复训练状态: {training_status['message']}")
        else:
            # 重新初始化未运行的训练状态
            training_status = {
                'running': False,
                'progress': 0,
                'message': '等待开始训练',
                'start_time': None,
                'elapsed_time': 0,
                'total_training_time': training_status.get('total_training_time', 0),
                'log': []
            }

            # 从日志文件恢复历史日志
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                    training_status['log'] = [line.strip() for line in log_lines[-100:]]  # 保留最近100条
            except FileNotFoundError:
                pass

    except FileNotFoundError:
        # 文件不存在，使用默认状态
        training_status['log'] = []
        try:
            # 尝试从日志文件恢复
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
                training_status['log'] = [line.strip() for line in log_lines[-100:]]  # 保留最近100条
        except FileNotFoundError:
            pass
    except Exception as e:
        print(f"加载任务状态时出错: {e}")
        training_status['log'] = []


def save_task_status():
    """保存任务状态"""
    try:
        with open(TASK_STATUS_FILE, 'w', encoding='utf-8') as f:
            # 创建可序列化版本的状态
            serializable_status = training_status.copy()
            # 移除不可序列化的项（如start_time如果它是datetime对象）
            if isinstance(serializable_status['start_time'], (float, int)):
                pass  # 可以序列化
            else:
                # 如果是datetime对象，转换为时间戳
                if hasattr(serializable_status['start_time'], 'timestamp'):
                    serializable_status['start_time'] = serializable_status['start_time'].timestamp()

            json.dump(serializable_status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存任务状态时出错: {e}")


def save_log_to_file(log_message):
    """保存日志到文件"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    except Exception as e:
        print(f"保存日志到文件时出错: {e}")


def clear_log_file():
    """清空日志文件"""
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            pass
    except Exception as e:
        print(f"清空日志文件时出错: {e}")


# 注册程序退出时的保存函数
def cleanup():
    if training_status['running']:
        save_task_status()
    else:
        # 仅保存非运行状态，防止意外保存错误状态
        temp_status = training_status.copy()
        temp_status['running'] = False
        try:
            with open(TASK_STATUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(temp_status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"退出时保存任务状态出错: {e}")


atexit.register(cleanup)

# 程序启动时加载任务状态
load_task_status()

# 从文件中加载累计训练时长
def load_total_training_time():
    try:
        with open('total_training_time.txt', 'r') as f:
            return float(f.read().strip())
    except:
        return 0

# 保存累计训练时长到文件
def save_total_training_time(time):
    with open('total_training_time.txt', 'w') as f:
        f.write(str(time))

# 初始化累计训练时长
training_status['total_training_time'] = load_total_training_time()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('.', 'favicon.ico')

@app.route('/train', methods=['POST'])
def start_training():
    global training_process, training_status

    try:
        if training_status['running']:
            return jsonify({'error': '训练已在进行中'}), 400

        # 获取参数
        data = request.json
        if not data:
            return jsonify({'error': '无效的JSON数据'}), 400
        data_dir = data.get('data_dir', './data/faces_with_labels/images')  # 更新默认数据目录
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)

        # 更新状态
        training_status['running'] = True
        training_status['progress'] = 0
        training_status['message'] = '正在启动人脸到SVG模型训练...'
        training_status['start_time'] = time.time()  # 记录开始时间
        training_status['elapsed_time'] = 0  # 重置本次训练时长

        # 构建训练命令 - 使用新的训练脚本
        import os
        train_script_path = os.path.abspath(os.path.join('..', 'train_face_to_svg.py'))
        data_dir_abs = os.path.abspath(os.path.join('..', data_dir))

        # 自动检测设备
        cmd = [
            sys.executable, train_script_path,
            '--data_dir', data_dir_abs,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--learning_rate', str(learning_rate)
            # 不指定--device参数，让脚本自动检测
        ]

        # 如果提供了resume_from参数，则添加到命令中
        resume_from = data.get('resume_from')
        if resume_from:
            cmd.extend(['--resume_from', resume_from])

        # 如果提供了model_save_path参数，则添加到命令中
        model_save_path = data.get('model_save_path')
        if model_save_path:
            # 确保目录存在
            model_dir = os.path.dirname(model_save_path)
            os.makedirs(model_dir, exist_ok=True)
            cmd.extend(['--model_path', model_save_path])
        else:
            # 使用默认路径
            default_path = './models/face2svg_final_model.pth'
            os.makedirs('./models', exist_ok=True)
            cmd.extend(['--model_path', default_path])

        # 更新训练状态
        training_status['current_epoch'] = 0
        training_status['total_epochs'] = int(epochs)
        training_status['resume_from_epoch'] = 0

        # 如果是从检查点恢复，计算起始epoch
        if resume_from:
            import re
            match = re.search(r'epoch_(\d+)', os.path.basename(resume_from))
            if match:
                resume_epoch = int(match.group(1))
                training_status['resume_from_epoch'] = resume_epoch
                training_status['current_epoch'] = resume_epoch

        print(f"训练命令: {' '.join(cmd)}")  # 调试信息

        # 启动训练进程（在后台线程中）
        thread = threading.Thread(target=run_training, args=(cmd,))
        thread.start()

        return jsonify({'message': '训练已启动'})
    except Exception as e:
        return jsonify({'error': f'启动训练时发生错误: {str(e)}'}), 500

@app.route('/train/status', methods=['GET'])
def get_training_status():
    # 如果训练正在进行，计算当前已用时间
    if training_status['running'] and training_status['start_time']:
        training_status['elapsed_time'] = time.time() - training_status['start_time']

    # 计算时间字符串
    elapsed_str = str(timedelta(seconds=int(training_status['elapsed_time'])))
    total_str = str(timedelta(seconds=int(training_status['total_training_time'])))

    response = {
        **training_status,
        'elapsed_time_str': elapsed_str,
        'total_training_time_str': total_str
    }

    return jsonify(response)

@app.route('/train/log', methods=['GET'])
def get_training_log():
    """获取训练日志"""
    return jsonify({'log': training_status['log']})

from datetime import timedelta

# 导入数据生成接口
from data_generator_api import add_data_generation_routes

# 添加数据生成路由
add_data_generation_routes(app)

def generate_data_from_celeba(input_dir, output_dir, max_count=None):
    """
    从CelebA数据集生成训练数据
    """
    import os
    import cv2
    import numpy as np
    from PIL import Image
    import json
    import glob
    import time
    import re

    print(f"从CelebA生成训练数据: {input_dir} -> {output_dir}")
    print(f"最大处理数量: {max_count if max_count else '无限制'}")

    # 创建输出目录
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # 获取输出目录中已存在的文件，确保跳过已处理的文件
    existing_files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        existing_files.update(glob.glob(os.path.join(images_output_dir, ext)))
        existing_files.update(glob.glob(os.path.join(images_output_dir, ext.upper())))

    # 提取已存在文件的文件名（不包含路径和扩展名）
    existing_basenames = set()
    existing_numeric_ids = set()  # 存储已存在的数字ID
    for file_path in existing_files:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        existing_basenames.add(basename)
        # 尝试提取数字ID
        try:
            existing_numeric_ids.add(int(basename))
        except ValueError:
            pass  # 非数字文件名

    print(f"输出目录中已存在 {len(existing_basenames)} 个文件")

    # 支持的图像格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"在源目录找到 {len(image_paths)} 张图像")

    # 如果已存在数字文件，从最大编号之后开始处理
    start_id = 1  # 默认从1开始
    if existing_numeric_ids:
        start_id = max(existing_numeric_ids) + 1  # 从最大已存在编号之后开始
        print(f"从编号 {start_id:06d} 开始处理")

    # 按数字编号收集源文件
    source_files_by_id = {}
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        try:
            file_id = int(basename)
            source_files_by_id[file_id] = img_path
        except ValueError:
            # 非数字文件名，暂时忽略，或者也加入处理
            pass

    # 确定要处理的文件范围
    if max_count:
        # 从start_id开始，尝试处理最多max_count个文件
        filtered_image_paths = []
        current_id = start_id
        processed_count = 0

        while processed_count < max_count:
            if current_id in source_files_by_id:
                src_path = source_files_by_id[current_id]
                # 检查目标文件是否已存在
                basename = os.path.splitext(os.path.basename(src_path))[0]
                if basename not in existing_basenames:
                    filtered_image_paths.append(src_path)
                    processed_count += 1
            current_id += 1

            # 防止无限循环
            if current_id > 300000:  # CelebA数据集的最大编号
                break
    else:
        # 如果没有限制，则处理所有未处理的文件
        filtered_image_paths = []
        for img_path in image_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            if basename not in existing_basenames:
                filtered_image_paths.append(img_path)

    print(f"需要处理 {len(filtered_image_paths)} 张新图像")

    # 限制处理数量 - 确保不超过指定的最大数量
    if max_count and len(filtered_image_paths) > max_count:
        filtered_image_paths = filtered_image_paths[:max_count]
        print(f"限制处理数量为 {max_count} 张")
    else:
        print(f"将处理 {len(filtered_image_paths)} 张图像")

    # 更新数据生成状态
    global data_generation_status
    data_generation_status['total_files'] = len(filtered_image_paths)
    data_generation_status['processed_files'] = 0

    processed_count = 0
    # skipped_count现在只统计本次处理中因各种原因跳过的文件数
    skipped_count = 0

    # 优先使用GPU进行处理
    import os
    # 不设置CUDA_VISIBLE_DEVICES，让PyTorch自动检测可用的GPU

    for i, img_path in enumerate(filtered_image_paths):
        if not data_generation_status['running']:  # 如果数据生成已停止，则退出
            print("数据生成被停止")
            break

        # 更新进度和当前文件
        data_generation_status['current_file'] = os.path.basename(img_path)
        data_generation_status['processed_files'] = i + 1
        data_generation_status['progress'] = int(((i + 1) / len(filtered_image_paths)) * 100) if len(filtered_image_paths) > 0 else 0
        data_generation_status['message'] = f'正在处理: {os.path.basename(img_path)} ({i+1}/{len(filtered_image_paths)})'

        print(f"处理图像 {i+1}/{len(filtered_image_paths)}: {os.path.basename(img_path)}")

        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                print(f"  无法读取图像: {os.path.basename(img_path)}")
                skipped_count += 1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 加载OpenCV预训练分类器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                print(f"  跳过 (未检测到人脸): {os.path.basename(img_path)}")
                skipped_count += 1
                continue

            # 取第一个检测到的人脸
            (x, y, w, h) = faces[0]

            # 在人脸区域内检测眼睛
            roi_gray = gray[y:y+h, x:x+w]

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
                face_params['eye_right_y'] = float(y + h * 0.7)

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

            # 读取并调整图像大小
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            # 构造输出路径，使用原文件名
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_img_path = os.path.join(images_output_dir, f"{base_name}.jpg")
            output_label_path = os.path.join(labels_output_dir, f"{base_name}.json")

            # 保存调整大小后的图像
            img.save(output_img_path, 'JPEG', quality=95)

            # 保存标签文件
            with open(output_label_path, 'w', encoding='utf-8') as f:
                json.dump(face_params, f, indent=2, ensure_ascii=False)

            processed_count += 1

            # 更新进度
            data_generation_status['processed_files'] = i + 1
            data_generation_status['progress'] = int(((i + 1) / len(filtered_image_paths)) * 100) if len(filtered_image_paths) > 0 else 0
            data_generation_status['message'] = f'已处理: {os.path.basename(img_path)} ({i+1}/{len(filtered_image_paths)})'

            if processed_count % 50 == 0:
                print(f"  已处理 {processed_count} 张图像")
                data_generation_status['message'] = f'CelebA数据生成: 已处理 {processed_count} 张图像'

        except Exception as e:
            print(f"处理图像时出错 {os.path.basename(img_path)}: {str(e)}")
            skipped_count += 1
            continue

    print(f"\nCelebA数据生成完成!")
    print(f"成功处理: {processed_count} 张")
    print(f"跳过: {skipped_count} 张")

    return processed_count, skipped_count

@app.route('/train/stop', methods=['POST'])
def stop_training():
    global training_process, training_status

    try:
        if training_process and training_process.poll() is None:
            # 终止进程树，确保所有子进程都被终止
            import psutil
            try:
                parent = psutil.Process(training_process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                gone, still_alive = psutil.wait_procs(children, timeout=3)
                for p in still_alive:
                    p.kill()
                parent.terminate()
                parent.wait(timeout=3)
            except psutil.NoSuchProcess:
                # 进程可能已经结束
                pass
            except psutil.TimeoutExpired:
                # 如果进程没有正常终止，强制杀死
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

            training_status['running'] = False
            training_status['message'] = '训练已停止'

        return jsonify({'message': '训练停止请求已发送'})
    except Exception as e:
        return jsonify({'error': f'停止训练时发生错误: {str(e)}'}), 500

@app.route('/generate_from_celeba', methods=['POST'])
def generate_from_celeba():
    """从CelebA数据集生成训练数据"""
    global data_generation_status

    try:
        data = request.json
        input_dir = data.get('input_dir', '../data/img_align_celeba/img_align_celeba')
        output_dir = data.get('output_dir', '../data/faces_with_labels')
        max_count = data.get('max_count', 2000)  # 默认处理2000张图片

        # 检查是否已有任务在运行（包括训练和数据生成）
        if training_status.get('running', False):
            return jsonify({'error': '有训练任务正在进行，无法生成新数据'}), 400
        if data_generation_status.get('running', False):
            return jsonify({'error': '有数据生成任务正在进行，无法启动新任务'}), 400

        # 更新状态
        data_generation_status['running'] = True
        data_generation_status['progress'] = 0
        data_generation_status['message'] = '正在从CelebA数据集生成训练数据...'

        # 在后台线程中启动数据生成
        thread = threading.Thread(
            target=run_data_generation,
            args=(input_dir, output_dir, max_count)
        )
        thread.start()

        return jsonify({'message': '数据生成已启动'})
    except Exception as e:
        data_generation_status['running'] = False
        data_generation_status['message'] = f'启动数据生成时出现错误: {str(e)}'
        return jsonify({'error': f'启动数据生成失败: {str(e)}'}), 500

def run_data_generation(input_dir, output_dir, max_count):
    """在后台线程中运行数据生成"""
    global data_generation_status
    try:
        # 启动数据生成
        processed_count, skipped_count = generate_data_from_celeba(
            input_dir=input_dir,
            output_dir=output_dir,
            max_count=max_count
        )

        # 完成后更新状态
        data_generation_status['running'] = False
        data_generation_status['progress'] = 100
        data_generation_status['message'] = f'CelebA数据生成完成! 成功处理: {processed_count} 张, 跳过: {skipped_count} 张'
        data_generation_status['processed_count'] = processed_count
        data_generation_status['skipped_count'] = skipped_count

        print(f"数据生成完成: 成功处理 {processed_count} 张, 跳过 {skipped_count} 张")
    except Exception as e:
        data_generation_status['running'] = False
        data_generation_status['message'] = f'数据生成过程中出现错误: {str(e)}'
        print(f"数据生成错误: {str(e)}")

@app.route('/data_generation/status', methods=['GET'])
def get_data_generation_status():
    """获取数据生成状态"""
    global data_generation_status
    return jsonify(data_generation_status)

@app.route('/data_generation/stop', methods=['POST'])
def stop_data_generation():
    """停止数据生成任务"""
    global data_generation_status
    try:
        # 更新状态，标记为停止
        data_generation_status['running'] = False
        data_generation_status['message'] = '数据生成已手动停止'

        return jsonify({'message': '数据生成停止请求已发送'})
    except Exception as e:
        return jsonify({'error': f'停止数据生成时发生错误: {str(e)}'}), 500

@app.route('/checkpoints', methods=['GET'])
def get_checkpoints():
    """获取可用的模型检查点列表"""
    import os
    import glob

    models_dir = '../models'
    checkpoint_pattern = os.path.join(models_dir, 'face2svg_checkpoint_*.pth')

    checkpoint_files = glob.glob(checkpoint_pattern)

    # 提取epoch数字并排序
    def extract_epoch(filename):
        import re
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0

    checkpoint_files.sort(key=lambda x: extract_epoch(x), reverse=True)

    checkpoints = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        checkpoints.append({
            'filename': filename,
            'path': filepath,
            'epoch': extract_epoch(filepath),
            'size': os.path.getsize(filepath),
            'modified': os.path.getmtime(filepath)
        })

    return jsonify({
        'checkpoints': checkpoints,
        'count': len(checkpoints)
    })

@app.route('/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """获取训练数据集统计信息"""
    import os
    import time

    # 检测人脸数据集目录
    faces_images_dir = os.path.join('..', 'data', 'faces_with_labels', 'images')
    faces_labels_dir = os.path.join('..', 'data', 'faces_with_labels', 'labels')

    image_count = 0
    label_count = 0
    latest_update = 0

    if os.path.exists(faces_images_dir):
        for filename in os.listdir(faces_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_count += 1
                # 检查文件修改时间
                file_time = os.path.getmtime(os.path.join(faces_images_dir, filename))
                if file_time > latest_update:
                    latest_update = file_time

    if os.path.exists(faces_labels_dir):
        for filename in os.listdir(faces_labels_dir):
            if filename.lower().endswith('.json'):
                label_count += 1
                # 检查文件修改时间
                file_time = os.path.getmtime(os.path.join(faces_labels_dir, filename))
                if file_time > latest_update:
                    latest_update = file_time

    # 转换时间戳为可读格式
    latest_update_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_update)) if latest_update > 0 else 'N/A'

    return jsonify({
        'image_count': image_count,
        'label_count': label_count,
        'latest_update': latest_update_str,
        'images_dir': faces_images_dir,
        'labels_dir': faces_labels_dir,
        'status': '就绪' if image_count > 0 else '缺少数据，请上传人脸图像或准备数据集'
    })

@app.route('/train/kill_all', methods=['POST'])
def kill_all_training():
    """终止所有训练相关进程"""
    try:
        import subprocess
        import signal
        import os

        # 使用pkill命令终止所有train_face_to_svg.py相关的进程
        result = subprocess.run(['pkill', '-f', 'train_face_to_svg.py'],
                                capture_output=True, text=True)

        # 同时终止当前管理的训练进程
        global training_process, training_status
        if training_process:
            try:
                # 获取进程ID
                pid = training_process.pid
                os.kill(pid, signal.SIGTERM)
            except:
                pass  # 如果无法终止特定进程，继续执行
            finally:
                training_process = None

        # 重置训练状态
        training_status['running'] = False
        training_status['message'] = '所有训练任务已终止'

        return jsonify({'message': '所有训练任务已终止'})
    except Exception as e:
        return jsonify({'error': f'终止所有训练任务时发生错误: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate_svg():
    try:
        # 检查是否有文件被上传
        if 'file' not in request.files:
            return jsonify({'error': '未提供输入图像'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # 获取其他参数
        model_path = request.form.get('model_path', './models/final_model.pth')
        output_path = request.form.get('output_path', './output.svg')
        method = request.form.get('method', 'face_aware')  # 新增方法参数
        mouth_type = request.form.get('mouth_type', None)  # 新增嘴型控制参数
        blink = request.form.get('blink', 'false').lower() == 'true'  # 新增眨眼控制参数

        # 保存上传的文件到临时位置
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            temp_input_path = tmp_file.name

        # 构建生成命令
        cmd = [
            sys.executable, '../inference.py',  # 修正路径到上一级目录
            '--input_image', temp_input_path,
            '--model_path', model_path,
            '--output_svg', output_path
            # 不指定--device参数，让脚本自动检测
        ]

        # 添加可选的控制参数
        if mouth_type:
            cmd.extend(['--mouth_type', mouth_type])
        if blink:
            cmd.append('--blink')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 增加到10分钟超时
        if result.returncode == 0:
            # 读取生成的SVG内容用于预览
            try:
                with open(output_path, 'r', encoding='utf-8') as svg_file:
                    svg_content = svg_file.read()
                return jsonify({
                    'message': 'SVG生成成功',
                    'output_path': output_path,
                    'svg_content': svg_content  # 返回SVG内容用于预览
                })
            except Exception as e:
                # 如果无法读取SVG文件，至少返回成功消息
                return jsonify({
                    'message': 'SVG生成成功，但无法读取预览内容',
                    'output_path': output_path
                })
        else:
            return jsonify({'error': f'SVG生成失败: {result.stderr}'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'SVG生成超时'}), 500
    except Exception as e:
        return jsonify({'error': f'生成SVG时发生错误: {str(e)}'}), 500
    finally:
        # 删除临时文件
        try:
            if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        except:
            pass  # 忽略删除临时文件时的错误

def run_training(cmd):
    global training_process, training_status

    try:
        # 计算总进度 - 基于epochs数
        total_epochs = 0
        for arg in cmd:
            if arg.isdigit():
                if cmd.index(arg) > 0 and cmd[cmd.index(arg)-1] == '--epochs':
                    total_epochs = int(arg)
                    break

        if total_epochs == 0:
            total_epochs = 50  # 默认值

        # 启动训练进程
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # 清空之前的日志
        training_status['log'] = []
        clear_log_file()  # 清空日志文件

        # 读取训练输出并更新进度
        for line in iter(training_process.stdout.readline, ''):
            # 检查训练进程是否还存在，防止在循环中被设置为None
            if training_process is None:
                break
            line = line.strip()
            if line:  # 只处理非空行
                print(line)  # 打印到服务器日志
                # 添加到日志列表，保留最新的100条记录
                training_status['log'].append(line)
                if len(training_status['log']) > 100:
                    training_status['log'] = training_status['log'][-100:]

                # 保存日志到文件
                save_log_to_file(line)

                # 保存当前状态
                save_task_status()

                # 更新进度 - 解析训练输出
                if 'INFO:' in line:
                    # 解析详细信息日志
                    if '使用设备' in line:
                        training_status['message'] = '正在初始化计算设备...'
                    elif '开始加载数据集' in line:
                        training_status['message'] = '正在加载数据集...'
                    elif '正在创建数据集' in line:
                        training_status['message'] = '正在创建数据集...'
                    elif '数据集创建完成' in line:
                        match = re.search(r'共 (\d+) 个样本', line)
                        if match:
                            training_status['message'] = f'数据集创建完成，共{match.group(1)}个样本'
                    elif '正在处理图像' in line:
                        # 检测"正在处理图像 X/Y"格式
                        match = re.search(r'正在处理图像 (\d+)/(\d+)', line)
                        if match:
                            current = int(match.group(1))
                            total = int(match.group(2))
                            # 估算这个阶段的进度
                            image_progress = int((current / total) * 20)  # 假设标签生成占20%的总进度
                            training_status['progress'] = min(image_progress, 20)
                            training_status['message'] = f'正在处理图像 {current}/{total}'
                    elif '标签生成完成' in line:
                        training_status['message'] = '图像标签生成完成'
                    elif '数据加载器创建完成' in line:
                        training_status['message'] = '数据加载器创建完成'
                    elif '正在创建模型' in line:
                        training_status['message'] = '正在创建模型...'
                    elif '模型、损失函数和优化器创建完成' in line:
                        training_status['message'] = '模型创建完成，开始训练...'
                    elif '开始训练' in line and '循环' in line:
                        training_status['message'] = '开始训练循环...'
                    elif '开始第' in line and '轮训练' in line:
                        # 解析轮次信息
                        match = re.search(r'第 (\d+)/(\d+) 轮', line)
                        if match:
                            current_epoch = int(match.group(1))
                            total_epochs = int(match.group(2))
                            training_status['current_epoch'] = current_epoch  # 更新当前epoch

                            # 计算进度：从检查点恢复时需要考虑起始点
                            resume_from = training_status['resume_from_epoch']
                            # 计算从resume_from开始的相对进度
                            if total_epochs > resume_from and resume_from > 0:
                                # 计算当前在总训练中的位置：20% for data prep + (current_epoch - resume_from) / (total_epochs - resume_from) * 80%
                                training_progress = int(20 + max(0, (current_epoch - resume_from) / max(1, total_epochs - resume_from) * 80))
                            else:
                                # 正常训练，没有从检查点恢复或resume_from为0
                                training_progress = int(20 + (current_epoch / total_epochs) * 80)

                            training_status['progress'] = max(training_status['progress'], training_progress)  # 确保进度不会回退
                            training_status['message'] = f'训练中 - 正在执行第 {current_epoch}/{total_epochs} 轮'
                    elif '正在处理第' in line and '当前批次' in line:
                        # 解析当前批次信息
                        epoch_match = re.search(r'第 (\d+) 轮', line)
                        batch_match = re.search(r'当前批次 (\d+)/(\d+)', line)
                        if epoch_match and batch_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(match.group(2)) if match else 50  # 默认值
                            current_batch = int(batch_match.group(1))
                            total_batches = int(batch_match.group(2))

                            # 计算更精确的进度
                            epoch_progress = (current_epoch - 1) / total_epochs
                            batch_progress = current_batch / (total_batches * total_epochs)
                            overall_progress = min(20 + (epoch_progress + batch_progress) * 80, 100)
                            training_status['progress'] = int(overall_progress)
                            training_status['message'] = f'训练中 - 第 {current_epoch}/{total_epochs} 轮, 批次 {current_batch}/{total_batches}'
                    elif '平均回归损失' in line or '平均分类损失' in line:
                        # 解析损失信息
                        epoch_match = re.search(r'第 (\d+)', line)
                        reg_loss_match = re.search(r'平均回归损失: ([\d.]+)', line)
                        cls_loss_match = re.search(r'平均分类损失: ([\d.]+)', line)

                        loss_info = ""
                        if reg_loss_match:
                            loss_info += f", 回归损失: {reg_loss_match.group(1)}"
                        if cls_loss_match:
                            loss_info += f", 分类损失: {cls_loss_match.group(1)}"

                        if epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            training_status['message'] = f'第 {current_epoch} 轮训练完成{loss_info}'
                elif 'TRAIN:' in line:
                    # 解析训练批次日志
                    epoch_match = re.search(r'Epoch \[(\d+)/(\d+)\]', line)
                    batch_match = re.search(r'Batch \[(\d+)/(\d+)\]', line)
                    if epoch_match and batch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        current_batch = int(batch_match.group(1))
                        total_batches = int(batch_match.group(2))
                        training_status['current_epoch'] = current_epoch  # 更新当前epoch

                        # 计算进度：从检查点恢复时需要考虑起始点
                        resume_from = training_status['resume_from_epoch']

                        if total_epochs > resume_from and resume_from > 0:
                            # 考虑从resume_from开始的进度，每轮内部也要按批次计算进度
                            epoch_completion = max(0, current_epoch - resume_from)  # 从resume_from开始完成的epoch数
                            total_epochs_to_complete = total_epochs - resume_from  # 总共需要完成的epoch数
                            epoch_progress = epoch_completion / max(1, total_epochs_to_complete) if total_epochs_to_complete > 0 else 0
                            batch_progress = current_batch / (total_batches * total_epochs_to_complete) if total_epochs_to_complete > 0 else 0
                            overall_progress = min(20 + (epoch_progress + batch_progress) * 80, 100)
                        else:
                            # 正常训练，没有从检查点恢复或resume_from为0
                            epoch_progress = (current_epoch - 1) / total_epochs
                            batch_progress = current_batch / (total_batches * total_epochs)
                            overall_progress = min(20 + (epoch_progress + batch_progress) * 80, 100)

                        training_status['progress'] = max(training_status['progress'], int(overall_progress))  # 确保进度不会回退

                        # 提取损失值
                        loss_match = re.search(r'Loss: ([\d.]+)', line)
                        reg_loss_match = re.search(r'Reg Loss: ([\d.]+)', line)
                        cls_loss_match = re.search(r'Cls Loss: ([\d.]+)', line)

                        loss_info = ""
                        if loss_match:
                            loss_info += f", 损失: {loss_match.group(1)}"
                        if reg_loss_match:
                            loss_info += f", 回归: {reg_loss_match.group(1)}"
                        if cls_loss_match:
                            loss_info += f", 分类: {cls_loss_match.group(1)}"

                        training_status['message'] = f'训练中 - Epoch {current_epoch}/{total_epochs}, Batch {current_batch}/{total_batches}{loss_info}'
                elif 'DONE:' in line:
                    # 解析完成日志
                    epoch_match = re.search(r'第 (\d+)/(\d+) 轮', line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        training_status['current_epoch'] = current_epoch  # 更新当前epoch

                        # 计算进度：从检查点恢复时需要考虑起始点
                        resume_from = training_status['resume_from_epoch']
                        if total_epochs > resume_from and resume_from > 0:
                            # 计算从resume_from开始的相对进度
                            training_status['progress'] = int(20 + max(0, (current_epoch - resume_from) / max(1, total_epochs - resume_from) * 80))
                        else:
                            # 正常训练，没有从检查点恢复或resume_from为0
                            training_status['progress'] = int(20 + (current_epoch / total_epochs) * 80)

                        training_status['message'] = f'第 {current_epoch} 轮训练完成'
                elif 'CHECKPOINT:' in line:
                    training_status['message'] = '正在保存模型检查点...'
                elif 'SUCCESS:' in line:
                    training_status['message'] = '训练完成！模型已保存'
                    training_status['progress'] = 100
                elif 'Epoch' in line and '/' in line and 'INFO:' not in line and 'TRAIN:' not in line:
                    # 兼容旧的日志格式
                    epoch_match = re.search(r'Epoch \[(\d+)/(\d+)\]', line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))

                        # 从输出中提取损失信息 - 支持新旧两种格式
                        loss_match = re.search(r'平均损失: ([\d.]+)', line) or re.search(r'Loss: ([\d.]+)', line)
                        reg_loss_match = re.search(r'回归损失: ([\d.]+)', line) or re.search(r'Reg Loss: ([\d.]+)', line)
                        cls_loss_match = re.search(r'分类损失: ([\d.]+)', line) or re.search(r'Cls Loss: ([\d.]+)', line)

                        loss_info = ""
                        if loss_match:
                            loss_info += f", 总损失: {loss_match.group(1)}"
                        if reg_loss_match:
                            loss_info += f", 回归损失: {reg_loss_match.group(1)}"
                        if cls_loss_match:
                            loss_info += f", 分类损失: {cls_loss_match.group(1)}"

                        # 如果有更详细的批次信息
                        batch_match = re.search(r'Batch \[(\d+)/(\d+)\]', line)
                        if batch_match:
                            current_batch = int(batch_match.group(1))
                            total_batches = int(batch_match.group(2))
                            training_status['current_epoch'] = current_epoch  # 更新当前epoch

                            # 计算进度：从检查点恢复时需要考虑起始点
                            resume_from = training_status['resume_from_epoch']
                            if total_epochs > resume_from and resume_from > 0:
                                # 基于epoch和batch双重信息计算进度，从resume_from开始
                                epoch_completion = max(0, current_epoch - resume_from)  # 从resume_from开始完成的epoch数
                                total_epochs_to_complete = total_epochs - resume_from  # 总共需要完成的epoch数
                                overall_progress = ((epoch_completion) + (current_batch / total_batches)) / max(1, total_epochs_to_complete)
                                training_status['progress'] = min(100, int(20 + overall_progress * 80))  # 将进度映射到20-100之间
                            else:
                                # 正常训练，没有从检查点恢复或resume_from为0
                                overall_progress = ((current_epoch - 1) + (current_batch / total_batches)) / total_epochs
                                training_status['progress'] = int(overall_progress * 100)

                            training_status['message'] = f'训练中 - Epoch {current_epoch}/{total_epochs}, Batch {current_batch}/{total_batches}{loss_info}'
                        else:
                            # 如果没有批次信息，就基于epoch计算进度
                            training_status['current_epoch'] = current_epoch  # 更新当前epoch

                            # 计算进度：从检查点恢复时需要考虑起始点
                            resume_from = training_status['resume_from_epoch']
                            if total_epochs > resume_from and resume_from > 0:
                                # 计算从resume_from开始的相对进度
                                training_status['progress'] = int(20 + max(0, (current_epoch - resume_from) / max(1, total_epochs - resume_from) * 80))
                            else:
                                # 正常训练，没有从检查点恢复或resume_from为0
                                training_status['progress'] = int((current_epoch / total_epochs) * 100)

                            training_status['message'] = f'训练中 - Epoch {current_epoch}/{total_epochs} 完成{loss_info}'

        # 等待进程结束
        if training_process:
            return_code = training_process.wait()
        else:
            return_code = 1  # 假设失败

        # 计算本次训练时长
        if training_status['start_time']:
            actual_elapsed_time = time.time() - training_status['start_time']
            training_status['elapsed_time'] = actual_elapsed_time
            # 将本次训练时长加到累计训练时长中
            training_status['total_training_time'] += actual_elapsed_time
            # 保存累计训练时长到文件
            save_total_training_time(training_status['total_training_time'])

        # 更新最终状态
        if return_code == 0:
            training_status['message'] = '训练完成！模型已保存到 ./models/final_model.pth'
            training_status['progress'] = 100
        else:
            training_status['message'] = f'训练失败，退出码: {return_code}'
            # 添加错误到日志中
            training_status['log'].append(f"训练失败，退出码: {return_code}")
            save_log_to_file(f"训练失败，退出码: {return_code}")
    except Exception as e:
        training_status['message'] = f'训练过程中出现错误: {str(e)}'
        training_status['log'].append(f"训练过程中出现错误: {str(e)}")
        save_log_to_file(f"训练过程中出现错误: {str(e)}")
    finally:
        training_status['running'] = False
        # 保存最终状态和日志
        save_task_status()
        training_process = None

@app.route('/system_stats', methods=['GET'])
def get_system_stats():
    """获取系统资源使用情况"""
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_total = memory.total
        memory_available = memory.available
        memory_used = memory.used

        # GPU使用情况（如果可用）
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {
                    'available': True,
                    'count': gpu_count,
                    'gpus': []
                }

                for i in range(gpu_count):
                    gpu_info['gpus'].append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_used': torch.cuda.memory_allocated(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory,
                        'memory_percent': (torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100
                    })
            else:
                gpu_info = {
                    'available': False,
                    'count': 0,
                    'gpus': []
                }
        except Exception:
            gpu_info = {
                'available': False,
                'count': 0,
                'gpus': []
            }

        # 返回系统资源信息
        return jsonify({
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_total': memory_total,
            'memory_available': memory_available,
            'memory_used': memory_used,
            'gpu_info': gpu_info
        })
    except Exception as e:
        return jsonify({'error': f'获取系统资源时发生错误: {str(e)}'}), 500

def init_training_monitoring():
    """初始化训练监控，检查是否已有训练进程在运行"""
    global training_process, training_status
    import psutil

    # 检查是否有训练进程已经在运行
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            # 检查是否是Python进程且运行的是train.py
            cmd_line = proc.info['cmdline']
            if (proc.info['name'] and 'python' in proc.info['name'].lower() and
                len(cmd_line) > 1 and any('train.py' in arg for arg in cmd_line)):

                # 找到了正在运行的训练进程
                training_process = proc  # 保存进程对象
                training_status['running'] = True
                training_status['progress'] = 0  # 初始化时进度未知，需要从日志或其他方式确定
                training_status['message'] = f'检测到运行中的训练进程 (PID: {proc.info["pid"]})'
                # 使用进程创建时间作为开始时间的近似值
                training_status['start_time'] = proc.info['create_time']
                print(f"检测到运行中的训练进程: PID {proc.info['pid']}")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略无法访问的进程
            continue

@app.route('/model/count', methods=['GET'])
def get_model_count():
    """获取模型数量"""
    try:
        model_dir = '../models'
        model_count = 0
        if os.path.exists(model_dir):
            model_count = len([f for f in os.listdir(model_dir)
                              if f.lower().endswith(('.pth', '.pt', '.h5', '.onnx'))])

        return jsonify({
            'count': model_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory('assets', filename)

if __name__ == '__main__':
    init_training_monitoring()  # 启动时初始化训练监控
    app.run(debug=True, host='0.0.0.0', port=5681)