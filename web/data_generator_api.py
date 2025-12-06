"""
数据生成器接口 - 更新为支持人脸数据集
"""
import os
from flask import Flask, request, jsonify
import subprocess
import sys
import json

def add_data_generation_routes(app):
    """为Flask应用添加数据生成相关路由"""

    @app.route('/generate_data', methods=['POST'])
    def generate_training_data():
        """生成训练数据 - 现在用于准备人脸数据集"""
        try:
            # 获取参数
            data = request.json
            if not data:
                return jsonify({'error': '无效的JSON数据'}), 400
            num_images = data.get('num_images', 100)  # 这个参数现在不直接使用，因为人脸数据集是预先收集的
            image_size = data.get('image_size', 224)

            # 原来的随机生成方法不再适用，现在是指导用户如何获取人脸数据集
            return jsonify({
                'message': '请参考文档获取人脸数据集，如CelebA、LaPa等公开数据集',
                'instructions': {
                    'CelebA': '下载地址: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html',
                    'LaPa': '下载地址: https://github.com/JDAI-CV/lapa-dataset',
                    'prepare_command': f'python ../prepare_dataset.py --method custom --input_dir /path/to/faces --output_dir ../data/faces'
                }
            })
        except Exception as e:
            return jsonify({'error': f'生成数据时发生错误: {str(e)}'}), 500

    @app.route('/data/status', methods=['GET'])
    def get_data_status():
        """获取数据状态 - 更新为检测人脸数据集"""
        try:
            # 检测人脸数据集目录
            faces_images_dir = os.path.join('..', 'data', 'faces_with_labels', 'images')
            faces_labels_dir = os.path.join('..', 'data', 'faces_with_labels', 'labels')

            image_count = 0
            label_count = 0

            if os.path.exists(faces_images_dir):
                image_count = len([f for f in os.listdir(faces_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

            if os.path.exists(faces_labels_dir):
                label_count = len([f for f in os.listdir(faces_labels_dir) if f.lower().endswith(('.json'))])

            return jsonify({
                'image_count': image_count,
                'label_count': label_count,
                'faces_images_dir': faces_images_dir,
                'faces_labels_dir': faces_labels_dir,
                'status': '就绪' if image_count > 0 else '缺少数据，请上传人脸图像或准备数据集'
            })
        except Exception as e:
            return jsonify({'error': f'获取数据状态时发生错误: {str(e)}'}), 500

    @app.route('/data/clear', methods=['POST'])
    def clear_training_data():
        """清空训练数据"""
        try:
            import shutil

            # 定义数据目录路径 - 与检查状态功能保持一致
            faces_images_dir = os.path.join('..', 'data', 'faces_with_labels', 'images')
            faces_labels_dir = os.path.join('..', 'data', 'faces_with_labels', 'labels')

            # 清空图像目录
            if os.path.exists(faces_images_dir):
                for filename in os.listdir(faces_images_dir):
                    file_path = os.path.join(faces_images_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # 清空标签目录
            if os.path.exists(faces_labels_dir):
                for filename in os.listdir(faces_labels_dir):
                    file_path = os.path.join(faces_labels_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            return jsonify({'message': '人脸数据和标签已清空完成'})
        except Exception as e:
            return jsonify({'error': f'清空数据时发生错误: {str(e)}'}), 500

    @app.route('/upload_celeba_zip', methods=['POST'])
    def upload_celeba_zip():
        """上传CelebA zip文件并自动处理"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': '未提供zip文件'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '未选择文件'}), 400

            if not file.filename.lower().endswith('.zip'):
                return jsonify({'error': '请上传zip格式的文件'}), 400

            import zipfile
            import os

            # 创建数据目录
            faces_dir = os.path.join('..', 'data', 'faces')
            celeba_dir = os.path.join('..', 'celeba_data')
            os.makedirs(celeba_dir, exist_ok=True)

            # 保存上传的zip文件
            zip_path = os.path.join(celeba_dir, 'img_align_celeba.zip')
            file.save(zip_path)

            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(celeba_dir)

            # 验证解压的文件
            img_dir = os.path.join(celeba_dir, 'img_align_celeba')
            if not os.path.exists(img_dir):
                return jsonify({'error': '解压后的目录结构不正确'}), 500

            # 统计图像数量
            import glob
            image_extensions = ['*.jpg', '*.jpeg', '*.png']
            all_images = []
            for ext in image_extensions:
                all_images.extend(glob.glob(os.path.join(img_dir, ext)))
                all_images.extend(glob.glob(os.path.join(img_dir, ext.upper())))

            image_count = len(all_images)

            return jsonify({
                'message': f'成功上传并解压CelebA数据集，共 {image_count} 张图像',
                'image_count': image_count,
                'status': 'uploaded'
            })
        except zipfile.BadZipFile:
            return jsonify({'error': '无效的zip文件'}), 500
        except Exception as e:
            return jsonify({'error': f'处理zip文件时发生错误: {str(e)}'}), 500

    @app.route('/process_celeba_data', methods=['POST'])
    def process_celeba_data():
        """处理CelebA数据，生成指定数量的训练样本"""
        try:
            import subprocess
            import sys
            import os

            # 获取参数
            data = request.json
            num_samples = data.get('num_samples', 1000) if data else 1000

            # 检查CelebA数据是否存在
            celeba_dir = os.path.join('..', 'celeba_data', 'img_align_celeba')
            if not os.path.exists(celeba_dir):
                return jsonify({'error': '未找到CelebA数据，请先上传zip文件'}), 400

            # 运行设置脚本
            script_path = os.path.join('..', 'setup_celeba_for_training.py')
            output_path = os.path.join('..', 'data', 'faces')

            cmd = [
                sys.executable, script_path,
                '--celeba_path', os.path.join('..', 'celeba_data'),
                '--output_path', output_path,
                '--num_samples', str(num_samples)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # 检查生成的图像数量
                import glob
                images_dir = os.path.join(output_path, 'images')
                image_extensions = ['*.jpg', '*.jpeg', '*.png']
                all_images = []
                for ext in image_extensions:
                    all_images.extend(glob.glob(os.path.join(images_dir, ext)))
                    all_images.extend(glob.glob(os.path.join(images_dir, ext.upper())))

                processed_count = len(all_images)

                return jsonify({
                    'message': f'成功处理CelebA数据，生成 {processed_count} 张训练图像',
                    'processed_count': processed_count,
                    'status': 'processed'
                })
            else:
                return jsonify({'error': f'处理CelebA数据失败: {result.stderr}'}), 500
        except subprocess.TimeoutExpired:
            return jsonify({'error': '处理CelebA数据超时'}), 500
        except Exception as e:
            return jsonify({'error': f'处理CelebA数据时发生错误: {str(e)}'}), 500