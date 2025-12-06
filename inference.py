"""
真人照片生成可控SVG - 统一接口
结合特征检测和深度学习模型
"""
import argparse
import os
import sys


def generate_svg_from_image(input_image_path, output_svg_path, **kwargs):
    """
    从输入图像生成SVG - 统一接口
    如果有训练好的模型，则使用深度学习方法
    否则使用特征检测方法
    """
    model_path = kwargs.get('model_path', './models/face2svg_final_model.pth')

    # 检查是否有训练好的模型
    if os.path.exists(model_path):
        # 使用深度学习模型
        from inference_face_to_svg import main as inference_main
        # 由于inference_face_to_svg需要命令行参数，我们直接调用生成函数
        from inference_face_to_svg import load_model, preprocess_image, predict_svg_params, create_svg_from_predictions
        import cv2
        import torch

        device = kwargs.get('device', 'cpu')

        try:
            # 加载模型
            model = load_model(model_path, device)

            # 获取原始图像尺寸
            original_image = cv2.imread(input_image_path)
            if original_image is None:
                print(f"错误: 无法读取图像 {input_image_path}")
                raise Exception(f"无法读取图像 {input_image_path}")

            # 预处理输入图像
            image_tensor = preprocess_image(input_image_path)

            # 预测SVG参数
            params = predict_svg_params(model, image_tensor, original_image.shape, device)

            # 如果用户指定了嘴型，覆盖预测的嘴型
            if kwargs.get('mouth_type'):
                params['mouth_type'] = kwargs['mouth_type']

            # 创建SVG
            svg_content = create_svg_from_predictions(params, original_image.shape)

            # 保存SVG
            with open(output_svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)

            return True, "使用深度学习模型生成SVG成功"

        except Exception as e:
            print(f"使用深度学习模型失败: {e}，回退到特征检测方法")
            # 继续使用特征检测方法
    else:
        print("未找到深度学习模型，使用特征检测方法")

    # 使用特征检测方法
    from controllable_features_svg import ControllableFeatureDetector

    detector = ControllableFeatureDetector()

    # 提取控制参数
    mouth_type = kwargs.get('mouth_type', None)
    blink = kwargs.get('blink', False)

    try:
        # 检测图像特征
        features, image_shape = detector.detect_features(input_image_path)

        if features is None:
            print("未检测到人脸，使用默认参数生成SVG")
            # 使用默认参数
            controls = {
                'mouth_type': mouth_type or 'E',
                'is_blinking': blink,
            }
            # 使用默认图像尺寸
            import cv2
            temp_image = cv2.imread(input_image_path)
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
                'mouth_type': mouth_type or features.get('mouth_type', 'E'),
                'is_blinking': blink or features.get('is_blinking', False),
            }
            svg_content = detector.generate_controllable_svg(features, image_shape, controls)

        # 保存SVG
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        return True, "使用特征检测方法生成SVG成功"
    except Exception as e:
        return False, f"生成SVG时出错: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='从真人照片生成可控SVG（支持嘴型和眨眼控制）')
    parser.add_argument('--input_image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--model_path', type=str, default='./models/face2svg_final_model.pth', help='模型路径')
    parser.add_argument('--output_svg', type=str, default='./output.svg', help='输出SVG路径')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--mouth_type', type=str, choices=['A', 'O', 'E'], default=None, help='强制设置嘴型 (A/O/E)')
    parser.add_argument('--blink', action='store_true', help='启用眨眼效果')

    args = parser.parse_args()

    kwargs = {
        'model_path': args.model_path,
        'mouth_type': args.mouth_type,
        'blink': args.blink,
        'device': args.device
    }

    success, message = generate_svg_from_image(args.input_image, args.output_svg, **kwargs)

    if success:
        print(message)
        print(f"SVG已保存到: {args.output_svg}")
    else:
        print(message)


if __name__ == "__main__":
    main()