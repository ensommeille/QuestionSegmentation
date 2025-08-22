#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试卷图像分割主程序

功能:
- 使用训练好的YOLOv8m模型检测试卷中的题目
- 根据检测结果分割出单独的题目图像
- 支持单张图片和批量处理
- 提供可视化结果
- 支持多种输出格式

作者: AI Assistant
日期: 2024
"""

import os
import sys
import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional

try:
    from ultralytics import YOLO
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请运行: pip install ultralytics opencv-python matplotlib pillow numpy")
    sys.exit(1)


class QuestionSegmenter:
    """试卷题目分割器"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        初始化分割器
        
        Args:
            model_path: 训练好的YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.setup_logging()
        self.load_model()
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"segment_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.logger.info(f"加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def detect_questions(self, image_path: str) -> List[Dict]:
        """
        检测图像中的题目
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            检测结果列表，每个元素包含边界框和置信度信息
        """
        try:
            # 进行推理
            results = self.model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # 获取边界框坐标 (xyxy格式)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': 'question',
                            'area': float((x2 - x1) * (y2 - y1))
                        }
                        detections.append(detection)
            
            self.logger.info(f"检测到 {len(detections)} 个题目")
            return detections
            
        except Exception as e:
            self.logger.error(f"检测过程中出现错误: {e}")
            return []
    
    def segment_questions(self, image_path: str, output_dir: str, 
                         save_individual: bool = True, save_annotated: bool = True,
                         min_area: int = 1000) -> Dict:
        """
        分割图像中的题目
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            save_individual: 是否保存单独的题目图像
            save_annotated: 是否保存标注后的原图
            min_area: 最小题目面积阈值
            
        Returns:
            分割结果字典
        """
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # 检测题目
            detections = self.detect_questions(image_path)
            
            # 过滤小面积检测结果
            filtered_detections = [d for d in detections if d['area'] >= min_area]
            
            self.logger.info(f"过滤后保留 {len(filtered_detections)} 个题目")
            
            # 按面积排序（从大到小）
            filtered_detections.sort(key=lambda x: x['area'], reverse=True)
            
            # 获取图像文件名（不含扩展名）
            image_name = Path(image_path).stem
            
            # 保存单独的题目图像
            question_files = []
            if save_individual:
                for i, detection in enumerate(filtered_detections):
                    x1, y1, x2, y2 = detection['bbox']
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(width, int(x2))
                    y2 = min(height, int(y2))
                    
                    # 裁剪题目区域
                    question_img = image_rgb[y1:y2, x1:x2]
                    
                    # 保存题目图像
                    question_filename = f"{image_name}_question_{i+1:03d}.jpg"
                    question_path = output_path / question_filename
                    
                    question_img_bgr = cv2.cvtColor(question_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(question_path), question_img_bgr)
                    
                    question_files.append({
                        'filename': question_filename,
                        'path': str(question_path),
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'area': detection['area']
                    })
            
            # 保存标注后的原图
            annotated_path = None
            if save_annotated:
                annotated_img = self.draw_annotations(image_rgb, filtered_detections)
                annotated_filename = f"{image_name}_annotated.jpg"
                annotated_path = output_path / annotated_filename
                
                annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(annotated_path), annotated_img_bgr)
            
            # 保存检测结果JSON
            results_data = {
                'image_info': {
                    'filename': Path(image_path).name,
                    'width': width,
                    'height': height,
                    'channels': image.shape[2] if len(image.shape) == 3 else 1
                },
                'detection_params': {
                    'model_path': self.model_path,
                    'conf_threshold': self.conf_threshold,
                    'iou_threshold': self.iou_threshold,
                    'min_area': min_area
                },
                'detections': filtered_detections,
                'question_files': question_files,
                'annotated_image': str(annotated_path) if annotated_path else None,
                'timestamp': datetime.now().isoformat()
            }
            
            results_filename = f"{image_name}_results.json"
            results_path = output_path / results_filename
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分割完成，结果保存到: {output_path}")
            
            return results_data
            
        except Exception as e:
            self.logger.error(f"分割过程中出现错误: {e}")
            return {}
    
    def draw_annotations(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像 (RGB格式)
            detections: 检测结果列表
            
        Returns:
            标注后的图像
        """
        annotated_img = image.copy()
        
        # 使用matplotlib绘制
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(annotated_img)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 选择颜色
            color = colors[i % len(colors)]
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            label = f"Q{i+1}: {confidence:.2f}"
            ax.text(
                x1, y1 - 5, label,
                fontsize=10, color=color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
            )
        
        ax.set_xlim(0, annotated_img.shape[1])
        ax.set_ylim(annotated_img.shape[0], 0)
        ax.axis('off')
        
        # 保存到内存
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return buf
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     image_extensions: List[str] = None) -> Dict:
        """
        批量处理图像
        
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            image_extensions: 支持的图像扩展名
            
        Returns:
            批量处理结果
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"找到 {len(image_files)} 个图像文件")
        
        batch_results = {
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'results': [],
            'errors': []
        }
        
        for image_file in image_files:
            try:
                self.logger.info(f"处理图像: {image_file.name}")
                
                # 为每个图像创建单独的输出目录
                image_output_dir = Path(output_dir) / image_file.stem
                
                result = self.segment_questions(
                    str(image_file),
                    str(image_output_dir)
                )
                
                if result:
                    batch_results['results'].append({
                        'image_file': str(image_file),
                        'output_dir': str(image_output_dir),
                        'question_count': len(result.get('detections', [])),
                        'result': result
                    })
                    batch_results['processed_images'] += 1
                else:
                    batch_results['failed_images'] += 1
                    batch_results['errors'].append({
                        'image_file': str(image_file),
                        'error': '处理失败'
                    })
                
            except Exception as e:
                self.logger.error(f"处理图像 {image_file.name} 时出现错误: {e}")
                batch_results['failed_images'] += 1
                batch_results['errors'].append({
                    'image_file': str(image_file),
                    'error': str(e)
                })
        
        # 保存批量处理结果
        batch_results_path = Path(output_dir) / "batch_results.json"
        with open(batch_results_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"批量处理完成: {batch_results['processed_images']}/{batch_results['total_images']} 成功")
        
        return batch_results
    
    def generate_report(self, results_data: Dict, output_path: str = None):
        """
        生成分割报告
        
        Args:
            results_data: 分割结果数据
            output_path: 报告输出路径
        """
        if output_path is None:
            output_path = "segmentation_report.txt"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("试卷题目分割报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"置信度阈值: {self.conf_threshold}\n")
                f.write(f"IoU阈值: {self.iou_threshold}\n")
                f.write("\n")
                
                if 'image_info' in results_data:
                    image_info = results_data['image_info']
                    f.write("图像信息:\n")
                    f.write(f"  文件名: {image_info['filename']}\n")
                    f.write(f"  尺寸: {image_info['width']} x {image_info['height']}\n")
                    f.write(f"  通道数: {image_info['channels']}\n")
                    f.write("\n")
                
                if 'detections' in results_data:
                    detections = results_data['detections']
                    f.write(f"检测结果: 共检测到 {len(detections)} 个题目\n")
                    f.write("\n")
                    
                    for i, detection in enumerate(detections):
                        f.write(f"题目 {i+1}:\n")
                        f.write(f"  边界框: {detection['bbox']}\n")
                        f.write(f"  置信度: {detection['confidence']:.4f}\n")
                        f.write(f"  面积: {detection['area']:.0f} 像素\n")
                        f.write("\n")
                
                if 'question_files' in results_data:
                    question_files = results_data['question_files']
                    f.write("输出文件:\n")
                    for qf in question_files:
                        f.write(f"  {qf['filename']}\n")
            
            self.logger.info(f"报告已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"生成报告时出现错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='试卷图像题目分割工具')
    parser.add_argument('--model', type=str, required=True,
                       help='训练好的YOLO模型路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像文件或目录路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值 (默认: 0.45)')
    parser.add_argument('--min-area', type=int, default=1000,
                       help='最小题目面积阈值 (默认: 1000)')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    parser.add_argument('--no-individual', action='store_true',
                       help='不保存单独的题目图像')
    parser.add_argument('--no-annotated', action='store_true',
                       help='不保存标注后的原图')
    
    args = parser.parse_args()
    
    # 创建分割器
    segmenter = QuestionSegmenter(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    try:
        if args.batch or os.path.isdir(args.input):
            # 批量处理模式
            print(f"批量处理目录: {args.input}")
            results = segmenter.batch_process(args.input, args.output)
            print(f"批量处理完成: {results['processed_images']}/{results['total_images']} 成功")
        else:
            # 单文件处理模式
            print(f"处理图像: {args.input}")
            results = segmenter.segment_questions(
                args.input,
                args.output,
                save_individual=not args.no_individual,
                save_annotated=not args.no_annotated,
                min_area=args.min_area
            )
            
            if results:
                question_count = len(results.get('detections', []))
                print(f"分割完成，检测到 {question_count} 个题目")
                
                # 生成报告
                report_path = Path(args.output) / "segmentation_report.txt"
                segmenter.generate_report(results, str(report_path))
            else:
                print("分割失败")
                sys.exit(1)
    
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()