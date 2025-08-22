#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试卷题目检测 - 数据集准备脚本
Exam Paper Question Detection - Dataset Preparation Script

功能：
1. 执行COCO到YOLO格式转换
2. 验证数据集完整性
3. 生成数据集统计报告
4. 创建类别映射文件
"""

import os
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from PIL import Image
import argparse

# 添加脚本目录到Python路径
sys.path.append(str(Path(__file__).parent))
from coco_to_yolo import COCOToYOLOConverter


class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, dataset_dir: str):
        """
        初始化验证器
        
        Args:
            dataset_dir: 数据集目录路径
        """
        self.dataset_dir = Path(dataset_dir)
        self.splits = ['train', 'val', 'test']
    
    def validate_dataset_structure(self) -> bool:
        """
        验证数据集目录结构
        
        Returns:
            验证是否通过
        """
        print("验证数据集目录结构...")
        
        required_dirs = []
        for split in self.splits:
            required_dirs.extend([
                self.dataset_dir / split / 'images',
                self.dataset_dir / split / 'labels'
            ])
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            print(f"错误: 缺少以下目录:")
            for missing_dir in missing_dirs:
                print(f"  - {missing_dir}")
            return False
        
        print("[OK] 数据集目录结构验证通过")
        return True
    
    def validate_image_label_pairs(self) -> Dict[str, Dict]:
        """
        验证图片和标注文件的对应关系
        
        Returns:
            验证结果统计
        """
        print("验证图片和标注文件对应关系...")
        
        results = {}
        
        for split in self.splits:
            images_dir = self.dataset_dir / split / 'images'
            labels_dir = self.dataset_dir / split / 'labels'
            
            # 获取所有图片和标注文件
            image_files = set(f.stem for f in images_dir.glob('*.jpg'))
            label_files = set(f.stem for f in labels_dir.glob('*.txt'))
            
            # 统计
            matched = len(image_files & label_files)
            missing_labels = image_files - label_files
            missing_images = label_files - image_files
            
            results[split] = {
                'total_images': len(image_files),
                'total_labels': len(label_files),
                'matched_pairs': matched,
                'missing_labels': list(missing_labels),
                'missing_images': list(missing_images)
            }
            
            print(f"{split.upper()}:")
            print(f"  图片数量: {len(image_files)}")
            print(f"  标注数量: {len(label_files)}")
            print(f"  匹配对数: {matched}")
            
            if missing_labels:
                print(f"  缺少标注的图片: {len(missing_labels)}")
            if missing_images:
                print(f"  缺少图片的标注: {len(missing_images)}")
        
        return results
    
    def validate_annotation_format(self) -> Dict[str, List]:
        """
        验证标注文件格式
        
        Returns:
            格式错误列表
        """
        print("验证标注文件格式...")
        
        format_errors = defaultdict(list)
        
        for split in self.splits:
            labels_dir = self.dataset_dir / split / 'labels'
            
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            format_errors[split].append(
                                f"{label_file.name}:{line_num} - 错误的字段数量: {len(parts)}"
                            )
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # 验证坐标范围
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                format_errors[split].append(
                                    f"{label_file.name}:{line_num} - 坐标超出范围 [0,1]"
                                )
                            
                            # 验证类别ID
                            if class_id < 0:
                                format_errors[split].append(
                                    f"{label_file.name}:{line_num} - 无效的类别ID: {class_id}"
                                )
                        
                        except ValueError as e:
                            format_errors[split].append(
                                f"{label_file.name}:{line_num} - 数值转换错误: {e}"
                            )
                
                except Exception as e:
                    format_errors[split].append(
                        f"{label_file.name} - 文件读取错误: {e}"
                    )
        
        # 打印验证结果
        total_errors = sum(len(errors) for errors in format_errors.values())
        if total_errors == 0:
            print("[OK] 所有标注文件格式验证通过")
        else:
            print(f"发现 {total_errors} 个格式错误:")
            for split, errors in format_errors.items():
                if errors:
                    print(f"  {split.upper()}: {len(errors)} 个错误")
                    for error in errors[:5]:  # 只显示前5个错误
                        print(f"    - {error}")
                    if len(errors) > 5:
                        print(f"    ... 还有 {len(errors) - 5} 个错误")
        
        return dict(format_errors)


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, dataset_dir: str):
        """
        初始化分析器
        
        Args:
            dataset_dir: 数据集目录路径
        """
        self.dataset_dir = Path(dataset_dir)
        self.splits = ['train', 'val', 'test']
    
    def analyze_dataset(self) -> Dict:
        """
        分析数据集统计信息
        
        Returns:
            分析结果字典
        """
        print("分析数据集统计信息...")
        
        analysis = {
            'image_stats': {},
            'annotation_stats': {},
            'class_distribution': {},
            'bbox_stats': {}
        }
        
        for split in self.splits:
            print(f"\n分析 {split.upper()} 数据集...")
            
            images_dir = self.dataset_dir / split / 'images'
            labels_dir = self.dataset_dir / split / 'labels'
            
            # 图片统计
            image_sizes = []
            for img_file in images_dir.glob('*.jpg'):
                try:
                    with Image.open(img_file) as img:
                        image_sizes.append(img.size)  # (width, height)
                except Exception as e:
                    print(f"警告: 无法读取图片 {img_file}: {e}")
            
            if image_sizes:
                widths, heights = zip(*image_sizes)
                analysis['image_stats'][split] = {
                    'count': len(image_sizes),
                    'width_mean': np.mean(widths),
                    'width_std': np.std(widths),
                    'height_mean': np.mean(heights),
                    'height_std': np.std(heights),
                    'min_size': (min(widths), min(heights)),
                    'max_size': (max(widths), max(heights))
                }
            
            # 标注统计
            class_counts = Counter()
            bbox_areas = []
            bbox_aspect_ratios = []
            total_annotations = 0
            
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            class_counts[class_id] += 1
                            bbox_areas.append(width * height)
                            bbox_aspect_ratios.append(width / height if height > 0 else 0)
                            total_annotations += 1
                
                except Exception as e:
                    print(f"警告: 无法读取标注文件 {label_file}: {e}")
            
            analysis['annotation_stats'][split] = {
                'total_annotations': total_annotations,
                'avg_annotations_per_image': total_annotations / len(image_sizes) if image_sizes else 0
            }
            
            analysis['class_distribution'][split] = dict(class_counts)
            
            if bbox_areas:
                analysis['bbox_stats'][split] = {
                    'area_mean': np.mean(bbox_areas),
                    'area_std': np.std(bbox_areas),
                    'area_min': min(bbox_areas),
                    'area_max': max(bbox_areas),
                    'aspect_ratio_mean': np.mean(bbox_aspect_ratios),
                    'aspect_ratio_std': np.std(bbox_aspect_ratios)
                }
        
        return analysis
    
    def save_analysis_report(self, analysis: Dict, output_path: str):
        """
        保存分析报告
        
        Args:
            analysis: 分析结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"分析报告已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集准备工具')
    parser.add_argument('--convert', action='store_true',
                       help='执行COCO到YOLO格式转换')
    parser.add_argument('--validate', action='store_true',
                       help='验证数据集')
    parser.add_argument('--analyze', action='store_true',
                       help='分析数据集')
    parser.add_argument('--all', action='store_true',
                       help='执行所有操作')
    parser.add_argument('--coco_json', type=str, default='data/Annotations/coco_info.json',
                       help='COCO标注文件路径')
    parser.add_argument('--images_dir', type=str, default='data/Images',
                       help='图片目录路径')
    parser.add_argument('--dataset_dir', type=str, default='datasets',
                       help='数据集目录路径')
    
    args = parser.parse_args()
    
    if args.all:
        args.convert = args.validate = args.analyze = True
    
    if not any([args.convert, args.validate, args.analyze]):
        print("请指定要执行的操作: --convert, --validate, --analyze 或 --all")
        return
    
    # 执行COCO到YOLO转换
    if args.convert:
        print("=== 执行COCO到YOLO格式转换 ===")
        converter = COCOToYOLOConverter(
            coco_json_path=args.coco_json,
            images_dir=args.images_dir,
            output_dir=args.dataset_dir
        )
        converter.convert()
    
    # 验证数据集
    if args.validate:
        print("\n=== 验证数据集 ===")
        validator = DatasetValidator(args.dataset_dir)
        
        # 验证目录结构
        if not validator.validate_dataset_structure():
            print("数据集目录结构验证失败，请检查目录结构")
            return
        
        # 验证图片和标注对应关系
        pair_results = validator.validate_image_label_pairs()
        
        # 验证标注格式
        format_errors = validator.validate_annotation_format()
        
        # 保存验证结果
        validation_results = {
            'pair_validation': pair_results,
            'format_errors': format_errors
        }
        
        validation_output = Path(args.dataset_dir) / 'validation_results.json'
        with open(validation_output, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        print(f"验证结果已保存到: {validation_output}")
    
    # 分析数据集
    if args.analyze:
        print("\n=== 分析数据集 ===")
        analyzer = DatasetAnalyzer(args.dataset_dir)
        analysis = analyzer.analyze_dataset()
        
        # 保存分析报告
        analysis_output = Path(args.dataset_dir) / 'dataset_analysis.json'
        analyzer.save_analysis_report(analysis, analysis_output)
        
        # 打印简要统计
        print("\n=== 数据集统计摘要 ===")
        for split in ['train', 'val', 'test']:
            if split in analysis['image_stats']:
                img_stats = analysis['image_stats'][split]
                ann_stats = analysis['annotation_stats'][split]
                print(f"{split.upper()}:")
                print(f"  图片数量: {img_stats['count']}")
                print(f"  总标注数量: {ann_stats['total_annotations']}")
                print(f"  平均每张图片标注数: {ann_stats['avg_annotations_per_image']:.2f}")
    
    print("\n数据集准备完成！")


if __name__ == '__main__':
    main()