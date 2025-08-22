#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试卷题目检测 - COCO到YOLO格式转换脚本
Exam Paper Question Detection - COCO to YOLO Format Converter

功能：
1. 读取COCO格式的标注文件
2. 转换为YOLO格式的标注文件
3. 划分训练集、验证集和测试集
4. 复制对应的图片文件
"""

import json
import os
import shutil
from pathlib import Path
import random
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm


class COCOToYOLOConverter:
    """COCO到YOLO格式转换器"""
    
    def __init__(self, coco_json_path: str, images_dir: str, output_dir: str):
        """
        初始化转换器
        
        Args:
            coco_json_path: COCO标注文件路径
            images_dir: 图片目录路径
            output_dir: 输出目录路径
        """
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # 加载COCO数据
        with open(self.coco_json_path, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"加载COCO数据完成:")
        print(f"  图片数量: {len(self.coco_data['images'])}")
        print(f"  标注数量: {len(self.coco_data['annotations'])}")
        print(f"  类别数量: {len(self.coco_data['categories'])}")
    
    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        将COCO格式的边界框转换为YOLO格式
        
        Args:
            bbox: COCO格式边界框 [x, y, width, height]
            img_width: 图片宽度
            img_height: 图片高度
        
        Returns:
            YOLO格式边界框 (x_center, y_center, width, height) - 归一化坐标
        """
        x, y, w, h = bbox
        
        # 转换为中心点坐标
        x_center = x + w / 2
        y_center = y + h / 2
        
        # 归一化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        return x_center_norm, y_center_norm, w_norm, h_norm
    
    def create_image_annotation_mapping(self) -> Dict[int, List[Dict]]:
        """
        创建图片ID到标注的映射
        
        Returns:
            图片ID到标注列表的映射字典
        """
        image_annotations = {}
        
        for annotation in self.coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)
        
        return image_annotations
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05) -> Tuple[List, List, List]:
        """
        划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        
        Returns:
            (训练集图片列表, 验证集图片列表, 测试集图片列表)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        images = self.coco_data['images'].copy()
        random.shuffle(images)
        
        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        print(f"数据集划分完成:")
        print(f"  训练集: {len(train_images)} 张图片")
        print(f"  验证集: {len(val_images)} 张图片")
        print(f"  测试集: {len(test_images)} 张图片")
        
        return train_images, val_images, test_images
    
    def convert_and_save(self, images: List[Dict], split_name: str, image_annotations: Dict[int, List[Dict]]):
        """
        转换并保存指定数据集分割
        
        Args:
            images: 图片信息列表
            split_name: 数据集分割名称 ('train', 'val', 'test')
            image_annotations: 图片ID到标注的映射
        """
        split_dir = self.output_dir / split_name
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n正在处理 {split_name} 数据集...")
        
        for image_info in tqdm(images, desc=f"转换{split_name}数据"):
            image_id = image_info['id']
            image_filename = image_info['file_name']
            img_width = image_info['width']
            img_height = image_info['height']
            
            # 复制图片文件
            src_image_path = self.images_dir / image_filename
            dst_image_path = images_dir / image_filename
            
            if src_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"警告: 图片文件不存在 {src_image_path}")
                continue
            
            # 创建YOLO标注文件
            label_filename = Path(image_filename).stem + '.txt'
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w', encoding='utf-8') as f:
                if image_id in image_annotations:
                    for annotation in image_annotations[image_id]:
                        bbox = annotation['bbox']
                        category_id = annotation['category_id']
                        
                        # 转换为YOLO格式
                        x_center, y_center, width, height = self.convert_bbox_to_yolo(
                            bbox, img_width, img_height
                        )
                        
                        # 写入标注文件 (类别ID从0开始)
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def convert(self, train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05):
        """
        执行完整的转换流程
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        print("开始COCO到YOLO格式转换...")
        
        # 创建图片到标注的映射
        image_annotations = self.create_image_annotation_mapping()
        
        # 划分数据集
        train_images, val_images, test_images = self.split_dataset(train_ratio, val_ratio, test_ratio)
        
        # 转换并保存各个数据集分割
        self.convert_and_save(train_images, 'train', image_annotations)
        self.convert_and_save(val_images, 'val', image_annotations)
        self.convert_and_save(test_images, 'test', image_annotations)
        
        print("\n转换完成！")
        print(f"输出目录: {self.output_dir}")
        
        # 统计转换结果
        self.print_conversion_stats()
    
    def print_conversion_stats(self):
        """
        打印转换统计信息
        """
        print("\n=== 转换统计 ===")
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg')))
                label_count = len(list(labels_dir.glob('*.txt')))
                
                # 统计标注数量
                total_annotations = 0
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        total_annotations += len(f.readlines())
                
                print(f"{split.upper()}:")
                print(f"  图片数量: {image_count}")
                print(f"  标注文件数量: {label_count}")
                print(f"  总标注数量: {total_annotations}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='COCO到YOLO格式转换工具')
    parser.add_argument('--coco_json', type=str, default='data/Annotations/coco_info.json',
                       help='COCO标注文件路径')
    parser.add_argument('--images_dir', type=str, default='data/Images',
                       help='图片目录路径')
    parser.add_argument('--output_dir', type=str, default='datasets',
                       help='输出目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建转换器并执行转换
    converter = COCOToYOLOConverter(
        coco_json_path=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )
    
    converter.convert(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == '__main__':
    main()