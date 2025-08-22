#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
用于验证试卷图像分割系统的各个组件是否正常工作
"""

import os
import sys
from pathlib import Path

def test_project_structure():
    """测试项目结构"""
    print("=== 测试项目结构 ===")
    
    required_dirs = [
        'scripts',
        'configs', 
        'datasets',
        'outputs',
        'datasets/yolo_format',
        'datasets/train/images',
        'datasets/train/labels',
        'datasets/val/images', 
        'datasets/val/labels',
        'datasets/test/images',
        'datasets/test/labels'
    ]
    
    required_files = [
        'scripts/coco_to_yolo.py',
        'scripts/prepare_dataset.py',
        'scripts/train_yolo.py',
        'scripts/segment_questions.py',
        'configs/dataset.yaml',
        'configs/train_config.yaml',
        'requirements.txt',
        'README.md',
        'quick_start.py'
    ]
    
    print("检查目录结构...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"[OK] {dir_path}")
        else:
            print(f"[MISSING] {dir_path}")
    
    print("\n检查文件...")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")

def test_data_conversion():
    """测试数据转换"""
    print("\n=== 测试数据转换 ===")
    
    # 检查原始数据
    if os.path.exists('data/Annotations/coco_info.json'):
        print("[OK] 找到COCO标注文件")
    else:
        print("[MISSING] COCO标注文件")
        return False
    
    if os.path.exists('data/Images'):
        image_count = len(list(Path('data/Images').glob('*.jpg')))
        print(f"[OK] 找到 {image_count} 张图片")
    else:
        print("[MISSING] 图片目录")
        return False
    
    # 检查转换后的数据
    yolo_dirs = ['datasets/train', 'datasets/val', 'datasets/test']
    for split_dir in yolo_dirs:
        images_dir = Path(split_dir) / 'images'
        labels_dir = Path(split_dir) / 'labels'
        
        if images_dir.exists() and labels_dir.exists():
            img_count = len(list(images_dir.glob('*.jpg')))
            label_count = len(list(labels_dir.glob('*.txt')))
            print(f"[OK] {split_dir}: {img_count} 图片, {label_count} 标注")
        else:
            print(f"[MISSING] {split_dir} 数据")
    
    return True

def test_dependencies():
    """测试依赖包"""
    print("\n=== 测试依赖包 ===")
    
    required_packages = [
        'ultralytics',
        'torch', 
        'cv2',
        'PIL',
        'numpy',
        'matplotlib',
        'yaml',
        'tqdm'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")

def test_config_files():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    
    # 测试dataset.yaml
    try:
        import yaml
        with open('configs/dataset.yaml', 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        print("[OK] dataset.yaml 格式正确")
        print(f"     类别数: {dataset_config.get('nc', 'N/A')}")
        print(f"     类别名: {dataset_config.get('names', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] dataset.yaml: {e}")
    
    # 测试train_config.yaml
    try:
        with open('configs/train_config.yaml', 'r', encoding='utf-8') as f:
            train_config = yaml.safe_load(f)
        print("[OK] train_config.yaml 格式正确")
        print(f"     模型: {train_config.get('model', 'N/A')}")
        print(f"     训练轮数: {train_config.get('epochs', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] train_config.yaml: {e}")

def main():
    """主函数"""
    print("试卷图像分割系统 - 系统测试")
    print("=" * 50)
    
    # 切换到项目根目录
    os.chdir(Path(__file__).parent)
    
    # 运行测试
    test_project_structure()
    test_dependencies()
    test_config_files()
    test_data_conversion()
    
    print("\n=== 测试完成 ===")
    print("如果所有项目都显示 [OK]，说明系统准备就绪")
    print("如果有 [MISSING] 或 [ERROR] 项目，请根据提示进行修复")

if __name__ == '__main__':
    main()