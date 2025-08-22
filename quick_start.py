#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试卷图像分割系统快速开始脚本

功能:
- 一键完成环境检查、数据准备、模型训练和图像分割
- 提供交互式配置选项
- 自动处理常见错误和异常情况
- 生成完整的运行报告

作者: AI Assistant
日期: 2024
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


class QuickStart:
    """快速开始助手"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.log_messages = []
        
    def log(self, message, level="INFO"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {level}: {message}"
        print(log_msg)
        self.log_messages.append(log_msg)
    
    def check_environment(self):
        """检查环境"""
        self.log("检查Python环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.log(f"Python版本过低: {python_version.major}.{python_version.minor}, 需要3.8+", "ERROR")
            return False
        
        self.log(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查必要的目录
        required_dirs = ['configs', 'scripts', 'datasets', 'data']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.log(f"缺少必要目录: {dir_name}", "ERROR")
                return False
        
        # 检查数据文件
        data_dir = self.project_root / 'data'
        images_dir = data_dir / 'Images'
        annotations_dir = data_dir / 'Annotations'
        
        if not images_dir.exists():
            self.log("未找到图片目录: data/Images", "WARNING")
        else:
            image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
            self.log(f"找到 {image_count} 张图片")
        
        if not annotations_dir.exists():
            self.log("未找到标注目录: data/Annotations", "WARNING")
        else:
            coco_file = annotations_dir / 'coco_info.json'
            if coco_file.exists():
                self.log("找到COCO标注文件")
            else:
                self.log("未找到COCO标注文件: coco_info.json", "WARNING")
        
        return True
    
    def install_dependencies(self, force=False):
        """安装依赖包"""
        self.log("检查依赖包...")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.log("未找到requirements.txt文件", "ERROR")
            return False
        
        # 检查是否需要安装
        try:
            import ultralytics
            import torch
            import cv2
            import matplotlib
            if not force:
                self.log("主要依赖包已安装")
                return True
        except ImportError:
            pass
        
        self.log("安装依赖包...")
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.log("依赖包安装成功")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"依赖包安装失败: {e}", "ERROR")
            return False
    
    def prepare_data(self):
        """准备数据"""
        self.log("准备数据集...")
        
        # 检查是否已经转换过
        yolo_dir = self.project_root / 'datasets' / 'yolo_format'
        if yolo_dir.exists() and any(yolo_dir.iterdir()):
            self.log("YOLO格式数据已存在，跳过转换")
            return True
        
        # 运行数据转换脚本
        coco_script = self.project_root / 'scripts' / 'coco_to_yolo.py'
        if not coco_script.exists():
            self.log("未找到数据转换脚本", "ERROR")
            return False
        
        try:
            cmd = [sys.executable, str(coco_script)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(self.project_root))
            self.log("数据转换完成")
            
            # 验证数据集
            prepare_script = self.project_root / 'scripts' / 'prepare_dataset.py'
            if prepare_script.exists():
                cmd = [sys.executable, str(prepare_script), '--validate']
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
                if result.returncode == 0:
                    self.log("数据集验证完成")
                else:
                    self.log(f"数据集验证警告: {result.stderr}", "WARNING")
            
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"数据准备失败: {e.stderr if hasattr(e, 'stderr') else str(e)}", "ERROR")
            return False
    
    def train_model(self, epochs=None, batch_size=None):
        """训练模型"""
        self.log("开始模型训练...")
        
        # 检查是否已有训练好的模型
        outputs_dir = self.project_root / 'outputs'
        best_model = None
        for model_dir in outputs_dir.glob('*/weights/best.pt'):
            if model_dir.exists():
                best_model = model_dir
                break
        
        if best_model:
            response = input(f"发现已训练的模型: {best_model}\n是否重新训练? (y/N): ")
            if response.lower() != 'y':
                self.log("使用现有模型，跳过训练")
                return str(best_model)
        
        # 运行训练脚本
        train_script = self.project_root / 'scripts' / 'train_yolo.py'
        if not train_script.exists():
            self.log("未找到训练脚本", "ERROR")
            return None
        
        try:
            # 修改配置文件中的训练参数
            config_file = self.project_root / 'configs' / 'train_config.yaml'
            if epochs or batch_size:
                import yaml
                
                # 读取现有配置
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 更新参数
                if epochs:
                    config['epochs'] = epochs
                    self.log(f"设置训练轮数: {epochs}")
                if batch_size:
                    config['batch'] = batch_size
                    self.log(f"设置批次大小: {batch_size}")
                
                # 写回配置文件
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            cmd = [sys.executable, str(train_script)]
            self.log("训练开始，这可能需要较长时间...")
            result = subprocess.run(cmd, cwd=str(self.project_root))
            
            if result.returncode == 0:
                self.log("模型训练完成")
                # 查找最新的best.pt文件
                for model_dir in outputs_dir.glob('*/weights/best.pt'):
                    if model_dir.exists():
                        return str(model_dir)
            else:
                self.log("模型训练失败", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"训练过程出错: {e}", "ERROR")
            return None
    
    def test_segmentation(self, model_path):
        """测试图像分割"""
        self.log("测试图像分割功能...")
        
        # 查找测试图片
        test_images = []
        data_images = self.project_root / 'data' / 'Images'
        if data_images.exists():
            test_images = list(data_images.glob('*.jpg'))[:3]  # 取前3张图片测试
        
        if not test_images:
            self.log("未找到测试图片", "WARNING")
            return False
        
        # 运行分割脚本
        segment_script = self.project_root / 'scripts' / 'segment_questions.py'
        if not segment_script.exists():
            self.log("未找到分割脚本", "ERROR")
            return False
        
        test_output = self.project_root / 'test_results'
        test_output.mkdir(exist_ok=True)
        
        try:
            for i, test_image in enumerate(test_images):
                self.log(f"测试图片 {i+1}/{len(test_images)}: {test_image.name}")
                
                cmd = [
                    sys.executable, str(segment_script),
                    '--model', model_path,
                    '--input', str(test_image),
                    '--output', str(test_output / f'test_{i+1}'),
                    '--conf', '0.5'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
                
                if result.returncode == 0:
                    self.log(f"图片 {test_image.name} 分割成功")
                else:
                    self.log(f"图片 {test_image.name} 分割失败: {result.stderr}", "WARNING")
            
            self.log(f"测试结果保存在: {test_output}")
            return True
            
        except Exception as e:
            self.log(f"测试分割功能时出错: {e}", "ERROR")
            return False
    
    def generate_report(self):
        """生成运行报告"""
        report_file = self.project_root / 'quick_start_report.txt'
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("试卷图像分割系统快速开始报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"项目路径: {self.project_root}\n")
                f.write("\n")
                
                f.write("运行日志:\n")
                f.write("-" * 30 + "\n")
                for log_msg in self.log_messages:
                    f.write(log_msg + "\n")
            
            self.log(f"运行报告已保存: {report_file}")
            
        except Exception as e:
            self.log(f"生成报告时出错: {e}", "ERROR")
    
    def run_interactive(self):
        """交互式运行"""
        print("=" * 60)
        print("欢迎使用试卷图像分割系统快速开始工具")
        print("=" * 60)
        
        # 环境检查
        if not self.check_environment():
            print("\n环境检查失败，请检查项目结构和数据文件")
            return False
        
        # 依赖安装
        install_deps = input("\n是否安装/更新依赖包? (Y/n): ")
        if install_deps.lower() != 'n':
            if not self.install_dependencies():
                print("\n依赖包安装失败，请手动安装")
                return False
        
        # 数据准备
        prepare_data = input("\n是否准备数据集? (Y/n): ")
        if prepare_data.lower() != 'n':
            if not self.prepare_data():
                print("\n数据准备失败，请检查数据文件")
                return False
        
        # 模型训练
        train_model = input("\n是否训练模型? (Y/n): ")
        model_path = None
        if train_model.lower() != 'n':
            # 询问训练参数
            epochs_input = input("训练轮数 (默认100): ")
            epochs = int(epochs_input) if epochs_input.strip() else None
            
            batch_input = input("批次大小 (默认16): ")
            batch_size = int(batch_input) if batch_input.strip() else None
            
            model_path = self.train_model(epochs, batch_size)
            if not model_path:
                print("\n模型训练失败")
                return False
        else:
            # 查找现有模型
            outputs_dir = self.project_root / 'outputs'
            for model_dir in outputs_dir.glob('*/weights/best.pt'):
                if model_dir.exists():
                    model_path = str(model_dir)
                    break
            
            if not model_path:
                print("\n未找到训练好的模型，请先训练模型")
                return False
        
        # 测试分割
        test_seg = input("\n是否测试图像分割功能? (Y/n): ")
        if test_seg.lower() != 'n':
            self.test_segmentation(model_path)
        
        # 生成报告
        self.generate_report()
        
        print("\n=" * 60)
        print("快速开始完成！")
        print(f"模型路径: {model_path}")
        print("使用以下命令进行图像分割:")
        print(f"python scripts/segment_questions.py --model {model_path} --input <图片路径> --output <输出目录>")
        print("=" * 60)
        
        return True
    
    def run_auto(self, skip_training=False):
        """自动运行"""
        print("自动运行模式")
        
        if not self.check_environment():
            return False
        
        if not self.install_dependencies():
            return False
        
        if not self.prepare_data():
            return False
        
        if not skip_training:
            model_path = self.train_model()
            if not model_path:
                return False
        else:
            # 查找现有模型
            outputs_dir = self.project_root / 'outputs'
            model_path = None
            for model_dir in outputs_dir.glob('*/weights/best.pt'):
                if model_dir.exists():
                    model_path = str(model_dir)
                    break
            
            if not model_path:
                self.log("未找到训练好的模型", "ERROR")
                return False
        
        self.test_segmentation(model_path)
        self.generate_report()
        
        print(f"\n自动运行完成！模型路径: {model_path}")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='试卷图像分割系统快速开始工具')
    parser.add_argument('--auto', action='store_true', help='自动运行模式')
    parser.add_argument('--skip-training', action='store_true', help='跳过模型训练')
    
    args = parser.parse_args()
    
    quick_start = QuickStart()
    
    try:
        if args.auto:
            success = quick_start.run_auto(skip_training=args.skip_training)
        else:
            success = quick_start.run_interactive()
        
        if success:
            print("\n✅ 快速开始成功完成！")
        else:
            print("\n❌ 快速开始过程中出现错误")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n未预期的错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()