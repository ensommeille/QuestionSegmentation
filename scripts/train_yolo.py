#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8m 试卷图像分割模型训练脚本

功能:
- 使用YOLOv8m模型进行试卷题目检测训练
- 支持自定义训练参数
- 提供训练过程监控和可视化
- 自动保存最佳模型
- 支持断点续训

作者: AI Assistant
日期: 2024
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging

try:
    from ultralytics import YOLO
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请运行: pip install ultralytics torch matplotlib numpy")
    sys.exit(1)


class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or "configs/train_config.yaml"
        self.config = self.load_config()
        self.setup_logging()
        self.model = None
        self.results = None
        
    def load_config(self):
        """加载训练配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"配置文件未找到: {self.config_path}")
            return self.get_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"配置文件格式错误: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """获取默认配置"""
        return {
            'model': {
                'name': 'yolov8m.pt',
                'pretrained': True
            },
            'data': {
                'yaml_path': 'configs/dataset.yaml'
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'imgsz': 640,
                'device': 'auto',
                'workers': 8,
                'patience': 20,
                'save_period': 10
            },
            'optimizer': {
                'lr0': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0
            },
            'validation': {
                'val': True,
                'split': 'val',
                'save_json': True,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300
            },
            'output': {
                'project': 'outputs',
                'name': 'yolov8m_question_detection',
                'exist_ok': True,
                'save': True,
                'save_txt': True,
                'save_conf': True
            }
        }
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_environment(self):
        """检查训练环境"""
        self.logger.info("检查训练环境...")
        
        # 检查CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"检测到 {gpu_count} 个GPU: {gpu_name}")
            self.logger.info(f"CUDA版本: {torch.version.cuda}")
        else:
            self.logger.warning("未检测到CUDA，将使用CPU训练（速度较慢）")
        
        # 检查数据集
        data_yaml = self.config.get('data', 'configs/dataset.yaml')
        if not os.path.exists(data_yaml):
            self.logger.error(f"数据集配置文件未找到: {data_yaml}")
            return False
        
        # 检查输出目录
        output_dir = Path(self.config.get('project', 'outputs'))
        output_dir.mkdir(exist_ok=True)
        
        return True
    
    def load_model(self):
        """加载YOLO模型"""
        model_name = self.config.get('model', 'yolov8m.pt')
        self.logger.info(f"加载模型: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            self.logger.info(f"模型加载成功: {model_name}")
            
            # 打印模型信息
            if hasattr(self.model.model, 'info'):
                self.model.model.info()
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def train(self, resume=None):
        """开始训练"""
        if not self.check_environment():
            return False
        
        if self.model is None:
            self.load_model()
        
        self.logger.info("开始训练...")
        
        # 准备训练参数
        train_args = {
            'data': self.config.get('data', 'configs/dataset.yaml'),
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.config.get('device', 0),
            'workers': self.config.get('workers', 8),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', 10),
            'project': self.config.get('project', 'outputs'),
            'name': self.config.get('name', 'yolov8m_exam'),
            'exist_ok': self.config.get('exist_ok', True),
            'optimizer': self.config.get('optimizer', 'AdamW'),
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': resume,
            'amp': True,  # 自动混合精度
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            # 优化器参数
            'lr0': self.config.get('lr0', 0.001),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            # 数据增强参数
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 5.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
            # 验证参数
            'val': self.config.get('val', True),
            # 保存参数
            'save': self.config.get('save', True)
        }
        
        try:
            # 开始训练
            self.results = self.model.train(**train_args)
            self.logger.info("训练完成！")
            
            # 保存训练结果
            self.save_training_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {e}")
            return False
    
    def validate(self, model_path=None):
        """验证模型"""
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            self.logger.error("请先训练模型或指定模型路径")
            return None
        
        self.logger.info("开始验证模型...")
        
        val_args = {
            'data': self.config.get('data', 'configs/dataset.yaml'),
            'imgsz': self.config.get('imgsz', 640),
            'batch': self.config.get('batch', 16),
            'conf': 0.25,  # 默认置信度阈值
            'iou': 0.45,   # 默认IoU阈值
            'max_det': 300, # 默认最大检测数
            'device': self.config.get('device', 0),
            'workers': self.config.get('workers', 8),
            'save_json': True,
            'save_hybrid': False,
            'verbose': True,
            'split': 'val',
            'dnn': False,
            'plots': True
        }
        
        try:
            results = self.model.val(**val_args)
            self.logger.info("验证完成！")
            
            # 打印验证结果
            if hasattr(results, 'box'):
                box_results = results.box
                self.logger.info(f"mAP50: {box_results.map50:.4f}")
                self.logger.info(f"mAP50-95: {box_results.map:.4f}")
                self.logger.info(f"Precision: {box_results.mp:.4f}")
                self.logger.info(f"Recall: {box_results.mr:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"验证过程中出现错误: {e}")
            return None
    
    def save_training_summary(self):
        """保存训练总结"""
        if self.results is None:
            return
        
        summary_dir = Path(self.config.get('project', 'outputs')) / self.config.get('name', 'yolov8m_exam')
        summary_file = summary_dir / "training_summary.txt"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("YOLOv8m 试卷题目检测训练总结\n")
                f.write("=" * 50 + "\n")
                f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"配置文件: {self.config_path}\n")
                f.write(f"数据集: {self.config.get('data', 'configs/dataset.yaml')}\n")
                f.write(f"训练轮数: {self.config.get('epochs', 100)}\n")
                f.write(f"批次大小: {self.config.get('batch', 16)}\n")
                f.write(f"图像尺寸: {self.config.get('imgsz', 640)}\n")
                f.write(f"设备: {self.config.get('device', 0)}\n")
                f.write("\n")
                
                # 如果有训练结果，添加性能指标
                if hasattr(self.results, 'results_dict'):
                    results_dict = self.results.results_dict
                    f.write("训练结果:\n")
                    for key, value in results_dict.items():
                        f.write(f"{key}: {value}\n")
            
            self.logger.info(f"训练总结已保存到: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"保存训练总结失败: {e}")
    
    def plot_results(self):
        """绘制训练结果"""
        if self.results is None:
            self.logger.warning("没有训练结果可绘制")
            return
        
        try:
            # 结果图片通常保存在训练输出目录中
            results_dir = Path(self.config.get('project', 'outputs')) / self.config.get('name', 'yolov8m_exam')
            
            # 检查是否有结果图片
            result_images = list(results_dir.glob("*.png"))
            if result_images:
                self.logger.info(f"训练结果图片已保存到: {results_dir}")
                for img in result_images:
                    self.logger.info(f"  - {img.name}")
            else:
                self.logger.warning("未找到训练结果图片")
                
        except Exception as e:
            self.logger.error(f"绘制结果时出现错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8m 试卷图像分割模型训练')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅进行验证，不训练')
    parser.add_argument('--model', type=str, default=None,
                       help='用于验证的模型路径')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLOTrainer(config_path=args.config)
    
    if args.validate_only:
        # 仅验证模式
        results = trainer.validate(model_path=args.model)
        if results:
            print("验证完成！")
        else:
            print("验证失败！")
    else:
        # 训练模式
        success = trainer.train(resume=args.resume)
        if success:
            print("训练完成！")
            # 训练完成后进行验证
            trainer.validate()
            # 绘制结果
            trainer.plot_results()
        else:
            print("训练失败！")
            sys.exit(1)


if __name__ == "__main__":
    main()