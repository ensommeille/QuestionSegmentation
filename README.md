# 试卷图像分割系统

基于YOLOv8m的试卷题目检测与分割系统，能够自动识别试卷中的题目区域并进行精确分割。

## 项目特点

- 🎯 **高精度检测**: 基于YOLOv8m模型，检测精度高
- 🚀 **快速处理**: 支持GPU加速，处理速度快
- 📊 **完整流程**: 从数据预处理到模型训练再到图像分割的完整解决方案
- 🔧 **易于使用**: 提供简单易用的命令行接口
- 📈 **可视化结果**: 支持检测结果可视化和详细报告生成
- 🔄 **批量处理**: 支持单张图片和批量图片处理

## 项目结构

```
quesitionSeg/
├── configs/                 # 配置文件目录
│   ├── dataset.yaml         # 数据集配置
│   └── train_config.yaml    # 训练配置
├── scripts/                 # 脚本文件目录
│   ├── coco_to_yolo.py      # COCO到YOLO格式转换
│   ├── prepare_dataset.py   # 数据集准备和验证
│   ├── train_yolo.py        # 模型训练脚本
│   └── segment_questions.py # 图像分割主程序
├── datasets/                # 数据集目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   ├── test/                # 测试集
│   └── yolo_format/         # YOLO格式数据
├── models/                  # 模型文件目录
├── outputs/                 # 训练输出目录
├── logs/                    # 日志文件目录
├── data/                    # 原始数据目录
│   ├── Images/              # 原始图片
│   └── Annotations/         # COCO标注文件
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明文档
```

## 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU训练推荐)
- 内存: 8GB+ (推荐16GB+)
- 存储: 10GB+ 可用空间

## 安装步骤

### 1. 克隆项目

```bash
git clone <项目地址>
cd quesitionSeg
```

### 2. 创建虚拟环境 (推荐)

```bash
# 使用conda
conda create -n yolo_seg python=3.9
conda activate yolo_seg

# 或使用venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖包

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "from ultralytics import YOLO; print('YOLOv8 安装成功')"
```

## 使用指南

### 第一步: 数据准备

1. **准备原始数据**
   - 将试卷图片放入 `data/Images/` 目录
   - 将COCO格式标注文件放入 `data/Annotations/` 目录

2. **转换数据格式**
   ```bash
   python scripts/coco_to_yolo.py
   ```

3. **验证数据集**
   ```bash
   python scripts/prepare_dataset.py --validate
   ```

### 第二步: 模型训练

1. **检查配置文件**
   - 编辑 `configs/train_config.yaml` 调整训练参数
   - 编辑 `configs/dataset.yaml` 确认数据路径

2. **开始训练**
   ```bash
   python scripts/train_yolo.py
   ```

3. **监控训练过程**
   - 查看 `logs/` 目录中的日志文件
   - 查看 `outputs/` 目录中的训练结果

4. **从检查点恢复训练**
   ```bash
   python scripts/train_yolo.py --resume outputs/yolov8m_question_detection/weights/last.pt
   ```

### 第三步: 图像分割

1. **单张图片分割**
   ```bash
   python scripts/segment_questions.py \
     --model outputs/yolov8m_question_detection/weights/best.pt \
     --input test_image.jpg \
     --output results/
   ```

2. **批量图片分割**
   ```bash
   python scripts/segment_questions.py \
     --model outputs/yolov8m_question_detection/weights/best.pt \
     --input test_images/ \
     --output results/ \
     --batch
   ```

3. **自定义参数**
   ```bash
   python scripts/segment_questions.py \
     --model outputs/yolov8m_question_detection/weights/best.pt \
     --input test_image.jpg \
     --output results/ \
     --conf 0.6 \
     --iou 0.5 \
     --min-area 2000
   ```

## 配置说明

### 训练配置 (train_config.yaml)

```yaml
training:
  epochs: 100          # 训练轮数
  batch_size: 16       # 批次大小
  imgsz: 640          # 输入图像尺寸
  device: 'auto'      # 设备选择 ('auto', 'cpu', '0', '0,1')
  workers: 8          # 数据加载线程数
  patience: 20        # 早停耐心值

optimizer:
  lr0: 0.01           # 初始学习率
  momentum: 0.937     # 动量
  weight_decay: 0.0005 # 权重衰减

augmentation:
  hsv_h: 0.015        # 色调增强
  hsv_s: 0.7          # 饱和度增强
  hsv_v: 0.4          # 明度增强
  degrees: 0.0        # 旋转角度
  translate: 0.1      # 平移比例
  scale: 0.5          # 缩放比例
  fliplr: 0.5         # 水平翻转概率
```

### 数据集配置 (dataset.yaml)

```yaml
path: datasets/yolo_format  # 数据集根目录
train: train/images         # 训练集图片路径
val: val/images            # 验证集图片路径
test: test/images          # 测试集图片路径

nc: 1                      # 类别数量
names:
  0: question              # 类别名称
```

## 性能优化建议

### 硬件优化
- **GPU**: 使用NVIDIA GPU可显著提升训练和推理速度
- **内存**: 增加系统内存可提高数据加载速度
- **存储**: 使用SSD可加快数据读取速度

### 软件优化
- **批次大小**: 根据GPU内存调整batch_size
- **图像尺寸**: 较小的imgsz可提高速度但可能降低精度
- **工作线程**: 调整workers数量以平衡CPU和I/O负载

### 模型优化
- **模型选择**: 根据精度和速度需求选择合适的YOLO模型
- **量化**: 使用模型量化技术减少模型大小
- **剪枝**: 移除不重要的模型参数

## 常见问题

### Q1: 训练时出现CUDA内存不足错误
**A**: 减少batch_size或imgsz，或者使用梯度累积:
```yaml
training:
  batch_size: 8  # 减少批次大小
  imgsz: 512     # 减少图像尺寸
```

### Q2: 检测精度不理想
**A**: 尝试以下方法:
- 增加训练轮数
- 调整学习率
- 增加数据增强
- 检查标注质量
- 使用更大的模型(如yolov8l)

### Q3: 推理速度太慢
**A**: 优化建议:
- 使用较小的模型(如yolov8n)
- 减少输入图像尺寸
- 使用模型量化
- 启用TensorRT加速

### Q4: 数据格式转换失败
**A**: 检查以下项目:
- COCO标注文件格式是否正确
- 图片路径是否存在
- 文件权限是否足够

## 输出结果说明

### 训练输出
- `outputs/yolov8m_question_detection/weights/best.pt`: 最佳模型权重
- `outputs/yolov8m_question_detection/weights/last.pt`: 最后一轮模型权重
- `outputs/yolov8m_question_detection/results.png`: 训练曲线图
- `outputs/yolov8m_question_detection/confusion_matrix.png`: 混淆矩阵

### 分割输出
- `{image_name}_question_001.jpg`: 分割出的题目图像
- `{image_name}_annotated.jpg`: 标注后的原图
- `{image_name}_results.json`: 检测结果JSON文件
- `segmentation_report.txt`: 分割报告

## 技术支持

如果遇到问题，请检查:
1. 依赖包是否正确安装
2. 配置文件是否正确设置
3. 数据格式是否符合要求
4. 系统资源是否充足

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 支持YOLOv8m模型训练
- 支持COCO到YOLO格式转换
- 支持图像分割和批量处理
- 提供完整的配置和使用文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交代码前，请确保:
1. 代码符合PEP8规范
2. 添加必要的注释和文档
3. 通过所有测试用例

---

**注意**: 本项目仅供学习和研究使用，请遵守相关法律法规。