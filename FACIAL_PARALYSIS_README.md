# 面瘫识别和分级系统

本项目基于DSAT（Dynamic Semantic Aggregation Transformer）模型，将其从人脸关键点检测任务迁移到面瘫识别和分级任务。

## 功能特性

- **二分类**：正常 vs 面瘫识别
- **多分类**：面瘫严重程度分级（正常、轻度、中度、重度）
- **数据增强**：支持旋转、翻转、亮度对比度调整等
- **模型评估**：准确率、精确率、召回率、F1分数
- **推理演示**：单张图像和批量图像预测

## 项目结构

```
├── main_facial_paralysis.py          # 面瘫分类主训练脚本
├── demo_facial_paralysis.py          # 面瘫分类推理演示
├── prepare_facial_paralysis_data.py  # 数据准备工具
├── facial_paralysis_dataset.py       # 面瘫数据集加载器
├── model.py                          # 修改后的模型（支持分类）
├── solver.py                         # 修改后的训练器（支持分类）
└── data/
    └── facial_paralysis/
        ├── train/
        │   ├── normal/               # 正常人脸
        │   ├── paralysis/            # 面瘫患者（二分类）
        │   ├── mild/                 # 轻度面瘫（多分类）
        │   ├── moderate/             # 中度面瘫（多分类）
        │   └── severe/               # 重度面瘫（多分类）
        └── test/
            ├── normal/
            ├── paralysis/
            ├── mild/
            ├── moderate/
            └── severe/
```

## 环境要求

```bash
Python 3.7+
PyTorch >= 1.7.0
torchvision
OpenCV
scikit-learn
numpy
PIL
```

## 安装依赖

```bash
pip install torch torchvision opencv-python scikit-learn pillow numpy
```

## 数据准备

### 1. 数据集结构

将您的面瘫数据集按照以下结构组织：

```
data/facial_paralysis/
├── train/
│   ├── normal/          # 正常人脸图像
│   └── paralysis/       # 面瘫患者图像
└── test/
    ├── normal/
    └── paralysis/
```

对于多分类任务：

```
data/facial_paralysis/
├── train/
│   ├── normal/          # 正常
│   ├── mild/            # 轻度面瘫
│   ├── moderate/        # 中度面瘫
│   └── severe/          # 重度面瘫
└── test/
    ├── normal/
    ├── mild/
    ├── moderate/
    └── severe/
```

### 2. 使用数据准备工具

如果您有分散的图像数据，可以使用数据准备工具自动组织：

```bash
# 二分类数据准备
python prepare_facial_paralysis_data.py \
    --root_dir ./data/facial_paralysis \
    --task_type binary \
    --normal_dir /path/to/normal/images \
    --paralysis_dir /path/to/paralysis/images \
    --train_ratio 0.8

# 多分类数据准备
python prepare_facial_paralysis_data.py \
    --root_dir ./data/facial_paralysis \
    --task_type multiclass \
    --normal_dir /path/to/normal/images \
    --mild_dir /path/to/mild/images \
    --moderate_dir /path/to/moderate/images \
    --severe_dir /path/to/severe/images \
    --train_ratio 0.8
```

## 训练模型

### 1. 二分类训练

```bash
python main_facial_paralysis.py \
    --task_type classification \
    --num_classes 2 \
    --train_data_dir ./data/facial_paralysis/train \
    --test_data_dir ./data/facial_paralysis/test \
    --batch_size 32 \
    --num_iters 10000 \
    --lr 1e-4 \
    --phase train
```

### 2. 多分类训练

```bash
python main_facial_paralysis.py \
    --task_type classification \
    --num_classes 4 \
    --train_data_dir ./data/facial_paralysis/train \
    --test_data_dir ./data/facial_paralysis/test \
    --batch_size 32 \
    --num_iters 15000 \
    --lr 1e-4 \
    --phase train
```

### 3. 训练参数说明

- `--task_type`: 任务类型，设置为 `classification`
- `--num_classes`: 类别数量（2为二分类，4为多分类）
- `--train_data_dir`: 训练数据目录
- `--test_data_dir`: 测试数据目录
- `--batch_size`: 批次大小
- `--num_iters`: 训练迭代次数
- `--lr`: 学习率
- `--phase`: `train` 或 `test`

## 模型评估

```bash
python main_facial_paralysis.py \
    --task_type classification \
    --num_classes 2 \
    --test_data_dir ./data/facial_paralysis/test \
    --phase test \
    --best_model checkpoint/models/best_checkpoint.pth.tar
```

## 推理演示

### 1. 单张图像预测

```bash
python demo_facial_paralysis.py \
    --model_path checkpoint/models/best_checkpoint.pth.tar \
    --input /path/to/image.jpg \
    --num_classes 2 \
    --return_probs
```

### 2. 批量预测（目录）

```bash
python demo_facial_paralysis.py \
    --model_path checkpoint/models/best_checkpoint.pth.tar \
    --input /path/to/images/directory \
    --num_classes 2 \
    --return_probs \
    --output results.txt
```

### 3. 推理参数说明

- `--model_path`: 训练好的模型路径
- `--input`: 输入图像路径或目录
- `--num_classes`: 类别数量
- `--return_probs`: 返回各类别概率
- `--output`: 结果保存文件（可选）

## 模型架构

修改后的FAN模型包含以下组件：

1. **特征提取器**：
   - 卷积层 + HourGlass模块
   - ChannelTransformer用于多尺度特征融合
   - maskSemhash用于语义掩码

2. **分类头**：
   - 自适应平均池化
   - 全连接层 + Dropout
   - Softmax输出

3. **损失函数**：
   - 交叉熵损失（CrossEntropyLoss）

## 评估指标

- **准确率（Accuracy）**：正确预测的比例
- **精确率（Precision）**：预测为正例中实际为正例的比例
- **召回率（Recall）**：实际正例中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均

## 训练技巧

1. **数据增强**：
   - 随机水平翻转
   - 亮度对比度调整
   - 随机旋转（-15°到+15°）

2. **学习率调度**：
   - 初始学习率：1e-4
   - 每2000次迭代衰减50%

3. **正则化**：
   - Dropout（0.5和0.3）
   - 权重衰减（1e-4）

## 性能优化建议

1. **数据集平衡**：确保各类别样本数量均衡
2. **图像质量**：使用高质量、清晰的人脸图像
3. **人脸对齐**：预处理时进行人脸检测和对齐
4. **超参数调优**：根据具体数据集调整学习率、批次大小等

## 故障排除

1. **CUDA内存不足**：
   - 减小batch_size
   - 使用梯度累积

2. **训练不收敛**：
   - 检查数据质量
   - 调整学习率
   - 增加训练迭代次数

3. **过拟合**：
   - 增加数据增强
   - 使用更大的Dropout
   - 增加权重衰减

## 扩展功能

1. **迁移学习**：使用预训练的人脸识别模型作为特征提取器
2. **多任务学习**：同时进行关键点检测和面瘫分类
3. **注意力机制**：添加注意力模块聚焦于面部关键区域
4. **模型集成**：训练多个模型进行集成预测

## 引用

如果您在研究中使用了本代码，请引用原始DSAT论文以及本项目的修改。

## 许可证

本项目遵循原始DSAT项目的许可证。