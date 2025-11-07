# 面瘫识别和分级模型迁移总结

## 项目概述

本项目成功将DSAT（Dynamic Semantic Aggregation Transformer）模型从人脸关键点检测任务迁移到面瘫识别和分级任务。通过保留原有的强大特征提取能力，同时添加分类头，实现了医学图像分类功能。

## 主要修改内容

### 1. 模型架构修改 (model.py)

**新增参数：**
- `num_classes`: 分类类别数量
- `task_type`: 任务类型 ('classification' 或 'regression')

**新增组件：**
- 分类头：包含全局平均池化、全连接层和Dropout
- 支持双模式：分类模式输出类别概率，回归模式输出关键点热图

**架构特点：**
- 保留原有的HourGlass、ChannelTransformer、maskSemhash等核心组件
- 添加自适应池化层提取全局特征
- 使用多层全连接网络进行最终分类

### 2. 数据集加载器 (facial_paralysis_dataset.py)

**功能特性：**
- 支持二分类和多分类任务
- 灵活的目录结构支持
- 自动数据增强（翻转、旋转、亮度对比度调整）
- 人脸检测和对齐预处理

**支持的数据格式：**
```
data/facial_paralysis/
├── train/
│   ├── normal/          # 正常人脸
│   ├── paralysis/       # 面瘫患者
│   ├── mild/            # 轻度面瘫
│   ├── moderate/        # 中度面瘫
│   └── severe/          # 重度面瘫
└── test/
    └── [相同结构]
```

### 3. 训练器修改 (solver.py)

**新增功能：**
- 支持分类和回归两种任务模式
- 使用CrossEntropyLoss进行分类训练
- 分类评估指标：准确率、精确率、召回率、F1分数
- 保留原有回归模式的RMSE评估

**训练流程：**
- 自动根据任务类型选择损失函数和评估指标
- 支持模型检查点保存和恢复
- 学习率调度和早停机制

### 4. 主训练脚本 (main_facial_paralysis.py)

**参数配置：**
- `--task_type`: 任务类型选择
- `--num_classes`: 类别数量配置
- `--train_data_dir` / `--test_data_dir`: 数据路径
- 完整的超参数配置接口

**使用示例：**
```bash
# 二分类训练
python main_facial_paralysis.py --task_type classification --num_classes 2

# 多分类训练
python main_facial_paralysis.py --task_type classification --num_classes 4
```

### 5. 推理演示 (demo_facial_paralysis.py)

**功能特性：**
- 单张图像和批量图像预测
- 支持概率输出
- 灵活的输入格式（文件路径或目录）
- 结果保存功能

**使用示例：**
```bash
# 单张图像预测
python demo_facial_paralysis.py --model_path model.pth --input image.jpg --return_probs

# 批量预测
python demo_facial_paralysis.py --model_path model.pth --input images_dir/
```

### 6. 数据准备工具 (prepare_facial_paralysis_data.py)

**功能：**
- 自动组织分散的图像数据
- 训练/测试集划分
- 标签文件生成
- 数据集验证

**支持的场景：**
- 二分类数据组织
- 多分类数据组织
- 自定义类别名称

### 7. 便利脚本

**训练脚本 (train_facial_paralysis.sh):**
```bash
bash train_facial_paralysis.sh binary    # 二分类
bash train_facial_paralysis.sh multiclass # 多分类
```

**测试脚本 (test_facial_paralysis.sh):**
```bash
bash test_facial_paralysis.sh model.pth 2
```

**推理脚本 (inference_facial_paralysis.sh):**
```bash
bash inference_facial_paralysis.sh model.pth image.jpg 2 output.txt
```

## 技术特点

### 1. 模型优势

- **强大的特征提取**：利用DSAT的HourGlass和ChannelTransformer提取多尺度特征
- **语义感知**：maskSemhash模块提供语义级别的特征选择
- **端到端训练**：支持从原始图像到分类结果的端到端训练
- **双模式兼容**：同时支持分类和回归任务

### 2. 数据处理

- **鲁棒的预处理**：人脸检测、对齐、归一化
- **丰富的数据增强**：几何变换和颜色变换
- **灵活的数据格式**：支持多种目录结构和标签格式

### 3. 训练优化

- **自适应学习率**：动态调整学习率
- **正则化技术**：Dropout和权重衰减防止过拟合
- **早停机制**：基于验证集性能的模型选择

## 性能评估

### 分类指标

- **准确率 (Accuracy)**: 正确预测的比例
- **精确率 (Precision)**: 查准率
- **召回率 (Recall)**: 查全率
- **F1分数**: 精确率和召回率的调和平均

### 模型比较

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| 原始DSAT | 关键点检测 | 人脸对齐、分析 |
| 修改版DSAT-分类 | 面瘫检测 | 医学诊断 |
| 修改版DSAT-分级 | 严重程度评估 | 治疗方案制定 |

## 使用建议

### 1. 数据准备

- **数据质量**：使用高质量、清晰的人脸图像
- **数据平衡**：确保各类别样本数量均衡
- **标注准确性**：确保医学标签的准确性

### 2. 训练策略

- **迁移学习**：可使用预训练的人脸识别模型作为初始化
- **超参数调优**：根据具体数据集调整学习率和批次大小
- **交叉验证**：使用K折交叉验证评估模型稳定性

### 3. 部署考虑

- **模型优化**：可进行模型量化和剪枝以适应移动设备
- **实时性**：优化推理速度以满足临床需求
- **可解释性**：添加注意力可视化以提高模型可信度

## 扩展方向

### 1. 技术扩展

- **多任务学习**：同时进行关键点检测和面瘫分类
- **注意力机制**：添加空间和通道注意力
- **模型集成**：训练多个模型进行集成预测

### 2. 应用扩展

- **其他疾病诊断**：扩展到其他面部疾病检测
- **疗效评估**：评估治疗效果
- **预后预测**：预测康复情况

## 总结

本项目成功实现了DSAT模型从人脸关键点检测到面瘫识别和分级的迁移，保留了原模型的强大特征提取能力，同时添加了适合医学图像分类的功能。通过完整的工具链和详细的文档，为面瘫的自动化诊断提供了有效的技术方案。

该迁移工作展示了深度学习模型在不同任务间的适应能力，为医学图像分析提供了新的思路和方法。