#!/bin/bash

# 面瘫识别和分级测试脚本
# 使用方法：bash test_facial_paralysis.sh model_path [num_classes]

MODEL_PATH=$1
NUM_CLASSES=${2:-2}

if [ -z "$MODEL_PATH" ]; then
    echo "错误：请提供模型路径"
    echo "使用方法：bash test_facial_paralysis.sh model_path [num_classes]"
    exit 1
fi

echo "开始面瘫分类测试..."
echo "模型路径: $MODEL_PATH"
echo "类别数量: $NUM_CLASSES"

python main_facial_paralysis.py \
    --task_type classification \
    --num_classes $NUM_CLASSES \
    --test_data_dir ./data/facial_paralysis/test \
    --phase test \
    --best_model $MODEL_PATH

echo "测试完成！"