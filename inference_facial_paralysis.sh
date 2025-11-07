#!/bin/bash

# 面瘫识别和分级推理脚本
# 使用方法：bash inference_facial_paralysis.sh model_path input_path [num_classes] [output_file]

MODEL_PATH=$1
INPUT_PATH=$2
NUM_CLASSES=${3:-2}
OUTPUT_FILE=${4:-""}

if [ -z "$MODEL_PATH" ] || [ -z "$INPUT_PATH" ]; then
    echo "错误：请提供模型路径和输入路径"
    echo "使用方法：bash inference_facial_paralysis.sh model_path input_path [num_classes] [output_file]"
    exit 1
fi

echo "开始面瘫分类推理..."
echo "模型路径: $MODEL_PATH"
echo "输入路径: $INPUT_PATH"
echo "类别数量: $NUM_CLASSES"

if [ -n "$OUTPUT_FILE" ]; then
    echo "输出文件: $OUTPUT_FILE"
    python demo_facial_paralysis.py \
        --model_path $MODEL_PATH \
        --input $INPUT_PATH \
        --num_classes $NUM_CLASSES \
        --return_probs \
        --output $OUTPUT_FILE
else
    python demo_facial_paralysis.py \
        --model_path $MODEL_PATH \
        --input $INPUT_PATH \
        --num_classes $NUM_CLASSES \
        --return_probs
fi

echo "推理完成！"