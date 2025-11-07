#!/bin/bash

# 面瘫识别和分级训练脚本示例
# 使用方法：bash train_facial_paralysis.sh [binary|multiclass]

TASK_TYPE=${1:-binary}

echo "开始面瘫分类训练..."
echo "任务类型: $TASK_TYPE"

if [ "$TASK_TYPE" = "binary" ]; then
    # 二分类：正常 vs 面瘫
    python main_facial_paralysis.py \
        --task_type classification \
        --num_classes 2 \
        --train_data_dir ./data/facial_paralysis/train \
        --test_data_dir ./data/facial_paralysis/test \
        --batch_size 32 \
        --num_iters 10000 \
        --lr 1e-4 \
        --beta1 0.9 \
        --beta2 0.999 \
        --weightDecay 1e-4 \
        --res 128 \
        --augment True \
        --phase train \
        --log_dir checkpoint/logs \
        --model_save_dir checkpoint/models \
        --log_step 100 \
        --model_save_step 1000 \
        --lr_update_step 2000

elif [ "$TASK_TYPE" = "multiclass" ]; then
    # 多分类：正常、轻度、中度、重度
    python main_facial_paralysis.py \
        --task_type classification \
        --num_classes 4 \
        --train_data_dir ./data/facial_paralysis/train \
        --test_data_dir ./data/facial_paralysis/test \
        --batch_size 32 \
        --num_iters 15000 \
        --lr 1e-4 \
        --beta1 0.9 \
        --beta2 0.999 \
        --weightDecay 1e-4 \
        --res 128 \
        --augment True \
        --phase train \
        --log_dir checkpoint/logs \
        --model_save_dir checkpoint/models \
        --log_step 100 \
        --model_save_step 1000 \
        --lr_update_step 2000

else
    echo "错误：不支持的任务类型。请使用 'binary' 或 'multiclass'"
    exit 1
fi

echo "训练完成！"