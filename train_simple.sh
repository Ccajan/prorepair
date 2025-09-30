#!/bin/bash

# 简单的单GPU训练脚本
# 用法: bash train_simple.sh [model_name] [gpu_id]

MODEL_NAME=${1:-codellama-7b}
GPU_ID=${2:-1}  # 默认使用GPU 1
CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================================"
echo "🚀 单GPU训练"
echo "============================================================"
echo "📦 模型: $MODEL_NAME"
echo "🎮 GPU ID: $GPU_ID"
echo "============================================================"
echo ""

# 直接运行Python脚本，不使用torchrun
python3 sft2.py \
    --model_name $MODEL_NAME \
    --training_mode dft_preference \
    --curriculum_stages 3 \
    --attn_implementation sdpa \
    --use_wandb \
    --wandb_project prorepair-training \
    --wandb_entity buaabarty

echo ""
echo "============================================================"
echo "✅ 训练完成"
echo "============================================================"
