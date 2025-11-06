#!/bin/bash

# LCS 加权训练启动脚本

# 配置参数
MODEL_NAME="$1"  # 如 "deepseek-ai/deepseek-coder-6.7b-instruct"
DATASET_PATH="$2"  # 如 "data/trainset/ds_llm_sorted_by_diff.json"
OUTPUT_NAME="$3"  # 如 "sft_deepseek7b_lcs"

# LCS 加权配置
# LCS_WEIGHT: LCS token 的损失权重
#   - 默认 2.0：LCS token 的权重是普通 token 的 2 倍
#   - 更高的值：更强调保留原有代码（如 3.0, 4.0）
#   - 1.0：等价于标准训练（无加权）

# ========== 默认配置：LCS 权重 2.0 ==========
echo "启动 LCS 加权训练"
echo "LCS token 权重: 2.0"
echo "策略: 对公共子序列 token 赋予 2 倍权重"

# 注意：使用单GPU训练，因为4bit量化与DataParallel不兼容
CUDA_VISIBLE_DEVICES=0 \
LCS_WEIGHT=2.0 \
    python3 SingleTrainWithLCS.py "$MODEL_NAME" "$DATASET_PATH" "$OUTPUT_NAME"

# ========== 其他可选配置 ==========
# 更强的 LCS 偏好（权重 3.0）
# CUDA_VISIBLE_DEVICES=0 LCS_WEIGHT=3.0 python3 SingleTrainWithLCS.py "$MODEL_NAME" "$DATASET_PATH" "${OUTPUT_NAME}_w3"

# 标准训练（无加权，baseline）
# CUDA_VISIBLE_DEVICES=0 LCS_WEIGHT=1.0 python3 SingleTrainWithLCS.py "$MODEL_NAME" "$DATASET_PATH" "${OUTPUT_NAME}_baseline"

