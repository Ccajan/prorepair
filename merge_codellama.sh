#!/bin/bash

# 合并 CodeLlama-7B 训练后的 adapter
# 用法: ./merge_codellama.sh

set -e  # 遇到错误立即退出

echo "🔄 开始合并 CodeLlama-7B adapter..."

# 配置
BASE_MODEL="codellama-7b"
ADAPTER_PATH="trained_models/trained_model_codellama_7b_curriculum"
OUTPUT_PATH="merged_models/codellama-7b-prorepair"

# 检查 adapter 是否存在
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ 错误: Adapter 路径不存在: $ADAPTER_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p merged_models

# 执行合并
python3 merge_adapter.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --output_path "$OUTPUT_PATH"

echo ""
echo "✅ 合并完成！"
echo "📁 合并后的模型: $OUTPUT_PATH"
echo ""
echo "你现在可以使用合并后的模型进行推理："
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  model = AutoModelForCausalLM.from_pretrained('$OUTPUT_PATH')"
echo "  tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_PATH')"

