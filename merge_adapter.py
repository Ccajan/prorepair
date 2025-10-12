"""
合并 LoRA Adapter 与基础模型
将训练后的 adapter 合并到基础模型，生成完整的模型
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def resolve_model_alias(name: str) -> str:
    """解析模型别名到完整路径"""
    mapping = {
        "Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    }
    return mapping.get(name, name)


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    device: str = "auto"
):
    """
    合并 LoRA adapter 与基础模型
    
    Args:
        base_model_name: 基础模型名称或路径
        adapter_path: adapter 文件夹路径
        output_path: 输出模型保存路径
        device: 设备配置
    """
    
    print("=" * 60)
    print("🔄 LoRA Adapter 合并工具")
    print("=" * 60)
    
    # 解析模型别名
    base_model_path = resolve_model_alias(base_model_name)
    
    print(f"📦 基础模型: {base_model_path}")
    print(f"🔧 Adapter 路径: {adapter_path}")
    print(f"💾 输出路径: {output_path}")
    print()
    
    # 检查 adapter 路径
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter 路径不存在: {adapter_path}")
    
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"未找到 adapter_config.json: {adapter_config}")
    
    # 1. 加载基础模型（不使用量化，以便合并）
    print("🚀 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 使用 fp16 以节省内存
        device_map=device,
        trust_remote_code=True,
    )
    print(f"✅ 基础模型加载完成")
    
    # 2. 加载 tokenizer
    print("🚀 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # 从 adapter 路径加载，保留训练时的配置
        trust_remote_code=True
    )
    print(f"✅ Tokenizer 加载完成")
    
    # 3. 加载 PEFT adapter
    print("🚀 加载 LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device
    )
    print(f"✅ LoRA adapter 加载完成")
    
    # 4. 合并 adapter 到基础模型
    print("🔄 合并 adapter 到基础模型...")
    merged_model = model.merge_and_unload()
    print(f"✅ 合并完成")
    
    # 5. 保存合并后的模型
    print(f"💾 保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # 使用 safetensors 格式
        max_shard_size="5GB"  # 如果模型太大，分片保存
    )
    
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ 模型保存完成")
    print()
    print("=" * 60)
    print("🎉 合并完成！")
    print(f"📁 合并后的模型位置: {output_path}")
    print("=" * 60)
    
    # 显示模型大小信息
    try:
        total_size = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if os.path.isfile(os.path.join(output_path, f))
        )
        print(f"💽 模型总大小: {total_size / (1024**3):.2f} GB")
    except Exception as e:
        print(f"⚠️ 无法计算模型大小: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='合并 LoRA Adapter 与基础模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用模型别名
  python merge_adapter.py \\
    --base_model codellama-7b \\
    --adapter_path trained_models/trained_model_codellama_7b_curriculum \\
    --output_path merged_models/codellama-7b-merged

  # 使用完整模型路径
  python merge_adapter.py \\
    --base_model codellama/CodeLlama-7b-Instruct-hf \\
    --adapter_path trained_models/trained_model_codellama_7b_curriculum \\
    --output_path merged_models/codellama-7b-merged
        """
    )
    
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='基础模型名称或路径 (支持别名: codellama-7b, Llama-3-8B-Instruct)'
    )
    
    parser.add_argument(
        '--adapter_path',
        type=str,
        required=True,
        help='LoRA adapter 文件夹路径'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='合并后模型的保存路径'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='设备配置 (默认: auto)'
    )
    
    args = parser.parse_args()
    
    try:
        merge_adapter(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()



