"""
测试合并后的模型
验证模型是否能正常加载和生成
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_model(model_path: str, test_prompt: str = None):
    """
    测试合并后的模型
    
    Args:
        model_path: 合并后的模型路径
        test_prompt: 测试提示词
    """
    
    print("=" * 60)
    print("🧪 测试合并后的模型")
    print("=" * 60)
    print(f"📁 模型路径: {model_path}")
    print()
    
    # 1. 加载 tokenizer
    print("🚀 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("✅ Tokenizer 加载完成")
    
    # 2. 加载模型
    print("🚀 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ 模型加载完成")
    
    # 3. 测试生成
    if test_prompt is None:
        test_prompt = """You are a code repair assistant. Please fix the following buggy Java code:

```java
public int divide(int a, int b) {
    return a / b;
}
```

The bug is that it doesn't handle division by zero. Please provide the fixed code."""
    
    print("\n" + "=" * 60)
    print("📝 测试提示词:")
    print("=" * 60)
    print(test_prompt)
    print()
    
    # 格式化输入
    messages = [{"role": "user", "content": test_prompt}]
    
    # 对于 CodeLlama，使用简单的格式
    formatted_input = f"<s>[INST] {test_prompt} [/INST] "
    
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    
    print("🤖 生成回复...")
    print("=" * 60)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 只显示生成的部分（去掉输入提示）
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    print(response)
    print()
    print("=" * 60)
    print("✅ 测试完成！模型运行正常。")
    print("=" * 60)
    
    # 显示模型信息
    print("\n📊 模型信息:")
    print(f"  • 模型类型: {model.__class__.__name__}")
    print(f"  • 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  • 设备: {next(model.parameters()).device}")
    print(f"  • 数据类型: {next(model.parameters()).dtype}")


def main():
    parser = argparse.ArgumentParser(
        description='测试合并后的模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试合并后的模型
  python test_merged_model.py --model_path merged_models/codellama-7b-prorepair

  # 使用自定义测试提示
  python test_merged_model.py \\
    --model_path merged_models/codellama-7b-prorepair \\
    --test_prompt "Fix this bug: def add(a, b): return a - b"
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='合并后的模型路径'
    )
    
    parser.add_argument(
        '--test_prompt',
        type=str,
        default=None,
        help='测试提示词（可选，默认使用内置测试）'
    )
    
    args = parser.parse_args()
    
    try:
        test_model(
            model_path=args.model_path,
            test_prompt=args.test_prompt
        )
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()



