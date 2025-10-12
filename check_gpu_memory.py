#!/usr/bin/env python3
"""
GPU显存诊断工具 - 用于检查模型加载时的显存使用
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os

def check_memory():
    """检查GPU显存使用"""
    if not torch.cuda.is_available():
        print("❌ 未检测到GPU")
        return
    
    device = torch.device("cuda:0")
    print(f"\n{'='*60}")
    print(f"🔍 GPU显存诊断")
    print(f"{'='*60}")
    
    # 初始状态
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated(0) / 1024**3
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"📊 GPU信息:")
    print(f"   设备: {torch.cuda.get_device_name(0)}")
    print(f"   总显存: {total_mem:.2f}GB")
    print(f"   初始使用: {initial_mem:.2f}GB")
    print(f"   可用显存: {(total_mem - initial_mem):.2f}GB")
    print()
    
    # 加载tokenizer
    print("📦 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    mem_after_tokenizer = torch.cuda.memory_allocated(0) / 1024**3
    print(f"✅ Tokenizer加载后: {mem_after_tokenizer:.2f}GB (增加 {mem_after_tokenizer - initial_mem:.2f}GB)")
    print()
    
    # 加载模型（BF16）
    print("📦 加载模型 (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf",
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    torch.cuda.empty_cache()
    
    mem_after_model = torch.cuda.memory_allocated(0) / 1024**3
    print(f"✅ 模型加载后: {mem_after_model:.2f}GB (增加 {mem_after_model - mem_after_tokenizer:.2f}GB)")
    print()
    
    # 应用LoRA (rank=16)
    print("🔧 应用 LoRA (rank=16)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16.0,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()
    
    mem_after_lora = torch.cuda.memory_allocated(0) / 1024**3
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA应用后: {mem_after_lora:.2f}GB (增加 {mem_after_lora - mem_after_model:.2f}GB)")
    print(f"   可训练参数: {trainable_params:,}")
    print()
    
    # 显存总结
    print(f"{'='*60}")
    print(f"📊 显存使用总结:")
    print(f"   模型总占用: {mem_after_lora:.2f}GB")
    print(f"   剩余可用: {(total_mem - mem_after_lora):.2f}GB")
    print(f"   使用率: {(mem_after_lora / total_mem * 100):.1f}%")
    print(f"{'='*60}")
    
    # 评估
    remaining = total_mem - mem_after_lora
    print(f"\n💡 评估:")
    if remaining > 15:
        print(f"   ✅ 显存充足，可以使用 batch_size=4")
    elif remaining > 10:
        print(f"   ⚠️  显存适中，建议 batch_size=2-3")
    elif remaining > 5:
        print(f"   ⚠️  显存紧张，建议 batch_size=1-2")
    else:
        print(f"   ❌ 显存不足，可能无法训练")
    
    print(f"\n   注意：DDP模式需要额外 ~2GB 用于进程通信")
    print()

if __name__ == "__main__":
    # 设置环境变量
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    check_memory()


