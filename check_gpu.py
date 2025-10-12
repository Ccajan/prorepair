#!/usr/bin/env python3
"""
GPU 状态检查和 Batch Size 推荐工具
"""

import torch
import sys

def format_size(bytes):
    """格式化字节为可读大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def check_gpu_status():
    """检查GPU状态并给出训练建议"""
    
    print("\n" + "="*60)
    print("🎮 GPU 状态检查")
    print("="*60)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA/GPU")
        print("   将使用 CPU 训练（非常慢，不推荐）")
        print("="*60 + "\n")
        return
    
    # GPU基本信息
    gpu_count = torch.cuda.device_count()
    print(f"\n✅ 检测到 {gpu_count} 个 GPU\n")
    
    total_memory_all = 0
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory
        total_memory_all += total_mem
        
        # 尝试获取当前使用情况
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = total_mem - allocated
        
        print(f"GPU {i}: {props.name}")
        print(f"   总显存: {format_size(total_mem)}")
        print(f"   已分配: {format_size(allocated)} ({allocated/total_mem*100:.1f}%)")
        print(f"   已保留: {format_size(reserved)} ({reserved/total_mem*100:.1f}%)")
        print(f"   可用:   {format_size(free)} ({free/total_mem*100:.1f}%)")
        print(f"   CUDA 能力: {props.major}.{props.minor}")
        print(f"   多处理器: {props.multi_processor_count}")
        print()
    
    print("="*60)
    print("📊 训练配置建议")
    print("="*60)
    
    # 根据显存大小推荐配置
    first_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if first_gpu_mem_gb >= 75:  # 80GB
        batch_size = 8
        grad_accum = 4
        gpu_type = "80GB (A100/H100)"
    elif first_gpu_mem_gb >= 45:  # 48GB
        batch_size = 6
        grad_accum = 4
        gpu_type = "48GB (A6000/A40)"
    elif first_gpu_mem_gb >= 38:  # 40GB
        batch_size = 4
        grad_accum = 4
        gpu_type = "40GB (A100)"
    elif first_gpu_mem_gb >= 22:  # 24GB
        batch_size = 2
        grad_accum = 8
        gpu_type = "24GB (RTX 3090/4090)"
    elif first_gpu_mem_gb >= 14:  # 16GB
        batch_size = 1
        grad_accum = 16
        gpu_type = "16GB (RTX 4060Ti)"
    else:  # < 16GB
        batch_size = 1
        grad_accum = 16
        gpu_type = f"{first_gpu_mem_gb:.0f}GB (显存较小)"
    
    effective_batch = batch_size * grad_accum * gpu_count
    
    print(f"\n检测到: {gpu_type}")
    print(f"\n推荐配置:")
    print(f"   --batch_size {batch_size}")
    print(f"   --gradient_accumulation_steps {grad_accum}")
    print(f"   有效 batch size = {effective_batch}")
    
    if gpu_count == 1:
        print(f"\n🚀 启动命令 (单GPU):")
        print(f"   python sft2.py \\")
        print(f"       --model_name \"CodeLlama-7b-Instruct\" \\")
        print(f"       --batch_size {batch_size} \\")
        print(f"       --gradient_accumulation_steps {grad_accum}")
        
        print(f"\n   或使用自动化脚本:")
        print(f"   ./train_single_gpu.sh")
    else:
        print(f"\n🚀 启动命令 (多GPU - 推荐):")
        print(f"   accelerate launch sft2.py \\")
        print(f"       --model_name \"CodeLlama-7b-Instruct\" \\")
        print(f"       --batch_size {batch_size} \\")
        print(f"       --gradient_accumulation_steps {grad_accum}")
        
        print(f"\n   或使用自动化脚本:")
        print(f"   ./train_multi_gpu.sh")
        
        # 预估加速比
        speedup = min(gpu_count * 0.85, gpu_count)  # 考虑通信开销
        print(f"\n   预计加速: ~{speedup:.1f}x (相比单GPU)")
    
    print("\n" + "="*60)
    print("💡 优化建议")
    print("="*60)
    
    suggestions = []
    
    # Flash Attention
    try:
        import flash_attn
        print("✅ Flash Attention 2 已安装 (推荐)")
    except ImportError:
        print("⚠️  Flash Attention 2 未安装")
        suggestions.append("安装 Flash Attention 2: pip install flash-attn --no-build-isolation")
    
    # BF16支持检查
    if torch.cuda.is_bf16_supported():
        print("✅ 支持 BF16 混合精度训练")
    else:
        print("⚠️  不支持 BF16，将使用 FP16")
    
    # Accelerate
    try:
        import accelerate
        print("✅ Accelerate 已安装 (多GPU训练)")
    except ImportError:
        if gpu_count > 1:
            print("⚠️  Accelerate 未安装 (多GPU训练需要)")
            suggestions.append("安装 Accelerate: pip install accelerate")
    
    # DeepSpeed
    try:
        import deepspeed
        print("✅ DeepSpeed 已安装 (高级优化)")
    except ImportError:
        print("ℹ️  DeepSpeed 未安装 (可选)")
    
    if suggestions:
        print(f"\n📦 安装建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        check_gpu_status()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
