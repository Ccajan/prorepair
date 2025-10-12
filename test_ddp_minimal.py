#!/usr/bin/env python3
"""
最小化 DDP 测试脚本 - 用于验证 NCCL 配置是否正确
运行: torchrun --nproc_per_node=2 test_ddp_minimal.py
"""

import os

# ==========================================
# 关键环境变量配置（必须在导入 torch 之前设置）
# ==========================================
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_DISTRIBUTED_BROADCAST_BUCKET_SIZE"] = str(25 * 1024 * 1024)
os.environ["NCCL_MIN_NCHANNELS"] = "1"
os.environ["NCCL_MAX_NCHANNELS"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def print_rank0(msg):
    """只在 rank 0 打印"""
    if dist.get_rank() == 0:
        print(msg)

def main():
    # 1. 初始化分布式环境
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    print(f"✅ Rank {local_rank}/{world_size} 初始化完成")
    
    # 2. 检查环境变量
    if local_rank == 0:
        print("\n" + "="*60)
        print("🔍 环境变量检查:")
        print(f"   NCCL_SHM_DISABLE: {os.environ.get('NCCL_SHM_DISABLE', 'NOT SET')}")
        print(f"   TORCH_DISTRIBUTED_BROADCAST_BUCKET_SIZE: {int(os.environ.get('TORCH_DISTRIBUTED_BROADCAST_BUCKET_SIZE', 0)) / (1024*1024):.0f} MB")
        print(f"   NCCL_MIN_NCHANNELS: {os.environ.get('NCCL_MIN_NCHANNELS', 'NOT SET')}")
        print(f"   NCCL_MAX_NCHANNELS: {os.environ.get('NCCL_MAX_NCHANNELS', 'NOT SET')}")
        print("="*60 + "\n")
    
    # 3. 创建测试模型（模拟 LoRA 参数量）
    print_rank0("📦 创建测试模型...")
    
    class TestModel(nn.Module):
        def __init__(self, hidden_size=4096, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) 
                for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = TestModel().to(f"cuda:{local_rank}")
    
    # 显示模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Rank {local_rank}: 模型参数量 = {total_params:,} ({total_params * 4 / (1024**3):.2f} GB in FP32)")
    
    # 4. 显示初始显存
    mem_before = torch.cuda.memory_allocated() / 1024**3
    peak_before = torch.cuda.max_memory_allocated() / 1024**3
    print(f"   Rank {local_rank}: DDP 包装前显存 = {mem_before:.2f} GB (峰值 {peak_before:.2f} GB)")
    
    # 5. 清理缓存（关键步骤）
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 6. DDP 包装
    print_rank0("\n🔧 开始 DDP 包装...")
    
    try:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
            static_graph=True,
        )
        print_rank0("✅ DDP 初始化成功 (static_graph=True, bucket_cap_mb=25)")
    except TypeError as e:
        if "static_graph" in str(e):
            ddp_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=25,
            )
            print_rank0("✅ DDP 初始化成功 (bucket_cap_mb=25)")
        else:
            ddp_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
            print_rank0("✅ DDP 初始化成功 (基础配置)")
    
    # 7. 显示 DDP 后显存
    mem_after = torch.cuda.memory_allocated() / 1024**3
    peak_after = torch.cuda.max_memory_allocated() / 1024**3
    print(f"   Rank {local_rank}: DDP 包装后显存 = {mem_after:.2f} GB (峰值 {peak_after:.2f} GB)")
    print(f"   Rank {local_rank}: DDP 包装增量 = {mem_after - mem_before:.2f} GB (峰值增量 {peak_after - peak_before:.2f} GB)")
    
    # 8. 测试前向传播
    print_rank0("\n🧪 测试前向传播...")
    
    x = torch.randn(2, 4096).to(f"cuda:{local_rank}")
    output = ddp_model(x)
    
    print(f"   Rank {local_rank}: 前向传播成功，输出形状 = {output.shape}")
    
    # 9. 测试反向传播
    print_rank0("🧪 测试反向传播...")
    
    loss = output.sum()
    loss.backward()
    
    print(f"   Rank {local_rank}: 反向传播成功")
    
    # 10. 测试梯度同步
    print_rank0("🧪 测试梯度同步...")
    
    # 检查梯度是否同步
    for name, param in ddp_model.named_parameters():
        if param.grad is not None:
            grad_sum = param.grad.sum().item()
            # 所有 rank 的梯度应该相同（DDP 自动同步）
            print(f"   Rank {local_rank}: {name[:30]:30s} grad_sum = {grad_sum:.4f}")
            break  # 只打印第一个参数
    
    # 11. 最终显存统计
    final_mem = torch.cuda.memory_allocated() / 1024**3
    final_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print_rank0("\n" + "="*60)
    print_rank0("📊 最终显存统计:")
    print(f"   Rank {local_rank}: 当前显存 = {final_mem:.2f} GB")
    print(f"   Rank {local_rank}: 峰值显存 = {final_peak:.2f} GB")
    print_rank0("="*60)
    
    # 同步所有进程
    dist.barrier()
    
    if local_rank == 0:
        print("\n✅ 所有测试通过！DDP 配置正确。")
        print("\n⚠️  关键检查项:")
        print("   1. NCCL 日志中应显示 'Using network Socket'")
        print("   2. NCCL 日志中应显示 'via direct/direct'（而非 'via SHM'）")
        print("   3. DDP 包装的峰值显存增量应 < 2GB")
    
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()


