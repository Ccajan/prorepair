#!/usr/bin/env python3
"""
测试多GPU主进程检测
用于验证修复是否生效
"""

import os
import sys

def check_main_process():
    """检测是否是主进程"""
    # 方法1: 检查环境变量
    local_rank = os.environ.get('LOCAL_RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    rank = os.environ.get('RANK', None)
    
    print("="*60)
    print("🔍 多GPU进程检测")
    print("="*60)
    print(f"环境变量:")
    print(f"  LOCAL_RANK: {local_rank}")
    print(f"  RANK: {rank}")
    print(f"  WORLD_SIZE: {world_size}")
    
    if local_rank is not None:
        is_main = (int(local_rank) == 0)
        print(f"\n检测结果:")
        print(f"  当前进程: {'主进程 (rank 0) ✅' if is_main else f'工作进程 (rank {local_rank})'}")
        print(f"  应该初始化 wandb: {'是' if is_main else '否'}")
        print(f"  应该打印日志: {'是' if is_main else '否'}")
    else:
        print(f"\n检测结果:")
        print(f"  单GPU模式或未使用分布式训练")
        print(f"  默认为主进程 ✅")
    
    print("="*60)
    
    # 方法2: 检查 torch.distributed
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            torch_rank = dist.get_rank()
            torch_world_size = dist.get_world_size()
            print(f"\nTorch Distributed 状态:")
            print(f"  已初始化: ✅")
            print(f"  Rank: {torch_rank}")
            print(f"  World Size: {torch_world_size}")
            print(f"  是主进程: {'是' if torch_rank == 0 else '否'}")
        else:
            print(f"\nTorch Distributed 状态:")
            print(f"  未初始化（单GPU模式）")
    except ImportError:
        print(f"\nTorch Distributed: 未安装")
    
    print("="*60)
    
    return local_rank

if __name__ == "__main__":
    check_main_process()
