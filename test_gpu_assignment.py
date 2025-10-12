#!/usr/bin/env python3
"""
测试GPU分配是否正确工作的脚本
"""
import torch
import sys
import os

def test_gpu_assignment(gpu_id):
    """测试GPU分配"""
    print(f"测试GPU {gpu_id}分配...")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"系统可用GPU数量: {gpu_count}")
    
    if gpu_id >= gpu_count:
        print(f"错误：GPU {gpu_id} 不存在")
        return False
    
    try:
        # 设置GPU设备
        torch.cuda.set_device(gpu_id)
        print(f"已设置PyTorch使用GPU {gpu_id}")
        
        # 获取当前设备
        current_device = torch.cuda.current_device()
        print(f"当前PyTorch设备: cuda:{current_device}")
        
        # 创建测试张量
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda(gpu_id)
        print(f"测试张量设备: {test_tensor.device}")
        
        # 验证张量确实在指定GPU上
        expected_device = f"cuda:{gpu_id}"
        if str(test_tensor.device) == expected_device:
            print(f"✅ 成功：张量在正确的设备 {expected_device} 上")
            return True
        else:
            print(f"❌ 失败：张量在设备 {test_tensor.device} 上，期望在 {expected_device} 上")
            return False
            
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_gpu_assignment.py <gpu_id>")
        sys.exit(1)
    
    gpu_id = int(sys.argv[1])
    success = test_gpu_assignment(gpu_id)
    
    if success:
        print(f"🎉 GPU {gpu_id} 分配测试成功！")
    else:
        print(f"💥 GPU {gpu_id} 分配测试失败！")
        sys.exit(1)