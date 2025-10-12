#!/usr/bin/env python3
"""
GPU状态检查脚本
显示系统中所有GPU的状态和使用情况
"""

import os
import torch
import subprocess
import psutil

def check_system_gpus():
    """检查系统GPU状态"""
    print("=== 系统GPU状态检查 ===")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"系统GPU数量: {gpu_count}")
    
    # 显示每个GPU的信息
    for i in range(gpu_count):
        print(f"\n--- GPU {i} ---")
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)
        
        print(f"名称: {gpu_name}")
        print(f"计算能力: {gpu_props.major}.{gpu_props.minor}")
        print(f"总内存: {gpu_props.total_memory / 1024**3:.1f}GB")
        print(f"多处理器数量: {gpu_props.multi_processor_count}")
        
        # 检查内存使用（需要先设置设备）
        try:
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"已分配内存: {allocated:.3f}GB")
                print(f"缓存内存: {cached:.3f}GB")
        except Exception as e:
            print(f"无法获取内存信息: {e}")

def check_gpu_processes():
    """检查GPU进程"""
    print("\n=== GPU进程检查 ===")
    
    try:
        # 尝试使用nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            print("运行中的GPU进程:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    pid, process_name, gpu_uuid, used_memory = parts[:4]
                    print(f"  PID: {pid}, 进程: {process_name}, 显存: {used_memory}MB")
        else:
            print("没有检测到GPU进程或nvidia-smi不可用")
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"无法运行nvidia-smi: {e}")

def check_gpu_locks():
    """检查GPU锁文件"""
    print("\n=== GPU锁文件检查 ===")
    
    lock_files = []
    for i in range(8):  # 检查GPU 0-7的锁文件
        lock_file = f"/tmp/gpu_{i}_lock"
        if os.path.exists(lock_file):
            lock_files.append((i, lock_file))
    
    if not lock_files:
        print("没有发现GPU锁文件")
        return
    
    for gpu_id, lock_file in lock_files:
        print(f"\nGPU {gpu_id} 锁文件存在: {lock_file}")
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read().strip()
            print(f"锁信息: {lock_info}")
            
            # 解析PID并检查进程状态
            if lock_info.startswith("PID:"):
                parts = lock_info.split(',')
                pid_str = parts[0].split(':')[1]
                try:
                    pid = int(pid_str)
                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        print(f"占用进程: PID {pid}, 名称: {proc.name()}, 状态: {proc.status()}")
                    else:
                        print(f"⚠️  进程 {pid} 已不存在，锁文件可能过期")
                except (ValueError, psutil.NoSuchProcess) as e:
                    print(f"⚠️  无法检查进程状态: {e}")
                    
        except Exception as e:
            print(f"⚠️  读取锁文件失败: {e}")

def main():
    print("GPU状态检查工具\n")
    
    check_system_gpus()
    check_gpu_processes()
    check_gpu_locks()
    
    print("\n=== 环境变量 ===")
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'PYTORCH_CUDA_ALLOC_CONF']
    for var in cuda_vars:
        value = os.environ.get(var, '未设置')
        print(f"{var}: {value}")

if __name__ == '__main__':
    main()

