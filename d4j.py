import os, sys
# 在导入torch前设置GPU，让整个进程只看到指定的GPU
if len(sys.argv) == 6:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]
import json
import torch
import time
import fcntl
import psutil
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from peft import PeftModel
import re

# 修复 vLLM 多进程问题
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass


def setup_gpu_environment():
    """设置GPU环境，包括冲突检测和预防"""
    if len(sys.argv) != 6:
        print("Usage: python d4j.py <model_key> <num_processes> <process_id> <gpu_id> <num_generations>")
        print("参数说明:")
        print("  model_key: 模型名称")
        print("  num_processes: 总进程数")
        print("  process_id: 当前进程ID (0开始)")
        print("  gpu_id: GPU编号 (0开始)")
        print("  num_generations: 生成次数")
        sys.exit(1)
    
    gpu_id = sys.argv[4]
    
    # 验证GPU编号格式
    try:
        gpu_num = int(gpu_id)
        if gpu_num < 0:
            raise ValueError("GPU编号不能为负数")
    except ValueError as e:
        print(f"错误：无效的GPU编号 '{gpu_id}': {e}")
        sys.exit(1)
    
    # 由于CUDA_VISIBLE_DEVICES已设置，PyTorch只能看到一个GPU（编号为0）
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"物理GPU {gpu_id} 已映射为 cuda:0 (PyTorch可见GPU数: {gpu_count})")
        
        # 测试GPU是否可用
        try:
            test_tensor = torch.tensor([1.0]).cuda(0)
            print(f"GPU 测试成功: {test_tensor.device}")
            del test_tensor
        except Exception as e:
            print(f"GPU 测试失败: {e}")
            sys.exit(1)
    else:
        print("错误：CUDA不可用")
        sys.exit(1)
    
    # GPU冲突检测和预防
    lock_file = f"/tmp/gpu_{gpu_id}_lock"
    try:
        # 尝试获取GPU锁
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        
        # 写入进程信息
        process_info = f"PID:{os.getpid()},TIME:{time.time()},SCRIPT:{sys.argv[0]}\n"
        os.write(lock_fd, process_info.encode())
        os.close(lock_fd)
        
        print(f"成功获取物理GPU {gpu_id} 的锁")
        
        # 注册清理函数
        import atexit
        def cleanup_gpu_lock():
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    print(f"已释放GPU {gpu_id} 锁")
            except:
                pass
        atexit.register(cleanup_gpu_lock)
        
    except FileExistsError:
        # GPU已被占用，检查占用进程状态
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read().strip()
            
            # 解析锁信息
            if lock_info.startswith("PID:"):
                parts = lock_info.split(',')
                pid_str = parts[0].split(':')[1]
                lock_pid = int(pid_str)
                
                # 检查进程是否还存在
                if psutil.pid_exists(lock_pid):
                    proc = psutil.Process(lock_pid)
                    if proc.is_running():
                        print(f"错误：GPU {gpu_id} 已被进程 {lock_pid} 占用")
                        print(f"占用进程信息: {proc.name()} (状态: {proc.status()})")
                        print("请等待该进程完成或手动终止该进程")
                        sys.exit(1)
                    else:
                        print(f"检测到僵尸锁文件，进程 {lock_pid} 已不存在，清理锁文件")
                        os.remove(lock_file)
                        return setup_gpu_environment()  # 递归重试
                else:
                    print(f"检测到过期锁文件，进程 {lock_pid} 已不存在，清理锁文件")
                    os.remove(lock_file)
                    return setup_gpu_environment()  # 递归重试
            else:
                print(f"检测到格式错误的锁文件，清理并重试")
                os.remove(lock_file)
                return setup_gpu_environment()  # 递归重试
                
        except Exception as e:
            print(f"检查GPU锁时出错: {e}")
            print("建议手动检查并清理 /tmp/gpu_*_lock 文件")
            sys.exit(1)
    
    except Exception as e:
        print(f"设置GPU锁时出错: {e}")
        sys.exit(1)
    
    print(f"GPU设置完成")


# 在导入其他模块前先设置GPU环境
if __name__ == '__main__':
    setup_gpu_environment()

# vLLM 环境变量优化 - WSL2兼容性
os.environ['VLLM_USE_MODELSCOPE'] = '0'  # 使用数字格式
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'  # 使用数字格式
# WSL2 CUDA 兼容性设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用以便调试
# os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'  # 让 vLLM 自动选择注意力后端
os.environ['VLLM_USE_V1'] = '0'  # 禁用V1引擎，使用更稳定的V0
# 离线模式设置 - 避免连接HuggingFace Hub
os.environ['HF_HUB_OFFLINE'] = '1'  # HuggingFace Hub离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Transformers离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'  # Datasets离线模式

import importlib
VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

# 性能优化设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 充分利用显存的优化设置
torch.backends.cuda.enable_flash_sdp(True)  # 启用 Flash Scaled Dot Product Attention
torch.backends.cuda.enable_math_sdp(True)   # 启用数学优化的注意力
torch.backends.cuda.enable_mem_efficient_sdp(True)  # 启用内存高效的注意力
torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
torch.backends.cudnn.deterministic = False  # 禁用确定性以获得更好性能
# 设置显存分配策略，充分利用大显存
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
# 如果支持 Flash Attention，启用相关优化
FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    # 尝试导入核心函数来验证安装
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print(f"FlashAttention 可用 - 版本: {flash_attn.__version__}")
except ImportError as e:
    print(f"FlashAttention 导入失败: {str(e)[:100]}...")
    print("原因可能是: CUDA 版本不匹配或编译问题")
    print("使用 PyTorch 内置的优化注意力机制")
except Exception as e:
    print(f"FlashAttention 初始化失败: {str(e)[:100]}...")
    print("使用 PyTorch 内置的优化注意力机制")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全局变量声明
model = None
tokenizer = None
USE_VLLM = False
BOF = None
EOF = None


def extract_first_java_code(s: str) -> str:
    matches = re.findall(r'```java(.*?)```', s, re.DOTALL)
    return matches[0].strip() if matches else ""


def monitor_gpu_status():
    """监控GPU状态，提供详细的GPU使用信息"""
    if not torch.cuda.is_available():
        return "CUDA不可用"
    
    try:
        # 获取当前使用的GPU设备
        current_device = torch.cuda.current_device()
        
        # 获取GPU信息
        gpu_name = torch.cuda.get_device_name(current_device)
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        cached_memory = torch.cuda.memory_reserved(current_device)
        
        # 计算使用率
        memory_usage = (allocated_memory / total_memory) * 100
        cache_usage = (cached_memory / total_memory) * 100
        
        # 显示GPU状态
        status = f"当前GPU cuda:{current_device} ({gpu_name}): "
        status += f"内存使用 {allocated_memory/1024**3:.1f}GB/{total_memory/1024**3:.1f}GB ({memory_usage:.1f}%), "
        status += f"缓存 {cached_memory/1024**3:.1f}GB ({cache_usage:.1f}%)"
        
        return status
    except Exception as e:
        return f"GPU状态监控失败: {e}"


def cleanup_resources():
    """清理资源，避免文件句柄泄露"""
    global model, tokenizer
    try:
        print("开始清理资源...")
        print(f"清理前GPU状态: {monitor_gpu_status()}")
        
        if USE_VLLM and model is not None:
            # vLLM 模型清理
            if hasattr(model, 'llm_engine') and model.llm_engine is not None:
                try:
                    model.llm_engine.shutdown()
                except:
                    pass
            del model
            model = None
            print("vLLM 模型资源已清理")
        elif model is not None:
            # 标准模型清理
            del model
            model = None
            print("标准模型资源已清理")
        
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
            
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"清理后GPU状态: {monitor_gpu_status()}")
            
    except Exception as e:
        print(f"资源清理时出错: {e}")


def set_file_limits():
    """设置文件句柄限制"""
    try:
        import resource
        # 获取当前限制
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"当前文件句柄限制: soft={soft}, hard={hard}")
        
        # 尝试提高软限制到硬限制
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"已提高文件句柄限制到: {hard}")
    except ImportError:
        print("无法导入resource模块，跳过文件句柄限制设置")
    except Exception as e:
        print(f"设置文件句柄限制时出错: {e}")



# 设置模型别名前缀对应的开头与结尾提示
MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>'),
    'codellama': ('[INST]', '[/INST]'),
    'llama': ('[INST]', '[/INST]'),
    'mistral': ('[INST]', '[/INST]'),
    'starchat': ('<|system|>\n<|end|>\n<|user|>', '<|end|>\n<|assistant|>'),
}

# 模型配置：支持基础模型和训练后模型两种版本
MODEL_CONFIGS = {
    # Qwen3 模型
    'qwen3-8b': {
        'base_model': '/data1/czj/model/qwen3-8b',
        'adapter_path': None
    },
    'qwen3-8b-trained': {
        'base_model': '/data1/czj/model/qwen3-8b',
        'adapter_path': '/data1/czj/model/trained_model_qwen3'
    },

    # CodeLlama 模型
    'codellama-7b': {
        'base_model': 'codellama/CodeLlama-7b-Instruct-hf',
        'adapter_path': None
    },
    'codellama-7b-trained': {
        'base_model': '/data1/czj/model/CodeLlama-7b-Instruct',
        'adapter_path': '/data1/czj/model/trained_model_llama'
    },

    'codellama-13b': {
        'base_model': '/data1/czj/model/codeLlama-13b-instruct',
        'adapter_path': None
    },
    'trained_model_codellama-v1': {
        'base_model': '/data1/czj/model/codeLlama-13b-instruct',
        'adapter_path': '/data1/czj/model/trained_model_codellama_13b-v1'
    },
    # Llama3.1 模型
    'llama3.1-8b': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': None
    },
    'llama3.1-8b-trained': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': '/data1/czj/model/trained_model_llama'
    },
    'llama3.1-8b-trained-v1': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': '/data1/czj/model/trained_model_llama-v1'
    },
    'sft-grpo-llama': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': '/data1/czj/model/trained_sft_grpo_llama_3_8b_instruct/sft'
    },

}


def get_prompt_format(model_key):
    for key in MODEL_PROMPT_FORMATS:
        if model_key.startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    # 默认格式
    return '[INST]', '[/INST]'


def merge_lora_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """
    合并 LoRA 适配器到基础模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: LoRA 适配器路径
        output_path: 合并后模型的输出路径
    
    Returns:
        str: 合并后的模型路径
    """
    print("=" * 60)
    print("🔄 检测到 LoRA 适配器，开始自动合并...")
    print("=" * 60)
    
    # 检查是否已经合并过
    if os.path.exists(output_path):
        config_file = os.path.join(output_path, "config.json")
        if os.path.exists(config_file):
            print(f"✅ 发现已合并的模型: {output_path}")
            print("   跳过合并步骤，直接使用已合并模型")
            return output_path
    
    print(f"📦 基础模型: {base_model_path}")
    print(f"🔗 LoRA 适配器: {adapter_path}")
    print(f"💾 输出路径: {output_path}")
    print()
    
    try:
        from peft import PeftModel
        
        print("📦 [1/4] 加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ 基础模型加载成功")
        
        print("\n🔗 [2/4] 加载 LoRA 适配器...")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16
        )
        print("✅ LoRA 适配器加载成功")
        
        print("\n🔄 [3/4] 合并 LoRA 权重...")
        model = model.merge_and_unload()
        print("✅ 合并成功")
        
        print(f"\n💾 [4/4] 保存合并后的模型到 {output_path}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # 保存 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
        print("✅ 保存成功")
        
        # 清理内存
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("✅ LoRA 合并完成！")
        print("=" * 60)
        return output_path
        
    except ImportError:
        print("❌ 错误：未安装 peft 库")
        print("请运行: pip install peft")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_vllm_model(model_config):
    from vllm import LLM
    """使用 vLLM 加载模型"""
    base_model_path = model_config['base_model']
    adapter_path = model_config.get('adapter_path')
    
    # 如果有 LoRA 适配器，先合并
    if adapter_path and os.path.exists(adapter_path):
        # 检查是否是 LoRA 适配器（检查 adapter_config.json）
        adapter_config_file = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_file):
            print(f"🔍 检测到 LoRA 适配器: {adapter_path}")
            
            # 生成合并后模型的路径
            adapter_name = os.path.basename(adapter_path.rstrip('/'))
            merged_model_path = os.path.join(
                os.path.dirname(adapter_path),
                f"{adapter_name}_merged"
            )
            
            # 合并 LoRA
            actual_model_path = merge_lora_adapter(
                base_model_path,
                adapter_path,
                merged_model_path
            )
            
            # 使用合并后的模型
            base_model_path = actual_model_path
            print(f"\n🚀 使用合并后的模型: {base_model_path}\n")
        else:
            print(f"⚠️ 路径存在但不是 LoRA 适配器，尝试直接加载: {adapter_path}")
    
    # vLLM 配置
    vllm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "max_model_len": 4096,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "disable_log_stats": True,
        "download_dir": None,
        "tensor_parallel_size": 1,
    }
    
    print(f"加载vLLM模型: {base_model_path}")
    llm = LLM(**vllm_config)
    
    # 加载对应的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return llm, tokenizer



def main():
    """主函数"""
    # 设置文件句柄限制
    set_file_limits()
    
    # 设置prompt格式
    global BOF, EOF, model, tokenizer, USE_VLLM
    BOF, EOF = get_prompt_format(sys.argv[1])

    try:
        # 检查vLLM是否可用
        if not VLLM_AVAILABLE:
            print("错误：vLLM未安装，请先安装vLLM")
            sys.exit(1)
        
        # 加载模型
        if sys.argv[1] in MODEL_CONFIGS:
            model_config = MODEL_CONFIGS[sys.argv[1]]
            model, tokenizer = load_vllm_model(model_config)
            USE_VLLM = True
        else:
            print(f"错误：未知的模型 '{sys.argv[1]}'")
            print(f"可用模型: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

        print('模型加载成功', flush=True)
        print(f"GPU状态: {monitor_gpu_status()}")
        
        # 执行主要处理逻辑
        process_files()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源清理
        cleanup_resources()

def process_files():
    """处理文件的主要逻辑"""
    base_dir = 'defects4j/dataset'
    base_fix_dir = f'defects4j/results/{sys.argv[1]}'

    cnt = 0

    try:
        for file_path in sorted(Path(base_dir).rglob('*.json'), reverse=True):  # 遍历 base_dir 目录下所有 .json 文件，并按文件名降序排序
            cnt += 1  # 计数器自增，用于分配任务
            if cnt % int(sys.argv[2]) != int(sys.argv[3]):  # 根据 sys.argv 传入的参数，判断当前文件是否属于当前进程需要处理的任务
                continue  # 如果不是当前进程要处理的文件，则跳过

            # 获取文件的完整路径
            full_path = str(file_path)
            print(full_path, flush=True)

            # 使用 with 语句确保文件正确关闭
            try:
                with open(full_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                json_data = json.loads(content)
                result_data = json_data.copy()  # 创建副本避免修改原数据

                for e in range(int(sys.argv[5])):  # 更新为sys.argv[5]，因为现在num_generations是第5个参数
                    # 创建固定的输出目录
                    fix_dir = f'{base_fix_dir}/fixed{e}'
                    os.makedirs(fix_dir, exist_ok=True)

                    # 获取文件名
                    file_name = os.path.basename(full_path)
                    fix_name = os.path.join(fix_dir, file_name)
                    print(f"Output path: {fix_name}", flush=True)
                    
                    if os.path.exists(fix_name) and os.path.exists(fix_name + '.log'):
                        print('result exists ...')
                        continue
                    
                    try:
                        full, res = cal(file_name.split('.')[0], json_data['buggy'], json_data['issue_title'],
                                        json_data['issue_description'], json_data['loc'])
                        if full is None:  # 如果 full 为 None，则说明修复失败，跳过此轮
                            print(f"修复失败，跳过 {file_name}")
                            continue
                        
                        result_data['fix'] = res  # 将修复后的代码存入 JSON 数据
                        
                        # 使用 with 语句确保文件正确关闭
                        with open(fix_name, 'w', encoding='utf-8') as output_file:
                            json.dump(result_data, output_file, indent=2, ensure_ascii=False)
                        
                        with open(fix_name + '.log', 'w', encoding='utf-8') as log_file:
                            print(full, file=log_file)
                            
                    except Exception as e:
                        print(f"处理文件 {file_name} 时出错: {e}")
                        continue
                        
            except (IOError, json.JSONDecodeError) as e:
                print(f"读取文件 {full_path} 时出错: {e}")
                continue
                
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"处理过程中出现未预期错误: {e}")
    finally:
        # 清理资源
        cleanup_resources()


def cal_vllm(bug_id, code, title, description, filename):
    """使用 vLLM 进行高性能推理"""
    from vllm import SamplingParams
    try:
        prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
        print(prompt, flush=True)
        
        # vLLM 采样参数 - 保守配置
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            max_tokens=512,  # 降低token数量减少显存压力
            repetition_penalty=1.1,
            stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
        )
        
        # vLLM 生成 - 添加错误处理
        try:
            outputs = model.generate([prompt], sampling_params)
            output = outputs[0]
            full_text = output.outputs[0].text
        except RuntimeError as e:
            if "CUDA" in str(e) or "unknown error" in str(e):
                print(f"vLLM CUDA错误，尝试清理缓存后重试: {e}")
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # 重试一次
                try:
                    outputs = model.generate([prompt], sampling_params)
                    output = outputs[0]
                    full_text = output.outputs[0].text
                except Exception as e2:
                    print(f"vLLM重试失败: {e2}")
                    return [None, None]
            else:
                print(f"vLLM生成错误: {e}")
                return [None, None]
        
        # 完整文本包含prompt
        complete_text = prompt + full_text
        print(complete_text)
        
        # 根据模型格式提取生成的 Java 代码部分
        try:
            ret = extract_first_java_code(full_text.split(EOF)[-1])
        except (IndexError, AttributeError):
            ret = extract_first_java_code(full_text)
        
        print('code:', ret, flush=True)
        return [complete_text, ret]
        
    except Exception as e:
        print(f"cal_vllm 执行出错: {e}")
        return [None, None]

def cal_vllm_batch(prompts):
    """vLLM 批处理推理 - 充分利用显存和并发"""
    from vllm import SamplingParams
    print(f"vLLM 批处理: {len(prompts)} 个样本")
    
    # vLLM 采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,
        repetition_penalty=1.1,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    
    # vLLM 批量生成 - 自动并发处理
    outputs = model.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        full_text = output.outputs[0].text
        complete_text = prompts[i] + full_text
        
        # 提取 Java 代码
        try:
            ret = extract_first_java_code(full_text.split('[/INST]')[-1])
        except IndexError:
            ret = extract_first_java_code(full_text)
        
        results.append([complete_text, ret])
        print(f"vLLM 批处理样本 {i+1} 完成")
    
    return results

def cal(bug_id, code, title, description, filename):
    """推理接口 - 使用vLLM"""
    return cal_vllm(bug_id, code, title, description, filename)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python d4j.py <model_key> <num_processes> <process_id> <gpu_id> <num_generations>")
        print("参数说明:")
        print("  model_key: 模型名称")
        print("  num_processes: 总进程数")
        print("  process_id: 当前进程ID (0开始)")
        print("  gpu_id: GPU编号 (0开始)")
        print("  num_generations: 生成次数")
        sys.exit(1)
    main()
