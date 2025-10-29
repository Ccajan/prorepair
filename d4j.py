import os, sys
import json

# 模型配置：支持基础模型和训练后模型两种版本（提前定义用于GPU判断）
MODEL_CONFIGS = {
    # DeepSeek API 模型
    'deepseek-v3': {
        'type': 'api',
        'model_name': 'deepseek-chat',
        'api_url': 'https://api.deepseek.com/v1/chat/completions',
        'api_key_env': 'DEEPSEEK_API_KEY',
    },
    
    # Qwen API 模型
    'qwen3-max': {
        'type': 'api',
        'model_name': 'qwen-max',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        'api_key_env': 'DASHSCOPE_API_KEY',  # 环境变量名称，不是 API Key 本身
    },
    
    # Qwen3 本地模型
     'qwen3-8b': {
        'model_path': 'model/qwen3-8b',
    },

    'qwen3-4b': {
        'model_path': 'model/qwen3-4b',
    },

    'qwen3-8b-trained': {
        'model_path': 'merged_models/qwen3-8b_merged_sft',
    },
    'qwen3-4b-trained': {
        'model_path': 'merged_models/qwen3-4b_merged_sft',
    },
    'qwen3-4b-trained-noprompt': {
        'model_path': 'merged_models/qwen3-4b-trained-noprompt',
    },
    'qwen3-4b-trained-useprompt': {
        'model_path': 'merged_models/qwen3-4b-trained-useprompt',
    },
    # Llama3.1 模型
    'llama3-8b': {
        'model_path': 'model/Llama-3-8B-Instruct',
    },
    'llama3-8b-trained': {
        'model_path': 'merged_models/llama3.1-8b-merged',
    },
    
    # DeepSeek 模型
    'deepseek-6.7b': {
        'model_path': 'model/deepseek-coder-6.7b-instruct',
    },
    'deepseek-1b': {
        'model_path': 'model/deepseek-1b',
    },
    'deepseek-6.7b-trained': {
        'model_path': 'merged_models/deepseek-6.7b-merged',
    },
    
    # StarCoder 模型（代码补全模型）
    'starcoder-7b': {
        'model_path': 'model/starcoder-7b',
    },
    'starcoder-7b-nopro': {
        'model_path': 'merged_models/starcoder-7b-nopro/starcoder-7b-nopro',
    },
    'starcoder-7b-par': {
        'model_path': 'merged_models/starcoder-7b-par',
    },

}

# 在导入torch前设置GPU，让整个进程只看到指定的GPU
# 但在重新提取模式或API模式下跳过GPU设置
if len(sys.argv) >= 6:
    # 检查是否为重新提取模式或API模式
    is_reextract = len(sys.argv) == 7 and sys.argv[6] == '--reextract'
    # 通过 MODEL_CONFIGS 判断是否为 API 模型
    is_api_model = (len(sys.argv) >= 2 and 
                    sys.argv[1] in MODEL_CONFIGS and 
                    MODEL_CONFIGS[sys.argv[1]].get('type') == 'api')
    
    if not is_reextract and not is_api_model:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]

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
import requests

# 修复 vLLM 多进程问题
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass


def setup_gpu_environment():
    """设置GPU环境，包括冲突检测和预防"""
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print("Usage: python d4j.py <model_key> <num_processes> <process_id> <gpu_id> <num_generations> [--reextract]")
        print("参数说明:")
        print("  model_key: 模型名称")
        print("  num_processes: 总进程数")
        print("  process_id: 当前进程ID (0开始)")
        print("  gpu_id: GPU编号 (0开始)")
        print("  num_generations: 生成次数")
        print("  --reextract: (可选) 从已有的log文件重新提取代码，不重新生成")
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
    # 使用项目目录下的锁文件，避免 /tmp 空间不足
    lock_dir = os.path.join(os.getcwd(), '.gpu_locks')
    os.makedirs(lock_dir, exist_ok=True)
    lock_file = os.path.join(lock_dir, f"gpu_{gpu_id}.lock")
    
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
            print(f"建议手动检查并清理 {lock_dir} 目录下的锁文件")
            sys.exit(1)

    except Exception as e:
        print(f"设置GPU锁时出错: {e}")
        sys.exit(1)

    print(f"GPU设置完成")


# 在导入其他模块前先设置GPU环境
if __name__ == '__main__':
    # 如果是重新提取模式或API模式，跳过GPU设置
    reextract_mode = len(sys.argv) == 7 and sys.argv[6] == '--reextract'
    # 通过 MODEL_CONFIGS 判断是否为 API 模式
    is_api_mode = (len(sys.argv) >= 2 and 
                   sys.argv[1] in MODEL_CONFIGS and 
                   MODEL_CONFIGS[sys.argv[1]].get('type') == 'api')
    
    if not reextract_mode and not is_api_mode:
        setup_gpu_environment()
    else:
        if reextract_mode:
            print("重新提取模式：跳过GPU环境设置")
        if is_api_mode:
            print("API模式：跳过GPU环境设置")

# vLLM 环境变量优化 - WSL2兼容性
os.environ['VLLM_USE_MODELSCOPE'] = '0'  # 使用数字格式
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'  # 使用数字格式
# 将 vllm 缓存和临时文件重定向到 /data1，避免根分区空间不足
os.environ['VLLM_CACHE_ROOT'] = '/data1/vllm_cache'
os.environ['TMPDIR'] = '/data1/tmp'
os.environ['TEMP'] = '/data1/tmp'
os.environ['TMP'] = '/data1/tmp'
# 禁用 vLLM 使用统计收集，避免磁盘空间不足问题
os.environ['VLLM_DO_NOT_TRACK'] = '1'  # 禁用使用统计跟踪
os.environ['VLLM_USAGE_STATS_SERVER'] = ''  # 禁用使用统计服务器
# 确保目录存在
os.makedirs('/data1/tmp', exist_ok=True)
os.makedirs('/data1/vllm_cache', exist_ok=True)
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
torch.backends.cuda.enable_math_sdp(True)  # 启用数学优化的注意力
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
    # 首先尝试匹配完整的代码块（有闭合标记）
    matches = re.findall(r'```java(.*?)```', s, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 如果没有闭合标记，尝试匹配从 ```java 开始到字符串结尾或到下一个 ``` 的内容
    match = re.search(r'```java\s*(.*?)(?:```|$)', s, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


def reextract_code_from_log(log_file_path):
    """从log文件中重新提取Java代码（从模型输出部分）"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # 找到模型输出的开始位置（在 EOF 标记之后）
        # 使用与 MODEL_PROMPT_FORMATS 一致的分隔符
        output_part = None
        separators = [
            # Qwen
            '<|im_end|>\n<|im_start|>assistant\n',
            '<|im_end|>',  # 兼容旧格式
            # Llama 3
            '<|end_of_text|>\n\n',
            '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',  # 标准 Llama 3.1
            # Llama 2
            '[/INST]',
            # DeepSeek - 由于EOF太短(只是\n)，使用BOF+特征定位
            '<|begin▁of▁sentence|>:\n',
            # 其他
            '<|end|>\n<|assistant|>',  # StarChat
            '### Response:\n',  # 通用 Instruct 格式
        ]
        for separator in separators:
            if separator in log_content:
                parts = log_content.split(separator)
                if len(parts) > 1:
                    output_part = parts[-1]  # 取最后一部分作为模型输出
                    break

        # 如果没有找到分隔符，尝试查找 "```java" 出现的位置
        # 假设第二个 ```java 是模型输出的代码
        if output_part is None:
            java_blocks = list(re.finditer(r'```java', log_content))
            if len(java_blocks) >= 2:
                # 从第二个 ```java 开始提取
                output_part = log_content[java_blocks[1].start():]
            else:
                print(f"警告: 无法在 {log_file_path} 中找到模型输出分隔符")
                return None

        # 从输出部分提取代码
        code = extract_first_java_code(output_part)

        if code:
            print(f"成功从 {log_file_path} 提取代码 (长度: {len(code)} 字符)")
            return code
        else:
            print(f"警告: 无法从 {log_file_path} 的输出部分提取代码")
            return None
    except Exception as e:
        print(f"读取log文件 {log_file_path} 时出错: {e}")
        return None


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
        status += f"内存使用 {allocated_memory / 1024 ** 3:.1f}GB/{total_memory / 1024 ** 3:.1f}GB ({memory_usage:.1f}%), "
        status += f"缓存 {cached_memory / 1024 ** 3:.1f}GB ({cache_usage:.1f}%)"

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


# 设置模型别名前缀对应的开头与结尾提示（必须与训练时格式一致！）
MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'),
    'llama3': ('<|begin_of_text|>\n\n', '<|end_of_text|>\n\n'),
    'deepseek': ('<|begin▁of▁sentence|>:\n', '\n'),  # DeepSeek Instruct 格式（需要验证）
    'starcoder': (None, None),  # StarCoder 不使用特殊格式，直接代码补全
}


def get_prompt_format(model_key):
    for key in MODEL_PROMPT_FORMATS:
        if model_key.startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    # 默认格式
    return '[INST]', '[/INST]'


def is_completion_model(model_key):
    """判断是否为代码补全模型（不需要指令格式）"""
    completion_prefixes = ['starcoder', 'starcode']
    return any(model_key.startswith(prefix) for prefix in completion_prefixes)


def load_vllm_model(model_config):
    from vllm import LLM
    """使用 vLLM 加载模型"""
    base_model_path = model_config['model_path']
    adapter_path = model_config.get('adapter_path')

    # 如果有 LoRA 适配器，报错退出
    if adapter_path and os.path.exists(adapter_path):
        print(f"错误：检测到 LoRA 适配器: {adapter_path}")
        print("vLLM 不支持 LoRA 适配器，请使用预合并模型")
        sys.exit(1)

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
        # 移除了 disable_usage_stats 参数，因为不是所有vLLM版本都支持
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

    # 检查是否为重新提取模式
    reextract_mode = len(sys.argv) == 7 and sys.argv[6] == '--reextract'

    # 检查是否为 API 模式
    model_key = sys.argv[1]
    model_config = MODEL_CONFIGS.get(model_key, {})
    is_api_mode = model_config.get('type') == 'api'

    # 设置prompt格式
    global BOF, EOF, model, tokenizer, USE_VLLM
    BOF, EOF = get_prompt_format(sys.argv[1])

    try:
        # 重新提取模式或API模式不需要加载本地模型
        if reextract_mode:
            print("重新提取模式：跳过模型加载")
            print("将直接从log文件提取代码...")
        elif is_api_mode:
            print(f"API模式：使用 {model_key}")
            print("跳过本地模型加载...")
            # 检查API key
            api_key_env = model_config.get('api_key_env', 'API_KEY')
            if not os.environ.get(api_key_env):
                print(f"警告：未设置 {api_key_env} 环境变量")
                print(f"请设置环境变量: export {api_key_env}='your-api-key'")
        else:
            # 检查vLLM是否可用
            if not VLLM_AVAILABLE:
                print("错误：vLLM未安装，请先安装vLLM")
                sys.exit(1)

            # 加载模型
            if sys.argv[1] in MODEL_CONFIGS:
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

    # 检查是否为重新提取模式
    reextract_mode = len(sys.argv) == 7 and sys.argv[6] == '--reextract'

    if reextract_mode:
        print("=" * 60)
        print("运行模式: 重新提取代码模式")
        print("将从已有的log文件中重新提取Java代码并保存到JSON")
        print("=" * 60)

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
                    log_name = fix_name + '.log'
                    print(f"Output path: {fix_name}", flush=True)

                    # 重新提取模式
                    if reextract_mode:
                        if os.path.exists(log_name):
                            print(f"重新提取模式: 处理 {log_name}")
                            # 从log文件提取代码
                            extracted_code = reextract_code_from_log(log_name)
                            if extracted_code:
                                result_data['fix'] = extracted_code
                                # 保存JSON文件
                                with open(fix_name, 'w', encoding='utf-8') as output_file:
                                    json.dump(result_data, output_file, indent=2, ensure_ascii=False)
                                print(f"✓ 已更新 {fix_name}")
                            else:
                                print(f"✗ 跳过 {file_name} (无法提取代码)")
                        else:
                            print(f"跳过 {file_name} (log文件不存在)")
                        continue

                    # 正常生成模式
                    if os.path.exists(fix_name) and os.path.exists(log_name):
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

                        with open(log_name, 'w', encoding='utf-8') as log_file:
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


def cal_api(bug_id, code, title, description, filename):
    """使用 API 进行推理（支持 DeepSeek、Qwen 等多种 API）"""
    # 从命令行参数获取模型key
    model_key = sys.argv[1]
    model_config = MODEL_CONFIGS.get(model_key)
    if not model_config:
        print(f"错误：未找到模型配置 {model_key}")
        return [None, None]
    
    # 获取 API 配置
    api_url = model_config['api_url']
    model_name = model_config['model_name']
    api_key_env = model_config.get('api_key_env', 'API_KEY')
    
    # 检查 API Key
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"错误：未设置 {api_key_env} 环境变量")
        print(f"请设置环境变量: export {api_key_env}='your-api-key'")
        return [None, None]
    
    prompt = f"# {title}\n{description}\n\nThis is an incorrect code ({filename}):\n```java\n{code}\n```\n\nYou are a software engineer. Can you repair the incorrect code? Please provide only the fixed Java code in a ```java code block."
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7,
        'max_tokens': 2048
    }
    
    try:
        print(f"调用 API: {model_key} ({model_name})", flush=True)
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        full_text = result['choices'][0]['message']['content']
        print(full_text, flush=True)
        
        # 提取 Java 代码
        ret = extract_first_java_code(full_text)
        print('code:', ret, flush=True)
        
        # 构造完整文本（模拟prompt+response格式）
        complete_text = f"{prompt}\n\nAssistant:\n{full_text}"
        
        return [complete_text, ret]
        
    except requests.exceptions.RequestException as e:
        print(f"{model_key} API 调用失败: {e}")
        return [None, None]
    except Exception as e:
        print(f"{model_key} API 处理错误: {e}")
        return [None, None]


def cal_vllm(bug_id, code, title, description, filename):
    """使用 vLLM 进行高性能推理"""
    from vllm import SamplingParams
    try:
        # 获取模型key
        model_key = sys.argv[1]
        
        # 根据模型类型构建不同的prompt
        if is_completion_model(model_key):
            # 代码补全模型：使用代码块标记，便于提取代码
            prompt = f" # {title}\n{description}\nThis is an incorrect code ({filename}):\n```java\n{code}\n```\n\n//You are a software engineer.  can you repair the incorrect java code?. Only output the fixed code\n```java\n"
        else:
            # 指令模型：使用BOF/EOF包装
            if BOF is None or EOF is None:
                # 如果没有设置BOF/EOF，回退到代码补全格式
                prompt = f"// Bug: # {title}\n{description}\nThis is an incorrect code ({filename}):\n```java\n{code}\n```\n\n// Fixed code:\n```java\n"
            else:
                prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer.  can you repair the incorrect java code?. Only output the fixed code. \n" + EOF + "\n```java\n"
        
        print(prompt, flush=True)

        # vLLM 采样参数 - 保守配置
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            max_tokens=2048,  # 降低token数量减少显存压力
            repetition_penalty=1.1,
            stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
        )

        # 循环生成，直到获取到合法的 Java 代码
        max_retries = 5
        retry_count = 0
        ret = ""
        complete_text = None
        model_key = sys.argv[1]
        
        while not ret and retry_count < max_retries:
            retry_count += 1
            print(f"尝试生成 Java 代码 (第 {retry_count}/{max_retries} 次)...", flush=True)
            
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
                        print(f"❌ vLLM 重试失败: {e2}", flush=True)
                        continue
                else:
                    print(f"❌ vLLM 生成错误: {e}", flush=True)
                    continue
            except Exception as e:
                print(f"❌ vLLM 生成异常: {e}", flush=True)
                continue
            
            # 组合完整文本
            complete_text = prompt + full_text
            
            # 根据模型类型提取代码
            ret = None
            
            if is_completion_model(model_key) or BOF is None or EOF is None:
                # 代码补全模型：从 "output the fixed code" 标记之后提取Java代码块
                marker = "output the fixed code"
                marker_pos = full_text.lower().find(marker)
                
                if marker_pos != -1:
                    # 找到标记，从标记之后提取
                    after_marker = full_text[marker_pos + len(marker):]
                    ret = extract_first_java_code(after_marker)
                else:
                    # 没找到标记，返回空，依靠重试机制
                    ret = None
                
                if not ret:
                    print(f"⚠️ 补全模型未生成有效代码，继续重试...", flush=True)
                    continue
            else:
                # 指令模型：从 EOF 标记之后提取
                if EOF in complete_text:
                    try:
                        after_eof = complete_text.split(EOF)[1]
                        ret = extract_first_java_code(after_eof)
                    except (IndexError, AttributeError) as e:
                        print(f"⚠️ 从 EOF 后提取失败: {e}", flush=True)
                        ret = None
                else:
                    print(f"⚠️ 未找到 EOF 标记，继续重试...", flush=True)
                    ret = None
            
            if ret:
                print(f'✅ 成功提取 Java 代码 (第 {retry_count} 次尝试)', flush=True)
                print('code:', ret, flush=True)
            else:
                print(f"⚠️ 提取的代码为空或无效，继续重试...", flush=True)
                ret = ""  # 重置为空字符串，继续循环
        
        # 检查最终结果
        if not ret or not ret.strip():
            print(f"❌ 经过 {max_retries} 次尝试仍未获取到有效的 Java 代码", flush=True)
            return [None, None]
        
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

        # 提取 Java 代码（从 EOF 之后）
        ret = extract_first_java_code(complete_text.split(EOF)[1])

        results.append([complete_text, ret])
        print(f"vLLM 批处理样本 {i + 1} 完成")

    return results


def cal(bug_id, code, title, description, filename):
    """推理接口 - 根据模型类型选择调用方式"""
    model_key = sys.argv[1]
    model_config = MODEL_CONFIGS.get(model_key, {})
    
    if model_config.get('type') == 'api':
        return cal_api(bug_id, code, title, description, filename)
    else:
        return cal_vllm(bug_id, code, title, description, filename)


if __name__ == '__main__':
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print("Usage: python d4j.py <model_key> <num_processes> <process_id> <gpu_id> <num_generations> [--reextract]")
        print("\n参数说明:")
        print("  model_key: 模型名称")
        print("  num_processes: 总进程数")
        print("  process_id: 当前进程ID (0开始)")
        print("  gpu_id: GPU编号 (0开始，API模式会被忽略)")
        print("  num_generations: 生成次数")
        print("  --reextract: (可选) 从已有的log文件重新提取代码，不重新生成")
        print("\n可用模型:")
        print("  API 模型 (无需 GPU):")
        print("    - qwen3-max      (Qwen3 Max API - 需要 DASHSCOPE_API_KEY)")
        print("    - deepseek-v3    (DeepSeek V3 API - 需要 DEEPSEEK_API_KEY)")
        print("  本地模型 (需要 GPU):")
        print("    - qwen3-8b, qwen3-4b")
        print("    - llama3-8b      (Llama 3.1-8B-Instruct)")
        print("    - deepseek-6.7b  (DeepSeek-Coder-6.7B-Instruct)")
        print("    - starcoder-7b   (StarCoder-7B 代码补全模型)")
        print("  训练后模型:")
        print("    - qwen3-8b-trained, qwen3-4b-trained")
        print("    - llama3-8b-trained")
        print("    - deepseek-6.7b-trained")
        print("    - starcoder-7b-trained")
        print("\n示例:")
        print("  # 使用 Qwen3 Max API")
        print("  python d4j.py qwen3-max 1 0 0 10")
        print("  # 使用 Llama 3.1 本地模型")
        print("  python d4j.py llama3-8b 4 0 0 10")
        print("  # 使用 DeepSeek 本地模型")
        print("  python d4j.py deepseek-6.7b 4 0 1 10")
        print("  # 使用训练后的模型")
        print("  python d4j.py llama3-8b-trained 4 0 0 10")
        sys.exit(1)
    main()