import os
import sys
# 在导入torch前设置GPU，让整个进程只看到指定的GPU
if len(sys.argv) >= 5:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]  # gpu_id 是第5个参数（索引4）
import re
import time
import psutil
import torch
from pathlib import Path
from typing import Tuple

# 修复 vLLM 多进程问题
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# vLLM 环境变量优化 - WSL2兼容性
os.environ['VLLM_USE_MODELSCOPE'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['VLLM_USE_V1'] = '0'
# 将 vllm 缓存和临时文件重定向到 /data1，避免根分区空间不足
os.environ['VLLM_CACHE_ROOT'] = '/data1/vllm_cache'
os.environ['TMPDIR'] = '/data1/tmp'
os.environ['TEMP'] = '/data1/tmp'
os.environ['TMP'] = '/data1/tmp'
# 确保目录存在
os.makedirs('/data1/tmp', exist_ok=True)
os.makedirs('/data1/vllm_cache', exist_ok=True)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入 vLLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 性能优化设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'


def setup_gpu_environment():
    """设置GPU环境，包括冲突检测和预防"""
    if len(sys.argv) < 5:
        print("Usage: python inference_java.py <model_key> <num_processes> <process_id> <gpu_id> [num_generations]")
        print("参数说明:")
        print("  model_key: 模型名称")
        print("  num_processes: 总进程数")
        print("  process_id: 当前进程ID (0开始)")
        print("  gpu_id: GPU编号 (0开始)")
        print("  num_generations: 生成次数 (可选，默认10)")
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
    
    # GPU冲突检测和预防（每个进程有独立的锁）
    # 使用项目目录下的锁文件，避免 /tmp 空间不足
    lock_dir = os.path.join(os.getcwd(), '.gpu_locks')
    os.makedirs(lock_dir, exist_ok=True)
    lock_file = os.path.join(lock_dir, f"gpu_{gpu_id}_proc_{sys.argv[3]}_java.lock")
    
    try:
        # 尝试获取GPU锁
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        
        # 写入进程信息
        process_info = f"PID:{os.getpid()},TIME:{time.time()},SCRIPT:{sys.argv[0]}\n"
        os.write(lock_fd, process_info.encode())
        os.close(lock_fd)
        
        print(f"成功获取物理GPU {gpu_id} 的锁 (锁文件: {lock_file})")
        
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
        print(f"锁文件目录: {lock_dir}")
        print(f"如果是空间不足问题，请检查该目录的磁盘空间")
        sys.exit(1)
    
    print(f"GPU设置完成")


def monitor_gpu_status():
    """监控GPU状态，提供详细的GPU使用信息"""
    if not torch.cuda.is_available():
        return "CUDA不可用"
    
    try:
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        cached_memory = torch.cuda.memory_reserved(current_device)
        
        memory_usage = (allocated_memory / total_memory) * 100
        cache_usage = (cached_memory / total_memory) * 100
        
        status = f"当前GPU cuda:{current_device} ({gpu_name}): "
        status += f"内存使用 {allocated_memory/1024**3:.1f}GB/{total_memory/1024**3:.1f}GB ({memory_usage:.1f}%), "
        status += f"缓存 {cached_memory/1024**3:.1f}GB ({cache_usage:.1f}%)"
        
        return status
    except Exception as e:
        return f"GPU状态监控失败: {e}"


def cleanup_resources():
    """清理资源，避免文件句柄泄露"""
    global llm
    try:
        print("开始清理资源...")
        print(f"清理前GPU状态: {monitor_gpu_status()}")
        
        if llm is not None:
            if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
                try:
                    llm.llm_engine.shutdown()
                except:
                    pass
            del llm
            llm = None
            print("vLLM 模型资源已清理")
        
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
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"当前文件句柄限制: soft={soft}, hard={hard}")
        
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"已提高文件句柄限制到: {hard}")
    except ImportError:
        print("无法导入resource模块，跳过文件句柄限制设置")
    except Exception as e:
        print(f"设置文件句柄限制时出错: {e}")


def extract_java_code(text: str) -> str:
    """从生成的文本中提取Java代码（提取最后一个代码块）"""
    # 首先尝试匹配完整的代码块（有闭合标记）
    matches = re.findall(r'```java(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # 返回最后一个代码块
    
    
    return ""

# 模型提示格式（必须与训练时格式一致！）
MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'),
    'llama3': ('<|start_header_id|>user<|end_header_id|>\n\n', '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'),
    'deepseek': ('###Instruction\n', '###response\n\n'),  # DeepSeek 格式
}


# 模型配置
MODEL_CONFIGS = {
    'qwen3-8b': {
        'model_path': 'model/qwen3-8b',
    },
    'qwen3-4b': {
        'model_path': 'model/qwen3-4b',
    },
    'qwen2.5-7b': {
        'model_path': 'model/qwen2.5-7b',
    },
    'qwen3-8b-trained': {
        'model_path': 'merged_models/qwen3-8b_merged_sft',
    },
    'qwen3-4b-trained': {
        'model_path': 'merged_models/qwen3-4b_merged_sft',
    },
    'llama3.1-8b': {
        'model_path': 'model/Llama-3-8B-Instruct',
    },
    'deepseek-6.7b': {
        'model_path': 'model/deepseek-coder-6.7b',
    },
    'deepseek-6.7b-nopro': {
        'model_path': 'merged_models/deepseek-6,7b-nopro',
    },
    'deepseek-6.7b-parepair': {
        'model_path': 'merged_models/deepseek-6.7b-prarepair',
    },

}

def get_prompt_format(model_key: str) -> tuple:
    """获取模型的提示格式"""
    for key in MODEL_PROMPT_FORMATS:
        if model_key.startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    return '[INST]', '[/INST]'  # 默认格式


def load_vllm_model(model_config):
    """使用 vLLM 加载模型"""
    from transformers import AutoTokenizer
    
    model_path = model_config['model_path']
    
    # 加载 tokenizer
    print("📝 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    
    llm_kwargs = {
        'model': model_path,
        'tokenizer': model_path,
        'tensor_parallel_size': 1,
        'dtype': 'bfloat16',
        'max_model_len': 4096,
        'gpu_memory_utilization': 0.9,
        'trust_remote_code': True,
        'enforce_eager': False,
    }
    
    print("🚀 使用 vLLM 加载模型...")
    try:
        llm = LLM(**llm_kwargs)
        print("✅ vLLM 模型加载成功")
    except Exception as e:
        print(f"❌ vLLM 模型加载失败: {e}")
        raise
    
    return llm, tokenizer, None

def generate_with_vllm(llm, prompt: str, model_key: str) -> Tuple[str, str]:
    """使用 vLLM 生成代码"""
    global tokenizer, EOF, BOF
    
    # vLLM 采样参数 - 保守配置
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        max_tokens=512,  # 确保生成完整代码（max_new_tokens，不包括输入）
        repetition_penalty=1.1,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    
    # 循环生成，直到获取到合法的 Java 代码
    max_retries = 5
    retry_count = 0
    ret = ""
    complete_text = None
    
    while not ret and retry_count < max_retries:
        retry_count += 1
        print(f"尝试生成 Java 代码 (第 {retry_count}/{max_retries} 次)...", flush=True)
        
        # vLLM 生成 - 添加错误处理
        try:
            outputs = llm.generate([prompt], sampling_params)
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
                    outputs = llm.generate([prompt], sampling_params)
                    output = outputs[0]
                    full_text = output.outputs[0].text
                except Exception as e2:
                    print(f"vLLM重试失败: {e2}")
                    continue
            else:
                print(f"vLLM生成错误: {e}")
                continue
        except Exception as e:
            print(f"vLLM生成异常: {e}")
            continue
        
        # 组合完整文本
        complete_text = prompt + full_text
        
        # 从 EOF 标记之后提取代码
        ret = None
        if EOF and EOF in complete_text:
            try:
                after_eof = complete_text.split(EOF)[1]
                ret = extract_java_code(after_eof)
            except (IndexError, AttributeError):
                ret = None
        else:
            ret = None
        
        if ret:
            print('code:', ret, flush=True)
        else:
            ret = ""  # 重置为空字符串，继续循环
    
    # 检查最终结果
    if not ret or not ret.strip():
        print(f"❌ 经过 {max_retries} 次尝试仍未获取到有效的 Java 代码", flush=True)
        return "", ""
    
    return complete_text, ret

# 添加 reextract 功能
def reextract_code_from_log(log_file_path, eof_marker=None, model_key=None):
    """
    从log文件中提取Java代码块
    新逻辑：从EOF之后找到public class，然后往前找最近的```java，从那里提取代码块
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 步骤1: 找到EOF标记的位置
        eof_pos = -1
        if eof_marker:
            eof_pos = log_content.find(eof_marker)
        
        # 如果没有找到EOF或者没有提供EOF，从头开始搜索
        search_start = eof_pos + len(eof_marker) if eof_pos != -1 and eof_marker else 0
        content_after_eof = log_content[search_start:]
        
        # 步骤2: 在EOF之后找到"public class"的位置
        public_class_match = re.search(r'public\s+class', content_after_eof)
        
        if public_class_match:
            # 找到了public class，计算在原始内容中的绝对位置
            public_class_pos = search_start + public_class_match.start()
            
            # 步骤3: 从public class的位置往前找最近的```java
            content_before_public_class = log_content[:public_class_pos]
            
            # 找到所有```java的位置
            java_markers = [m.start() for m in re.finditer(r'```java', content_before_public_class)]
            
            if java_markers:
                # 取最后一个（最接近public class的）```java
                nearest_java_marker = java_markers[-1]
                
                # 步骤4: 从这个```java开始提取代码块
                # 找到```java之后的内容
                start_pos = nearest_java_marker + len('```java')
                remaining_content = log_content[start_pos:]
                
                # 查找结束的```
                end_match = re.search(r'```', remaining_content)
                if end_match:
                    extracted_code = remaining_content[:end_match.start()].strip()
                    print(f"✓ 从 {os.path.basename(log_file_path)} 提取代码块 (从public class往前定位, 长度:{len(extracted_code)}字符)")
                    return extracted_code
                else:
                    # 没有找到结束标记，提取到文件末尾
                    extracted_code = remaining_content.strip()
                    print(f"✓ 从 {os.path.basename(log_file_path)} 提取未闭合代码块 (从public class往前定位, 长度:{len(extracted_code)}字符)")
                    return extracted_code
            else:
                print(f"✗ {os.path.basename(log_file_path)} 在public class之前找不到```java标记")
                return None
        else:
            print(f"✗ {os.path.basename(log_file_path)} 在EOF之后找不到public class")
            return None
        
    except Exception as e:
        print(f"✗ 读取 {os.path.basename(log_file_path)} 时出错: {e}")
        return None

# 在导入其他模块前先设置GPU环境
if __name__ == '__main__':
    # 检查是否为 reextract 模式（支持两种参数顺序）
    reextract_mode = False
    result_tag = None
    
    if len(sys.argv) >= 3:
        if sys.argv[1] == '--reextract':
            # 格式1: python inference_java.py --reextract qwen3-4b
            reextract_mode = True
            result_tag = sys.argv[2]
        elif sys.argv[2] == '--reextract':
            # 格式2: python inference_java.py qwen3-4b --reextract
            reextract_mode = True
            result_tag = sys.argv[1]
    
    if reextract_mode:
        # Reextract 模式：只需要结果标签
        if not result_tag:
            print(f"使用方法: python {sys.argv[0]} --reextract <result_tag>")
            print(f"    或者: python {sys.argv[0]} <result_tag> --reextract")
            print("示例: python inference_java.py --reextract qwen3-4b")
            print("      python inference_java.py qwen3-4b --reextract")
            print("将从 EOF 之后找 public class，然后往前定位 ```java 代码块")
            sys.exit(1)
        
        result_base_dir = f'evalrepair-java-res/{result_tag}'
        
        if not os.path.exists(result_base_dir):
            print(f"错误: 结果目录不存在: {result_base_dir}")
            sys.exit(1)
        
        # 根据 result_tag 获取对应的 EOF 标记
        BOF, EOF = get_prompt_format(result_tag)
        
        print("=" * 60)
        print(f"📝 重新提取模式: {result_tag}")
        print(f"📂 目标目录: {result_base_dir}")
        print(f"🎯 提取策略: EOF之后 → public class → 往前找最近的```java")
        print(f"📌 EOF标记: {EOF if EOF else '(无EOF标记)'}")
        print("=" * 60)
        
        # 遍历所有 fixed* 目录
        total_processed = 0
        total_success = 0
        
        for fixed_dir in sorted(Path(result_base_dir).glob('fixed*')):
            print(f"\n📁 处理目录: {fixed_dir.name}")
            for log_file in sorted(fixed_dir.glob('*.log')):
                total_processed += 1
                java_file = str(log_file)[:-4]  # 移除 .log 扩展名
                
                # 传入 EOF 标记和 model_key
                extracted_code = reextract_code_from_log(str(log_file), eof_marker=EOF, model_key=result_tag)
                
                if extracted_code:
                    with open(java_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_code)
                    total_success += 1
                else:
                    pass  # 错误信息已在函数内打印
        
        print("\n" + "=" * 60)
        print(f"重新提取完成！")
        print(f"总计处理: {total_processed} 个文件")
        print(f"成功提取: {total_success} 个文件")
        print(f"失败: {total_processed - total_success} 个文件")
        print("=" * 60)
        sys.exit(0)
    else:
        # 正常模式：设置GPU环境
        setup_gpu_environment()
        set_file_limits()

# 初始化模型
if len(sys.argv) < 5 or sys.argv[1] not in MODEL_CONFIGS:
    print(f"使用方法: python {sys.argv[0]} <model_key> <num_processes> <process_id> <gpu_id> [num_generations]")
    print(f"可用模型: {list(MODEL_CONFIGS.keys())}")
    print()
    print(f"或者: python {sys.argv[0]} --reextract <result_tag>")
    print("  重新提取已有log文件中的代码（不重新生成）")
    sys.exit(1)

# 全局变量
llm = None
tokenizer = None

model_key = sys.argv[1]
model_config = MODEL_CONFIGS[model_key]
BOF, EOF = get_prompt_format(model_key)

# 加载 vLLM 模型
llm, tokenizer, _ = load_vllm_model(model_config)
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"GPU状态: {monitor_gpu_status()}")

def generate_fix(code: str, filename: str, model_key: str) -> Tuple[str, str]:
    """生成代码修复"""
    # 使用指令模型格式构建 prompt
    prompt = f"{BOF}This is an incorrect code ({filename}):\n```java\n{code}\n```\nYou are a software engineer. Can you repair the incorrect java code?\n{EOF}\n```java\n"
    
    print(prompt, flush=True)
    
    return generate_with_vllm(llm, prompt, model_key)

# 配置路径
base_dir = 'evalrepair-java/origin/'
result_base_dir = f'evalrepair-java-res/{model_key}/'

# 参数解析 (注意：现在参数顺序变了，gpu_id是第4个参数)
total_processes = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
process_id = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
gpu_id = sys.argv[4] if len(sys.argv) >= 5 else "0"
num_generations = int(sys.argv[5]) if len(sys.argv) >= 6 else 10

print(f"处理配置: 进程 {process_id + 1}/{total_processes}, 生成 {num_generations} 个版本")

# 处理文件
cnt = 0
for file_path in sorted(Path(base_dir).rglob('*.java'), reverse=True):
    cnt += 1
    if total_processes > 1 and cnt % total_processes != process_id:
        continue
    
    full_path = str(file_path)
    print(full_path, flush=True)
    
    with open(full_path, 'r', encoding='utf-8') as file:
        content = file.read()
    print(content)
    
    file_name = os.path.basename(full_path)
    
    for e in range(num_generations):
        fix_subdir = os.path.join(result_base_dir, f'fixed{e}')
        fix_name = os.path.join(fix_subdir, file_name)
        log_name = fix_name + '.log'
        
        os.makedirs(fix_subdir, exist_ok=True)
        print(f"Output path: {fix_name}", flush=True)
        
        # 跳过已存在的文件
        if os.path.exists(fix_name) and os.path.exists(log_name):
            print('result exists ...')
            continue
        
        # 生成修复代码
        full_text, code_result = generate_fix(content, file_name, model_key)
        if not full_text:
            continue
        
        # 保存结果
        with open(fix_name, 'w', encoding='utf-8') as f:
            f.write(code_result)
        with open(log_name, 'w', encoding='utf-8') as f:
            f.write(full_text)

print("✅ Java 代码修复完成！")

# 清理资源
cleanup_resources()