import os
import sys
# 在导入torch前设置GPU，让整个进程只看到指定的GPU
if len(sys.argv) >= 5:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]  # gpu_id 是第5个参数（索引4）

import time
import tqdm
import json
import argparse
import psutil
import torch
import re

# 修复 vLLM 多进程问题
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# vLLM 环境变量优化
os.environ['VLLM_USE_MODELSCOPE'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['VLLM_USE_V1'] = '0'

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_CONFIGS = {
    'qwen3-8b': {
        'model_path': 'model/qwen3-8b',
    },
    'qwen3-4b': {
        'model_path': 'model/qwen3-4b',
    },
    'qwen3-8b-trained-noprompt': {
        'model_path': 'merged_models/qwen3-8b-trained-noprompt',
    },
    'qwen3-4b-trained-noprompt': {
        'model_path': 'merged_models/qwen3-4b-trained-noprompt',
    },
    'qwen3-4b-trained-parepair': {
        'model_path': 'merged_models/qwen3-4b-parepair',
    },
    'llama3.1-8b': {
        'model_path': 'model/Llama-3-8B-Instruct',
    },
    'llama3.1-8b-nopro': {
        'model_path': 'merged_models/llama3.1-8b-nopro',
    },
    'llama3.1-8b-trained-parepair': {
        'model_path': 'merged_models/llama3.1-8b-parepair',
    },
    'deepseek-6.7b': {
        'model_path': 'model/deepseek-coder-6.7b',
    },
    'deepseek-6.7b-nopro': {
        'model_path': 'merged_models/deepseek-6.7b-nopro',
    },
    'deepseek-6.7b-parepair': {
        'model_path': 'merged_models/deepseek-6.7b-parepair',
    },
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

# Global variables for model and tokenizer
model = None
tokenizer = None


def setup_gpu_environment():
    """设置GPU环境，包括冲突检测和预防"""
    if len(sys.argv) < 5:
        print("Usage: python gen_apr.py <model_name> <num_processes> <process_id> <gpu_id>")
        print("参数说明:")
        print("  model_name: 模型名称")
        print("  num_processes: 总进程数")
        print("  process_id: 当前进程ID (0开始)")
        print("  gpu_id: GPU编号 (0开始)")
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
    lock_dir = os.path.join(os.getcwd(), '.gpu_locks')
    os.makedirs(lock_dir, exist_ok=True)
    lock_file = os.path.join(lock_dir, f"gpu_{gpu_id}_proc_{sys.argv[3]}_apr.lock")
    
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
        # GPU已被占用
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read().strip()
            
            if lock_info.startswith("PID:"):
                parts = lock_info.split(',')
                pid_str = parts[0].split(':')[1]
                lock_pid = int(pid_str)
                
                if psutil.pid_exists(lock_pid):
                    proc = psutil.Process(lock_pid)
                    if proc.is_running():
                        print(f"错误：GPU {gpu_id} 已被进程 {lock_pid} 占用")
                        sys.exit(1)
                    else:
                        print(f"检测到僵尸锁文件，清理并重试")
                        os.remove(lock_file)
                        return setup_gpu_environment()
                else:
                    print(f"检测到过期锁文件，清理并重试")
                    os.remove(lock_file)
                    return setup_gpu_environment()
        except Exception as e:
            print(f"检查GPU锁时出错: {e}")
            sys.exit(1)
    
    print(f"GPU设置完成")

# 模型提示格式（必须与训练时格式一致！）
MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'),
    'llama3': ('<|start_header_id|>user<|end_header_id|>\n\n', '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'),
    'deepseek': ('###Instruction\n', '###response\n\n'),  # DeepSeek 格式
    'opencoder': ('<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'),
}


def get_prompt_format(model_key: str) -> tuple:
    """获取模型的提示格式"""
    for key in MODEL_PROMPT_FORMATS:
        if model_key.startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    return ('', '')  # 默认格式（无特殊标记）


def extract_cpp_code(text: str) -> str:
    """从生成的文本中提取第一个完整的代码块（必须有闭合标记）"""
    # 只匹配完整的代码块（有闭合标记），没有回退策略
    match = re.search(r'```(?:c\+\+)?\s*(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def load_model(model_name):
    """Load model and tokenizer using vLLM"""
    global model, tokenizer
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in MODEL_CONFIGS")
    
    model_path = MODEL_CONFIGS[model_name]['model_path']
    print(f"Loading model from {model_path}...")
    
    # 加载 tokenizer
    print("📝 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    
    # vLLM 配置
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
        model = LLM(**llm_kwargs)
        print("✅ vLLM 模型加载成功")
    except Exception as e:
        print(f"❌ vLLM 模型加载失败: {e}")
        raise


def gen_with_retry(prompt, nsample, max_retries=5):
    """使用重试机制生成代码（与inference_cpp.py保持一致）"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    # vLLM 采样参数 - 与 inference_cpp.py 一致
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,  # 与inference_cpp.py一致
        repetition_penalty=1.1,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    
    successful_responses = []
    
    for sample_idx in range(nsample):
        retry_count = 0
        extracted_code = ""
        
        while not extracted_code and retry_count < max_retries:
            retry_count += 1
            print(f"样本 {sample_idx + 1}/{nsample}, 尝试 {retry_count}/{max_retries}...", flush=True)
            
            try:
                # vLLM 生成
                outputs = model.generate([prompt], sampling_params)
                output = outputs[0]
                generated_text = output.outputs[0].text.strip()
                
                # 从生成的文本中提取C++代码
                extracted_code = extract_cpp_code(generated_text)
                
                if extracted_code:
                    print(f"✅ 成功提取C++代码 (长度: {len(extracted_code)} 字符)")
                    successful_responses.append({"message": {"content": extracted_code}})
                    break
                else:
                    print(f"⚠️ 第 {retry_count} 次尝试未找到C++代码块")
                    
            except Exception as e:
                print(f"生成错误 (尝试 {retry_count}): {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        if not extracted_code:
            print(f"❌ 样本 {sample_idx + 1} 经过 {max_retries} 次尝试仍未获取到有效代码")
    
    if not successful_responses:
        print(f"❌ 所有样本都未能生成有效的C++代码")
        return None
    
    # Format response similar to OpenAI API
    result = {
        "choices": successful_responses,
        "prompt": prompt
    }
    return result


def gen(prompt, nsample):
    """Generate responses using vLLM with retry mechanism"""
    return gen_with_retry(prompt, nsample)


def process_prompt(dt, nsample, output_dir, index, model_key, dry_run=0):
    language = dt["lang_cluster"]
    file_path = os.path.join(output_dir, f"{index}_1.0_{language}.json")
    log_file_path = file_path.replace('.json', '.log')
    
    # 只有当JSON和log文件都存在时才跳过
    if not (os.path.exists(file_path) and os.path.exists(log_file_path)):
        # 获取模型特定的提示格式
        BOF, EOF = get_prompt_format(model_key)
        
        # 直接构建完整的提示词
        raw_prompt = f"Description: {dt['prob_desc_description']}\n"
        raw_prompt += f"This is an incorrect code:\n```c++\n{dt['bug_source_code']}\n```\n"
        raw_prompt += f"You are a software engineer. Can you repair the incorrect cpp code?\n"
        
        # 将原始 prompt 包装在模型格式中，并添加代码块开始标记
        formatted_prompt = f"{BOF}{raw_prompt}{EOF}\n```c++\n"
        
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(formatted_prompt, indent=4)}")
        else:
            out = gen(formatted_prompt, nsample)
            if out is not None:
                export_data = {"model_response": out, "source_data": dt}
                open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")
                
                # 保存完整的log文件，记录10次生成的输入输出
                log_file_path = file_path.replace('.json', '.log')
                with open(log_file_path, "w", encoding='utf-8') as log_f:
                    log_f.write("=" * 80 + "\n")
                    log_f.write(f"SAMPLE INDEX: {index}\n")
                    log_f.write(f"LANGUAGE: {language}\n")
                    log_f.write(f"MODEL: {model_key}\n")
                    log_f.write(f"NSAMPLE: {nsample}\n")
                    log_f.write("=" * 80 + "\n\n")
                    
                    log_f.write("INPUT PROMPT:\n")
                    log_f.write("-" * 40 + "\n")
                    log_f.write(formatted_prompt)
                    log_f.write("\n" + "-" * 40 + "\n\n")
                    
                    log_f.write("GENERATED OUTPUTS:\n")
                    log_f.write("-" * 40 + "\n")
                    for i, choice in enumerate(out["choices"], 1):
                        log_f.write(f"[OUTPUT {i}/{len(out['choices'])}]\n")
                        log_f.write(choice["message"]["content"])
                        log_f.write(f"\n{'-' * 20}\n\n")
                    
                    log_f.write("=" * 80 + "\n")
                    log_f.write("END OF LOG\n")
                    log_f.write("=" * 80 + "\n")
                
                print(f"✅ Saved: {file_path} and {log_file_path}")
            else:
                print(f"Failed to generate response for index {index}")


def main():
    # 检查是否使用多进程模式
    if len(sys.argv) >= 5:
        # 多进程模式: python gen_apr.py <model_name> <num_processes> <process_id> <gpu_id>
        setup_gpu_environment()
        
        model_name = sys.argv[1]
        num_processes = int(sys.argv[2])
        process_id = int(sys.argv[3])
        gpu_id = sys.argv[4]
        
        # 使用默认参数
        output_dir = f"xcodeeval/result/{model_name}"
        dataset_path = "xcodeeval/xcodeeval_merged.json"
        nsample = 10
        dry_run = 0
        
        print(f"多进程模式: 进程 {process_id + 1}/{num_processes}, GPU {gpu_id}")
    else:
        # 单进程模式: 使用argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output-dir",
            default=None,
            type=str,
            help="Output Folder to save the model outputs. If not specified, will use result/{model_name}/",
        )
        parser.add_argument(
            "--dry-run",
            default=0,
            help="Dry run mode (0 or 1).",
        )
        parser.add_argument(
            "--nsample",
            default=10,
            type=int,
            help="Number of samples to generate per prompt.",
        )
        parser.add_argument(
            "--model-name",
            default="qwen3-8b",
            type=str,
            help="Model name from MODEL_CONFIGS to use for generation.",
        )
        parser.add_argument(
            "--dataset-path",
            default="xcodeeval/xcodeeval_merged.json",
            type=str,
            help="Local path to the JSON dataset file.",
        )
        args = parser.parse_args()
        
        model_name = args.model_name
        output_dir = args.output_dir if args.output_dir else f"xcodeeval/result/{model_name}"
        dataset_path = args.dataset_path
        nsample = args.nsample
        dry_run = args.dry_run
        num_processes = 1
        process_id = 0
    
    print(f"Output directory: {output_dir}")
    
    # Load the model before processing
    load_model(model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load dataset from JSON file
    print(f"Loading dataset from JSON file: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        apr_dataset = json.load(f)
    
    print(f"Total dataset size: {len(apr_dataset)}")
    
    # 多进程模式：按进程ID分配数据
    if num_processes > 1:
        # 只处理分配给当前进程的数据
        process_dataset = []
        for idx, dt in enumerate(apr_dataset):
            if idx % num_processes == process_id:
                process_dataset.append((idx, dt))
        print(f"Process {process_id} handling {len(process_dataset)} samples")
    else:
        process_dataset = list(enumerate(apr_dataset))
    
    for idx, dt in tqdm.tqdm(
        process_dataset,
        desc=f"Generating with {model_name} (Process {process_id})",
    ):
        try:
            process_prompt(
                dt,
                nsample,
                output_dir,
                idx,
                model_name,
                dry_run,
            )
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")


if __name__ == "__main__":
    main()
    print("✅ 代码生成完成！")
