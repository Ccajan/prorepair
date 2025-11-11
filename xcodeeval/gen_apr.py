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
from promptsource.templates import Template


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
    'starcoder': ('', ''),  # StarCoder 不需要特殊格式
}


def get_prompt_format(model_key: str) -> tuple:
    """获取模型的提示格式"""
    for key in MODEL_PROMPT_FORMATS:
        if model_key.startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    return ('', '')  # 默认格式（无特殊标记）


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


def gen(prompt, nsample):
    """Generate responses using vLLM"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    try:
        # vLLM 采样参数 - 与 inference_java.py 一致
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,  # 修复：与README中说明的1024一致
            repetition_penalty=1.1,
            stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
            n=nsample,  # 一次生成多个样本
        )
        
        # vLLM 生成
        outputs = model.generate([prompt], sampling_params)
        output = outputs[0]
        
        # 提取所有生成的样本
        responses = []
        for generated_output in output.outputs:
            generated_text = generated_output.text.strip()
            responses.append({"message": {"content": generated_text}})
        
        # Format response similar to OpenAI API
        result = {
            "choices": responses,
            "prompt": prompt
        }
        return result
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return None


xcodeeval_prompt_template = {
    "apr": [
        "Fix a buggy program written in {{lang_cluster}} language to solve the following programming problem:\nDescription: {{prob_desc_description}}\nInput Specification: {{prob_desc_input_spec}}\nOutput Specification: {{prob_desc_output_spec}}\n{% for input, output in zip(prob_desc_sample_inputs, prob_desc_sample_outputs) %}\nSample Input:\n{{input}}\nSample Output:\n{{output}}\n{% endfor %}\nNotes: {{prob_desc_notes}}\nTake input from {{prob_desc_input_from}} and output to {{prob_desc_output_to}}\n\nHere is the code with a bug of {{bug_exec_outcome}}:\n\n{{bug_source_code}}\n\nProvide the fixed {{lang_cluster}} code without any description or extra tokens.\n\nFixed source code:\n ||END-of-SRC|| "
    ]
}


def process_prompt(dt, template, nsample, output_dir, index, model_key, dry_run=0):
    language = dt["lang_cluster"]
    file_path = os.path.join(output_dir, f"{index}_1.0_{language}.json")
    if not os.path.exists(file_path):
        # Sample inputs/outputs are already arrays in the JSON, no need to parse
        lm_io = template.apply(dt)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        
        # 获取模型特定的提示格式
        BOF, EOF = get_prompt_format(model_key)
        
        # 将原始 prompt 包装在模型格式中
        formatted_prompt = f"{BOF}{lm_io[0]}{EOF}"
        
        if dry_run:
            open(file_path, "w").write(f"{json.dumps(formatted_prompt, indent=4)}")
        else:
            out = gen(formatted_prompt, nsample)
            if out is not None:
                export_data = {"model_response": out, "source_data": dt}
                open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")
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
    templates = [
        Template(f"apr_{idx}", template, "xCodeEval", delimeter="||END-of-SRC||")
        for idx, template in enumerate(xcodeeval_prompt_template["apr"])
    ]
    template = templates[0]

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
                template,
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
