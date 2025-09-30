import json
import os
import sys
import torch
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

# vLLM 环境变量优化 - WSL2兼容性
os.environ['VLLM_USE_MODELSCOPE'] = '0'  # 使用数字格式
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'  # 使用数字格式
# WSL2 CUDA 兼容性设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用以便调试
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 明确指定GPU设备
# os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'  # 让 vLLM 自动选择注意力后端
os.environ['VLLM_USE_V1'] = '0'  # 禁用V1引擎，使用更稳定的V0
# 离线模式设置 - 避免连接HuggingFace Hub
os.environ['HF_HUB_OFFLINE'] = '1'  # HuggingFace Hub离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Transformers离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'  # Datasets离线模式

# vLLM 推理引擎导入
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("vLLM 可用 - 将使用高性能推理引擎")
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM 不可用 - 使用标准 transformers")

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


def cleanup_resources():
    """清理资源，避免文件句柄泄露"""
    global model, tokenizer
    try:
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


def load_vllm_model(model_config, retry_with_lower_memory=True):
    """使用 vLLM 加载模型以获得最佳性能"""
    base_model_path = model_config['base_model']
    adapter_path = model_config.get('adapter_path')
    
    # 如果有 LoRA 适配器，需要先合并
    if adapter_path and os.path.exists(adapter_path):
        print(f"vLLM 模式：检测到 LoRA 适配器: {adapter_path}")
        print("警告：vLLM 不直接支持 LoRA。降级到标准模式以支持 LoRA。")
        raise RuntimeError("vLLM 不支持 LoRA 适配器，请使用预合并模型或降级到标准模式")
    
    # vLLM WSL2 兼容配置 - 解决 CUDA 驱动错误和文件句柄问题
    vllm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": 0.8,  # 进一步降低显存利用率以避免CUDA错误
        "trust_remote_code": True,
        "max_model_len": 2048,  # 进一步降低序列长度
        "enforce_eager": True,  # 禁用 CUDA Graph
        "disable_custom_all_reduce": True,  # 禁用自定义 all-reduce
        "disable_log_stats": True,  # 禁用统计日志
        "download_dir": None,  # 不下载，使用本地缓存
        "load_format": "auto",  # 自动检测格式
        # 添加更多稳定性配置
        "max_num_seqs": 8,  # 降低并发序列数以减少资源使用
        "block_size": 16,  # 使用较小的块大小
        "disable_sliding_window": True,  # 禁用滑动窗口
        # WSL2 特殊配置
        "enable_chunked_prefill": False,  # 禁用分块预填充
        "use_v2_block_manager": False,  # 使用旧版块管理器
        "swap_space": 0,  # 禁用交换空间
        "cpu_offload_gb": 0,  # 禁用CPU卸载
    }
    
    print(f"vLLM WSL2兼容配置: 显存利用率80%, 最大序列长度2048, 降低并发数")
    
    llm = LLM(**vllm_config)
    
    # 加载对应的 tokenizer - 离线模式
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        local_files_only=True  # 仅使用本地文件
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return llm, tokenizer

def load_model_with_adapter(model_config):
    """加载带LoRA适配器的模型（fallback方案）"""
    base_model_path = model_config['base_model']
    adapter_path = model_config.get('adapter_path')

    # 加载tokenizer - 离线模式
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        local_files_only=True  # 仅使用本地文件
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 加载基础模型，启用 FlashAttention - 离线模式
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True  # 仅使用本地文件
    }
    
    # 如果支持 FlashAttention，启用相关设置
    if FLASH_ATTENTION_AVAILABLE:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )

    # 如果有适配器路径，加载并合并LoRA适配器
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Original model type: {type(model).__name__}")

        # 合并LoRA权重到基础模型中
        model = model.merge_and_unload()
        print(f"Merged model type: {type(model).__name__}")
        print("LoRA weights merged into base model")

    return model, tokenizer


def main():
    """主函数 - 解决多进程问题"""
    # 设置文件句柄限制
    set_file_limits()
    
    # 设置prompt格式
    global BOF, EOF, model, tokenizer, USE_VLLM
    BOF, EOF = get_prompt_format(sys.argv[1])

    try:
        # 模型加载 - 优先使用 vLLM
        if sys.argv[1] in MODEL_CONFIGS:
            model_config = MODEL_CONFIGS[sys.argv[1]]
            
            # 优先尝试使用 vLLM
            if VLLM_AVAILABLE:
                try:
                    model, tokenizer = load_vllm_model(model_config)
                    USE_VLLM = True
                    print("成功加载 vLLM 引擎")
                except RuntimeError as e:
                    error_str = str(e)
                    if "multiprocessing" in error_str or "freeze_support" in error_str:
                        print(f"vLLM 多进程错误，降级到标准模式: {e}")
                    elif "CUDA" in error_str or "cuda" in error_str or "unknown error" in error_str:
                        print(f"vLLM CUDA/驱动错误，降级到标准模式: {e}")
                        print("建议：检查显卡显存是否足够，或尝试降低 gpu_memory_utilization")
                    else:
                        print(f"vLLM 运行时错误，降级到标准模式: {e}")
                    model, tokenizer = load_model_with_adapter(model_config)
                    USE_VLLM = False
                except Exception as e:
                    print(f"vLLM 加载失败，降级到标准模式: {e}")
                    model, tokenizer = load_model_with_adapter(model_config)
                    USE_VLLM = False
            else:
                model, tokenizer = load_model_with_adapter(model_config)
                USE_VLLM = False
                
                # 尝试编译模型以获得更好的性能（仅在非vLLM模式）
                try:
                    if hasattr(torch, 'compile'):
                        model = torch.compile(model, mode="reduce-overhead")
                        print("模型已编译优化")
                except Exception as e:
                    print(f"模型编译失败，使用原始模型: {e}")
        else:
            raise ValueError(f"Unknown model key: {sys.argv[1]}. Available models: {list(MODEL_CONFIGS.keys())}")

        print('load model success ..', flush=True)
        
        # 执行主要处理逻辑
        process_files()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
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

                for e in range(int(sys.argv[-1])):
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
        
        # 提取生成的 Java 代码部分
        try:
            ret = extract_first_java_code(full_text.split('[/INST]')[-1])
        except IndexError:
            ret = extract_first_java_code(full_text)
        
        print('code:', ret, flush=True)
        return [complete_text, ret]
        
    except Exception as e:
        print(f"cal_vllm 执行出错: {e}")
        return [None, None]

def cal_vllm_batch(prompts):
    """vLLM 批处理推理 - 充分利用显存和并发"""
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
    """统一推理接口 - 自动选择 vLLM 或标准模式"""
    if USE_VLLM:
        return cal_vllm(bug_id, code, title, description, filename)
    else:
        try:
            # 标准 transformers 推理
            prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
            print(prompt, flush=True)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)  # 降低长度
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            generation_config = {
                "max_new_tokens": 512,  # 降低生成长度
                "temperature": 1.0,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
                "early_stopping": False,
            }
            
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_config)
                
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(full_text)
                
                try:
                    ret = extract_first_java_code(full_text.split('[/INST]')[1])
                except IndexError:
                    ret = extract_first_java_code(full_text)
                
                print('code:', ret, flush=True)
                return [full_text, ret]
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"CUDA/内存错误，尝试清理缓存: {e}")
                    # 清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    return [None, None]
                else:
                    print(f"生成错误: {e}")
                    return [None, None]
                    
        except Exception as e:
            print(f"cal 标准模式执行出错: {e}")
            return [None, None]


if __name__ == '__main__':
    main()