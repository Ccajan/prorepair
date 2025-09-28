import os
import json
import math
import random
import tempfile
import subprocess
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# ==========================================
# 配置
# ==========================================

MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>'),
    'codellama': ('[INST]', '[/INST]'),
    'llama': ('[INST]', '[/INST]'),
    'mistral': ('[INST]', '[/INST]'),
}

def get_prompt_format(model_key):
    """根据模型key获取prompt格式"""
    for key in MODEL_PROMPT_FORMATS:
        if model_key.lower().startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    return '[INST]', '[/INST]'

@dataclass
class TrainArgs:
    model_name: str
    model_base_path: str = "/data1/czj/prorepair/base_model"
    train_file: str = "/data1/czj/prorepair/data/trainset/sft_dataset.json"
    output_base_path: str = "/data1/czj/prorepair/train_model"
    
    sft_epochs: int = 5
    grpo_epochs: int = 2
    sft_lr: float = 3e-5
    grpo_lr: float = 1.5e-5
    sft_batch: int = 1
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    compile_model: bool = True
    
    generation_batch_size: int = 1
    use_kv_cache: bool = True
    
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    max_new_tokens: int = 512
    grad_clip: float = 1.0
    seed: int = 42
    
    use_dft: bool = True
    use_neftune: bool = True
    neftune_alpha: float = 5.0
    use_8bit: bool = False
    use_4bit: bool = True
    
    clip_param: float = 0.2
    kl_coeff: float = 0.01
    use_kl_penalty: bool = True
    advantage_normalization: bool = True
    advantage_clip: float = 10.0
    
    external_testcase_file: str = "/data1/czj/prorepair/data/trainset/testcases_sorted.json"
    problem_id_field: str = "problem_id"
    
    @property
    def base_model(self) -> str:
        return os.path.join(self.model_base_path, self.model_name)
    
    @property
    def out_dir(self) -> str:
        model_short_name = self.model_name.lower().replace("-", "_")
        return os.path.join(self.output_base_path, f"sft_grpo_{model_short_name}")

@dataclass
class CurriculumCfg:
    start_temp: float = 1.2
    end_temp: float = 0.8
    start_top_p: float = 0.95
    end_top_p: float = 0.8
    start_num: int = 3
    end_num: int = 3

    def interp(self, ratio: float) -> Tuple[float, float, int]:
        ratio = min(max(ratio, 0.0), 1.0)
        temp = self.start_temp + (self.end_temp - self.start_temp) * ratio
        top_p = self.start_top_p + (self.end_top_p - self.start_top_p) * ratio
        num = int(round(self.start_num + (self.end_num - self.start_num) * ratio))
        return temp, top_p, max(1, num)

@dataclass
class RewardCfg:
    diff_clip_low: float = -1.0
    diff_clip_high: float = 1.0
    use_test_cases: bool = True
    test_weight: float = 0.6
    text_weight: float = 0.4
    test_timeout: float = 5.0
    external_testcase_file: str = ""
    problem_id_field: str = "problem_id"

# ==========================================
# 数据加载
# ==========================================

class CodeDataset(Dataset):
    def __init__(self, path: str):
        self.data = []
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                self.data = obj if isinstance(obj, list) else [obj]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        for k in ["prompt", "chosen"]:
            if k not in item:
                raise ValueError(f"Sample {idx} missing field: {k}")
        item.setdefault("explanation", "")
        return item

# ==========================================
# 工具函数
# ==========================================

def extract_code(text: str) -> str:
    """从文本中提取代码"""
    import re
    text = str(text) if text is not None else ''
    
    # 尝试提取C++代码块
    patterns = [r'```cpp(.*?)```', r'```c\+\+(.*?)```', r'```(.*?)```']
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            if code and ('#include' in code or 'int main' in code):
                return code
    
    # 如果没有代码块但包含#include，返回整个文本
    if '#include' in text:
        return text.strip()
    
    return text.strip()

def run_code_with_testcases(code: str, testcases: List[Dict], timeout: float = 5.0) -> float:
    """执行代码并运行测试用例"""
    if not testcases:
        return 0.0
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            cpp_file = f.name
        
        import platform
        exe_file = cpp_file.replace('.cpp', '.exe' if platform.system() == 'Windows' else '')
        
        # 编译
        compile_result = subprocess.run(
            ['g++', '-o', exe_file, cpp_file, '-std=c++17'],
            capture_output=True, timeout=timeout
        )
        
        if compile_result.returncode != 0:
            os.unlink(cpp_file)
            return 0.0
        
        # 运行测试用例
        passed = 0
        for test in testcases:
            try:
                input_data = str(test.get('input', ''))
                expected_output = str(test.get('output', '')).strip()
                
                result = subprocess.run(
                    [exe_file], input=input_data, capture_output=True, 
                    text=True, timeout=timeout
                )
                
                if result.returncode == 0 and result.stdout.strip() == expected_output:
                    passed += 1
            except:
                continue
        
        # 清理文件
        try:
            os.unlink(cpp_file)
            os.unlink(exe_file)
        except:
            pass
            
        return passed / len(testcases)
        
    except:
        return 0.0

def compute_text_quality(ref_text: str, cand_text: str) -> Tuple[float, float, float]:
    """计算文本质量"""
    import difflib
    
    ref_text = str(ref_text) if ref_text is not None else ''
    cand_text = str(cand_text) if cand_text is not None else ''
    
    if not cand_text.strip():
        return 0.0, 1.0, -1.0
    if not ref_text.strip():
        return 0.0, 0.0, 0.0
    
    ref_lines = ref_text.splitlines()
    cand_lines = cand_text.splitlines()
    k = len(cand_lines)
    if k == 0:
        return 0.0, 1.0, -1.0
    
    matcher = difflib.SequenceMatcher(None, ref_lines, cand_lines)
    matching_blocks = matcher.get_matching_blocks()
    l = sum(block.size for block in matching_blocks if block.size > 0)
    retention = l / k
    
    similarity = matcher.ratio()
    diff_ratio = 1.0 - similarity
    score = retention - diff_ratio
    
    return retention, diff_ratio, score

def cleanup_cuda_memory(force_sync=False, aggressive=False):
    """激进的内存清理"""
    if torch.cuda.is_available():
        try:
            if force_sync:
                torch.cuda.synchronize()
            
            torch.cuda.empty_cache()
            gc.collect()
            
            if aggressive:
                for round_num in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"[WARNING] CUDA cleanup error: {e}")

def cleanup_memory():
    """简化版内存清理"""
    cleanup_cuda_memory()

def aggressive_cache_cleanup():
    """激进的内存清理"""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            for round_num in range(5):
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"[WARNING] CUDA cleanup error: {e}")

def get_adaptive_memory_config():
    """自适应获取GPU内存配置"""
    if not torch.cuda.is_available():
        return {}, 1, 4
    
    memory_config = {}
    n_gpus = torch.cuda.device_count()
    min_available_memory = float('inf')
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        
        used_memory = max(allocated, cached)
        free_memory = total_memory - used_memory
        
        # 保守分配策略
        safety_buffer = 2.0
        usable_memory = max(free_memory - safety_buffer, 4.0)
        allocated_memory = min(usable_memory * 0.75, total_memory * 0.85)
        
        memory_config[i] = f"{allocated_memory:.1f}GiB"
        min_available_memory = min(min_available_memory, allocated_memory)
    
    # 根据内存调整batch size和梯度累积
    if min_available_memory >= 15:
        suggested_batch = 2
        suggested_accumulation = 2
    elif min_available_memory >= 10:
        suggested_batch = 1
        suggested_accumulation = 4
    elif min_available_memory >= 6:
        suggested_batch = 1
        suggested_accumulation = 2
    else:
        suggested_batch = 1
        suggested_accumulation = 1
    
    return memory_config, suggested_batch, suggested_accumulation

def print_memory_usage(prefix=""):
    """打印内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"{prefix} GPU {i}: {allocated:.1f}GB/{total:.1f}GB (缓存:{cached:.1f}GB)")

# ==========================================
# Prompt调度器
# ==========================================

class PromptScheduler:
    def __init__(self, model_key: str):
        self.model_key = model_key
        
        self.innovative_prompts = [
            "Please try to propose multiple different repair methods, which can have major changes, but ensure that the code can be compiled and run:",
        ]
        
        self.conservative_prompts = [
            "Please generate a minimal repair patch that preserves the original code structure and only modifies the necessary parts to fix the error:",
        ]
        
        self.base_instruction = (
            "You are an expert software engineer. Analyze the incorrect code carefully and provide a correct implementation. "
            "Generate ONLY the corrected C++ code inside a code block, without explanations outside the code."
        )
    
    def get_diverse_prompt(self, base_prompt: str, epoch_ratio: float, candidate_idx: int = 0) -> Tuple[str, str]:
        BOF, EOF = get_prompt_format(self.model_key)
        
        if epoch_ratio < 0.6:
            strategy_prompts = self.innovative_prompts
            strategy_type = "innovative"
        else:
            strategy_prompts = self.conservative_prompts
            strategy_type = "conservative"
        
        prompt_idx = candidate_idx % len(strategy_prompts)
        diversity_instruction = strategy_prompts[prompt_idx]
        
        full_prompt = (
            BOF + "\n" + 
            base_prompt.strip() + "\n\n" +
            diversity_instruction + "\n" +
            self.base_instruction + "\n" +
            EOF + "\n```cpp\n"
        )
        
        return full_prompt, strategy_type

# ==========================================
# 概率计算
# ==========================================

def compute_log_probs_with_grad(model, tokenizer, prompts, responses):
    """计算序列对数概率(保留梯度)"""
    all_log_probs = []
    
    for prompt, response in zip(prompts, responses):
        try:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            if len(response_tokens) == 0:
                all_log_probs.append(torch.tensor(0.0, requires_grad=True))
                continue
                
            full_tokens = prompt_tokens + response_tokens
            input_ids = torch.tensor([full_tokens], dtype=torch.long)
            
            if input_ids.size(1) > 1024:
                all_log_probs.append(torch.tensor(0.0, requires_grad=True))
                continue
            
            # 移动到模型设备
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits

            # 计算response部分的对数概率
            prompt_len = len(prompt_tokens)
            start = max(0, prompt_len - 1)
            end = start + len(response_tokens)
            response_logits = logits[0, start:end, :]
            response_token_ids = torch.tensor(response_tokens, dtype=torch.long, device=response_logits.device)
            
            ce = F.cross_entropy(response_logits, response_token_ids, reduction='sum')
            all_log_probs.append(-ce)
            
            cleanup_memory()
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            all_log_probs.append(torch.tensor(0.0, requires_grad=True))
    
    return torch.stack(all_log_probs)

def compute_ref_log_probs(model, tokenizer, prompts, responses):
    """计算参考模型的序列对数概率（无梯度）"""
    all_log_probs = []
    
    with torch.inference_mode():
        for prompt, response in zip(prompts, responses):
            try:
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                
                if len(response_tokens) == 0:
                    all_log_probs.append(torch.tensor(0.0))
                    continue
                    
                full_tokens = prompt_tokens + response_tokens
                input_ids = torch.tensor([full_tokens], dtype=torch.long)
                
                if input_ids.size(1) > 512:
                    all_log_probs.append(torch.tensor(0.0))
                    continue
                
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits

                prompt_len = len(prompt_tokens)
                start = max(0, prompt_len - 1)
                end = start + len(response_tokens)
                response_logits = logits[0, start:end, :]
                response_token_ids = torch.tensor(response_tokens, dtype=torch.long, device=response_logits.device)
                
                ce = F.cross_entropy(response_logits, response_token_ids, reduction='sum')
                all_log_probs.append(-ce)
                
                cleanup_memory()
                    
            except Exception as e:
                print(f"Error processing sample: {e}")
                all_log_probs.append(torch.tensor(0.0))
    
    return torch.stack(all_log_probs)

# ==========================================
# 奖励计算
# ==========================================

class RewardComputer:
    def __init__(self, cfg: RewardCfg):
        self.cfg = cfg
        self.external_testcases = {}
        
        # 加载外部测试用例
        if cfg.external_testcase_file and os.path.exists(cfg.external_testcase_file):
            try:
                with open(cfg.external_testcase_file, 'r', encoding='utf-8') as f:
                    if cfg.external_testcase_file.endswith('.jsonl'):
                        for line in f:
                            line = line.strip()
                            if line:
                                data = json.loads(line)
                                if cfg.problem_id_field in data and 'testcases' in data:
                                    problem_id = data[cfg.problem_id_field]
                                    self.external_testcases[problem_id] = data['testcases']
                    else:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for problem_id, content in data.items():
                                if isinstance(content, dict) and 'testcases' in content:
                                    self.external_testcases[problem_id] = content['testcases']
                                elif isinstance(content, list):
                                    self.external_testcases[problem_id] = content
                print(f"[Reward] Loaded {len(self.external_testcases)} external test cases")
            except Exception as e:
                print(f"[Reward] Failed to load external test cases: {e}")
                self.external_testcases = {}

    def __call__(self, sample: Dict, generated_text: str, epoch_ratio: float) -> Dict:
        generated_text = str(generated_text) if generated_text is not None else ''
        chosen_text = str(sample.get('chosen', '')) if sample.get('chosen') is not None else ''
        
        generated_code = extract_code(generated_text)
        chosen_code = extract_code(chosen_text)
        
        compare_generated = generated_code if generated_code.strip() else generated_text
        compare_chosen = chosen_code if chosen_code.strip() else chosen_text
        
        retention, diff_ratio, text_score = compute_text_quality(compare_chosen, compare_generated)
        
        # 测试用例评估
        test_pass_rate = 0.0
        testcases = None
        testcase_source = "none"
        
        if self.cfg.use_test_cases:
            if 'testcases' in sample and sample['testcases']:
                testcases = sample['testcases']
                testcase_source = "inline"
            elif (self.cfg.problem_id_field in sample and 
                  sample[self.cfg.problem_id_field] in self.external_testcases):
                problem_id = sample[self.cfg.problem_id_field]
                testcases = self.external_testcases[problem_id]
                testcase_source = "external"
        
        if testcases:
            try:
                code = extract_code(generated_text)
                test_pass_rate = run_code_with_testcases(code, testcases, self.cfg.test_timeout)
            except Exception as e:
                print(f"Test execution failed: {e}")
                test_pass_rate = 0.0
        
        # 计算最终分数
        if testcases and self.cfg.use_test_cases:
            final_score = (self.cfg.test_weight * test_pass_rate + 
                          self.cfg.text_weight * text_score)
        else:
            final_score = text_score
        
        clipped_score = max(self.cfg.diff_clip_low, min(self.cfg.diff_clip_high, final_score))
        
        return {
            'reward': float(clipped_score),
            'retention': float(retention),
            'diff_ratio': float(diff_ratio),
            'text_score': float(text_score),
            'test_pass_rate': float(test_pass_rate),
            'final_score': float(final_score),
            'testcase_source': testcase_source,
            'used_tests': testcases is not None,
            'num_testcases': len(testcases) if testcases else 0,
        }

# ==========================================
# 训练器
# ==========================================

class SFT_GRPO_Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        self.BOF, self.EOF = get_prompt_format(args.model_name)
        self.prompt_scheduler = PromptScheduler(args.model_name)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 内存清理和自适应配置
        print("🧹 清理内存...")
        cleanup_cuda_memory(force_sync=True, aggressive=True)
        
        adaptive_memory, suggested_batch, suggested_accumulation = get_adaptive_memory_config()
        
        # 动态调整batch size
        original_batch = args.sft_batch
        args.sft_batch = min(suggested_batch, original_batch)
        if args.sft_batch != original_batch:
            print(f"🔄 调整batch size: {original_batch} → {args.sft_batch}")
        
        # 量化配置
        quantization_config = None
        if args.use_8bit and args.use_4bit:
            raise ValueError("Cannot use both 8bit and 4bit quantization")
        elif args.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        elif args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # 模型加载配置
        extra_kwargs = {
            "max_memory": adaptive_memory,
            "offload_folder": "./offload_tmp",
            "offload_state_dict": True,
        } if adaptive_memory else {}
        
        # 加载基础模型
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            **extra_kwargs
        )
        
        # 准备量化训练
        if args.use_8bit or args.use_4bit:
            base = prepare_model_for_kbit_training(base)
        
        # LoRA配置
        if args.use_8bit or args.use_4bit:
            target_modules = self._find_all_linear_names(base, int8=args.use_8bit, int4=args.use_4bit)
        else:
            target_modules = self._get_target_modules(args.base_model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(base, lora_config)
        
        # 启用梯度检查点
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        
        # 性能优化：编译模型
        if args.compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("🚀 模型编译已启用")
            except Exception as e:
                print(f"⚠️ 模型编译失败: {e}")
                args.compile_model = False
            self.model.gradient_checkpointing_enable()
        
        # NEFTune
        if args.use_neftune:
            self._apply_neftune(args.neftune_alpha)
        
        self.reference_model = None
        self._setup_inference_pipeline()
        
        # 确保只有LoRA参数可训练
        self._fix_trainable_params()
        
        # 验证模块设置
        self._verify_lora_modules()
        
        print(f"✓ Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
    
    def _fix_trainable_params(self):
        """确保只有LoRA参数可训练"""
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def _verify_lora_modules(self):
        """验证LoRA模块是否正确应用"""
        lora_modules = []
        total_params = 0
        trainable_params = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules.append(name)
            
            for param in module.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        if lora_modules:
            print(f"✓ {len(lora_modules)} LoRA modules")
        
        print(f"📊 Parameters: {trainable_params:,}/{total_params:,} trainable ({100 * trainable_params / total_params:.1f}%)")
        
        if len(lora_modules) == 0:
            print("⚠️ 未找到LoRA模块")

    def _find_all_linear_names(self, model, int4=False, int8=False) -> List[str]:
        """找到模型中所有线性层名称用于量化训练"""
        cls = torch.nn.Linear
        if int4 or int8:
            import bitsandbytes as bnb
            if int4:
                cls = bnb.nn.Linear4bit
            elif int8:
                cls = bnb.nn.Linear8bitLt
        
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                # 最后一层不加入LoRA
                if 'lm_head' in name or 'output_layer' in name:
                    continue
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(lora_module_names)
    
    def _get_target_modules(self, model_name: str) -> List[str]:
        """根据模型名称选择target_modules"""
        model_name = model_name.lower()
        if "llama" in model_name or "qwen" in model_name:
            return ["q_proj", "v_proj"]
        else:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]

    def _apply_neftune(self, alpha: float):
        """应用NEFTune噪声"""
        import math
        
        def neftune_forward_hook(module, input, output):
            if module.training and alpha > 0:
                dims = output.size(-1)
                noise_std = alpha / math.sqrt(output.numel())
                noise = torch.randn_like(output) * noise_std
                return output + noise
            return output
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                module.register_forward_hook(neftune_forward_hook)
        
        print(f"✓ NEFTune enabled: α={alpha}")

    def _setup_inference_pipeline(self):
        """设置推理pipeline"""
        try:
            self.inference_pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=self.args.generation_batch_size,
                return_full_text=False,
                clean_up_tokenization_spaces=False
            )
            
            # 优化生成配置
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
            self.model.generation_config.use_cache = self.args.use_kv_cache
            self.model.generation_config.do_sample = True
            
            print("✓ 推理pipeline就绪")
        except Exception as e:
            print(f"⚠️ 推理pipeline创建失败: {e}")
            raise e

    def _create_reference_model(self):
        """从 SFT 训练后的模型创建参考模型"""
        try:
            print("📋 Merging LoRA weights for reference model...")
            
            # 创建当前模型的深拷贝
            import copy
            temp_model = copy.deepcopy(self.model)
            
            # 在拷贝上进行合并操作
            merged_model = temp_model.merge_and_unload()
            
            # 保存合并后的权重
            merged_state_dict = merged_model.state_dict()
            
            # 创建新的参考模型实例（CPU）
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True
            )
            
            # 安全加载合并权重到参考模型
            missing_keys, unexpected_keys = self.reference_model.load_state_dict(merged_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ 参考模型缺少键: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"⚠️ 参考模型多余键: {len(unexpected_keys)} 个")
            
            # 清理临时变量
            del temp_model, merged_model, merged_state_dict
            cleanup_cuda_memory()
            
            # 冻结参考模型参数
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            
            print("✓ Reference model created from SFT checkpoint")
            
        except Exception as e:
            print(f"❌ 参考模型创建失败: {e}")
            raise e

    def run_sft(self, dataset: CodeDataset):
        """SFT训练"""
        print("Starting SFT training...")
        self.model.train()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.sft_lr,
            weight_decay=0.1
        )
        
        total_steps = self.args.sft_epochs * len(dataset) // self.args.batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self._collate_sft
        )
        
        global_step = 0
        
        for epoch in range(self.args.sft_epochs):
            epoch_loss = 0.0
            
            for step, batch in enumerate(loader):
                loss = self._sft_forward_step(batch)
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * self.args.gradient_accumulation_steps
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % 100 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"[SFT] epoch {epoch} step {global_step} loss={loss.item() * self.args.gradient_accumulation_steps:.4f} lr={current_lr:.2e}")
                        cleanup_memory()
                        
                    if global_step % 200 == 0:
                        cleanup_cuda_memory()
            
            print(f"[SFT] Epoch {epoch} avg_loss={epoch_loss / len(loader):.4f}")
        
        # 保存SFT模型
        sft_dir = os.path.join(self.args.out_dir, 'sft')
        os.makedirs(sft_dir, exist_ok=True)
        self.model.save_pretrained(sft_dir)
        self.tokenizer.save_pretrained(sft_dir)
        
        # 创建参考模型
        self._create_reference_model()
        
        print("✓ SFT training completed")

    def _sft_forward_step(self, batch):
        """SFT前向传播"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        if self.args.use_dft:
            return self._dft_forward_step(input_ids, attention_mask, labels)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )
            return outputs.loss

    def _dft_forward_step(self, input_ids, attention_mask, labels):
        """DFT前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        # 处理labels
        labels_for_gather = labels.clone()
        labels_for_gather[labels == -100] = 0
        
        # 获取目标token的概率
        target_log_probs = log_probs.gather(dim=-1, index=labels_for_gather.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=labels_for_gather.unsqueeze(-1)).squeeze(-1)
        
        # DFT损失：P(token) * (-log P(token))
        standard_losses = -target_log_probs
        dft_losses = standard_losses * target_probs
        
        # 应用mask
        loss_mask = (labels != -100).float()
        dft_losses = dft_losses * loss_mask
        
        total_loss = dft_losses.sum()
        total_tokens = loss_mask.sum()
        
        if total_tokens > 0:
            return total_loss / total_tokens
        else:
            return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    def _collate_sft(self, batch: List[Dict]):
        """SFT数据整理"""
        max_length = 2048
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for sample in batch:
            prompt = str(sample['prompt'])
            chosen = str(sample['chosen'])
            explanation = str(sample.get('explanation', ''))
            
            # 构建完整响应
            if explanation.strip():
                full_response = f"{explanation.strip()}\n\n{chosen.strip()}"
            else:
                full_response = chosen.strip()
            
            # 格式化文本
            enhanced_prompt = f"{prompt}\n\nYou are a software engineer. Can you repair the incorrect code?"
            formatted_text = f"{self.BOF} {enhanced_prompt} {self.EOF} {full_response}"
            
            # Tokenize
            tokenized = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)
            
            # 计算prompt长度
            prompt_text = f"{self.BOF} {enhanced_prompt} {self.EOF} "
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            
            # 构建labels
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            
            input_ids_list.append(input_ids.tolist())
            attention_mask_list.append(attention_mask.tolist())
            labels_list.append(labels.tolist())
        
        # Padding
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            
            padded_input_ids.append(
                input_ids_list[i] + [self.tokenizer.pad_token_id] * pad_len
            )
            padded_attention_mask.append(
                attention_mask_list[i] + [0] * pad_len
            )
            padded_labels.append(
                labels_list[i] + [-100] * pad_len
            )
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }

    def run_grpo(self, dataset: CodeDataset, curriculum: CurriculumCfg, reward_cfg: RewardCfg):
        """GRPO训练"""
        print("Starting GRPO training...")
        self.model.train()
        
        rewarder = RewardComputer(reward_cfg)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.grpo_lr,
            weight_decay=0.01
        )
        
        total_steps = self.args.grpo_epochs * len(dataset)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=min(self.args.dataloader_num_workers, 4),
            pin_memory=self.args.pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
            prefetch_factor=2 if self.args.dataloader_num_workers > 0 else None
        )
        
        global_step = 0
        accumulation_count = 0
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("✓ 启用自动混合精度训练")
        
        for epoch in range(self.args.grpo_epochs):
            epoch_policy_loss = 0.0
            epoch_reward = 0.0
            epoch_steps = 0
            
            print(f"[GRPO] Starting epoch {epoch}/{self.args.grpo_epochs}")
            print_memory_usage(f"💾 [Epoch {epoch}]")
            cleanup_cuda_memory(force_sync=True)
            
            for step, sample in enumerate(loader):
                sample = sample[0] if isinstance(sample, list) else sample
                
                # 课程学习进度
                epoch_ratio = (epoch + step / len(loader)) / self.args.grpo_epochs
                temp, top_p, K = curriculum.interp(epoch_ratio)
                
                # 生成候选
                candidates_data = self._generate_candidates(sample, temp, top_p, K, epoch_ratio)
                if not candidates_data or len(candidates_data['texts']) < 2:
                    continue
                
                # 计算奖励和概率
                rewards_info = self._compute_rewards_and_probs(
                    sample, candidates_data, rewarder, epoch_ratio
                )
                if not rewards_info:
                    continue
                
                # 计算GRPO损失
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        loss_info = self._compute_grpo_loss(rewards_info)
                else:
                    loss_info = self._compute_grpo_loss(rewards_info)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss = loss_info['total_loss']
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️ 跳过异常loss: {total_loss.item()}")
                    cleanup_cuda_memory()
                    continue
                
                loss_scaled = total_loss / self.args.gradient_accumulation_steps
                
                if self.args.use_amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                
                accumulation_count += 1
                
                # 只有累积到指定步数才更新权重
                if accumulation_count % self.args.gradient_accumulation_steps == 0:
                    if self.args.use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.grad_clip
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.grad_clip
                        )
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                else:
                    grad_norm = 0.0
                
                # 统计信息
                epoch_policy_loss += loss_info['policy_loss'].item()
                epoch_reward += rewards_info['rewards'].mean().item()
                epoch_steps += 1
                
                # 进度输出和内存监控
                if step % 50 == 0:
                    print(f"[GRPO] ep{epoch} step{step}/{len(loader)}")
                    cleanup_cuda_memory(force_sync=True, aggressive=True)
                elif step % 20 == 0:
                    cleanup_cuda_memory()
                
                if global_step % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    reward_mean = rewards_info['rewards'].mean().item()
                    reward_std = rewards_info['rewards'].std().item()
                    print(
                        f"[GRPO] ep{epoch} gs{global_step} "
                        f"loss={total_loss.item():.4f} pl={loss_info['policy_loss'].item():.4f} "
                        f"R={reward_mean:.3f}±{reward_std:.3f} "
                        f"lr={current_lr:.2e} gn={grad_norm:.2f}"
                    )
            
            # Epoch统计
            if epoch_steps > 0:
                avg_policy_loss = epoch_policy_loss / epoch_steps
                avg_reward = epoch_reward / epoch_steps
                print(f"[GRPO] epoch {epoch} avg_policy_loss={avg_policy_loss:.4f} avg_reward={avg_reward:.3f}")
        
        # 保存GRPO模型
        grpo_dir = os.path.join(self.args.out_dir, 'grpo')
        os.makedirs(grpo_dir, exist_ok=True)
        self.model.save_pretrained(grpo_dir)
        self.tokenizer.save_pretrained(grpo_dir)
        
        print("✓ GRPO training completed")

    def _generate_candidates(self, sample: Dict, temp: float, top_p: float, K: int, epoch_ratio: float) -> Dict:
        """生成候选文本"""
        base_prompt = str(sample['prompt'])
        
        diverse_prompt, strategy_type = self.prompt_scheduler.get_diverse_prompt(
            base_prompt, epoch_ratio, 0
        )
        
        was_training = self.model.training
        self.model.eval()
        
        texts = []
        
        with torch.no_grad():
            try:
                outputs = self.inference_pipe(
                    diverse_prompt,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=max(0.1, min(2.0, temp)),
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=K,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                if isinstance(outputs, list):
                    for o in outputs:
                        if not o or 'generated_text' not in o:
                            continue
                        full_text = str(o['generated_text'])
                        # pipeline已返回生成部分，但保持兼容逻辑
                        if self.EOF in full_text:
                            text = full_text.split(self.EOF)[-1].strip()
                        else:
                            text = full_text.strip()
                        if not text:
                            continue
                        # 仅保留包含有效C++代码的候选
                        if extract_code(text):
                            texts.append(text)
                else:
                    # 兼容性兜底
                    full_text = str(outputs)
                    if self.EOF in full_text:
                        text = full_text.split(self.EOF)[-1].strip()
                    else:
                        text = full_text.strip()
                    if extract_code(text):
                        texts.append(text)
                
            except Exception as e:
                print(f"Generation error: {e}")
                texts = []
        
        if was_training:
            self.model.train()
        
        cleanup_cuda_memory(force_sync=False, aggressive=False)
        
        if len(texts) == 0:
            print(f"[GRPO] Candidate generation failed (0 valid)")
            return {}
        
        return {
            'texts': texts,
            'base_prompt': base_prompt,
            'epoch_ratio': epoch_ratio
        }

    def _compute_rewards_and_probs(self, sample: Dict, candidates_data: Dict, 
                                  rewarder: RewardComputer, epoch_ratio: float) -> Dict:
        """计算奖励和概率"""
        texts = candidates_data['texts']
        base_prompt = candidates_data['base_prompt']
        
        if len(texts) < 2:
            return {}
        
        # 计算奖励（完全在CPU上进行）
        rewards = []
        extras = []
        with torch.no_grad():
            for i, text in enumerate(texts):
                reward_info = rewarder(sample, str(text), epoch_ratio)
                rewards.append(reward_info['reward'])
                extras.append(reward_info)
                # 每计算一个奖励后清理一次内存
                if i % 2 == 0:
                    cleanup_memory()
        
        # 生成prompt
        diverse_prompt, _ = self.prompt_scheduler.get_diverse_prompt(
            base_prompt, epoch_ratio, 0
        )
        diverse_prompts = [diverse_prompt] * len(texts)
        
        # 概率计算前进行深度内存清理
        print("🧹 准备计算对数概率，进行深度内存清理...")
        cleanup_cuda_memory(force_sync=True, aggressive=True)
        
        # 计算参考概率
        print(f"📊 计算 {len(texts)} 个候选的对数概率（参考模型）...")
        ref_log_probs = compute_ref_log_probs(self.reference_model, self.tokenizer, diverse_prompts, texts)
        cleanup_memory()
        
        # 所有统计计算在CPU上进行
        rewards_cpu = torch.tensor(rewards, dtype=torch.float32, device='cpu')
        baseline_cpu = rewards_cpu.mean()
        advantages_cpu = rewards_cpu - baseline_cpu
        
        # 改进的优势函数归一化和裁剪
        if len(advantages_cpu) > 1:
            adv_mean = advantages_cpu.mean()
            adv_std = advantages_cpu.std()
            
            # 更保守的归一化
            if adv_std > 1e-6:
                scale_factor = 1.0 / (adv_std + 1e-6)
                scale_factor = torch.clamp(scale_factor, max=10.0)
                advantages_cpu = (advantages_cpu - adv_mean) * scale_factor
            else:
                advantages_cpu = advantages_cpu - adv_mean
        
        # 更严格的裁剪优势函数
        clip_value = 2.0
        advantages_cpu = torch.clamp(advantages_cpu, -clip_value, clip_value)
        
        # 保持在CPU，策略阶段再转到策略设备
        baseline = baseline_cpu.item()
        
        return {
            'rewards': rewards_cpu,
            'ref_logps': ref_log_probs,
            'advantages': advantages_cpu,
            'extras': extras,
            'baseline': baseline,
            'texts': texts,
            'prompts': diverse_prompts,
        }

    def _compute_grpo_loss(self, rewards_info: Dict) -> Dict:
        """计算GRPO损失"""
        # 计算策略概率
        prompts = rewards_info['prompts']
        texts = rewards_info['texts']
        policy_log_probs = compute_log_probs_with_grad(self.model, self.tokenizer, prompts, texts)

        ref_log_probs = rewards_info['ref_logps']
        advantages = rewards_info['advantages']

        # 设备对齐
        device = policy_log_probs.device
        if ref_log_probs.device != device:
            ref_log_probs = ref_log_probs.to(device)
        if advantages.device != device:
            advantages = advantages.to(device)
        
        if not policy_log_probs.requires_grad:
            return {}
        
        # 计算概率比率
        log_ratio = policy_log_probs - ref_log_probs.detach()
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
        ratio = torch.exp(log_ratio)
        
        # PPO损失
        clipped_ratio = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param)
        advantages_clamped = torch.clamp(advantages.detach(), min=-5.0, max=5.0)
        
        policy_loss_1 = -advantages_clamped * ratio
        policy_loss_2 = -advantages_clamped * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # KL惩罚
        kl_penalty = torch.clamp(log_ratio.mean(), min=-2.0, max=2.0)
        
        # 总损失
        total_loss = policy_loss + self.args.kl_coeff * kl_penalty
        
        # 安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.abs() > 100.0:
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)
            policy_loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'kl_penalty': kl_penalty,
        }

# ==========================================
# 主函数
# ==========================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SFT + GRPO Training')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    args = parser.parse_args()

    # 创建训练参数
    train_args = TrainArgs(model_name=args.model_name)
    
    print("🚀 SFT + GRPO Training")
    print(f"📁 Model: {train_args.model_name}")
    print(f"📂 Data: {train_args.train_file}")
    print(f"💾 Output: {train_args.out_dir}")
    
    # 确保输出目录存在
    os.makedirs(train_args.out_dir, exist_ok=True)

    # 加载数据集
    dataset = CodeDataset(train_args.train_file)
    print(f"📖 Dataset: {len(dataset)} samples")
    
    # 初始化训练器
    trainer = SFT_GRPO_Trainer(train_args)

    # 检查是否已有模型
    sft_model_path = os.path.join(train_args.out_dir, 'sft')
    grpo_model_path = os.path.join(train_args.out_dir, 'grpo')
    
    if os.path.exists(grpo_model_path):
        print("✓ GRPO模型已存在")
        return
    elif os.path.exists(sft_model_path):
        print("✓ SFT模型已存在，跳过SFT阶段，直接进行GRPO")
        
        try:
            # 加载已有的SFT模型继续GRPO训练
            print("🔧 Loading existing SFT model for GRPO...")
            cleanup_cuda_memory(force_sync=True, aggressive=True)
            
            adaptive_memory, suggested_batch, suggested_accumulation = get_adaptive_memory_config()
            
            # 动态调整参数
            train_args.sft_batch = min(suggested_batch, train_args.sft_batch)
            train_args.gradient_accumulation_steps = max(suggested_accumulation, train_args.gradient_accumulation_steps)
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                train_args.base_model,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
                max_memory=adaptive_memory
            )
            
            # 加载SFT适配器作为策略模型
            trainer.model = PeftModel.from_pretrained(
                base_model,
                sft_model_path,
                dtype=torch.float16,
                is_trainable=True
            )
            
            trainer._fix_trainable_params()
            trainer.model.train()
            
            # 创建推理pipeline
            trainer.inference_pipe = pipeline(
                "text-generation", 
                model=trainer.model, 
                tokenizer=trainer.tokenizer,
                return_full_text=False
            )
            
            # 创建参考模型
            ref_base_model = AutoModelForCausalLM.from_pretrained(
                train_args.base_model,
                dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True
            )
            
            trainer.reference_model = PeftModel.from_pretrained(
                ref_base_model,
                sft_model_path,
                dtype=torch.float32,
                is_trainable=False
            )
            
            for param in trainer.reference_model.parameters():
                param.requires_grad = False
            trainer.reference_model.eval()
            
            print("✓ SFT模型加载完成，准备GRPO训练")
            
        except Exception as e:
            print(f"❌ SFT模型加载失败: {e}")
            return
    else:
        # SFT训练
        print('\n🎓 [Stage 1] SFT Training')
        trainer.run_sft(dataset)

    # GRPO训练
    print('\n🎯 [Stage 2] GRPO Training')
    print("-" * 60)
    
    # 使用固定的课程学习和奖励配置
    curriculum_cfg = CurriculumCfg(
        start_temp=1.2, end_temp=0.8,
        start_top_p=0.95, end_top_p=0.8,
        start_num=3, end_num=3
    )
    
    reward_cfg = RewardCfg(
        diff_clip_low=-1.0, diff_clip_high=1.0,
        use_test_cases=True,
        test_weight=0.6,
        text_weight=0.4,
        test_timeout=5.0
    )
    
    try:
        trainer.run_grpo(dataset, curriculum_cfg, reward_cfg)
    except Exception as e:
        print(f"❌ GRPO训练失败: {e}")

    print('\n🎉 Training Completed!')
    print("=" * 60)
    print(f"📁 Models saved under: {train_args.out_dir}")
    print(f"  - SFT model: {train_args.out_dir}/sft/")
    print(f"  - Final GRPO model: {train_args.out_dir}/grpo/")
    print("🔬 Ready for experiments and evaluation!")
    print("=" * 60)

if __name__ == '__main__':
    import warnings
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 激进的缓存池管理策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
    
    # 设置使用GPU 1和2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    
    # 禁用CUDA调试以提升性能
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"
    
    # 额外的性能优化
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["TORCH_CUDNN_ALLOW_TF32"] = "1"
    
    print("🚀 SFT + GRPO 训练性能优化已启用")
    
    # 显示GPU设备信息
    if torch.cuda.is_available():
        print(f"🎯 CUDA_VISIBLE_DEVICES设置为: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        print(f"📊 PyTorch检测到 {torch.cuda.device_count()} 个GPU:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    else:
        print("❌ CUDA不可用")
    
    # 过滤LoRA相关的meta parameter警告
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    warnings.filterwarnings("ignore", message=".*Did you mean to pass.*assign=True.*")
    
    main()
