


import os
import io
import re
import json
import math
import time
import difflib
import random
import tempfile
import subprocess
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# 设置模型别名前缀对应的开头与结尾提示
MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>'),
    'codellama': ('[INST]', '[/INST]'),
    'llama': ('[INST]', '[/INST]'),
    'mistral': ('[INST]', '[/INST]'),
    'starchat': ('<|system|>\n<|end|>\n<|user|>', '<|end|>\n<|assistant|>'),
}

def get_prompt_format(model_key):
    """根据模型key获取prompt格式"""
    for key in MODEL_PROMPT_FORMATS:
        if model_key.lower().startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    # 默认格式
    return '[INST]', '[/INST]'

# 内存优化设置 - 针对内存不足问题的优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
# 设置更小的分割大小和启用可扩展段来减少碎片

# 优化注意力内核：默认启用高效SDPA/Flash，遇到兼容问题时可通过环境变量关闭
try:
    from torch.backends.cuda import sdp_kernel
    _disable_sdp = os.environ.get("DISABLE_SDP", "0") == "1"
    if _disable_sdp:
        # 显式禁用，保障兼容性
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        print("⚙️ SDPA/Flash 关闭 (DISABLE_SDP=1)")
    else:
        # 启用高效实现提升生成速度
        sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        print("⚙️ SDPA/Flash 已启用 (可通过 DISABLE_SDP=1 关闭)")
except Exception as _sdp_e:
    print(f"[WARNING] SDPA/Flash 配置失败，使用默认实现: {_sdp_e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, prepare_model_for_kbit_training

# =========================
# 数据加载
# =========================
class CodeDataset(Dataset):
    def __init__(self, path: str):
        self.data = []
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.data.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    self.data = obj
                else:
                    self.data = [obj]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 必需字段校验
        for k in ["prompt", "chosen"]:
            if k not in item:
                raise ValueError(f"Sample {idx} missing field: {k}")
        item.setdefault("explanation", "")
        return item

# =========================
# 工具：文本质量评估
# =========================

def extract_first_cpp_code(s: str) -> str:
    """从生成的文本中提取C++代码块，增强版本处理格式不规范的情况"""
    # 首先尝试标准的```cpp格式
    matches = re.findall(r'```cpp(.*?)```', s, re.DOTALL)
    
    # 如果没有标准格式，尝试```c++格式
    if not matches:
        matches = re.findall(r'```c\+\+(.*?)```', s, re.DOTALL)
    
    # 如果还没有，尝试纯```格式
    if not matches:
        matches = re.findall(r'```(.*?)```', s, re.DOTALL)
    
    if not matches:
        # 最后尝试：如果没有代码块但包含#include，可能是裸代码
        if '#include' in s:
            return s.strip()
        return ""
    
    def is_valid_cpp_code(code):
        """判断是否为有效的C++代码"""
        code = code.strip()
        if not code:
            return False
        # 包含C++特征
        cpp_features = ['#include', 'using namespace', 'int main', 'void ', 'class ', 'struct ']
        return any(feature in code for feature in cpp_features)
    
    def clean_code(code):
        """清理代码格式"""
        lines = code.strip().split('\n')
        # 移除空白行和过多的空格
        cleaned_lines = []
        for line in lines:
            line = line.rstrip()  # 移除行尾空格
            if line or (cleaned_lines and cleaned_lines[-1]):  # 保留有意义的空行
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    # 尝试每个匹配的代码块
    for i, code in enumerate(matches):
        cleaned = clean_code(code)
        if is_valid_cpp_code(cleaned):
            return cleaned
    
    # 如果所有代码块都不理想，返回第一个清理后的代码
    if matches:
        return clean_code(matches[0])
    
    return ""

def extract_code(text: str) -> str:
    """从生成的文本中提取代码"""
    # 确保输入是字符串
    text = str(text) if text is not None else ''
    
    # 使用增强的C++代码提取逻辑
    cpp_code = extract_first_cpp_code(text)
    if cpp_code:
        return cpp_code
    
    # 如果没有找到代码块，返回整个文本（去除空行）
    return '\n'.join(line for line in text.splitlines() if line.strip())

def run_code_with_testcases(code: str, testcases: List[Dict], timeout: float = 5.0) -> float:
    """执行C++代码并运行测试用例，返回通过率"""
    if not testcases:
        return 0.0
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            cpp_file = f.name
        
        import platform
        exe_file = cpp_file.replace('.cpp', '.exe' if platform.system() == 'Windows' else '')
        
        compile_result = subprocess.run(
            ['g++', '-o', exe_file, cpp_file, '-std=c++17'],
            capture_output=True, timeout=timeout
        )
        
        if compile_result.returncode != 0:
            os.unlink(cpp_file)
            return 0.0
        
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
        
        try:
            os.unlink(cpp_file)
            os.unlink(exe_file)
        except:
            pass
            
        return passed / len(testcases)
        
    except:
        return 0.0

def compute_text_quality(ref_text: str, cand_text: str) -> Tuple[float, float, float]:
    """计算文本质量：retention, diff_ratio, score"""
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

# =========================
# 课程学习调度器
# =========================
@dataclass
class CurriculumCfg:
    start_temp: float = 1.2
    end_temp: float = 0.8
    start_top_p: float = 0.95
    end_top_p: float = 0.8
    start_num: int = 3  # 降低初始候选数量，减少内存使用
    end_num: int = 3  # 后期也维持3个候选

    def interp(self, ratio: float) -> Tuple[float, float, int]:
        """返回生成参数（温度、top_p、候选数量）"""
        ratio = min(max(ratio, 0.0), 1.0)
        temp = self.start_temp + (self.end_temp - self.start_temp) * ratio
        top_p = self.start_top_p + (self.end_top_p - self.start_top_p) * ratio
        num = int(round(self.start_num + (self.end_num - self.start_num) * ratio))
        return temp, top_p, max(1, num)

class PromptScheduler:
    """分早晚期的多元化prompt调度器"""
    
    def __init__(self, model_key: str):
        self.model_key = model_key
        
        # 创新型修复：鼓励大胆改动和多种方案
        self.innovative_prompts = [
            "Please try to propose multiple different repair methods, which can have major changes, but ensure that the code can be compiled and run:",
        ]
        
        # 保守型修复：最小化改动，保持原有结构  
        self.conservative_prompts = [
            "Please generate a minimal repair patch that preserves the original code structure and only modifies the necessary parts to fix the error:",
        ]
        
        # 通用的任务描述
        self.base_instruction = (
            "You are an expert software engineer. Analyze the incorrect code carefully and provide a correct implementation. "
            "Generate ONLY the corrected C++ code inside a code block, without explanations outside the code."
        )
    
    def get_diverse_prompt(self, base_prompt: str, epoch_ratio: float, candidate_idx: int = 0) -> str:
        """根据训练进度生成prompt"""
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
    
    def get_exploration_boost(self, epoch_ratio: float) -> float:
        """获取探索增强系数"""
        if epoch_ratio < 0.3:
            return 0.2
        elif epoch_ratio < 0.6:
            return 0.1
        else:
            return 0.0

# =========================
# 概率/损失相关：logprob、KL、优势
# =========================
def _is_sharded(model):
    """检测模型是否被分片到多个设备"""
    return hasattr(model, "hf_device_map") and model.hf_device_map is not None

def compute_log_probs_with_grad_sequential(model, tokenizer, prompts, responses, device=None):
    """计算序列对数概率(保留梯度) - 顺序处理版本"""
    if device is None:
        try:
            device = model.get_output_embeddings().weight.device
        except Exception:
            device = next(model.parameters()).device

    # 计算前进行深度内存清理
    print("[LOGPROB_GRAD] 开始计算前清理显存...")
    aggressive_cache_cleanup()
    cleanup_cuda_memory(force_sync=True, aggressive=True)
    
    all_log_probs = []
    
    # 逐个处理每个样本（标准做法：单样本前向，向量化聚合response段）
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        try:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            
            if len(response_tokens) == 0:
                all_log_probs.append(torch.tensor(0.0, device=device, requires_grad=True))
                continue
                
            full_tokens = prompt_tokens + response_tokens
            # 在CPU上构造输入，避免将整段一次性放到单卡
            input_ids = torch.tensor([full_tokens], dtype=torch.long)  
            attention_mask = torch.ones_like(input_ids)
            
            # 添加长度检查，避免过长序列导致OOM
            if input_ids.size(1) > 1024:  # 非常严格的长度限制，避免OOM4
                print(f"[LOGPROB_GRAD] 跳过过长序列 (长度: {input_ids.size(1)})")
                all_log_probs.append(torch.tensor(0.0, device=device, requires_grad=True))
                continue
            
            # 移动到GPU并进行前向传播
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # 单个样本前向传播 (保留梯度)；配合AMP和梯度检查点降低显存
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    # 启用梯度检查点以节省显存
                    if hasattr(model, 'gradient_checkpointing_enable'):
                        model.gradient_checkpointing_enable()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits

            # 计算response部分的对数概率（标准向量化实现）
            prompt_len = len(prompt_tokens)
            start = max(0, prompt_len - 1)
            end = start + len(response_tokens)
            response_logits = logits[0, start:end, :]
            # 将label放到logits所在设备上
            response_token_ids = torch.tensor(response_tokens, dtype=torch.long, device=response_logits.device)
            # 使用融合的交叉熵，内存更省（等价于逐token -log_softmax 的和）
            ce = F.cross_entropy(response_logits, response_token_ids, reduction='sum')
            all_log_probs.append((-ce).to(device))
            
            # 每个样本后都进行内存清理以避免OOM
            cleanup_cuda_memory(force_sync=True)
            
            # 手动释放中间变量
            del outputs, logits, response_logits
            if 'input_ids' in locals():
                del input_ids
            if 'attention_mask' in locals():
                del attention_mask
                
        except Exception as e:
            print(f"[LOGPROB_GRAD] Error processing sample {i}: {e}")
            all_log_probs.append(torch.tensor(0.0, device=device, requires_grad=True))
            cleanup_cuda_memory()
    
    return torch.stack(all_log_probs)

def compute_ref_log_probs_sequential(model, tokenizer, prompts, responses, device=None):
    """计算参考模型的序列对数概率（无梯度）- 顺序处理版本"""
    if device is None:
        # 参考模型可能在CPU
        try:
            device = model.get_output_embeddings().weight.device
        except Exception:
            device = next(model.parameters()).device

    # 计算前进行深度内存清理
    print("[REF_LOGPROB] 开始计算前清理显存...")
    aggressive_cache_cleanup()
    cleanup_cuda_memory(force_sync=True, aggressive=True)
    
    all_log_probs = []
    
    # 逐个处理每个样本（参考模型不需要梯度；标准向量化实现）
    with torch.inference_mode():
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            try:
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                
                if len(response_tokens) == 0:
                    all_log_probs.append(torch.tensor(0.0, device=device))
                    continue
                    
                full_tokens = prompt_tokens + response_tokens
                # 在CPU上构造输入，避免将整段一次性放到单卡
                input_ids = torch.tensor([full_tokens], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                
                # 移动到模型设备并进行前向传播（无梯度）
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # 添加内存检查，如果输入太长就跳过
                if input_ids.size(1) > 512:  # 非常严格的长度限制，避免OOM
                    print(f"[REF_LOGPROB] 跳过过长序列 (长度: {input_ids.size(1)})")
                    all_log_probs.append(torch.tensor(0.0, device=device))
                    continue
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits

                # 计算response部分的对数概率（标准向量化实现）
                prompt_len = len(prompt_tokens)
                start = max(0, prompt_len - 1)
                end = start + len(response_tokens)
                response_logits = logits[0, start:end, :]
                response_token_ids = torch.tensor(response_tokens, dtype=torch.long, device=response_logits.device)
                ce = F.cross_entropy(response_logits, response_token_ids, reduction='sum')
                all_log_probs.append((-ce).to(device))
                
                # 每个样本后都进行内存清理以避免OOM
                cleanup_cuda_memory(force_sync=True)
                
                # 手动释放中间变量
                del outputs, logits, response_logits
                if 'input_ids' in locals():
                    del input_ids
                if 'attention_mask' in locals():
                    del attention_mask
                    
            except Exception as e:
                print(f"[REF_LOGPROB] Error processing sample {i}: {e}")
                all_log_probs.append(torch.tensor(0.0, device=device))
                cleanup_cuda_memory()
    
    return torch.stack(all_log_probs)


def cleanup_cuda_memory(force_sync=False, aggressive=False):
    """激进的内存清理 - 专门清理缓存池"""
    if torch.cuda.is_available():
        try:
            # 记录清理前的缓存池大小
            before_cache = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                before_cache[i] = reserved - allocated
            
            if force_sync:
                torch.cuda.synchronize()
            
            # 标准清理
            torch.cuda.empty_cache()
            gc.collect()
            
            # 激进清理缓存池
            if aggressive:
                # 多轮清理，确保缓存池彻底释放
                for round_num in range(5):
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    
            # 再次清理确保彻底
            torch.cuda.empty_cache()
            
            # 显示清理效果
            if force_sync or aggressive:
                total_freed = 0
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    after_cache = reserved - allocated
                    freed = before_cache[i] - after_cache
                    total_freed += freed
                
                if total_freed > 0.5:
                    print(f"✅ 释放缓存 {total_freed:.1f}GB")
                        
        except RuntimeError as e:
            print(f"[WARNING] CUDA cleanup error: {e}")

def aggressive_cache_cleanup():
    """超级激进的缓存池清理"""
    if not torch.cuda.is_available():
        return
    
    # 清理前状态
    before_stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        cache_size = reserved - allocated
        before_stats[i] = {'allocated': allocated, 'reserved': reserved, 'cache': cache_size}
    
    # 超级清理流程
    torch.cuda.synchronize()
    
    # 多轮激进清理
    for round_num in range(10):
        torch.cuda.empty_cache()
        gc.collect()
        if round_num % 2 == 0:
            torch.cuda.synchronize()
    
    # 清理后统计
    total_cache_freed = 0
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        cache_size = reserved - allocated
        cache_freed = before_stats[i]['cache'] - cache_size
        if cache_freed > 0:
            total_cache_freed += cache_freed
    
    if total_cache_freed > 0:
        print(f"✅ 释放缓存: {total_cache_freed:.1f}GB")

def handle_cuda_error(e, context=""):
    """处理CUDA错误"""
    print(f"[CUDA ERROR] {context}: {e}")
    
    if "illegal memory access" in str(e):
        print("🔧 检测到非法内存访问，尝试恢复...")
        
        # 强制同步和清理
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            print("✓ CUDA恢复操作完成")
        except Exception as cleanup_e:
            print(f"❌ CUDA恢复失败: {cleanup_e}")
            
    elif "out of memory" in str(e):
        print("🔧 检测到内存不足，执行激进清理...")
        cleanup_cuda_memory(force_sync=True)
        
    return False  # 表示错误未完全恢复

def check_memory_pressure():
    """检查内存压力，返回是否需要更激进的内存管理"""
    if not torch.cuda.is_available():
        return False
    
    high_pressure = False
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        
        # 如果任何GPU使用超过90%内存，就认为是高压力
        usage_ratio = cached / total
        if usage_ratio > 0.9:
            high_pressure = True
            print(f"⚠️ GPU {i} 内存压力高: {usage_ratio:.1%}")
    
    return high_pressure

def print_memory_usage(prefix=""):
    """打印内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - cached
            print(f"{prefix} GPU {i}: {allocated:.1f}GB/{total:.1f}GB (缓存:{cached:.1f}GB)")

def get_adaptive_memory_config():
    """自适应获取每个GPU的最佳内存分配"""
    if not torch.cuda.is_available():
        return {}, 2  # 默认batch size
    
    memory_config = {}
    n_gpus = torch.cuda.device_count()
    min_available_memory = float('inf')
    
    print(f"🔍 检测GPU显存: {n_gpus}个GPU")
    
    for i in range(n_gpus):
        # 获取GPU属性
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3  # 转换为GB
        
        # 获取当前使用情况
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        
        # 计算可用内存
        used_memory = max(allocated, cached)
        free_memory = total_memory - used_memory
        
        # 保守分配策略：使用75%的可用内存，至少保留2GB给系统
        safety_buffer = 2.0  # GB
        usable_memory = max(free_memory - safety_buffer, 4.0)  # 最少4GB
        allocated_memory = min(usable_memory * 0.75, total_memory * 0.85)  # 最多85%总内存
        
        memory_config[i] = f"{allocated_memory:.1f}GiB"
        min_available_memory = min(min_available_memory, allocated_memory)
        
        print(f"📊 GPU {i}: 总计{total_memory:.1f}GB, 可用{free_memory:.1f}GB, 分配{allocated_memory:.1f}GB")
    
    # 根据最小可用内存自动调整batch size和梯度累积
    if min_available_memory >= 15:
        suggested_batch = 2
        suggested_accumulation = 2  # 有效batch=4
    elif min_available_memory >= 10:
        suggested_batch = 1
        suggested_accumulation = 4  # 有效batch=4
    elif min_available_memory >= 6:
        suggested_batch = 1
        suggested_accumulation = 2  # 有效batch=2
    else:
        suggested_batch = 1
        suggested_accumulation = 1  # 有效batch=1
    
    effective_batch = suggested_batch * suggested_accumulation
    print(f"💡 建议配置: batch={suggested_batch}, 累积={suggested_accumulation}, 有效batch={effective_batch}")
    
    return memory_config, suggested_batch, suggested_accumulation


# =========================
# 奖励函数（基于文本质量评估）
# =========================
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
                        if isinstance(data, list):
                            for item in data:
                                if cfg.problem_id_field in item and 'testcases' in item:
                                    problem_id = item[cfg.problem_id_field]
                                    self.external_testcases[problem_id] = item['testcases']
                        elif isinstance(data, dict):
                            for problem_id, content in data.items():
                                if isinstance(content, dict) and 'testcases' in content:
                                    self.external_testcases[problem_id] = content['testcases']
                                elif isinstance(content, list):
                                    self.external_testcases[problem_id] = content
                total_problems = len(self.external_testcases)
                total_testcases = sum(len(testcases) for testcases in self.external_testcases.values())
                print(f"[Reward] Loaded external test cases:")
                print(f"[Reward]   - {total_problems} problems with test cases")
                print(f"[Reward]   - {total_testcases} total test cases")
                if total_problems > 0:
                    avg_tests = total_testcases / total_problems
                    print(f"[Reward]   - {avg_tests:.1f} test cases per problem on average")
            except Exception as e:
                print(f"[Reward] Failed to load external test cases: {e}")
                self.external_testcases = {}

    def __call__(self, sample: Dict, generated_text: str, epoch_ratio: float) -> Dict:
        generated_text = str(generated_text) if generated_text is not None else ''
        chosen_text = str(sample.get('chosen', '')) if sample.get('chosen') is not None else ''
        
        if chosen_text.startswith("['") and chosen_text.endswith("']"):
            chosen_text = chosen_text[2:-2]
            chosen_text = chosen_text.replace("\\n", "\n")
            chosen_text = chosen_text.replace("\\'", "'")
            chosen_text = chosen_text.replace('\\"', '"')
        
        generated_code = extract_code(generated_text)
        chosen_code = extract_code(chosen_text)
        
        compare_generated = generated_code if generated_code.strip() else generated_text
        compare_chosen = chosen_code if chosen_code.strip() else chosen_text
        
        retention, diff_ratio, text_score = compute_text_quality(compare_chosen, compare_generated)
        
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
        
        test_pass_rate = 0.0
        if testcases:
            try:
                code = extract_code(generated_text)
                test_pass_rate = run_code_with_testcases(code, testcases, self.cfg.test_timeout)
            except Exception as e:
                print(f"[Reward] Test execution failed: {e}")
                test_pass_rate = 0.0
        
        if testcases:
            final_score = (self.cfg.test_weight * test_pass_rate + 
                          self.cfg.text_weight * text_score)
            used_tests = True
        else:
            final_score = text_score
            used_tests = False
        
        clipped_score = max(self.cfg.diff_clip_low, min(self.cfg.diff_clip_high, final_score))
        
        return {
            'reward': float(clipped_score),
            'retention': float(retention),
            'diff_ratio': float(diff_ratio),
            'text_score': float(text_score),
            'test_pass_rate': float(test_pass_rate),
            'final_score': float(final_score),
            'testcase_source': testcase_source,  # 测试用例来源
            'used_tests': used_tests,            # 是否使用了测试用例
            'num_testcases': len(testcases) if testcases else 0,  # 测试用例数量
        }

# =========================
# 训练器：SFT + GRPO
# =========================
@dataclass
class TrainArgs:
    model_name: str
    model_base_path: str = "/home/liu01/projects/base_model"
    train_file: str = "/home/liu01/projects/prorepair/data/trainset/sft_dataset.json"
    output_base_path: str = "/home/liu01/projects/train_model"
    device: str = 'auto'  # 设置了CUDA_VISIBLE_DEVICES="0,2"，将自动使用GPU 0和2
    
    sft_epochs: int = 5
    grpo_epochs: int = 2
    sft_lr: float = 3e-5
    grpo_lr: float = 1.5e-5
    sft_batch: int = 1
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    
    dataloader_num_workers: int = 4  # 增加数据加载并行度
    pin_memory: bool = True
    compile_model: bool = True
    
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    max_new_tokens: int = 512
    grad_clip: float = 1.0
    seed: int = 42
    
    generation_batch_size: int = 1  # 恢复为1以解决OOM问题
    use_kv_cache: bool = True
    reduce_memory_cleanup: bool = True
    
    use_dft: bool = True
    use_neftune: bool = True
    neftune_alpha: float = 5.0
    use_8bit: bool = True
    use_4bit: bool = False
    external_testcase_file: str = "/home/liu01/projects/prorepair/data/trainset/testcases_sorted.json"
    problem_id_field: str = "problem_id"
    clip_param: float = 0.2
    kl_coeff: float = 0.01
    use_kl_penalty: bool = True
    advantage_normalization: bool = True
    advantage_clip: float = 10.0
    
    @property
    def base_model(self) -> str:
        """动态生成完整的模型路径"""
        return os.path.join(self.model_base_path, self.model_name)
    
    @property
    def out_dir(self) -> str:
        model_short_name = self.model_name.lower().replace("-", "_")
        return os.path.join(self.output_base_path, f"sft_grpo_{model_short_name}")

class SFT_GRPO_Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        random.seed(args.seed); torch.manual_seed(args.seed)
        
        # 获取模型的prompt格式
        self.BOF, self.EOF = get_prompt_format(args.model_name)
        
        # 初始化多元化prompt调度器
        self.prompt_scheduler = PromptScheduler(args.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 配置量化
        quantization_config = None
        if args.use_8bit and args.use_4bit:
            raise ValueError("Cannot use both 8bit and 4bit quantization simultaneously")
        elif args.use_8bit:
            print(f"🔧 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif args.use_4bit:
            print(f"🔧 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        n_gpus = torch.cuda.device_count()
        # 统一使用auto模式
        devmap = "auto"
        print(f"🔧 加载模型: {n_gpus}个GPU")
        
        try:
            # 模型加载前清理内存
            print("🧹 清理内存...")
            for i in range(5):
                torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            aggressive_cache_cleanup()  # 激进清理缓存池
            print("✅ 内存清理完成")
            
            adaptive_memory, suggested_batch, suggested_accumulation = get_adaptive_memory_config()
            
            # 为模型加载预留更多内存 - 更保守的分配策略
            conservative_memory = {}
            for device_id, memory_str in adaptive_memory.items():
                memory_gb = float(memory_str.replace("GiB", ""))
                # 更保守的内存分配（减少30%分配）
                conservative_gb = memory_gb * 0.7
                conservative_memory[device_id] = f"{conservative_gb:.1f}GiB"
            
            # 增加更多CPU卸载选项
            extra_kwargs = {
                "max_memory": conservative_memory,
                "offload_folder": "./offload_tmp",
                "offload_state_dict": True,
            } if conservative_memory else {
                "offload_folder": "./offload_tmp",
                "offload_state_dict": True,
            }
            
            # 动态调整batch size
            original_batch = args.sft_batch
            args.sft_batch = min(suggested_batch, original_batch)
            if args.sft_batch != original_batch:
                print(f"🔄 调整batch size: {original_batch} → {args.sft_batch}")
            
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                dtype=torch.bfloat16 if not (args.use_8bit or args.use_4bit) else "auto",
                device_map=devmap,
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                **extra_kwargs
            )
            pass
        except Exception as e:
            raise e
        
        # 如果使用量化，需要准备模型
        if args.use_8bit or args.use_4bit:
            try:
                base = prepare_model_for_kbit_training(base)
            except Exception as e:
                raise e
        
        # 根据模型类型自动选择target_modules
        if args.use_8bit or args.use_4bit:
            # 量化模型使用更全面的target_modules
            target_modules = self._find_all_linear_names(base, int8=args.use_8bit, int4=args.use_4bit)
        else:
            # 非量化模型使用基本的target_modules
            target_modules = self._get_target_modules(args.base_model)
        
        print(f"🔧 LoRA配置: r={args.lora_r}, α={args.lora_alpha}")
        
        try:
            lora = LoraConfig(
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout,
                target_modules=target_modules, 
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(base, lora)  # 不移动已经用device_map分布的模型
            # 启用梯度检查点，显著降低激活显存占用
            try:
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "config"):
                    self.model.config.use_cache = False
                print("✓ Gradient checkpointing enabled (use_cache=False)")
            except Exception as _gce:
                print(f"⚠️ Failed to enable gradient checkpointing: {_gce}")
            
            # 性能优化：编译模型
            if self.args.compile_model:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("🚀 模型编译已启用")
                except Exception as e:
                    print(f"⚠️ 模型编译失败: {e}")
                    self.args.compile_model = False
            print("✓ LoRA applied")
        except Exception as e:
            raise e
        
        # 参考模型将在SFT训练完成后创建
        self.reference_model = None
        # 创建用于GRPO推理的pipeline
        print("🚀 创建推理pipeline...")
        
        try:
            # 优化的pipeline配置
            self.inference_pipe = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                batch_size=args.generation_batch_size,  # 启用批量生成
                return_full_text=False,  # 只返回生成部分
                clean_up_tokenization_spaces=False  # 减少后处理开销
            )
            
            # 优化生成配置
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
            self.model.generation_config.use_cache = args.use_kv_cache
            self.model.generation_config.do_sample = True
            
            print("✓ 推理pipeline就绪")
        except Exception as e:
            raise e
        
        # 显示设备映射信息
        if hasattr(base, 'hf_device_map'):
            print(f"📱 多GPU映射: {len(base.hf_device_map)}模块")
        
        # 确保只有LoRA参数可训练
        self._fix_trainable_params()
        
        # 应用NEFTune噪声（如果启用）
        if args.use_neftune:
            self._apply_neftune(args.neftune_alpha)
        
        # 验证模块是否正确应用
        self._verify_lora_modules()

    def _fix_trainable_params(self):
        """确保只有LoRA参数可训练"""
        
        for name, param in self.model.named_parameters():
            # 标准LoRA: 只有lora相关参数可训练
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _find_all_linear_names(self, model, int4=False, int8=False) -> List[str]:
        """Find all linear layer names in the model for quantized training. Reference from QLoRA paper."""
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
                # last layer is not add to lora_module_names
                if 'lm_head' in name:
                    continue
                if 'output_layer' in name:
                    continue
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(lora_module_names)

    def _get_target_modules(self, model_name: str) -> List[str]:
        """根据模型名称自动选择target_modules"""
        model_name = model_name.lower()
        
        if "llama" in model_name:
            # Llama, Llama2, Llama3 系列
            return ["q_proj", "v_proj"]
        elif "qwen" in model_name:
            # Qwen, Qwen2 系列 (注意Qwen可能有两种架构)
            # 检查具体的Qwen版本
            return ["q_proj","v_proj"]  # Qwen2使用标准命名
            
        else:
            # 默认使用最通用的Llama格式命名
            return ["q_proj", "v_proj"]
    
    def _verify_lora_modules(self):
        """验证LoRA模块是否正确应用"""
        lora_modules = []
        total_params = 0
        trainable_params = 0
        
        for name, module in self.model.named_modules():
            # 检查标准LoRA模块
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules.append(name)
            
            # 计算参数统计
            for param in module.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        if lora_modules:
            print(f"✓ {len(lora_modules)} LoRA modules")
        
        print(f"📊 Parameters: {trainable_params:,}/{total_params:,} trainable ({100 * trainable_params / total_params:.1f}%)")
        
        if len(lora_modules) == 0:
            print("⚠️ 未找到LoRA模块")
    
    def _apply_neftune(self, alpha: float):
        """应用NEFTune噪声到embedding层
        
        参考MOTrain.py中的neftune_noise_alpha=5实现
        在embedding层的forward过程中添加高斯噪声
        """
        import math
        
        def neftune_forward_hook(module, input, output):
            """NEFTune前向钩子函数"""
            if module.training and alpha > 0:
                # 计算噪声标准差：alpha / sqrt(embedding_dim * batch_size * seq_len)
                dims = output.size(-1)  # embedding dimension
                noise_std = alpha / math.sqrt(output.numel())
                
                # 生成高斯噪声并添加到输出
                noise = torch.randn_like(output) * noise_std
                return output + noise
            return output
        
        # 查找并应用到所有embedding层
        neftune_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                # 注册前向钩子
                module.register_forward_hook(neftune_forward_hook)
                neftune_modules.append(name)
        
        if neftune_modules:
            print(f"🔊 NEFTune已启用: α={alpha}")
        else:
            print("⚠️ 未找到embedding层，NEFTune未启用")

    # ---------- 阶段1：SFT（CoT） ----------
    def run_sft(self, dataset: CodeDataset):
        """完整的SFT训练实现"""
        self.model.train()
        
        # 优化器设置
        optim = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.sft_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )
        
        # 学习率调度器：warmup + cosine decay
        total_steps = self.args.sft_epochs * len(dataset) // self.args.sft_batch
        warmup_steps = int(0.15 * total_steps)   # 15% warmup，小数据集需要更温和的启动
        
        def lr_scheduler_fn(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_scheduler_fn)
        
        # 数据加载器
        loader = DataLoader(
            dataset, 
            batch_size=self.args.sft_batch, 
            shuffle=True, 
            collate_fn=self._collate_sft,
            drop_last=True
        )
        
        # 使用配置中的梯度累积步数
        accumulation_steps = self.args.gradient_accumulation_steps
        
        effective_batch_size = self.args.sft_batch * accumulation_steps
        print(f"[SFT] steps={total_steps} bs={self.args.sft_batch} acc={accumulation_steps} eff={effective_batch_size}")
        
        # 训练前清理内存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        global_step = 0
        
        for ep in range(self.args.sft_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(loader):
                try:
                    # 前向传播
                    loss = self._sft_forward_step(batch)
                    
                    # 梯度累积
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    epoch_loss += loss.item() * accumulation_steps
                    epoch_steps += 1
                    
                    # 梯度更新
                    if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                        # 梯度裁剪
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.args.grad_clip
                        )
                        
                        # 优化器步骤
                        optim.step()
                        scheduler.step()
                        optim.zero_grad()
                        
                        global_step += 1
                        
                        if global_step % 200 == 0:
                            cleanup_cuda_memory()
                        
                        # 日志输出
                        if global_step % 100 == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            print(f"[SFT] ep{ep} gs{global_step}/{total_steps} loss={loss.item() * accumulation_steps:.4f} lr={current_lr:.2e} gn={grad_norm:.2f}")
                
                except Exception as e:
                    print(f"[SFT] Error at step {step}: {e}")
                    cleanup_cuda_memory()
                    continue
            
            # Epoch 统计
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"[SFT] epoch {ep} avg_loss={avg_loss:.4f}")
        
        print("[SFT] Saving model...")
        
        # 保存 SFT 模型（供 GRPO 参考）
        os.makedirs(self.args.out_dir, exist_ok=True)
        sft_dir = os.path.join(self.args.out_dir, 'sft')
        
        self.model.save_pretrained(sft_dir)
        self.tokenizer.save_pretrained(sft_dir)
        print(f"✓ SFT completed: {sft_dir}")
        
        # 创建参考模型（使用SFT训练后的状态）
        print("🔧 Creating reference model from SFT checkpoint...")
        self._create_reference_model_from_sft()
        
        # 清理SFT阶段的内存，为GRPO准备
        torch.cuda.empty_cache()

    def _create_reference_model_from_sft(self):
        """从SFT训练后的模型创建参考模型"""
        try:
            print("📋 Merging LoRA weights for reference model...")
            
            # 创建当前模型的深拷贝，避免影响原模型
            import copy
            temp_model = copy.deepcopy(self.model)
            
            # 在拷贝上进行合并操作
            merged_model = temp_model.merge_and_unload()
            
            # 保存合并后的权重
            merged_state_dict = merged_model.state_dict()
            
            # 创建新的参考模型实例
            # 将参考模型加载到CPU，降低GPU显存占用
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True
            )
            
            # 🔧 安全加载合并权重到参考模型
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
            
            print("✓ Reference model created from SFT checkpoint with auto device mapping")
            
        except Exception as e:
            print(f"❌ 参考模型创建失败: {e}")
            raise e

    def _sft_forward_step(self, batch):
        """SFT前向传播步骤 (支持DFT改进)"""
        try:
            # auto模式下让模型自动处理设备分配
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask'] 
            labels = batch['labels']
            
            if self.args.use_dft:
                # DFT: Dynamic Fine-Tuning 实现
                return self._dft_forward_step(input_ids, attention_mask, labels)
            else:
                # 标准SFT
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False
                )
                return outputs.loss
                
        except Exception as e:
            raise e

    def _dft_forward_step(self, input_ids, attention_mask, labels):
        """DFT (Dynamic Fine-Tuning) 前向传播实现
        
        基于论文原理：
        - 标准SFT: gradient ∝ 1/P(token) (逆概率加权，导致方差无界)  
        - DFT: 用P(token)重新缩放目标函数，中和逆概率依赖
        - 效果: 从不稳定的概率依赖机制转为稳定的均匀加权更新
        """
        # 前向传播获取logits (不计算loss)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 计算概率
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)          # [batch_size, seq_len, vocab_size]
        
        # 选择目标token的概率
        # 将labels中的-100替换为0以避免索引错误，后面会mask掉
        labels_for_gather = labels.clone()
        labels_for_gather[labels == -100] = 0
        
        # 获取目标token的log概率和概率
        target_log_probs = log_probs.gather(dim=-1, index=labels_for_gather.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=labels_for_gather.unsqueeze(-1)).squeeze(-1)
        
        # DFT核心：用token概率重新缩放标准SFT目标
        # 标准SFT loss: -log P(token)
        # DFT loss: -log P(token) * P(token) = P(token) * (-log P(token))
        # 这样消除了梯度中的1/P(token)项，使更新变得稳定
        standard_losses = -target_log_probs  # 标准SFT损失
        dft_losses = standard_losses * target_probs  # 用概率重新缩放
        
        # 应用mask (忽略labels为-100的位置)
        loss_mask = (labels != -100).float()
        dft_losses = dft_losses * loss_mask
        
        # 计算平均损失
        total_loss = dft_losses.sum()
        total_tokens = loss_mask.sum()
        
        if total_tokens > 0:
            return total_loss / total_tokens
        else:
            return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    def _collate_sft(self, batch: List[Dict]):
        """完整的SFT数据整理函数（修复版）"""
        max_length = 2048
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for sample in batch:
            prompt = sample['prompt']
            chosen = sample['chosen']
            explanation = sample.get('explanation', '')
            
            # 确保数据是字符串类型
            prompt = str(prompt) if prompt is not None else ''
            chosen = str(chosen) if chosen is not None else ''
            explanation = str(explanation) if explanation is not None else ''
            
            # 构建完整的响应
            if explanation and explanation.strip():
                full_response = f"{explanation.strip()}\n\n{chosen.strip()}"
            else:
                full_response = chosen.strip()
            
            # 添加角色和任务指导，使用模型特定的prompt格式
            enhanced_prompt = f"{prompt}\n\nYou are a software engineer. Can you repair the incorrect code?"
            formatted_text = f"{self.BOF} {enhanced_prompt} {self.EOF} {full_response}"
            
            # Tokenize完整文本
            tokenized = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)
            
            # 计算prompt部分的长度（需要排除）
            prompt_text = f"{self.BOF} {enhanced_prompt} {self.EOF} "
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            
            # 构建labels：只对response部分计算损失
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # Mask掉prompt部分
            
            input_ids_list.append(input_ids.tolist())
            attention_mask_list.append(attention_mask.tolist())
            labels_list.append(labels.tolist())
        
        # Padding到batch中的最大长度
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            
            # Padding
            padded_input_ids.append(
                input_ids_list[i] + [self.tokenizer.pad_token_id] * pad_len
            )
            padded_attention_mask.append(
                attention_mask_list[i] + [0] * pad_len
            )
            padded_labels.append(
                labels_list[i] + [-100] * pad_len
            )
        
        # 让auto模式自动处理设备分配
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }

    # ---------- 阶段2：GRPO ----------
    def run_grpo(self, dataset: CodeDataset, curriculum: CurriculumCfg, reward_cfg: RewardCfg):
        """完整的GRPO训练实现"""
        self.model.train()
        
        # 检查可训练参数
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        if trainable_count == 0:
            print("❌ 错误：没有可训练参数！")
            return
        print(f"✓ GRPO模型有 {trainable_count} 个可训练参数")
        print_memory_usage("💾 [Before GRPO]")
        
        rewarder = RewardComputer(reward_cfg)
        
        
        optim = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.grpo_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # 学习率调度器 - 更温和的衰减
        total_steps = self.args.grpo_epochs * len(dataset)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        def grpo_lr_scheduler_fn(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # 不完全衰减到0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, grpo_lr_scheduler_fn)
        
        # 数据加载器 - 性能优化
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=min(self.args.dataloader_num_workers, 4),  # 限制worker数量
            pin_memory=self.args.pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
            prefetch_factor=2 if self.args.dataloader_num_workers > 0 else None
        )
        
        effective_batch_size = 1 * self.args.gradient_accumulation_steps  # GRPO使用batch_size=1
        print(f"[GRPO] steps={total_steps} bs=1 acc={self.args.gradient_accumulation_steps} eff={effective_batch_size}")
        
        global_step = 0
        accumulation_count = 0
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("✓ 启用自动混合精度训练")
        
        # 训练统计
        epoch_stats = {
            'policy_losses': [],
            'rewards': [],
            'advantages': []
        }
        
        for ep in range(self.args.grpo_epochs):
            epoch_policy_loss = 0.0
            epoch_reward = 0.0
            epoch_steps = 0
            
            print(f"[GRPO] Starting epoch {ep}/{self.args.grpo_epochs}")
            print_memory_usage(f"💾 [Epoch {ep}]")
            cleanup_cuda_memory(force_sync=True)
            
            for step, sample in enumerate(loader):
                try:
                    sample = sample[0] if isinstance(sample, list) else sample
                    
                    # 减少内存清理频率以提升速度
                    if step % 50 == 0:  # 降低到每50步激进清理一次
                        cleanup_cuda_memory(force_sync=True, aggressive=True)
                        print(f"🧹 Step {step}: 激进内存清理完成")
                    elif step % 20 == 0:
                        cleanup_cuda_memory()  # 每20步常规清理
                    
                    # 进度输出和内存监控
                    if step % 50 == 0:  # 增加输出频率便于监控进度
                        print(f"[GRPO] ep{ep} step{step}/{len(loader)}")
                    
                    # 课程学习进度
                    epoch_ratio = (ep + step / max(1, len(loader))) / max(1, self.args.grpo_epochs)
                    temp, top_p, K = curriculum.interp(epoch_ratio)
                    
                    # 生成多元化候选组
                    candidates_data = self._generate_candidate_group(sample, temp, top_p, K, epoch_ratio)
                    
                    if not candidates_data or len(candidates_data['texts']) < 2:
                        print(f"[GRPO] Warning: Insufficient valid candidates at step {step} (got {len(candidates_data.get('texts', []))} candidates), skipping")
                        continue
                    
                    # 确保有足够的候选进行有效比较
                    actual_candidates = len(candidates_data['texts'])
                    if actual_candidates < K * 0.5:  # 如果少于期望数量的50%
                        print(f"[GRPO] Warning: Only {actual_candidates}/{K} candidates generated at step {step}")
                    
                    # 计算奖励和概率
                    rewards_info = self._compute_rewards_and_probs(
                        sample, candidates_data, rewarder, epoch_ratio
                    )
                    
                    if not rewards_info:
                        print(f"[GRPO] Warning: Failed to compute rewards at step {step}, skipping")
                        cleanup_cuda_memory()  # 失败时清理内存
                        continue
                    
                    # 减少概率计算后的清理频率
                    if step % 10 == 0:
                        cleanup_cuda_memory(force_sync=False, aggressive=False)
                    
                    # 计算GRPO损失
                    if self.args.use_amp:
                        with torch.amp.autocast('cuda'):
                            loss_info = self._compute_grpo_loss(rewards_info)
                    else:
                        loss_info = self._compute_grpo_loss(rewards_info)
                    
                    # 减少损失计算后的清理频率
                    if step % 20 == 0:
                        cleanup_cuda_memory()
                    
                    # 反向传播 - 添加数值安全检查
                    optim.zero_grad()
                    # 🔧 梯度累积：除以累积步数，添加安全检查
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
                            # 混合精度梯度裁剪和更新
                            scaler.unscale_(optim)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.args.grad_clip
                            )
                            scaler.step(optim)
                            scaler.update()
                        else:
                            # 常规梯度裁剪和更新
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.args.grad_clip
                            )
                            optim.step()
                        
                        optim.zero_grad()
                        scheduler.step()
                        global_step += 1
                    else:
                        grad_norm = 0.0  # 未更新时设为0
                    
                    # 进一步降低内存清理频率
                    if global_step % 100 == 0:  # 每100步
                        if check_memory_pressure():
                            cleanup_cuda_memory(force_sync=True, aggressive=False)
                        elif global_step % 50 == 0:
                            cleanup_cuda_memory(force_sync=False, aggressive=False)
                    
                    # 统计信息
                    epoch_policy_loss += loss_info['policy_loss'].item()
                    epoch_reward += rewards_info['rewards'].mean().item()
                    epoch_steps += 1
                    
                    # 详细日志
                    if global_step % 20 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        self._log_grpo_step(
                            ep, global_step, total_steps, loss_info, rewards_info, 
                            current_lr, grad_norm, temp, top_p, K
                        )
                    
                    # 存储统计信息
                    epoch_stats['policy_losses'].append(loss_info['policy_loss'].item())
                    epoch_stats['rewards'].append(rewards_info['rewards'].mean().item())
                    epoch_stats['advantages'].append(rewards_info['advantages'].std().item())
                    
                except Exception as e:
                    print(f"[GRPO] Error at step {step}: {str(e)}")
                    continue
            
            # Epoch统计
            if epoch_steps > 0:
                avg_policy_loss = epoch_policy_loss / epoch_steps
                avg_reward = epoch_reward / epoch_steps
                
                print(f"[GRPO] epoch {ep} avg_policy_loss={avg_policy_loss:.4f} avg_reward={avg_reward:.3f}")
        
        print("[GRPO] Saving model...")
        
        # 保存最终模型
        grpo_dir = os.path.join(self.args.out_dir, 'grpo')
        os.makedirs(grpo_dir, exist_ok=True)
        self.model.save_pretrained(grpo_dir)
        self.tokenizer.save_pretrained(grpo_dir)
        
        print(f"✓ GRPO completed: {grpo_dir}")

    def _generate_candidate_group(self, sample: Dict, temp: float, top_p: float, K: int, epoch_ratio: float = 0.0) -> Dict:
        """生成K个候选文本（单次批量生成，加速）"""
        base_prompt = sample['prompt']
        base_prompt = str(base_prompt) if base_prompt is not None else ''
        
        # 获取prompt（所有候选都用同一个prompt）
        diverse_prompt, strategy_type = self.prompt_scheduler.get_diverse_prompt(
            base_prompt, epoch_ratio, 0
        )
        
        # 获取探索增强系数（保留但不逐条扰动以提升速度）
        _ = self.prompt_scheduler.get_exploration_boost(epoch_ratio)
        
        # 设置模型为eval模式（只设置一次）
        was_training = self.model.training
        self.model.eval()
        
        # 适度清理一次，避免频繁同步
        cleanup_cuda_memory(force_sync=False, aggressive=False)
        texts = []
        
        # 单次批量生成K个候选（更快）
        with torch.no_grad():
            try:
                outputs = self.inference_pipe(
                    diverse_prompt,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=max(0.1, min(2.0, temp)),
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=max(2, K),
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
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
                        if extract_first_cpp_code(text):
                            texts.append(text)
                else:
                    # 兼容性兜底
                    full_text = str(outputs)
                    if self.EOF in full_text:
                        text = full_text.split(self.EOF)[-1].strip()
                    else:
                        text = full_text.strip()
                    if extract_first_cpp_code(text):
                        texts.append(text)
            except Exception as e:
                print(f"[GRPO] Batch generation error: {e}")
                texts = []
        
        # 适度清理一次
        cleanup_cuda_memory(force_sync=False, aggressive=False)
        
        # 恢复训练模式（只恢复一次）
        if was_training:
            self.model.train()
        
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
        """计算奖励和概率 - CPU优化版本"""
        texts = candidates_data['texts']
        base_prompt = candidates_data['base_prompt']
        
        if len(texts) < 2:
            return {}
        
        # 奖励计算完全在CPU上进行（不需要梯度）
        rewards = []
        extras = []
        with torch.no_grad():  # 明确表示这是评估阶段，不需要梯度
            for i, text in enumerate(texts):
                reward_info = rewarder(sample, str(text), epoch_ratio)
                rewards.append(reward_info['reward'])
                extras.append(reward_info)
                # 每计算一个奖励后清理一次内存
                if i % 2 == 0:  # 每2个清理一次
                    cleanup_cuda_memory()
        
        # 使用与生成时相同的prompt（确保生成和评估一致性）
        diverse_prompt, strategy_type = self.prompt_scheduler.get_diverse_prompt(
            base_prompt, epoch_ratio, 0
        )
        diverse_prompts = [diverse_prompt] * len(texts)  # 所有候选用同一个prompt
        strategy_types = [strategy_type] * len(texts)
        
        # 概率计算前进行超级激进清理
        print("🧹 准备计算对数概率，进行深度内存清理...")
        aggressive_cache_cleanup()
        cleanup_cuda_memory(force_sync=True, aggressive=True)
        
        # 计算对数概率（先仅计算参考模型；策略模型在损失阶段逐候选计算以降低显存峰值）
        print(f"📊 计算 {len(texts)} 个候选的对数概率（参考模型）...")
        ref_log_probs = compute_ref_log_probs_sequential(self.reference_model, self.tokenizer, diverse_prompts, texts)
        cleanup_cuda_memory()

        # 所有统计计算在CPU上进行
        rewards_cpu = torch.tensor(rewards, dtype=torch.float32, device='cpu')
        baseline_cpu = rewards_cpu.mean()
        advantages_cpu = rewards_cpu - baseline_cpu
        
        # 🔧 改进的优势函数归一化和裁剪 - 更保守的处理
        if len(advantages_cpu) > 1:  # 需要至少2个样本才能计算std
            adv_mean = advantages_cpu.mean()
            adv_std = advantages_cpu.std()
            
            # 更保守的归一化：防止除零和极端标准化
            if adv_std > 1e-6:  # 提高最小标准差阈值
                # 限制标准化因子，防止过度缩放
                scale_factor = 1.0 / (adv_std + 1e-6)
                scale_factor = torch.clamp(scale_factor, max=10.0)  # 限制最大缩放倍数
                advantages_cpu = (advantages_cpu - adv_mean) * scale_factor
            else:
                advantages_cpu = advantages_cpu - adv_mean
        
        # 🔧 更严格的裁剪优势函数防止极端值
        clip_value = 2.0  # 大幅降低裁剪阈值，防止优势爆炸
        advantages_cpu = torch.clamp(advantages_cpu, -clip_value, clip_value)
        
        # 保持在CPU，策略阶段再转到策略设备
        baseline = baseline_cpu.item()
        return {
            'rewards': rewards_cpu,
            'ref_logps': ref_log_probs,  # 可能在CPU
            'advantages': advantages_cpu,
            'extras': extras,
            'baseline': baseline,
            'strategy_types': strategy_types,
            'texts': texts,
            'prompts': diverse_prompts,
        }

    def _compute_grpo_loss(self, rewards_info: Dict) -> Dict:
        """计算标准GRPO/PPO损失 - 包含clipped surrogate objective和KL惩罚"""
        # 按需计算策略logprobs，避免与参考logprobs同时常驻GPU导致峰值高
        if 'policy_logps' in rewards_info:
            policy_log_probs = rewards_info['policy_logps']
        else:
            prompts = rewards_info['prompts']
            texts = rewards_info['texts']
            policy_log_probs = compute_log_probs_with_grad_sequential(self.model, self.tokenizer, prompts, texts)

        ref_log_probs = rewards_info['ref_logps']
        advantages = rewards_info['advantages']

        # 设备对齐：将参考logprobs、优势放到策略logprobs所在设备
        target_device = policy_log_probs.device
        if ref_log_probs.device != target_device:
            ref_log_probs = ref_log_probs.to(target_device)
        if advantages.device != target_device:
            advantages = advantages.to(target_device)
        
        if not policy_log_probs.requires_grad:
            return {}
        
        # 1. 计算概率比率 r = exp(policy_log - ref_log) - 添加数值稳定性
        log_ratio = policy_log_probs - ref_log_probs.detach()
        # 🔧 更严格的裁剪：防止exp爆炸，exp(-5)≈0.007, exp(5)≈148
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
        ratio = torch.exp(log_ratio)
        
        # 2. PPO风格的clipped surrogate objective
        clip_param = getattr(self.args, 'clip_param', 0.2)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        
        # 🔧 确保advantages在合理范围内，防止损失爆炸
        advantages_clamped = torch.clamp(advantages.detach(), min=-5.0, max=5.0)
        
        # 计算两种损失并取最小值（最保守的更新）
        policy_loss_1 = -advantages_clamped * ratio
        policy_loss_2 = -advantages_clamped * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # 添加损失裁剪防止极端值
        policy_loss = torch.clamp(policy_loss, -1000.0, 1000.0)
        
        # 3. KL散度惩罚（可选）- 添加数值稳定性
        kl_penalty = torch.tensor(0.0, device=policy_log_probs.device)
        if getattr(self.args, 'use_kl_penalty', True):
            kl_penalty = torch.clamp(log_ratio.mean(), min=-2.0, max=2.0)  # 限制KL惩罚
        
        # 4. 总损失 - 添加数值稳定性检查
        kl_coeff = getattr(self.args, 'kl_coeff', 0.01)
        total_loss = policy_loss + kl_coeff * kl_penalty
        
        # 🔧 最终安全检查：确保loss在合理范围内
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.abs() > 100.0:
            print(f"⚠️ 检测到异常损失值: {total_loss.item():.4f}, 重置为1.0")
            total_loss = torch.tensor(1.0, device=policy_log_probs.device, requires_grad=True)
            policy_loss = torch.tensor(1.0, device=policy_log_probs.device, requires_grad=True)
        
        # 5. 计算一些有用的统计量
        with torch.no_grad():
            approx_kl = log_ratio.mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_param).float().mean()
            ratio_mean = ratio.mean()
            ratio_std = ratio.std()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'kl_penalty': kl_penalty,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'clip_param': clip_param
        }

    def _log_grpo_step(self, epoch: int, step: int, total_steps: int, 
                      loss_info: Dict, rewards_info: Dict, lr: float, 
                      grad_norm: float, temp: float, top_p: float, K: int):
        """记录GRPO训练步骤信息"""
        reward_mean = rewards_info['rewards'].mean().item()
        reward_std = rewards_info['rewards'].std().item()
        advantage_std = rewards_info['advantages'].std().item()
        
        # 确定当前策略阶段
        epoch_ratio = epoch / max(1, self.args.grpo_epochs)
        current_strategy = "innovative" if epoch_ratio < 0.6 else "conservative"
        
        # 获取GRPO统计信息
        policy_loss = loss_info.get('policy_loss', loss_info['total_loss']).item()
        kl_penalty = loss_info.get('kl_penalty', torch.tensor(0.0)).item()
        approx_kl = loss_info.get('approx_kl', torch.tensor(0.0)).item()
        clip_fraction = loss_info.get('clip_fraction', torch.tensor(0.0)).item()
        ratio_mean = loss_info.get('ratio_mean', torch.tensor(1.0)).item()
        
        print(
            f"[GRPO] ep{epoch} s{step}/{total_steps} "
            f"loss={loss_info['total_loss'].item():.4f} pl={policy_loss:.4f} kl={kl_penalty:.3f} "
            f"R={reward_mean:.3f}±{reward_std:.3f} adv={advantage_std:.3f} "
            f"kl~{approx_kl:.3f} cf={clip_fraction:.2f} r={ratio_mean:.2f} "
            f"lr={lr:.2e} gn={grad_norm:.2f} T={temp:.2f} p={top_p:.2f} K={K} {current_strategy}"
        )
        
        # 打印第一个候选的详细信息
        if rewards_info['extras']:
            first_extra = rewards_info['extras'][0]
            # 根据是否有测试用例决定输出格式
            if first_extra.get('used_tests', False):
                testcase_source = first_extra.get('testcase_source', 'unknown')
                num_tests = first_extra.get('num_testcases', 0)
                print(
                    f"[GRPO] cand0 test={first_extra['test_pass_rate']:.2f} "
                    f"text={first_extra['text_score']:.3f} final={first_extra['final_score']:.3f} "
                    f"tests={num_tests}({testcase_source}) ret={first_extra['retention']:.3f} diff={first_extra['diff_ratio']:.3f}"
                )
            else:
                print(
                    f"[GRPO] cand0 text={first_extra.get('text_score', 0):.3f} (no_tests) "
                    f"ret={first_extra['retention']:.3f} diff={first_extra['diff_ratio']:.3f}"
                )

# =========================
# 入口
# =========================
def main():
    import argparse
    p = argparse.ArgumentParser(description='SFT + GRPO Training with DFT and Test Cases')
    p.add_argument('--model_name', type=str, required=True, 
                   help='Model name (will be joined with base path, e.g., "Llama-3-8B-Instruct")')
    args = p.parse_args()

    # 创建固定配置的训练参数
    targs = TrainArgs(model_name=args.model_name)
    
    print("🚀 SFT + GRPO Training")
    print(f"📁 Model: {targs.model_name}")
    print(f"📂 Data: {targs.train_file}")
    print(f"💾 Output: {targs.out_dir}")
    print(f"🎯 Epochs: SFT={targs.sft_epochs}, GRPO={targs.grpo_epochs}")
    print(f"⚙️ Device: {targs.device}")
    
    # 确保输出目录存在
    os.makedirs(targs.out_dir, exist_ok=True)

    # 加载数据集
    dataset = CodeDataset(targs.train_file)
    print(f"📖 Dataset: {len(dataset)} samples")
    
    # 初始化训练器
    print("🔧 Initializing trainer...")
    trainer = SFT_GRPO_Trainer(targs)

    # 检查是否已有SFT模型
    sft_model_path = os.path.join(targs.out_dir, 'sft')
    grpo_model_path = os.path.join(targs.out_dir, 'grpo')
    
    if os.path.exists(grpo_model_path):
        print("✓ GRPO模型已存在，跳过训练")
        print(f"📁 最终模型路径: {grpo_model_path}")
        return
    elif os.path.exists(sft_model_path):
        print("✓ SFT模型已存在，跳过SFT阶段，直接进行GRPO")
        
        # 有SFT模型时，直接进行GRPO
        try:
            # 步骤1：加载原始base model
            print("🔧 Loading base model...")
            # 自适应内存分配
            print("🧹 执行超级缓存清理...")
            aggressive_cache_cleanup()  # 超级激进清理缓存池
            
            adaptive_memory, suggested_batch, suggested_accumulation = get_adaptive_memory_config()
            
            # 动态调整batch size和梯度累积
            original_batch = targs.sft_batch
            original_accumulation = targs.gradient_accumulation_steps
            
            # 调整batch size
            targs.sft_batch = min(suggested_batch, original_batch)
            if targs.sft_batch != original_batch:
                print(f"🔄 自动调整batch size: {original_batch} → {targs.sft_batch}")
            
            # 调整梯度累积步数
            targs.gradient_accumulation_steps = max(suggested_accumulation, original_accumulation)
            if targs.gradient_accumulation_steps != original_accumulation:
                print(f"🔄 自动调整梯度累积: {original_accumulation} → {targs.gradient_accumulation_steps}")
            
            # 计算有效batch size
            effective_batch = targs.sft_batch * targs.gradient_accumulation_steps
            print(f"📊 最终配置: batch_size={targs.sft_batch}, accumulation={targs.gradient_accumulation_steps}, 有效batch={effective_batch}")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                targs.base_model,
                dtype=torch.float16,  # 使用16位精度
                device_map="auto",  # 恢复auto模式
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,  # 减少CPU内存使用
                max_memory=adaptive_memory,  # 自适应内存分配
                attn_implementation="eager"  # 使用eager attention避免一些meta tensor问题
            )
            
            # 步骤2：加载SFT训练好的LoRA适配器作为策略模型
            print(f"🔧 Loading policy model: base model + SFT LoRA adapter")
            print(f"📁 SFT适配器路径: {sft_model_path}")
            from peft import PeftModel
            
            # 直接加载SFT训练好的LoRA适配器作为策略模型
            trainer.model = PeftModel.from_pretrained(
                base_model,
                sft_model_path,
                dtype=torch.float16,
                is_trainable=True  # 策略模型继续训练SFT的LoRA参数
            )
            
            # 确保只有LoRA参数可训练
            trainer._fix_trainable_params()
            trainer.model.train()
            
            # 创建新的pipeline用于GRPO生成
            print("🚀 Creating fresh inference pipeline for GRPO...")
            # 统一使用auto模式，不指定device让pipeline自己处理
            trainer.inference_pipe = pipeline("text-generation", model=trainer.model, tokenizer=trainer.tokenizer)
            print("✓ Fresh inference pipeline ready")
            
            # 验证LoRA参数设置结果
            lora_param_count = sum(1 for name, p in trainer.model.named_parameters() if 'lora' in name.lower() and p.requires_grad)
            total_trainable = sum(1 for p in trainer.model.parameters() if p.requires_grad)
            print(f"✓ LoRA参数: {lora_param_count}个可训练")
            print(f"✓ 总可训练参数: {total_trainable}个")
            
            if total_trainable == 0:
                print("❌ 警告：没有可训练参数！")
            
            print("✓ 策略模型加载完成")
            
            # 步骤3：创建参考模型（策略模型的冻结副本）
            print("🔧 Creating reference model: frozen copy of policy model...")
            
            # 加载相同的基础模型和SFT适配器作为参考模型
            # 参考模型固定在CPU
            ref_base_model = AutoModelForCausalLM.from_pretrained(
                targs.base_model,
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
                is_trainable=False  # 参考模型完全冻结
            )
            
            # 确保参考模型所有参数都冻结
            for param in trainer.reference_model.parameters():
                param.requires_grad = False
            trainer.reference_model.eval()
            
            print("✓ 参考模型创建完成（策略模型的冻结副本）")
            
            # 最终验证参数状态
            final_lora_count = sum(1 for name, p in trainer.model.named_parameters() if 'lora' in name.lower() and p.requires_grad)
            final_total_trainable = sum(1 for p in trainer.model.parameters() if p.requires_grad)
            print(f"✓ 策略模型 - LoRA参数: {final_lora_count}个可训练")
            print(f"✓ 策略模型 - 总可训练参数: {final_total_trainable}个")
            
        except Exception as e:
            print(f"❌ SFT模型加载失败: {e}")
            return
    else:
        print('\n🎓 [Stage 1] SFT Training (CoT: explanation + chosen)')
        print("-" * 60)
        trainer.run_sft(dataset)

    print('\n🎯 [Stage 2] GRPO Training (group-relative + curriculum)')
    print("-" * 60)
    # 使用固定的课程学习和奖励配置
    curriculum_cfg = CurriculumCfg(
        start_temp=1.2, end_temp=0.8,
        start_top_p=0.95, end_top_p=0.8,
        start_num=3, end_num=3  # 减少候选为3个，并且后期也维持三个
    )
    
    reward_cfg = RewardCfg(
        diff_clip_low=-1.0, diff_clip_high=1.0,
        use_test_cases=True,                              # 启用测试用例
        test_weight=0.6,                                  # 测试用例权重60%
        text_weight=0.4,                                  # 文本质量权重40%
        test_timeout=5.0,                                 # 5秒超时
        external_testcase_file=targs.external_testcase_file,  # 外部测试用例文件
        problem_id_field=targs.problem_id_field            # problem_id字段名
    )
    
    try:
        trainer.run_grpo(dataset, curriculum_cfg, reward_cfg)
    except Exception as e:
        print(f"❌ GRPO训练失败: {e}")

    print('\n🎉 Training Completed!')
    print("=" * 60)
    print(f"📁 Models saved under: {targs.out_dir}")
    print(f"  - SFT model: {targs.out_dir}/sft/")
    print(f"  - Final GRPO model: {targs.out_dir}/grpo/")
    print("🔬 Ready for paper experiments and evaluation!")
    print("=" * 60)

if __name__ == '__main__':
    import os
    import warnings
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 激进的缓存池管理策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32,roundup_power2_divisions:32,garbage_collection_threshold:0.6"
    
    # 强制禁用内存缓存，减少缓存池积累
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    # 设置使用GPU 0和2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    
    # 禁用CUDA调试以提升性能
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 禁用同步，提升性能
    os.environ["TORCH_USE_CUDA_DSA"] = "0"   # 禁用设备断言，提升性能
    
    # 额外的性能优化
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # 启用cuDNN v8 API
    os.environ["TORCH_CUDNN_ALLOW_TF32"] = "1"      # 允许TF32精度提速
    
    print("🚀 训练性能优化已启用")
    print("⚡ 批量计算、模型编译、数据并行加载等优化功能已开启")
    
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