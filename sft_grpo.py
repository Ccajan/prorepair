


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

# 内存优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    """
    执行C++代码并运行测试用例，返回通过率
    testcases格式: [{"input": "...", "output": "...", "testcase_id": ..., ...}, ...]
    支持包含testcase_id, complexity, features等额外字段
    """
    if not testcases:
        return 0.0
    
    try:
        # 创建临时C++文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            cpp_file = f.name
        
        # 编译C++代码
        import platform
        if platform.system() == 'Windows':
            exe_file = cpp_file.replace('.cpp', '.exe')
        else:
            exe_file = cpp_file.replace('.cpp', '')
        
        # 使用g++编译器
        try:
            compile_result = subprocess.run(
                ['g++', '-o', exe_file, cpp_file, '-std=c++17'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            compile_success = compile_result.returncode == 0
        except FileNotFoundError:
            # g++编译器不存在
            compile_success = False
        
        if not compile_success:
            # 编译失败，清理文件并返回
            try:
                os.unlink(cpp_file)
            except:
                pass
            return 0.0
        
        passed = 0
        total = len(testcases)
        
        for i, test in enumerate(testcases):
            try:
                # 提取输入输出（兼容各种格式）
                input_data = str(test.get('input', ''))
                expected_output = str(test.get('output', '')).strip()
                testcase_id = test.get('testcase_id', i)
                
                # 确保输入数据格式正确
                if input_data and not input_data.endswith('\n'):
                    input_data += '\n'
                
                # 执行编译后的程序
                result = subprocess.run(
                    [exe_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    actual_output = result.stdout.strip()
                    if actual_output == expected_output:
                        passed += 1
                        
            except subprocess.TimeoutExpired:
                # 测试用例超时
                continue
            except Exception as e:
                # 其他执行错误
                continue
        
        # 清理临时文件
        try:
            os.unlink(cpp_file)
            os.unlink(exe_file)
        except:
            pass
            
        return passed / total if total > 0 else 0.0
        
    except Exception:
        return 0.0

def compute_text_quality(ref_text: str, cand_text: str) -> Tuple[float, float, float]:
    """
    计算生成文本与参考文本的质量评估
    返回 (retention, diff_ratio, score)，其中：
    - retention = l/k，l为候选代码中与参考代码匹配的行数，k为候选代码总行数（越高越好）
    - diff_ratio = 1 - similarity，整体差异程度（越小越好，表示改动越少）
    - score = retention - diff_ratio，综合质量评分
    """
    # 确保输入是字符串
    ref_text = str(ref_text) if ref_text is not None else ''
    cand_text = str(cand_text) if cand_text is not None else ''
    
    # 特殊情况处理
    if not cand_text.strip():
        return 0.0, 1.0, -1.0
    if not ref_text.strip():
        return 0.0, 0.0, 0.0
    
    # 按行分割进行比较
    ref_lines = ref_text.splitlines()
    cand_lines = cand_text.splitlines()
    
    k = len(cand_lines)  # 候选代码总行数
    if k == 0:
        return 0.0, 1.0, -1.0
    
    # 使用difflib计算匹配
    matcher = difflib.SequenceMatcher(None, ref_lines, cand_lines)
    
    # l = 匹配的行数（匹配块的大小之和）
    matching_blocks = matcher.get_matching_blocks()
    l = sum(block.size for block in matching_blocks if block.size > 0)
    
    # retention = l/k：候选代码中与参考代码匹配的行数比例
    retention = l / k
    
    # diff_ratio = 1 - similarity：整体差异程度，越小表示改动越少
    similarity = matcher.ratio()
    diff_ratio = 1.0 - similarity
    
    # score = retention - diff_ratio：综合评分
    # 高保留率好，低差异度好
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
    start_num: int = 6
    end_num: int = 2

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
            "Explore completely different approaches to fix this code. Consider major refactoring or alternative algorithms:",
            "Think creatively and propose a significantly different solution. Major code restructuring is acceptable:",
            "Generate an innovative fix using a different programming paradigm or data structure:",
            "Approach this problem from a completely new angle. Feel free to make substantial changes:",
        ]
        
        # 保守型修复：最小化改动，保持原有结构  
        self.conservative_prompts = [
            "Please generate a minimal repair patch that preserves the original code structure and only modifies the necessary parts to fix the error:",
            "Provide the smallest possible fix that maintains the existing code structure:",
            "Generate a minimal patch - change only what's absolutely necessary to fix the bug:",
            "Create a conservative fix that preserves the original design and makes minimal changes:",
            "Deliver a targeted fix that keeps the original code structure intact:",
        ]
        
        # 通用的任务描述
        self.base_instruction = (
            "You are an expert software engineer. Analyze the incorrect code carefully and provide a correct implementation. "
            "Generate ONLY the corrected C++ code inside a code block, without explanations outside the code."
        )
    
    def get_diverse_prompt(self, base_prompt: str, epoch_ratio: float, candidate_idx: int = 0) -> str:
        """根据训练进度生成创新型或保守型修复prompt"""
        # 获取模型特定的前缀后缀
        BOF, EOF = get_prompt_format(self.model_key)
        
        # 早期使用创新型，晚期使用保守型
        if epoch_ratio < 0.6:  # 早期：创新型修复
            strategy_prompts = self.innovative_prompts
            strategy_type = "innovative"
        else:  # 晚期：保守型修复
            strategy_prompts = self.conservative_prompts
            strategy_type = "conservative"
        
        # 从对应策略中选择prompt变体
        prompt_idx = candidate_idx % len(strategy_prompts)
        diversity_instruction = strategy_prompts[prompt_idx]
        
        # 构建完整prompt
        full_prompt = (
            BOF + "\n" + 
            base_prompt.strip() + "\n\n" +
            diversity_instruction + "\n" +
            self.base_instruction + "\n" +
            EOF + "\n```cpp\n"
        )
        
        return full_prompt, strategy_type
    
    def get_exploration_boost(self, epoch_ratio: float) -> float:
        """获取探索增强系数（用于调整温度）"""
        if epoch_ratio < 0.3:
            return 0.2  # 早期大幅增强探索
        elif epoch_ratio < 0.6:
            return 0.1  # 中期适度增强
        else:
            return 0.0  # 晚期不增强

# =========================
# 概率/损失相关：logprob、KL、优势
# =========================
def compute_log_probs(model, tokenizer, prompts, responses, device=None):
    """计算序列对数概率（无梯度）"""
    if device is None:
        device = next(model.parameters()).device
    
    log_probs = []
    batch_size = len(prompts)
    
    for i in range(batch_size):
        prompt = prompts[i] if isinstance(prompts, list) else prompts
        response = responses[i] if isinstance(responses, list) else responses
        
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        response_logits = logits[0, len(prompt_tokens)-1:len(prompt_tokens)+len(response_tokens)-1, :]
        response_token_ids = torch.tensor(response_tokens, dtype=torch.long, device=device)
        
        log_probs_dist = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs_dist.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        sequence_log_prob = token_log_probs.sum()
        log_probs.append(sequence_log_prob)
    
    return torch.stack(log_probs)

def compute_log_probs_with_grad(model, tokenizer, prompts, responses, device=None):
    """计算序列对数概率（保留梯度）"""
    if device is None:
        device = next(model.parameters()).device
    
    log_probs = []
    batch_size = len(prompts)
    
    for i in range(batch_size):
        prompt = prompts[i] if isinstance(prompts, list) else prompts
        response = responses[i] if isinstance(responses, list) else responses
        
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        
        response_logits = logits[0, len(prompt_tokens)-1:len(prompt_tokens)+len(response_tokens)-1, :]
        response_token_ids = torch.tensor(response_tokens, dtype=torch.long, device=device)
        
        log_probs_dist = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs_dist.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        sequence_log_prob = token_log_probs.sum()
        log_probs.append(sequence_log_prob)
    
    return torch.stack(log_probs)

def cleanup_cuda_memory():
    """清理CUDA内存"""
    torch.cuda.empty_cache()
    gc.collect()


# =========================
# 奖励函数（基于文本质量评估）
# =========================
@dataclass
class RewardCfg:
    diff_clip_low: float = -1.0  # 文本质量分数的下界
    diff_clip_high: float = 1.0  # 文本质量分数的上界
    # 测试用例权重配置
    use_test_cases: bool = True           # 是否使用测试用例
    test_weight: float = 0.6              # 测试用例权重 (60%)
    text_weight: float = 0.4              # 文本质量权重 (40%)
    test_timeout: float = 5.0             # 测试超时时间(秒)
    # 外部测试用例配置
    external_testcase_file: str = ""      # 外部测试用例文件路径
    problem_id_field: str = "problem_id"  # problem_id字段名

class RewardComputer:
    def __init__(self, cfg: RewardCfg):
        self.cfg = cfg
        self.external_testcases = {}  # 缓存外部测试用例
        
        # 加载外部测试用例文件
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
                            # 格式: {problem_id: {"testcases": [...]}}
                            for problem_id, content in data.items():
                                if isinstance(content, dict) and 'testcases' in content:
                                    self.external_testcases[problem_id] = content['testcases']
                                elif isinstance(content, list):
                                    # 直接是testcase列表
                                    self.external_testcases[problem_id] = content
                                    
                # 统计信息
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
        # 确保输入是字符串
        generated_text = str(generated_text) if generated_text is not None else ''
        chosen_text = str(sample.get('chosen', '')) if sample.get('chosen') is not None else ''
        
        # 清理chosen_text：移除列表格式的字符串表示
        if chosen_text.startswith("['") and chosen_text.endswith("']"):
            # 移除列表的字符串表示形式
            chosen_text = chosen_text[2:-2]  # 移除 ['...']
            chosen_text = chosen_text.replace("\\n", "\n")  # 恢复换行符
            chosen_text = chosen_text.replace("\\'", "'")   # 恢复单引号
            chosen_text = chosen_text.replace('\\"', '"')   # 恢复双引号
        
        # 从代码块中提取纯代码
        generated_code = extract_code(generated_text)
        chosen_code = extract_code(chosen_text)
        
        # 使用提取的代码进行质量比较，如果提取失败则使用原文本
        compare_generated = generated_code if generated_code.strip() else generated_text
        compare_chosen = chosen_code if chosen_code.strip() else chosen_text
        
        retention, diff_ratio, text_score = compute_text_quality(compare_chosen, compare_generated)
        
        # 获取测试用例（优先级：样本内 > 外部文件 > 无）
        testcases = None
        testcase_source = "none"
        
        if self.cfg.use_test_cases:
            # 1. 优先使用样本内的测试用例
            if 'testcases' in sample and sample['testcases']:
                testcases = sample['testcases']
                testcase_source = "inline"
            
            # 2. 尝试从外部文件通过problem_id获取
            elif (self.cfg.problem_id_field in sample and 
                  sample[self.cfg.problem_id_field] in self.external_testcases):
                problem_id = sample[self.cfg.problem_id_field]
                testcases = self.external_testcases[problem_id]
                testcase_source = "external"
        
        # 计算测试通过率
        test_pass_rate = 0.0
        if testcases:
            try:
                code = extract_code(generated_text)
                test_pass_rate = run_code_with_testcases(code, testcases, self.cfg.test_timeout)
            except Exception as e:
                print(f"[Reward] Test execution failed: {e}")
                test_pass_rate = 0.0
        
        # 综合奖励计算
        if testcases:
            # 有测试用例：测试用例 + 文本质量的加权组合
            final_score = (self.cfg.test_weight * test_pass_rate + 
                          self.cfg.text_weight * text_score)
            used_tests = True
        else:
            # 无测试用例：仅使用文本质量
            final_score = text_score
            used_tests = False
        
        # 应用裁剪
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
    model_name: str  # 动态的模型名称（会拼接到基础路径）
    
    # 固定的路径配置
    model_base_path: str = "/data1/czj/model"  # 模型基础路径
    train_file: str = "/data1/czj/prorepair/data/trainset/sft_dataset.json"  # 与sft1.py保持一致
    output_base_path: str = "/data1/czj/model"  # 输出基础路径
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 适合小数据集(1500条)的训练配置 - 内存优化版
    sft_epochs: int = 5          # 增加epoch补偿小batch size
    grpo_epochs: int = 3         # 增加GRPO训练轮数
    sft_lr: float = 3e-5         # 适中学习率，小数据集+小batch
    grpo_lr: float = 1.5e-5      # 适中GRPO学习率
    sft_batch: int = 2           # 减小batch size节省显存
    
    # LoRA 配置 - 标准LoRA
    lora_r: int = 8                # rank=8标准配置
    lora_alpha: int = 16           # 2倍rank的缩放因子
    lora_dropout: float = 0.1      # 防过拟合的dropout
    
    # 生成和优化固定配置
    max_new_tokens: int = 1024
    grad_clip: float = 1.0
    seed: int = 42
    
    # DFT (Dynamic Fine-Tuning) 配置 - 基于论文精确实现
    use_dft: bool = True           # 是否使用DFT改进（简单的概率重新缩放）
    
    # NEFTune 配置 - 噪声嵌入改进指令微调
    use_neftune: bool = True       # 是否使用NEFTune噪声
    neftune_alpha: float = 5.0     # NEFTune噪声强度，参考MOTrain.py
    
    # 量化配置 - 8bit量化节省显存
    use_8bit: bool = True          # 是否使用8bit量化
    use_4bit: bool = False         # 是否使用4bit量化（与8bit互斥）
    
    # 静态配置：外部测试用例
    external_testcase_file: str = "/data1/czj/prorepair/data/trainset/testcases_sorted.json"  # 外部测试用例文件路径
    problem_id_field: str = "problem_id"  # problem_id字段名
    
    @property
    def base_model(self) -> str:
        """动态生成完整的模型路径"""
        return os.path.join(self.model_base_path, self.model_name)
    
    @property
    def out_dir(self) -> str:
        """动态生成输出路径，与sft1.py保持一致"""
        model_short_name = self.model_name.lower().replace("-", "_")
        return os.path.join(self.output_base_path, f"trained_sft_grpo_{model_short_name}")

class SFT_GRPO_Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        random.seed(args.seed); torch.manual_seed(args.seed)
        
        # 获取模型的prompt格式
        self.BOF, self.EOF = get_prompt_format(args.model_name)
        print(f"🎯 Using prompt format for {args.model_name}: '{self.BOF}' ... '{self.EOF}'")
        
        # 初始化多元化prompt调度器
        self.prompt_scheduler = PromptScheduler(args.model_name)
        print(f"🎲 Initialized diverse prompt scheduler:")
        print(f"   - Early stage (0-60%): {len(self.prompt_scheduler.innovative_prompts)} innovative repair strategies")
        print(f"   - Late stage (60-100%): {len(self.prompt_scheduler.conservative_prompts)} conservative repair strategies")
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 配置量化
        quantization_config = None
        if args.use_8bit and args.use_4bit:
            raise ValueError("Cannot use both 8bit and 4bit quantization simultaneously")
        elif args.use_8bit:
            print(f"🔧 Using 8-bit quantization for memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif args.use_4bit:
            print(f"🔧 Using 4-bit quantization for maximum memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        print(f"🔧 Loading model with device_map='auto' for multi-GPU support")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if not (args.use_8bit or args.use_4bit) else "auto",
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        
        # 如果使用量化，需要准备模型
        if args.use_8bit or args.use_4bit:
            base = prepare_model_for_kbit_training(base)
        
        # 根据模型类型自动选择target_modules
        if args.use_8bit or args.use_4bit:
            # 量化模型使用更全面的target_modules
            target_modules = self._find_all_linear_names(base, int8=args.use_8bit, int4=args.use_4bit)
            print(f"Using quantized target_modules: {target_modules}")
        else:
            # 非量化模型使用基本的target_modules
            target_modules = self._get_target_modules(args.base_model)
            print(f"Using standard target_modules: {target_modules}")
        
        # 使用标准LoRA（移除复杂的Chain LoRA）
        print(f"🔧 Using standard LoRA: rank={args.lora_r}, alpha={args.lora_alpha}")
        lora = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=target_modules, 
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base, lora)  # 不移动已经用device_map分布的模型
        print(f"✓ Standard LoRA applied to {len(target_modules)} module types")
        
        # 创建用于GRPO推理的pipeline，模仿d4j.py的做法
        print("🚀 Creating inference pipeline for GRPO generation...")
        self.inference_pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("✓ Inference pipeline ready")
        
        # 显示设备映射信息
        if hasattr(base, 'hf_device_map'):
            print(f"📱 Multi-GPU device mapping:")
            device_counts = {}
            for module_name, device in base.hf_device_map.items():
                device_counts[device] = device_counts.get(device, 0) + 1
            for device, count in sorted(device_counts.items()):
                if isinstance(device, int):
                    print(f"   GPU {device}: {count} modules")
                else:
                    print(f"   {device}: {count} modules")
        
        # 确保只有LoRA参数可训练
        self._fix_trainable_params()
        
        # 应用NEFTune噪声（如果启用）
        if args.use_neftune:
            self._apply_neftune(args.neftune_alpha)
        
        # 验证模块是否正确应用
        self._verify_lora_modules()

    def _fix_trainable_params(self):
        """确保只有LoRA参数可训练"""
        print("🔧 修复可训练参数设置...")
        
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
            print(f"⚠️ Unknown model type: {model_name}, using default Llama target_modules")
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
            print(f"✓ Standard LoRA modules: {len(lora_modules)}")
            print(f"  First 5 modules:")
            for module_name in lora_modules[:5]:
                print(f"  - {module_name}")
            if len(lora_modules) > 5:
                print(f"  ... and {len(lora_modules) - 5} more modules")
        
        print(f"📊 Parameter Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        if len(lora_modules) == 0:
            print("❌ Warning: No LoRA modules found! Check target_modules configuration.")
    
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
            print(f"🔊 NEFTune applied to {len(neftune_modules)} embedding layers with alpha={alpha}")
            print(f"   Embedding layers: {', '.join(neftune_modules[:3])}" + 
                  (f" and {len(neftune_modules)-3} more..." if len(neftune_modules) > 3 else ""))
        else:
            print("⚠️ Warning: No embedding layers found for NEFTune!")

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
        
        # 梯度累积步数（针对小数据集优化）
        accumulation_steps = max(1, 16 // self.args.sft_batch)  # 降低累积步数，小数据集用较小有效batch
        
        print(f"[SFT] Starting training: {self.args.sft_epochs} epochs, {len(loader)} steps/epoch")
        print(f"[SFT] Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        print(f"[SFT] Gradient accumulation steps: {accumulation_steps}")
        
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
                        
                        if global_step % 50 == 0:
                            cleanup_cuda_memory()
                        
                        # 日志输出
                        if global_step % 20 == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            # 计算实际处理的样本数
                            actual_samples = global_step * accumulation_steps
                            actual_epoch = actual_samples / len(loader)
                            print(f"[SFT] ep{ep} step{step+1}/{len(loader)} global_step{global_step}/{total_steps} (样本:{actual_samples}, 实际ep:{actual_epoch:.1f}) "
                                  f"loss={loss.item() * accumulation_steps:.4f} "
                                  f"lr={current_lr:.2e} grad_norm={grad_norm:.3f}")
                
                except Exception as e:
                    print(f"[SFT] Error at step {step}: {e}")
                    cleanup_cuda_memory()
                    continue
            
            # Epoch 统计
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"[SFT] Epoch {ep} completed: avg_loss={avg_loss:.4f}")
        
        print("[SFT] Training completed, saving model...")
        
        # 保存 SFT 模型（供 GRPO 参考）
        os.makedirs(self.args.out_dir, exist_ok=True)
        sft_dir = os.path.join(self.args.out_dir, 'sft')
        
        # 保存LoRA适配器（在GRPO阶段再合并）
        print("[SFT] Saving LoRA adapter...")
        self.model.save_pretrained(sft_dir)
        self.tokenizer.save_pretrained(sft_dir)
        print(f"[SFT] Saved LoRA adapter to: {sft_dir}")
        
        print("✓ SFT training completed, no reference model needed for simplified GRPO")
        
        # 清理SFT阶段的内存，为GRPO准备
        torch.cuda.empty_cache()
        print("[SFT] SFT phase completed successfully!")

    def _sft_forward_step(self, batch):
        """SFT前向传播步骤 (支持DFT改进)"""
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch['labels'].to(self.args.device)
        
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
        
        rewarder = RewardComputer(reward_cfg)
        
        # 优化器设置 - 通常GRPO使用更小的学习率
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
        
        # 数据加载器 - 每次处理一个样本进行组内比较
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"[GRPO] Starting training: {self.args.grpo_epochs} epochs, {len(loader)} steps/epoch")
        print(f"[GRPO] Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        global_step = 0
        
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
            
            for step, sample in enumerate(loader):
                try:
                    sample = sample[0] if isinstance(sample, list) else sample
                    
                    # 简单进度输出
                    if step % 50 == 0:
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
                        continue
                    
                    # 计算GRPO损失
                    loss_info = self._compute_grpo_loss(rewards_info)
                    
                    # 反向传播
                    optim.zero_grad()
                    loss_info['total_loss'].backward()
                    
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.args.grad_clip
                    )
                    
                    optim.step()
                    scheduler.step()
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        cleanup_cuda_memory()
                    
                    # 统计信息
                    epoch_policy_loss += loss_info['policy_loss'].item()
                    epoch_reward += rewards_info['rewards'].mean().item()
                    epoch_steps += 1
                    
                    # 详细日志
                    if global_step % 10 == 0:
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
                
                print(f"[GRPO] Epoch {ep} completed: "
                      f"avg_policy_loss={avg_policy_loss:.4f} "
                      f"avg_reward={avg_reward:.3f}")
        
        print("[GRPO] Training completed, saving final model...")
        
        # 保存最终模型
        grpo_dir = os.path.join(self.args.out_dir, 'grpo')
        os.makedirs(grpo_dir, exist_ok=True)
        self.model.save_pretrained(grpo_dir)
        self.tokenizer.save_pretrained(grpo_dir)
        
        # 保存训练统计
        stats_file = os.path.join(grpo_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in epoch_stats.items()}, f, indent=2)
        
        print("[GRPO] GRPO phase completed successfully!")

    def _generate_d4j_style(self, prompt: str, max_new_tokens: int = 1024, 
                           temperature: float = 1.0, top_p: float = 0.9) -> str:
        """生成单个候选，使用温度和top_p控制多样性"""
        try:
            output = self.inference_pipe(
                prompt, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            full_text = output[0]['generated_text']
            return full_text
        except Exception as e:
            print(f"[GRPO] Error in generation: {e}")
            return ""

    def _generate_candidate_group(self, sample: Dict, temp: float, top_p: float, K: int, epoch_ratio: float = 0.0) -> Dict:
        """生成多元化候选组"""
        base_prompt = sample['prompt']
        base_prompt = str(base_prompt) if base_prompt is not None else ''
        
        cleanup_cuda_memory()
        
        texts = []
        max_attempts = K + 2
        attempt = 0
        
        while len(texts) < max(2, K) and attempt < max_attempts:
            try:
                cleanup_cuda_memory()
                
                # 获取当前候选的索引，用于prompt多样性
                candidate_idx = len(texts)
                
                # 使用多元化prompt调度器生成不同的prompt
                diverse_prompt, strategy_type = self.prompt_scheduler.get_diverse_prompt(
                    base_prompt, epoch_ratio, candidate_idx
                )
                
                # 获取探索增强系数
                exploration_boost = self.prompt_scheduler.get_exploration_boost(epoch_ratio)
                
                # 应用温度扰动和探索增强
                temp_noise = temp + random.uniform(-0.1, 0.1) + exploration_boost
                temp_noise = max(0.1, min(2.0, temp_noise))
                
                # 生成候选
                full_text = self._generate_d4j_style(
                    diverse_prompt, 
                    max_new_tokens=self.args.max_new_tokens, 
                    temperature=temp_noise,
                    top_p=top_p
                )
                
                if not full_text:
                    attempt += 1
                    continue
                
                # 提取生成的部分（去除原始prompt）
                # 模仿d4j.py的处理方式
                if self.EOF in full_text:
                    text = full_text.split(self.EOF)[1].strip()
                else:
                    # 如果没有找到EOF，尝试从prompt之后提取
                    original_prompt_part = prompt.split(self.EOF)[0] if self.EOF in prompt else prompt
                    if original_prompt_part in full_text:
                        text = full_text.replace(original_prompt_part, "").strip()
                    else:
                        text = full_text.strip()
                
                text = str(text).strip()
                
                # 检查是否包含有效的C++代码
                extracted_code = extract_first_cpp_code(text)
                if not extracted_code:
                    continue
                
                texts.append(text)
                
                # 每生成一个候选后轻量清理  
                cleanup_cuda_memory()
                
            except Exception as e:
                print(f"[GRPO] Error generating candidate: {e}")
                cleanup_cuda_memory()
                time.sleep(1.0)
            
            attempt += 1
        
        if len(texts) == 0:
            print(f"[GRPO] All candidate generation failed after {attempt} attempts")
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
        
        # 计算奖励
        rewards = []
        extras = []
        for text in texts:
            reward_info = rewarder(sample, str(text), epoch_ratio)
            rewards.append(reward_info['reward'])
            extras.append(reward_info)
        
        # 为每个候选重新生成对应的多元化prompt（用于概率计算）
        diverse_prompts = []
        strategy_types = []
        for i in range(len(texts)):
            diverse_prompt, strategy_type = self.prompt_scheduler.get_diverse_prompt(
                base_prompt, epoch_ratio, i
            )
            diverse_prompts.append(diverse_prompt)
            strategy_types.append(strategy_type)
        
        # 计算对数概率
        policy_log_probs = compute_log_probs_with_grad(self.model, self.tokenizer, diverse_prompts, texts)
        
        if torch.isnan(policy_log_probs).any() or torch.isinf(policy_log_probs).any():
            return {}
        
        # 计算优势
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=policy_log_probs.device)
        baseline = rewards_t.mean()
        advantages = rewards_t - baseline
        
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        return {
            'rewards': rewards_t,
            'logps': policy_log_probs,
            'advantages': advantages,
            'extras': extras,
            'baseline': baseline,
            'strategy_types': strategy_types
        }

    def _compute_grpo_loss(self, rewards_info: Dict) -> Dict:
        """计算GRPO损失"""
        policy_log_probs = rewards_info['logps']
        advantages = rewards_info['advantages']
        
        if not policy_log_probs.requires_grad:
            return {}
        
        policy_loss = -(policy_log_probs * advantages.detach()).mean()
        
        return {
            'total_loss': policy_loss,
            'policy_loss': policy_loss
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
        
        print(f"[GRPO] ep{epoch} step{step}/{total_steps} "
              f"loss={loss_info['total_loss'].item():.4f} "
              f"R_mean={reward_mean:.3f}±{reward_std:.3f} "
              f"adv_std={advantage_std:.3f} "
              f"lr={lr:.2e} grad_norm={grad_norm:.3f} "
              f"temp={temp:.2f} top_p={top_p:.2f} K={K} "
              f"strategy={current_strategy}")
        
        # 打印第一个候选的详细信息
        if rewards_info['extras']:
            first_extra = rewards_info['extras'][0]
            # 根据是否有测试用例决定输出格式
            if first_extra.get('used_tests', False):
                testcase_source = first_extra.get('testcase_source', 'unknown')
                num_tests = first_extra.get('num_testcases', 0)
                print(f"[GRPO]   -> test_pass={first_extra['test_pass_rate']:.2f} "
                      f"text_score={first_extra['text_score']:.3f} "
                      f"final={first_extra['final_score']:.3f} "
                      f"tests={num_tests}({testcase_source}) "
                      f"ret={first_extra['retention']:.3f} "
                      f"diff={first_extra['diff_ratio']:.3f}")
            else:
                print(f"[GRPO]   -> text_score={first_extra.get('text_score', 0):.3f} "
                      f"(no_tests) "
                  f"ret={first_extra['retention']:.3f} "
                  f"diff={first_extra['diff_ratio']:.3f}")

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
    
    print("🚀 SFT + GRPO Training Pipeline with DFT & Test Cases")
    print("=" * 60)
    print(f"🏷️ Model Name: {targs.model_name}")
    print(f"📁 Model Path: {targs.base_model}")
    print(f"📂 Data File: {targs.train_file}")
    print(f"💾 Output Dir: {targs.out_dir}")
    print(f"🔗 LoRA: Rank={targs.lora_r}, Alpha={targs.lora_alpha}")
    print(f"🎯 Training: SFT={targs.sft_epochs} epochs, GRPO={targs.grpo_epochs} epochs")
    print(f"⚙️ Device: {targs.device}")
    if targs.use_dft:
        print(f"🧠 DFT: Enabled - Dynamic Fine-Tuning (Probability Rescaling)")
    else:
        print(f"📚 SFT: Standard Supervised Fine-Tuning")
    
    if targs.use_neftune:
        print(f"🔊 NEFTune: Enabled - Noisy Embeddings (alpha={targs.neftune_alpha})")
    else:
        print(f"🔇 NEFTune: Disabled")
    
    if targs.use_8bit:
        print(f"⚡ Quantization: 8-bit (Memory Optimized)")
    elif targs.use_4bit:
        print(f"⚡ Quantization: 4-bit (Maximum Memory Saving)")
    else:
        print(f"💾 Quantization: Disabled (Full Precision)")
    
    print(f"🧪 External Test Cases: {targs.external_testcase_file}")
    print(f"🏷️ Problem ID Field: {targs.problem_id_field}")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(targs.out_dir, exist_ok=True)

    # 加载数据集
    print(f"📖 Loading dataset from: {targs.train_file}")
    dataset = CodeDataset(targs.train_file)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # 初始化训练器
    print("🔧 Initializing SFT + GRPO trainer...")
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
        
        try:
            # 步骤1：加载原始base model
            print("🔧 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                targs.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            
            # 步骤2：重新创建LoRA配置并加载SFT权重
            print("🔧 Loading SFT LoRA adapter...")
            from peft import PeftModel, LoraConfig, get_peft_model
            target_modules = trainer._get_target_modules(targs.base_model)
            
            # 加载训练好的LoRA模型
            sft_lora_model = PeftModel.from_pretrained(
                base_model,
                sft_model_path,
                torch_dtype=torch.bfloat16
            )
            
            # 步骤3：合并LoRA权重到base model（这是关键步骤）
            print("🔧 Merging LoRA weights into base model...")
            merged_model = sft_lora_model.merge_and_unload()
            
            # 步骤4：在合并后的模型上重新应用LoRA用于GRPO训练
            print("🔧 Applying fresh LoRA for GRPO training...")
            lora_config = LoraConfig(
                r=targs.lora_r,
                lora_alpha=targs.lora_alpha,
                lora_dropout=targs.lora_dropout,
                target_modules=target_modules,
                task_type="CAUSAL_LM"
            )
            trainer.model = get_peft_model(merged_model, lora_config)
            
            # 创建新的pipeline用于GRPO生成
            print("🚀 Creating fresh inference pipeline for GRPO...")
            trainer.inference_pipe = pipeline("text-generation", model=trainer.model, tokenizer=trainer.tokenizer)
            print("✓ Fresh inference pipeline ready")
            
            # 修复：手动设置LoRA参数为可训练
            trainable_count = 0
            for name, param in trainer.model.named_parameters():
                if 'lora_' in name.lower():
                    param.requires_grad = True
                    trainable_count += 1
            
            trainer.model.train()
            
            # 立即验证设置结果
            verification_count = sum(1 for p in trainer.model.parameters() if p.requires_grad)
            print(f"✓ 设置了 {trainable_count} 个LoRA参数为可训练")
            print(f"✓ 验证：当前有 {verification_count} 个可训练参数")
            
            if verification_count == 0:
                print("❌ 警告：参数设置后验证发现没有可训练参数！")
            
            # 再次验证主模型的可训练参数状态
            final_trainable = sum(1 for p in trainer.model.parameters() if p.requires_grad)
            print(f"✓ 主模型有 {final_trainable} 个可训练参数，无需参考模型")
            
            print("✓ SFT模型加载成功，权重已合并，新LoRA已应用")
            
        except Exception as e:
            print(f"❌ SFT模型加载失败: {e}")
            import traceback
            print(f"❌ 详细错误信息:")
            traceback.print_exc()
            
            print("\n⚠️  SFT模型加载失败，可能的原因：")
            print("   1. 模型文件损坏")
            print("   2. PEFT版本不兼容") 
            print("   3. 模型格式问题")
            print("   4. 内存不足")
            
            print(f"\n❌ 无法继续GRPO训练，因为需要SFT模型")
            print(f"📁 SFT模型目录保留在: {sft_model_path}")
            print(f"💡 请检查并修复SFT模型后重新运行")
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
        start_num=4, end_num=2  # 平衡候选数量和内存使用
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
        print(f"\n❌ GRPO训练失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n📁 SFT模型保留在: {sft_model_path}")
        print(f"💡 GRPO训练失败不影响已训练的SFT模型")

    print('\n🎉 Training Completed!')
    print("=" * 60)
    print(f"📁 Models saved under: {targs.out_dir}")
    print(f"  - SFT model: {targs.out_dir}/sft/")
    print(f"  - Final GRPO model: {targs.out_dir}/grpo/")
    print("🔬 Ready for paper experiments and evaluation!")
    print("=" * 60)

if __name__ == '__main__':
    main()
