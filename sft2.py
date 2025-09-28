"""
多模式偏好训练框架 - 支持SFT/DFT/DFT+偏好三种训练模式
"""

# ==========================================
# 系统导入与环境设置
# ==========================================
import os
import json
import random
import time
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass

os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

# ==========================================
# 深度学习框架导入
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# ==========================================
# 预训练模型与微调框架
# ==========================================
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ==========================================
# 配置类
# ==========================================

@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 模型配置
    model_base_path: str = "/data1/czj/model"
    output_base_path: str = "/data1/czj/train_model"
    
    # 数据配置
    train_file: str = "/data1/czj/prorepair/data/trainset/sft_dataset1.json"
    max_length: int = 2048
    num_negatives: int = 2
    chat_template: str = "llama"
    
    # 训练模式: "sft", "dft", "dft_preference"
    training_mode: str = "dft_preference"
    
    # 训练参数
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_clipping: float = 0.3
    
    # 课程学习
    curriculum_stages: int = 3
    
    # QLoRA配置
    lora_rank: int = 32
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    
    # 量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # 损失函数
    preference_beta: float = 0.5
    
    # 优化配置
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # NEFTune噪声
    use_neftune: bool = True
    neftune_alpha: float = 5.0
    
    # 日志配置
    logging_steps: int = 100
    save_steps: int = 500
    
    
    # 其他必需属性
    model_name_only: str = ""  # 由命令行参数指定
    use_curriculum: bool = True
    curriculum_strategy: str = "bug_count"
    
    def __post_init__(self) -> None:
        """初始化后处理"""
        self.model_name = os.path.join(self.model_base_path, self.model_name_only)
        model_short_name = self.model_name_only.lower().replace("-", "_")
        self.output_dir = os.path.join(
            self.output_base_path, 
            f"trained_model_{model_short_name}_curriculum"
        )
        if self.target_modules is None:
            self.target_modules = self._get_default_target_modules()
    
    def _get_default_target_modules(self) -> List[str]:
        """自动选择QLoRA目标模块"""
        model_name_lower = self.model_name_only.lower()
        if "llama" in model_name_lower or "qwen" in model_name_lower:
            return ["q_proj", "v_proj"]
        else:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]

# ==========================================
# 数据处理
# ==========================================

class ChatTemplate:
    """聊天模板处理器"""
    
    def __init__(self, template_type: str = "llama") -> None:
        self.template_type = template_type.lower()
        if self.template_type not in ["llama", "qwen"]:
            raise ValueError(f"不支持的模板类型: {template_type}，支持: llama, qwen")
    
    def format_conversation(self, prompt: str, response: str, explanation: str = "") -> str:
        """格式化对话为训练文本"""
        full_response = response.strip()
        if explanation and explanation.strip():
            full_response = f"{full_response}\n\nExplanation: {explanation.strip()}"
        
        if self.template_type == "qwen":
            return (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{full_response}<|im_end|>"
            )
        else:  # llama格式
            return f"<s>[INST] {prompt} [/INST] {full_response}</s>"

class CurriculumSampler(Sampler):
    """🎓 课程学习采样器 - 动态调整采样范围"""
    def __init__(self, dataset, curriculum_scheduler):
        self.dataset = dataset
        self.curriculum_scheduler = curriculum_scheduler
        self.current_stage = 0
        
    def update_stage(self, stage: int):
        """更新当前阶段"""
        self.current_stage = stage
        
    def __iter__(self):
        # 根据当前阶段获取可用的数据索引
        stage_data = self.curriculum_scheduler.get_stage_data(self.dataset.sorted_data, self.current_stage)
        indices = list(range(len(stage_data)))
        # 随机打乱
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        stage_data = self.curriculum_scheduler.get_stage_data(self.dataset.sorted_data, self.current_stage)
        return len(stage_data)

class CurriculumScheduler:
    """🎓 课程学习调度器"""
    def __init__(self, config: TrainingConfig, total_samples: int):
        self.config = config
        self.total_samples = total_samples
        self.current_stage = 0
        self.max_stages = config.curriculum_stages
        
    def get_difficulty_score(self, sample):
        """计算样本难度分数 - 基于bug_hunks_count"""
        return sample.get('bug_hunks_count', 0)
    
    def sort_by_difficulty(self, data):
        """按难度排序数据 - 从简单到困难"""
        # 计算每个样本的难度分数
        scored_data = [(sample, self.get_difficulty_score(sample)) for sample in data]
        
        # 按难度排序（简单到困难）
        scored_data.sort(key=lambda x: x[1])
        
        return [sample for sample, _ in scored_data]
    
    def get_stage_data(self, sorted_data, stage: int):
        """获取当前阶段的数据"""
        # 计算当前阶段应该包含的数据比例
        stage_ratio = (stage + 1) / self.max_stages
        stage_size = int(len(sorted_data) * stage_ratio)
        
        # 返回从简单到当前难度的所有数据
        return sorted_data[:stage_size]
    
    def update_stage(self, current_step: int, total_steps: int):
        """更新课程学习阶段"""
        # 计算当前应该处于哪个阶段
        progress = current_step / total_steps
        new_stage = min(int(progress * self.max_stages), self.max_stages - 1)
        
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            return True  # 阶段发生变化
        return False
    
    def get_difficulty_distribution(self, data):
        """获取数据难度分布统计"""
        difficulty_scores = [self.get_difficulty_score(sample) for sample in data]
        
        if not difficulty_scores:
            return {}
        
        return {
            'min_difficulty': min(difficulty_scores),
            'max_difficulty': max(difficulty_scores),
            'avg_difficulty': sum(difficulty_scores) / len(difficulty_scores),
            'total_samples': len(difficulty_scores)
        }

class PreferenceDataset(Dataset):
    """偏好学习数据集 - 支持课程学习"""
    def __init__(self, data_path: str, tokenizer, chat_template: ChatTemplate,
                 max_length: int = 2048, num_negatives: int = 3, 
                 curriculum_scheduler: CurriculumScheduler = None):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.curriculum_scheduler = curriculum_scheduler
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                self.raw_data = [json.loads(line) for line in f if line.strip()]
            else:
                self.raw_data = json.load(f)
        
        print(f"Loaded {len(self.raw_data)} samples")
        
        # 🎓 课程学习：按难度排序
        self.sorted_data = curriculum_scheduler.sort_by_difficulty(self.raw_data)
        
        # 打印难度分布统计
        full_stats = curriculum_scheduler.get_difficulty_distribution(self.sorted_data)
        stage_0_data = curriculum_scheduler.get_stage_data(self.sorted_data, 0)
        stage_stats = curriculum_scheduler.get_difficulty_distribution(stage_0_data)
        
        print(f"Curriculum Learning: {stage_stats['total_samples']}/{full_stats['total_samples']} samples in stage 0")
        
        # 使用完整的排序数据，由sampler控制实际使用的数据
        self.data = self.sorted_data
    
    def update_curriculum_stage(self, stage: int):
        """更新课程学习阶段"""
        self.current_stage_data = self.curriculum_scheduler.get_stage_data(self.sorted_data, stage)
        self.data = self.current_stage_data
        
        stage_stats = self.curriculum_scheduler.get_difficulty_distribution(self.current_stage_data)
        print(f"Curriculum Stage {stage}: {stage_stats['total_samples']} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def _process_sample_with_prompt_masking(tokenizer, chat_template, prompt, response, explanation, max_length):
    """处理样本并正确掩码prompt部分"""
    if explanation and explanation.strip():
        full_response = f"{explanation.strip()}\n\n{response.strip()}"
    else:
        full_response = response.strip()
    
    full_text = chat_template.format_conversation(prompt, full_response)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = tokenized['input_ids'].squeeze(0)
    attention_mask = tokenized['attention_mask'].squeeze(0)
    
    # 修复：更准确地计算prompt长度
    # 方法1：使用完整的prompt模板来计算长度
    if chat_template.template_type == "qwen":
        prompt_template = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama格式
        prompt_template = f"<s>[INST] {prompt} [/INST] "
    
    # 使用相同的tokenizer设置来计算prompt长度
    prompt_tokenized = tokenizer(
        prompt_template,
        add_special_tokens=False,  # 因为模板中已经包含了特殊token
        return_tensors="pt"
    )
    prompt_len = prompt_tokenized['input_ids'].size(1)
    
    labels = input_ids.clone()
    # 确保不会超出序列长度
    prompt_len = min(prompt_len, len(labels))
    labels[:prompt_len] = -100  # 只对response部分计算损失
    
    # 调试信息：检查有多少个token用于损失计算
    valid_tokens = (labels != -100).sum().item()
    total_tokens = len(labels)
    # print(f"Debug: prompt_len={prompt_len}, valid_tokens={valid_tokens}, total_tokens={total_tokens}")
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def collate_fn(batch, tokenizer, chat_template, max_length, num_negatives):
    """批处理函数"""
    chosen_batch = []
    rejected_batch = []
    
    for sample in batch:
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected_list = sample['rejected']
        explanation = sample.get('explanation', '')
        
        # 处理rejected列表
        if isinstance(rejected_list, str):
            rejected_list = [rejected_list]
        
        # 采样negatives
        if len(rejected_list) > num_negatives:
            rejected_list = random.sample(rejected_list, num_negatives)
        
        # 处理chosen
        chosen_item = _process_sample_with_prompt_masking(
            tokenizer, chat_template, prompt, chosen, explanation, max_length
        )
        chosen_batch.append(chosen_item)
        
        # 处理rejected
        for rejected in rejected_list:
            rejected_item = _process_sample_with_prompt_masking(
                tokenizer, chat_template, prompt, rejected, "", max_length
            )
            rejected_batch.append(rejected_item)
    
    # Padding function with separate handling for chosen and rejected
    def pad_batch(batch_list, is_chosen=True):
        if not batch_list:
            return None
            
        max_len = max(item['input_ids'].size(0) for item in batch_list)  # 修复：使用size(0)而不是size(1)
        batch_size = len(batch_list)
        
        input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, item in enumerate(batch_list):
            seq_len = item['input_ids'].size(0)  # 修复：使用size(0)
            input_ids[i, :seq_len] = item['input_ids']  # 修复：不需要squeeze
            attention_mask[i, :seq_len] = item['attention_mask']  # 修复：不需要squeeze
            
            # chosen使用预处理的labels，rejected全部为-100
            if is_chosen:
                labels[i, :seq_len] = item['labels']
            # 修复：padding部分的labels保持为-100（已经在初始化时设置）
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return {
        'chosen': pad_batch(chosen_batch, is_chosen=True),
        'rejected': pad_batch(rejected_batch, is_chosen=False)
    }

# ==========================================
# 损失函数
# ==========================================

class SFTLoss(nn.Module):
    """标准SFT损失：交叉熵损失"""
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, model, chosen_batch, rejected_batch=None):
        """计算标准SFT损失"""
        if chosen_batch is None:
            device = next(model.parameters()).device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {'lm_loss': zero_loss, 'total_loss': zero_loss}
        
        outputs = model(
            input_ids=chosen_batch['input_ids'],
            attention_mask=chosen_batch['attention_mask'],
            labels=chosen_batch['labels']
        )
        lm_loss = outputs.loss
        return lm_loss, {'lm_loss': lm_loss, 'total_loss': lm_loss}

class DFTLoss(nn.Module):
    """纯DFT损失：E[P(token) * (-log P(token))]"""
    def __init__(self):
        super().__init__()
    
    def forward(self, model, chosen_batch, rejected_batch=None):
        """计算DFT损失"""
        if chosen_batch is None:
            device = next(model.parameters()).device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {'lm_loss': zero_loss, 'total_loss': zero_loss}
        
        lm_loss = self._compute_dft_loss(model, chosen_batch)
        return lm_loss, {'lm_loss': lm_loss, 'total_loss': lm_loss}
    
    def _compute_dft_loss(self, model, batch):
        """DFT风格LM损失：E[P(token) * (-log P(token))]"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=False
        )
        logits = outputs.logits
        labels = batch['labels']
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        # 修复：只使用非-100的位置作为mask，不要fallback到attention_mask
        mask = (labels != -100).float()
        
        # 如果mask全为0，说明数据处理有问题，返回一个小的非零损失而不是0
        if mask.sum() == 0:
            print("Warning: No valid tokens for DFT loss calculation, returning small loss")
            device = next(model.parameters()).device
            return torch.tensor(1e-6, device=device, requires_grad=True)
        
        labels_safe = labels.clone()
        labels_safe[labels == -100] = 0
        target_log_probs = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        token_losses = (-target_log_probs) * target_probs
        token_losses = token_losses * mask
        return token_losses.sum() / mask.sum().clamp(min=1.0)

class HybridLoss(nn.Module):
    """混合损失：DFT风格LM损失 + 偏好对比损失
    
    - LM路径固定为DFT样式：loss = E[P(token) * (-log P(token))]
    - 再加权偏好对比损失（chosen vs rejected）
    """
    def __init__(self, lm_weight: float = 1.0, preference_weight: float = 0.5, preference_beta: float = 0.5):
        super().__init__()
        self.lm_weight = lm_weight
        self.preference_weight = preference_weight
        self.preference_beta = preference_beta
        
    def forward(self, model, chosen_batch, rejected_batch):
        """计算混合损失"""
        device = next(model.parameters()).device
        individual_losses = {}
        
        # 1) DFT风格的LM损失（仅对chosen）
        if chosen_batch is not None:
            lm_loss = self._compute_lm_loss(model, chosen_batch)
            individual_losses['lm_loss'] = lm_loss
            total_loss = self.lm_weight * lm_loss
        else:
            individual_losses['lm_loss'] = torch.tensor(0.0, device=device)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 2) 偏好对比损失（pairwise logistic）
        if chosen_batch is not None and rejected_batch is not None:
            # 计算chosen的log概率
            chosen_log_prob = self._get_sequence_log_prob(model, chosen_batch)
            
            # 计算rejected的log概率（不产生梯度）
            with torch.no_grad():
                rejected_log_prob = self._get_sequence_log_prob(model, rejected_batch)
            
            # 处理维度不匹配
            if len(rejected_log_prob) != len(chosen_log_prob):
                num_chosen = len(chosen_log_prob)
                rejected_log_prob = rejected_log_prob.view(num_chosen, -1).mean(dim=1)
            
            # 偏好损失：希望chosen的概率大于rejected
            diff = self.preference_beta * (chosen_log_prob - rejected_log_prob)
            preference_loss = -F.logsigmoid(diff).mean()
            
            total_loss = total_loss + self.preference_weight * preference_loss
            individual_losses['preference_loss'] = preference_loss
        else:
            individual_losses['preference_loss'] = torch.tensor(0.0, device=device)
        
        individual_losses['total_loss'] = total_loss
        return total_loss, individual_losses
    
    def _get_sequence_log_prob(self, model, batch):
        """计算序列的log概率"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # 修复：使用batch['labels']而不是重新计算shift
        # 这样可以保持与mask处理的一致性
        logits = outputs.logits
        labels = batch['labels']
        
        # 确保logits和labels的维度匹配
        if logits.size(1) != labels.size(1):
            min_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 修复：只使用非-100的位置作为mask，与DFT loss保持一致
        mask = (labels != -100).float()
        
        # 如果mask全为0，返回一个小的log概率
        if mask.sum() == 0:
            device = next(model.parameters()).device
            batch_size = labels.size(0)
            return torch.full((batch_size,), -10.0, device=device)
        
        # 收集目标token的log概率
        labels_safe = labels.clone()
        labels_safe[labels == -100] = 0  # 防止index out of range
        
        token_log_probs = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * mask
        
        # 计算序列级别的平均log概率
        seq_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return seq_log_probs

    def _compute_lm_loss(self, model, batch):
        """DFT风格LM损失：E[P(token) * (-log P(token))]，与sft_grpo.py对齐。"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=False
        )
        # 与 sft_grpo.py 的 _dft_forward_step 对齐：不进行 token shift
        logits = outputs.logits
        labels = batch['labels']
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        # 修复：只使用非-100的位置作为mask，不要fallback到attention_mask
        mask = (labels != -100).float()
        
        # 如果mask全为0，说明数据处理有问题，返回一个小的非零损失而不是0
        if mask.sum() == 0:
            print("Warning: No valid tokens for LM loss calculation, returning small loss")
            device = next(model.parameters()).device
            return torch.tensor(1e-6, device=device, requires_grad=True)
        
        labels_safe = labels.clone()
        labels_safe[labels == -100] = 0
        target_log_probs = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        token_losses = (-target_log_probs) * target_probs
        token_losses = token_losses * mask
        return token_losses.sum() / mask.sum().clamp(min=1.0)

# ==========================================
# 训练器
# ==========================================

class CurriculumPreferenceTrainer:
    """🎓 课程学习偏好训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # 🎓 课程学习组件
        self.curriculum_scheduler = None
        self.dataset = None
        
        # 根据训练模式选择损失函数
        self.loss_fn = self._create_loss_function(config)
        
        # 训练状态
        self.global_step = 0
        self.loss_history = []
    
    def _create_loss_function(self, config: TrainingConfig) -> nn.Module:
        """根据训练模式创建损失函数"""
        if config.training_mode == "sft":
            print("🎯 训练模式: 标准SFT")
            return SFTLoss()
        elif config.training_mode == "dft":
            print("🎯 训练模式: 纯DFT")
            return DFTLoss()
        elif config.training_mode == "dft_preference":
            print(f"🎯 训练模式: DFT + 偏好损失")
            return HybridLoss(
                lm_weight=1.0,
                preference_weight=config.preference_beta,
                preference_beta=config.preference_beta
            )
        else:
            raise ValueError(f"不支持的训练模式: {config.training_mode}")
    
    def setup_model(self):
        """设置模型"""
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            use_fast=True, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # QLoRA 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
        
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",  
            trust_remote_code=True,
            use_cache=False
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        
        # 准备k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        # QLoRA配置
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.target_modules
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # 梯度检查点
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # 需要为输入启用梯度以兼容检查点
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        
        # 确保只有LoRA参数需要梯度
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 参数统计
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params:,}")
        
        # 验证LoRA参数
        lora_params = sum(p.numel() for name, p in self.model.named_parameters() if 'lora' in name.lower() and p.requires_grad)
        if lora_params == 0:
            raise RuntimeError("No LoRA parameters found with requires_grad=True")
        
        # NEFTune噪声
        if getattr(self.config, 'use_neftune', True) and self.config.neftune_alpha > 0:
            alpha = float(self.config.neftune_alpha)
            def neftune_forward_hook(module, input, output):
                if module.training and alpha > 0:
                    noise = torch.randn_like(output) * (alpha / (output.numel() ** 0.5))
                    return output + noise
                return output
            neftune_modules = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Embedding):
                    module.register_forward_hook(neftune_forward_hook)
                    neftune_modules.append(name)
            if not neftune_modules:
                print("⚠️ NEFTune: no embedding modules found")
    
    def setup_data(self):
        """设置数据 - 课程学习"""
        chat_template = ChatTemplate(self.config.chat_template)
        
        # 🎓 初始化课程学习调度器
        # 先加载数据来计算总样本数
        with open(self.config.train_file, 'r', encoding='utf-8') as f:
            if self.config.train_file.endswith('.jsonl'):
                temp_data = [json.loads(line) for line in f if line.strip()]
            else:
                temp_data = json.load(f)
        
        self.curriculum_scheduler = CurriculumScheduler(self.config, len(temp_data))
        print(f"Curriculum stages: {self.config.curriculum_stages}")
        
        # 创建数据集（支持课程学习）
        self.dataset = PreferenceDataset(
            self.config.train_file,
            self.tokenizer,
            chat_template,
            self.config.max_length,
            self.config.num_negatives,
            self.curriculum_scheduler
        )
        
        # 🎓 创建课程学习采样器
        self.curriculum_sampler = CurriculumSampler(self.dataset, self.curriculum_scheduler)
        
        # 创建数据加载器 - 使用自定义采样器
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=self.curriculum_sampler,  # 使用自定义采样器替代shuffle
            collate_fn=lambda batch: collate_fn(
                batch, self.tokenizer, chat_template,
                self.config.max_length, self.config.num_negatives
            ),
            num_workers=0
        )
    
    def setup_optimizer(self):
        """设置优化器"""
        # 修复：只优化需要梯度的参数（LoRA参数）
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Training steps: {total_steps}, warmup: {warmup_steps}")
    
    def train_step(self, batch):
        """训练步骤 - 使用混合损失（CrossEntropyLoss + PreferenceLoss）"""
        # 移动数据到设备并确保需要梯度
        for key in ['chosen', 'rejected']:
            if key in batch and batch[key] is not None:
                for tensor_key in batch[key]:
                    if isinstance(batch[key][tensor_key], torch.Tensor):
                        batch[key][tensor_key] = batch[key][tensor_key].to(self.device)
                        # 对于input_ids，确保需要梯度以支持梯度检查点
                        if tensor_key == 'input_ids':
                            batch[key][tensor_key] = batch[key][tensor_key].requires_grad_(False)  # input_ids不需要梯度
        
        # 确保模型在训练模式
        self.model.train()
        
        # 使用选定的损失函数
        if batch['chosen'] is not None:
            total_loss, loss_dict = self.loss_fn(self.model, batch['chosen'], batch.get('rejected'))
            
            # 提取各个损失组件
            lm_loss = loss_dict['lm_loss']
            preference_loss = loss_dict.get('preference_loss', torch.tensor(0.0))
            
            # 应用梯度累积
            total_loss = total_loss / self.config.gradient_accumulation_steps
            
            return total_loss, lm_loss, preference_loss
        else:
            # 如果没有chosen数据，返回零损失
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero_loss, zero_loss, zero_loss
    
    def train(self):
        """🎓 课程学习训练主循环"""
        print("🚀 Starting Curriculum Preference Training...")
        
        # 设置模型、数据、优化器
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        self.model.train()
        
        # 训练时间记录
        start_time = time.time()
        start_datetime = datetime.now()
        
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        print(f"📊 Training Info:")
        print(f"   Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total steps: {total_steps}")
        if self.config.use_curriculum:
            print(f"   🎓 Curriculum: {self.config.curriculum_stages} stages, {self.config.curriculum_strategy} strategy")
        print("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # 训练步骤 - 现在返回lm_loss和preference_loss
                total_loss, lm_loss, preference_loss = self.train_step(batch)
                
                # 反向传播
                total_loss.backward()
                
                # 优化器步骤
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
                    
                    # 🎓 课程学习：检查是否需要更新阶段
                    if self.curriculum_scheduler.update_stage(self.global_step, total_steps):
                        stage = self.curriculum_scheduler.current_stage
                        # 修复：只更新采样器，不重建DataLoader
                        self.curriculum_sampler.update_stage(stage)
                        print(f"🎓 Updated curriculum to stage {stage}")
                    
                    # 日志记录
                    if self.global_step % self.config.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed_time = time.time() - start_time
                        progress = (self.global_step / total_steps) * 100
                        
                        # 添加课程学习信息到日志
                        stage = self.curriculum_scheduler.current_stage
                        stage_data = self.curriculum_scheduler.get_stage_data(self.dataset.sorted_data, stage)
                        samples_count = len(stage_data)
                        total_samples = len(self.dataset.raw_data)
                        curriculum_info = f" | 🎓 Stage: {stage} ({samples_count}/{total_samples})"
                        
                        self.loss_history.append({
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            'total_loss': total_loss.item(),
                            'lm_loss': lm_loss.item(),
                            'preference_loss': preference_loss.item(),
                            'lr': lr,
                            'curriculum_stage': self.curriculum_scheduler.current_stage,
                            'curriculum_samples': samples_count
                        })
                        
                        print(f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                              f"Step {self.global_step}/{total_steps} ({progress:.1f}%) | "
                              f"LM: {lm_loss.item():.4f} | Pref: {preference_loss.item():.4f} | "
                              f"Total: {total_loss.item():.4f} | LR: {lr:.2e}{curriculum_info}")
                
                epoch_loss += total_loss.item()
            
            print(f"✅ Epoch {epoch+1} completed | Avg Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        # 保存模型
        self.save_model()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        end_datetime = datetime.now()
        
        print("🎉 Curriculum Preference Training completed!")
        print(f"📊 Training Summary:")
        print(f"   • Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Total time: {total_time/60:.1f} minutes")
        print(f"   • Total steps: {self.global_step}")
        print(f"   • Final curriculum stage: {self.curriculum_scheduler.current_stage}")
        print("=" * 60)
    
    def save_model(self):
        """保存模型"""
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"💾 Model saved to {self.config.output_dir}")
        
        # 保存训练统计
        if self.loss_history:
            stats_file = os.path.join(self.config.output_dir, "training_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
            print(f"📊 Training stats saved to {stats_file}")
        
        # 保存课程学习配置
        curriculum_config = {
            'curriculum_stages': self.config.curriculum_stages,
            'final_stage': self.curriculum_scheduler.current_stage,
            'strategy': 'bug_count'
        }
        curriculum_file = os.path.join(self.config.output_dir, "curriculum_config.json")
        with open(curriculum_file, 'w') as f:
            json.dump(curriculum_config, f, indent=2)
        print(f"🎓 Curriculum config saved to {curriculum_file}")

# ==========================================
# 主程序
# ==========================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🎓 Curriculum Preference Training')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name (e.g., "Llama-3-8B-Instruct")')
    parser.add_argument('--curriculum_stages', type=int, default=3,
                       help='Number of curriculum stages')
    parser.add_argument('--training_mode', type=str, default='dft_preference',
                       choices=['sft', 'dft', 'dft_preference'],
                       help='Training mode: sft (standard), dft (pure DFT), dft_preference (DFT + preference)')
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(model_name_only=args.model_name)
    config.curriculum_stages = args.curriculum_stages
    config.training_mode = args.training_mode
    
    print("Curriculum Preference Training")
    print(f"Model: {config.model_name_only}")
    print(f"Mode: {config.training_mode}")
    print(f"LoRA: r={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"Epochs: {config.num_epochs}, LR: {config.learning_rate}")
    print(f"Output: {config.output_dir}")
    
    try:
        # 开始训练
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        trainer = CurriculumPreferenceTrainer(config)
        trainer.train()
        
        print(f"\nTraining completed! Model saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
