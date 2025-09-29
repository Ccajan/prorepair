import os
import json
import random
import time

os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)   # 取消你之前设置的 allocator 调整
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"           # 精确定位发生错误的 kernel（会慢）
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 配置类
# ==========================================

@dataclass


class Config:
    """训练配置"""
    # 动态参数 - 只有模型名称用于拼接路径
    model_name_only: str = "Llama-3-8B-Instruct"
    
    # 固定路径配置
    model_base_path: str = "/data1/czj/model"
    train_file: str = "data/trainset/sft_dataset.json"
    output_dir: str = "/data1/czj/model/trained_model_llama"
    
    # 固定训练参数
    max_length: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    seed: int = 42
    
    # LoRA配置 - 使用标准LoRA避免复杂性
    use_chain_lora: bool = False    # 使用标准LoRA而非Chain LoRA
    chain_depth: int = 3            # Chain LoRA参数（保留但不使用）
    lora_rank: int = 8              # rank=8标准配置
    lora_alpha: float = 16.0        # 2倍rank的缩放因子
    lora_dropout: float = 0.1       # 防过拟合的dropout
    target_modules: List[str] = None
    
    # 固定损失参数
    preference_beta: float = 0.5    # pairwise loss温度
    dft_weight: float = 0.3         # DFT loss权重
    use_dft: bool = True
    num_negatives: int = 2          # 每个chosen使用的rejected数量
    
    # 固定训练优化 - 使用fp16避免bf16兼容性问题
    fp16: bool = True               # 改用fp16，更好的driver兼容性
    bf16: bool = False              # 禁用bf16避免优化器状态tensor问题
    gradient_checkpointing: bool = True   # 启用梯度检查点节省内存
    gradient_clipping: float = 1.0
    
    # 固定日志设置
    logging_steps: int = 10
    save_steps: int = 500
    
    # 固定聊天模板
    chat_template: str = "llama"
    
    def __post_init__(self):
        # 拼接完整的模型路径
        self.model_name = os.path.join(self.model_base_path, self.model_name_only)
        
        if self.target_modules is None:
            # 自动设置target_modules
            if "llama" in self.model_name_only.lower():
                self.target_modules = ["q_proj", "v_proj"]
            elif "qwen" in self.model_name_only.lower():
                self.target_modules = ["q_proj", "v_proj"]
            else:
                self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ==========================================
# 数据处理
# ==========================================

class ChatTemplate:
    """聊天模板"""
    def __init__(self, template_type: str = "llama"):
        self.template_type = template_type.lower()
    
    def format_conversation(self, prompt: str, response: str, explanation: str = "") -> str:
        full_response = response.strip()
        if explanation and explanation.strip():
            full_response = f"{full_response}\n\nExplanation: {explanation.strip()}"
        
        if self.template_type == "qwen":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{full_response}<|im_end|>"
        else:  # llama
            return f"<s>[INST] {prompt} [/INST] {full_response}</s>"

class PreferenceDataset(Dataset):
    """偏好学习数据集"""
    def __init__(self, data_path: str, tokenizer, chat_template: ChatTemplate,
                 max_length: int = 2048, num_negatives: int = 3):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.num_negatives = num_negatives
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                self.data = [json.loads(line) for line in f if line.strip()]
            else:
                self.data = json.load(f)
        
        print(f"✅ Loaded {len(self.data)} preference samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
        
        # 采样negatives - 确保每个样本都有固定数量
        if len(rejected_list) > num_negatives:
            rejected_list = random.sample(rejected_list, num_negatives)
        elif len(rejected_list) < num_negatives:
            # 如果rejected不够，重复采样补足
            rejected_list = rejected_list * ((num_negatives // len(rejected_list)) + 1)
            rejected_list = rejected_list[:num_negatives]
        
        # Tokenize chosen
        chosen_text = chat_template.format_conversation(prompt, chosen, explanation)
        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        chosen_batch.append(chosen_tokens)
        
        # Tokenize rejected  
        for rejected in rejected_list:
            rejected_text = chat_template.format_conversation(prompt, rejected)
            rejected_tokens = tokenizer(
                rejected_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            rejected_batch.append(rejected_tokens)
    
    # Padding
    def pad_batch(batch_list):
        if not batch_list:
            return None
            
        max_len = max(item['input_ids'].size(1) for item in batch_list)
        batch_size = len(batch_list)
        
        input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, item in enumerate(batch_list):
            seq_len = item['input_ids'].size(1)
            input_ids[i, :seq_len] = item['input_ids'].squeeze(0)
            attention_mask[i, :seq_len] = item['attention_mask'].squeeze(0)
            labels[i, :seq_len] = item['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return {
        'chosen': pad_batch(chosen_batch),
        'rejected': pad_batch(rejected_batch)
    }

# ==========================================
# 损失函数
# ==========================================

class PreferenceLoss(nn.Module):
    """偏好损失"""
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta
    
    def get_log_probs(self, model, batch):
        """计算log概率"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        logits = outputs.logits[:, :-1, :]
        labels = batch['labels'][:, 1:].contiguous()
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        mask = (labels != -100).float()
        labels_safe = labels.clone()
        labels_safe[labels == -100] = 0
        
        token_log_probs = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * mask
        
        seq_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return seq_log_probs
    
    def forward(self, model, chosen_batch, rejected_batch):
        """计算偏好损失"""
        chosen_log_probs = self.get_log_probs(model, chosen_batch)
        rejected_log_probs = self.get_log_probs(model, rejected_batch)
        
        # 处理维度不匹配 - 现在应该总是能整除
        if len(rejected_log_probs) != len(chosen_log_probs):
            num_chosen = len(chosen_log_probs)
            rejected_per_chosen = len(rejected_log_probs) // num_chosen
            rejected_log_probs = rejected_log_probs.view(num_chosen, rejected_per_chosen).mean(dim=1)
        
        # Pairwise logistic loss
        diff = self.beta * (chosen_log_probs - rejected_log_probs)
        loss = -F.logsigmoid(diff).mean()
        return loss

class DFTLoss(nn.Module):
    """DFT损失"""
    def __init__(self, weight: float = 0.3):
        super().__init__()
        self.weight = weight
    
    def forward(self, model, batch):
        """计算DFT损失"""
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        logits = outputs.logits[:, :-1, :]
        labels = batch['labels'][:, 1:].contiguous()
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        mask = (labels != -100).float()
        labels_safe = labels.clone()
        labels_safe[labels == -100] = 0
        
        target_probs = probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        target_log_probs = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
        
        dft_loss = -(target_probs.detach() * target_log_probs) * mask
        
        return (dft_loss.sum() / mask.sum().clamp(min=1.0)) * self.weight

# ==========================================
# 训练器
# ==========================================

class PreferenceTrainer:
    """偏好学习训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # 损失函数
        self.preference_loss = PreferenceLoss(config.preference_beta)
        self.dft_loss = DFTLoss(config.dft_weight)
        
        # 训练状态
        self.global_step = 0
        self.loss_history = []
    
    def setup_model(self):
        """设置模型"""
        print(f"🔧 Loading model: {self.config.model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            use_fast=True, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model - 直接GPU加载
        print("🔧 Loading model directly to GPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # 直接指定GPU 1
            trust_remote_code=True,
            use_cache=False
        )
        
        # 启用梯度检查点
        if self.config.gradient_checkpointing:
            print("🔧 Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()
        
        # 应用LoRA
        print(f"🔧 Applying LoRA: rank={self.config.lora_rank}")
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.target_modules
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total params: {total_params:,}")
        print(f"📊 Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def setup_data(self):
        """设置数据"""
        chat_template = ChatTemplate(self.config.chat_template)
        
        dataset = PreferenceDataset(
            self.config.train_file,
            self.tokenizer,
            chat_template,
            self.config.max_length,
            self.config.num_negatives
        )
        
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(
                batch, self.tokenizer, chat_template,
                self.config.max_length, self.config.num_negatives
            ),
            num_workers=0
        )
    
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
        
        print(f"📈 Total training steps: {total_steps}")
        print(f"🔥 Warmup steps: {warmup_steps}")
    
    def train_step(self, batch):
        """训练步骤"""
        # 移动数据到设备 - 自动检测模型设备
        model_device = next(self.model.parameters()).device
        for key in ['chosen', 'rejected']:
            if key in batch and batch[key] is not None:
                for tensor_key in batch[key]:
                    if isinstance(batch[key][tensor_key], torch.Tensor):
                        batch[key][tensor_key] = batch[key][tensor_key].to(model_device)
        
        # 计算损失
        total_loss = 0.0
        
        # Preference loss
        if batch['chosen'] is not None and batch['rejected'] is not None:
            pref_loss = self.preference_loss(self.model, batch['chosen'], batch['rejected'])
            total_loss += pref_loss
        else:
            pref_loss = torch.tensor(0.0, device=model_device)
        
        # DFT loss
        if batch['chosen'] is not None and self.config.use_dft:
            dft_loss = self.dft_loss(self.model, batch['chosen'])
            total_loss += dft_loss
        else:
            dft_loss = torch.tensor(0.0, device=model_device)
        
        total_loss = total_loss / self.config.gradient_accumulation_steps
        
        return total_loss, pref_loss, dft_loss
    
    def train(self):
        """训练主循环"""
        print("🚀 Starting preference training...")
        
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
        print("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # 训练步骤
                total_loss, pref_loss, dft_loss = self.train_step(batch)
                
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
                    
                    # 日志记录
                    if self.global_step % self.config.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed_time = time.time() - start_time
                        progress = (self.global_step / total_steps) * 100
                        
                        self.loss_history.append({
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            'total_loss': total_loss.item(),
                            'pref_loss': pref_loss.item(),
                            'dft_loss': dft_loss.item(),
                            'lr': lr
                        })
                        
                        print(f"[Epoch {epoch+1}/{self.config.num_epochs}] "
                              f"Step {self.global_step}/{total_steps} ({progress:.1f}%) | "
                              f"Pref: {pref_loss.item():.4f} | DFT: {dft_loss.item():.4f} | "
                              f"Total: {total_loss.item():.4f} | LR: {lr:.2e}")
                
                epoch_loss += total_loss.item()
            
            print(f"✅ Epoch {epoch+1} completed | Avg Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        # 保存模型
        self.save_model()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        end_datetime = datetime.now()
        
        print("🎉 Training completed!")
        print(f"📊 Training Summary:")
        print(f"   • Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Total time: {total_time/60:.1f} minutes")
        print(f"   • Total steps: {self.global_step}")
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

# ==========================================
# 主程序
# ==========================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preference Training')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name (e.g., "Llama-3-8B-Instruct")')
    args = parser.parse_args()
    
    # 创建配置
    config = Config(model_name_only=args.model_name)
    
    print("🚀 Chain of LoRA Preference Training Pipeline")
    print("=" * 60)
    print(f"🏷️ Model Name: {config.model_name_only}")
    print(f"📁 Model Path: {config.model_name}")
    print(f"📂 Data File: {config.train_file}")
    print(f"💾 Output Dir: {config.output_dir}")
    print(f"🔗 Chain of LoRA: Use={config.use_chain_lora}, Depth={config.chain_depth}, Rank={config.lora_rank}, Alpha={config.lora_alpha}")
    print(f"🎯 Training: {config.num_epochs} epochs, Batch={config.batch_size}, LR={config.learning_rate}")
    print(f"📊 Loss: Beta={config.preference_beta}, DFT Weight={config.dft_weight}")
    print(f"💬 Template: {config.chat_template}")
    print("=" * 60)
    
    try:
        # 开始训练
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        trainer = PreferenceTrainer(config)
        trainer.train()
        
        print("\n🎉 Chain of LoRA Preference Training Completed!")
        print("=" * 60)
        print(f"📁 Final model saved: {config.output_dir}")
        print("🔬 Ready for evaluation and paper experiments!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
