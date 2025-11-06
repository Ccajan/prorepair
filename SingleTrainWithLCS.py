"""
单任务训练脚本 + LCS 加权：鼓励保留原有代码结构

核心思想：
    代码修复的本质是"最小化编辑"。通过计算 buggy code 和 fixed code 的
    最长公共子序列(LCS)，我们可以明确标识哪些部分应该保留。
    对 LCS 对应的 token 赋予更高的损失权重，使模型更倾向于保留原有内容。

损失函数：
    L = (1/T) Σ_{t=1}^T w_t * CrossEntropy(y_t, ŷ_t)
    
    其中：
    w_t = {
        w_lcs    如果 y_t 在 LCS 中（应该保留）
        1.0      否则（需要修改）
    }
    
    - w_lcs: LCS token 的权重（默认 2.0，可配置）
    - 更高的权重 → 模型更重视正确预测 LCS token → 倾向保留原有内容

环境变量配置：
    - LCS_WEIGHT: LCS token 的权重，默认 2.0
    
使用示例：
    # 默认 LCS 权重 2.0
    python SingleTrainWithLCS.py model_name dataset.json output_name
    
    # 自定义 LCS 权重
    LCS_WEIGHT=3.0 python SingleTrainWithLCS.py model_name dataset.json output_name

特性：
    - 简单直观，无需额外超参调优
    - 稳定优化，不引入额外损失项
    - 计算高效，LCS 只需预计算一次
    - wandb 自动记录 LCS 覆盖率统计
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
import sys
import torch.nn as nn
import wandb
from difflib import SequenceMatcher

max_len = 2048

model_name = sys.argv[1]
original_model_name = model_name

include_prompt_in_loss = True
EPOCH_NUM = 3
print(f"配置：是否将 prompt 算入 loss: {include_prompt_in_loss}")

# 智能加载模型
if not os.path.exists(model_name):
    cache_model_name = "models--" + model_name.replace("/", "--")
    local_cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/{cache_model_name}/snapshots")
    
    if os.path.exists(local_cache_path):
        snapshots = [d for d in os.listdir(local_cache_path) if os.path.isdir(os.path.join(local_cache_path, d))]
        if snapshots:
            latest_snapshot = max(snapshots, key=lambda x: os.path.getmtime(os.path.join(local_cache_path, x)))
            model_name = os.path.join(local_cache_path, latest_snapshot)
            print(f"Loading model from local cache: {model_name}")
        else:
            print(f"Using HuggingFace model: {model_name}")
    else:
        print(f"Using HuggingFace model: {model_name} (will download if needed)")
else:
    print(f"Loading model from local path: {model_name}")

full_dataset = load_dataset("json", data_files=sys.argv[2], split="train")
output_dir = 'models/' + sys.argv[3]

# 根据模型类型设置数据格式标记
if 'opencoder' in original_model_name.lower():
    # OpenCoder 模型使用 <|im_start|> / <|im_end|> 格式（与 Qwen 相同）
    BUGGY_START_MARKER = "This is an incorrect code to the problem:\n```"
    BUGGY_END_MARKER = "```\n<|im_end|>"
    print(f"检测到 OpenCoder 模型，使用 OpenCoder 数据格式标记")
elif 'qwen' in original_model_name.lower():
    # Qwen 模型使用 <|im_start|> / <|im_end|> 格式
    BUGGY_START_MARKER = "This is an incorrect code to the problem:\n```"
    BUGGY_END_MARKER = "```\n<|im_end|>"
    print(f"检测到 Qwen 模型，使用 Qwen 数据格式标记")
elif 'llama3' in original_model_name.lower():
    BUGGY_START_MARKER = "This is an incorrect code to the problem:\n```"
    BUGGY_END_MARKER = "```\n<|eot_id|>"
    print(f"检测到 Llama3 模型，使用 Llama3 数据格式标记")
else:
    # 其他模型使用 <|EOT|> 格式
    BUGGY_START_MARKER = "This is an incorrect code to the problem:\n```"
    BUGGY_END_MARKER = "```\n<|EOT|>"
    print(f"使用默认数据格式标记")

# 检查断点续训
resume_from_checkpoint = None
wandb_run_id = None

manual_wandb_id = os.environ.get('WANDB_RUN_ID', None)
if manual_wandb_id:
    wandb_run_id = manual_wandb_id
    print(f"使用手动指定的 wandb run ID: {wandb_run_id}")

if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
        if os.path.exists(latest_checkpoint):
            resume_from_checkpoint = latest_checkpoint
            print(f"发现 checkpoint，将从 {resume_from_checkpoint} 恢复训练")
            
            if not wandb_run_id:
                trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
                if os.path.exists(trainer_state_file):
                    import json
                    try:
                        with open(trainer_state_file, 'r') as f:
                            trainer_state = json.load(f)
                            if 'log_history' in trainer_state and len(trainer_state['log_history']) > 0:
                                for log_entry in trainer_state['log_history']:
                                    if '_wandb' in log_entry or 'wandb_run_id' in log_entry:
                                        wandb_run_id = log_entry.get('wandb_run_id', None)
                                        if wandb_run_id:
                                            print(f"从 checkpoint 的 trainer_state.json 找到 wandb run ID: {wandb_run_id}")
                                            break
                    except Exception as e:
                        print(f"读取 trainer_state.json 时出错: {e}")
                
                if not wandb_run_id:
                    wandb_id_file = os.path.join(output_dir, "wandb_run_id.txt")
                    if os.path.exists(wandb_id_file):
                        with open(wandb_id_file, 'r') as f:
                            wandb_run_id = f.read().strip()
                        print(f"从 wandb_run_id.txt 找到 wandb run ID: {wandb_run_id}")
                
                if not wandb_run_id:
                    wandb_dir = os.path.join(output_dir, "wandb")
                    if os.path.exists(wandb_dir):
                        run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
                        if run_dirs:
                            run_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x)), reverse=True)
                            latest_run_dir = run_dirs[0]
                            run_id_parts = latest_run_dir.split('-')
                            if len(run_id_parts) >= 3:
                                wandb_run_id = run_id_parts[-1]
                                print(f"从 wandb 目录找到 wandb run ID: {wandb_run_id}")
        else:
            print("未找到有效的 checkpoint，将从头开始训练")
    else:
        print("未找到 checkpoint，将从头开始训练")
else:
    print("首次训练，将从头开始")

# LCS 加权配置
LCS_WEIGHT = float(os.environ.get('LCS_WEIGHT', '2.0'))

print(f"LCS 加权配置:")
print(f"  LCS token 权重: {LCS_WEIGHT}")
print(f"  非 LCS token 权重: 1.0")
print(f"  策略: 对 LCS 匹配的 token 赋予更高权重，鼓励保留原有代码结构")

# 初始化 wandb
if wandb_run_id:
    wandb.init(
        project="morepair-lcs-training",
        name=sys.argv[3],
        id=wandb_run_id,
        resume="must",
        config={
            "model_name": model_name,
            "max_len": max_len,
            "learning_rate": 2e-4,
            "num_epochs": EPOCH_NUM,
            "batch_size": 1,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "include_prompt_in_loss": include_prompt_in_loss,
            "lcs_weight": LCS_WEIGHT,
            "method": "lcs_weighted",
        }
    )
else:
    wandb.init(
        project="morepair-lcs-training",
        name=sys.argv[3],
        config={
            "model_name": model_name,
            "max_len": max_len,
            "learning_rate": 2e-4,
            "num_epochs": EPOCH_NUM,
            "batch_size": 1,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "include_prompt_in_loss": include_prompt_in_loss,
            "lcs_weight": LCS_WEIGHT,
            "method": "lcs_weighted",
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "wandb_run_id.txt"), 'w') as f:
        f.write(wandb.run.id)
    print(f"保存 wandb run ID: {wandb.run.id}")

# load as 4bit model, prepare for qlora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# 根据模型类型设置正确的 eos_token_id
if 'opencoder' in original_model_name.lower():
    # OpenCoder 模型使用 <|im_end|> 作为分隔符（与 Qwen 相同）
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id == tokenizer.unk_token_id:
        # 如果 convert_tokens_to_ids 失败，尝试编码后提取
        eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    print(f"OpenCoder 模型使用 <|im_end|> 作为分隔符，token ID: {eos_token_id}")
elif 'qwen' in original_model_name.lower():
    # Qwen 模型使用 <|im_end|> 作为分隔符
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id == tokenizer.unk_token_id:
        # 如果 convert_tokens_to_ids 失败，尝试编码后提取
        eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    print(f"Qwen 模型使用 <|im_end|> 作为分隔符，token ID: {eos_token_id}")
elif 'llama3' in original_model_name.lower():
    # Llama3 模型使用 <|eot_id|> 作为分隔符
    eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eos_token_id == tokenizer.unk_token_id:
        # 如果 convert_tokens_to_ids 失败，尝试编码后提取
        eos_token_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
    print(f"Llama3 使用 <|eot_id|> 作为分隔符，token ID: {eos_token_id}")
else:
    print(f"使用默认 eos_token_id: {eos_token_id}")

# ========== LCS 计算函数 ==========
def compute_lcs_tokens(buggy_tokens: List[int], fixed_tokens: List[int]) -> set:
    """
    计算 buggy code 和 fixed code 的 LCS（最长公共子序列）
    
    Args:
        buggy_tokens: buggy code 的 token IDs
        fixed_tokens: fixed code 的 token IDs
    
    Returns:
        set: fixed code 中属于 LCS 的 token 位置集合（0-based）
    """
    if not buggy_tokens or not fixed_tokens:
        return set()
    
    # 使用 SequenceMatcher 计算 LCS
    # SequenceMatcher 基于 Ratcliff/Obershelp 算法，效率高且准确
    matcher = SequenceMatcher(None, buggy_tokens, fixed_tokens)
    
    # 获取所有匹配块（LCS 片段）
    lcs_positions = set()
    for match in matcher.get_matching_blocks():
        i, j, n = match  # i: buggy 起始, j: fixed 起始, n: 匹配长度
        if n > 0:  # 跳过 sentinel 块
            # 记录 fixed code 中的 LCS 位置
            for pos in range(j, j + n):
                lcs_positions.add(pos)
    
    return lcs_positions

# tokenize dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding=False, truncation=False)
    
    # 提取 buggy code 的 token（用于 LCS 计算）
    buggy_code_token_ids = []
    
    for text in examples["text"]:
        # 提取 buggy code（使用全局配置的标记）
        buggy_start = text.find(BUGGY_START_MARKER)
        if buggy_start != -1:
            code_start = buggy_start + len(BUGGY_START_MARKER)
            lang_end = text.find('\n', code_start)
            if lang_end != -1:
                code_start = lang_end + 1
            
            buggy_end = text.find(BUGGY_END_MARKER, code_start)
            if buggy_end != -1:
                buggy_code = text[code_start:buggy_end]
                buggy_tokens = tokenizer(buggy_code, add_special_tokens=False)['input_ids']
                buggy_code_token_ids.append(buggy_tokens)
            else:
                buggy_code_token_ids.append([])
        else:
            buggy_code_token_ids.append([])
    
    tokenized['buggy_code_token_ids'] = buggy_code_token_ids
    return tokenized

print(f"Dataset size: {len(full_dataset)}")
full_dataset = full_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 增强的 data collator，计算 LCS 并生成权重
class LCSWeightedDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, include_prompt_in_loss=True, lcs_weight=2.0):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.include_prompt_in_loss = include_prompt_in_loss
        self.lcs_weight = lcs_weight
        self._system_prompt_detected = False  # 用于首次检测时输出提示
    
    def __call__(self, features, return_tensors=None):
        processed_features = []
        prompt_lengths = []
        loss_weights_list = []  # 存储每个样本的 loss 权重
        lcs_coverage_list = []  # 存储 LCS 覆盖率（用于统计）
        
        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            buggy_tokens = feature.get('buggy_code_token_ids', [])
            
            split_indices = [i for i, x in enumerate(input_ids) if x == eos_token_id]

            if len(split_indices) < 2:
                print('data illegal, not enough split points!')
                sys.exit(0)
            
            # 自动处理 system prompt：
            # - 如果有 >=3 个分割点，假设有 system prompt
            #   格式：system<|im_end|> user<|im_end|> assistant<|im_end|>
            #   prompt = system + user, response = assistant
            # - 如果只有 2 个分割点，没有 system prompt
            #   格式：user<|im_end|> assistant<|im_end|>
            #   prompt = user, response = assistant
            if len(split_indices) >= 3:
                # 有 system prompt：使用倒数第二个和最后一个分割点
                if not self._system_prompt_detected:
                    print(f"检测到 system prompt 格式（{len(split_indices)} 个分割点），将使用 system+user 作为 prompt")
                    self._system_prompt_detected = True
                prompt_ids = input_ids[:split_indices[-2]]  # system + user
                response_ids = input_ids[split_indices[-2]+1:split_indices[-1]]  # assistant
                attention_mask_prompt = attention_mask[:split_indices[-2]]
                attention_mask_response = attention_mask[split_indices[-2]+1:split_indices[-1]]
            else:
                # 无 system prompt：使用第一个和第二个分割点
                if not self._system_prompt_detected:
                    print(f"检测到标准格式（{len(split_indices)} 个分割点），将使用 user 作为 prompt")
                    self._system_prompt_detected = True
                prompt_ids = input_ids[:split_indices[0]]  # user
                response_ids = input_ids[split_indices[0]+1:split_indices[1]]  # assistant
                attention_mask_prompt = attention_mask[:split_indices[0]]
                attention_mask_response = attention_mask[split_indices[0]+1:split_indices[1]]
            
            # 数据质量检查
            if len(prompt_ids) >= max_len - 1:
                print(f'Warning: prompt length ({len(prompt_ids)}) >= max_len-1 ({max_len-1}), response will be truncated!')
            
            combined_input_ids = (prompt_ids + response_ids)[:max_len-1] + [eos_token_id]
            combined_attention_mask = (attention_mask_prompt + attention_mask_response)[:max_len-1] + [1]
            
            processed_feature = {
                'input_ids': combined_input_ids,
                'attention_mask': combined_attention_mask
            }
            
            actual_prompt_len = min(len(prompt_ids), len(combined_input_ids) - 1)
            prompt_lengths.append(actual_prompt_len)
            
            # ========== 计算 LCS 并生成权重 ==========
            loss_weights = [1.0] * len(combined_input_ids)  # 默认权重 1.0
            
            if buggy_tokens:
                # 关键修复：使用 response_ids 作为 fixed code 的 tokens
                # 因为 response_ids 已经是从完整文本 tokenize 后得到的，与 input_ids 一致
                response_len_in_combined = min(len(response_ids), len(combined_input_ids) - actual_prompt_len)
                actual_response_tokens = response_ids[:response_len_in_combined]
                
                # 计算 buggy_tokens 和 actual_response_tokens 的 LCS
                lcs_positions = compute_lcs_tokens(buggy_tokens, actual_response_tokens)
                
                # 计算 LCS 覆盖率（用于监控）
                if len(actual_response_tokens) > 0:
                    lcs_coverage = len(lcs_positions) / len(actual_response_tokens)
                else:
                    lcs_coverage = 0.0
                lcs_coverage_list.append(lcs_coverage)
                
                # 将 LCS 位置映射到 combined_input_ids 中的 response 部分
                # response 在 combined_input_ids 中的起始位置是 actual_prompt_len
                for i in range(response_len_in_combined):
                    # i 是 response_ids 中的位置
                    # actual_prompt_len + i 是 combined_input_ids 中的位置
                    if i in lcs_positions:
                        loss_weights[actual_prompt_len + i] = self.lcs_weight
            else:
                lcs_coverage_list.append(0.0)
            
            loss_weights_list.append(loss_weights)
            processed_features.append(processed_feature)
        
        batch = super().__call__(processed_features, return_tensors)
        
        if not self.include_prompt_in_loss:
            for i, prompt_len in enumerate(prompt_lengths):
                batch['labels'][i, :prompt_len] = -100
        
        # 将权重转换为 tensor 并附加到 batch
        # 需要与 labels 对齐（shift 后的长度）
        max_len_in_batch = batch['labels'].shape[1]
        weight_tensor = torch.ones_like(batch['labels'], dtype=torch.float32)
        
        for i, weights in enumerate(loss_weights_list):
            # weights 对应 input_ids，labels 是 shift 后的（labels[t] = input_ids[t+1]）
            # 所以 weight_tensor[i, t] 对应 input_ids[i, t+1] 的权重
            for t in range(min(len(weights) - 1, max_len_in_batch)):
                weight_tensor[i, t] = weights[t + 1]  # shift 对齐
        
        batch['prompt_lengths'] = torch.tensor(prompt_lengths, dtype=torch.long)
        batch['loss_weights'] = weight_tensor
        batch['lcs_coverage'] = torch.tensor(lcs_coverage_list, dtype=torch.float32)  # 用于统计
        
        return batch

# QLoRA parameters selection function
def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

# QLoRA config
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=find_all_linear_names(base_model, int4=True),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
base_model = get_peft_model(base_model, peft_config)

# trainer config
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    prediction_loss_only=False,
    max_grad_norm=0.3,
    num_train_epochs=EPOCH_NUM,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=100,
    logging_steps=100,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="constant",
    warmup_ratio=0.05,
    remove_unused_columns=False,
    neftune_noise_alpha=5,
    report_to="wandb",
    logging_first_step=True,
    logging_strategy="steps",
)

# data collator
data_collator = LCSWeightedDataCollator(
    tokenizer=tokenizer, 
    mlm=False, 
    include_prompt_in_loss=include_prompt_in_loss,
    lcs_weight=LCS_WEIGHT
)

# ========== 自定义 Trainer 实现 LCS 加权损失 ==========
class LCSWeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算 LCS 加权损失
        """
        # 提取额外信息
        loss_weights = inputs.pop("loss_weights")  # [batch_size, seq_len]
        lcs_coverage = inputs.pop("lcs_coverage")  # list of floats
        prompt_lengths = inputs.pop("prompt_lengths")
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 计算加权交叉熵损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()  # 权重也需要 shift
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 应用权重
        weights = shift_weights.view(-1).to(loss_per_token.device)
        mask = (shift_labels.view(-1) != -100).float()
        
        weighted_loss = loss_per_token * weights * mask
        total_loss = weighted_loss.sum() / (mask.sum() + 1e-8)
        
        # 记录统计信息到 wandb
        if self.state.global_step % self.args.logging_steps == 0:
            if isinstance(lcs_coverage, torch.Tensor):
                avg_lcs_coverage = lcs_coverage.float().mean().item()
            else:
                avg_lcs_coverage = float(sum(lcs_coverage) / (len(lcs_coverage) + 1e-8)) if lcs_coverage else 0.0
            
            # 计算 LCS token 和非 LCS token 的平均损失（用于分析）
            lcs_mask = (weights > 1.0) & (mask > 0)
            non_lcs_mask = (weights <= 1.0) & (mask > 0)
            
            lcs_loss = (loss_per_token * lcs_mask.float()).sum() / (lcs_mask.sum() + 1e-8)
            non_lcs_loss = (loss_per_token * non_lcs_mask.float()).sum() / (non_lcs_mask.sum() + 1e-8)
            
            wandb.log({
                "loss": total_loss.item(),
                "lcs_coverage": avg_lcs_coverage,
                "lcs_token_loss": lcs_loss.item() if lcs_mask.sum() > 0 else 0.0,
                "non_lcs_token_loss": non_lcs_loss.item() if non_lcs_mask.sum() > 0 else 0.0,
                "lcs_token_ratio": lcs_mask.sum().item() / (mask.sum().item() + 1e-8),
            }, step=self.state.global_step)
        
        return (total_loss, outputs) if return_outputs else total_loss

# 使用自定义 Trainer
trainer = LCSWeightedTrainer(
    model=base_model,
    train_dataset=full_dataset,
    data_collator=data_collator,
    args=training_args,
)

num_gpus = torch.cuda.device_count()
print(f"Training dataset size: {len(trainer.train_dataset)}")
print(f"Number of GPUs: {num_gpus}")
print(f"Epochs: {training_args.num_train_epochs}, Batch size per device: {training_args.per_device_train_batch_size}")
if num_gpus > 1:
    steps_per_epoch = len(trainer.train_dataset) // (training_args.per_device_train_batch_size * num_gpus)
    print(f"Steps per epoch per device: ~{steps_per_epoch}, Total steps: {steps_per_epoch * training_args.num_train_epochs}")
else:
    steps_per_epoch = len(trainer.train_dataset) // training_args.per_device_train_batch_size
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {steps_per_epoch * training_args.num_train_epochs}")

# 检查训练是否已完成
training_completed = False
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

if os.path.exists(final_checkpoint_dir):
    adapter_model_path = os.path.join(final_checkpoint_dir, "adapter_model.safetensors")
    adapter_config_path = os.path.join(final_checkpoint_dir, "adapter_config.json")
    
    if os.path.exists(adapter_model_path) and os.path.exists(adapter_config_path):
        print(f"发现已完成的训练 checkpoint: {final_checkpoint_dir}")
        training_completed = True
    else:
        print(f"final_checkpoint 存在但不完整，将重新训练")

if not training_completed:
    if resume_from_checkpoint:
        print(f"从 checkpoint 恢复训练: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("从头开始训练")
        trainer.train()
    trainer.save_model(output_dir)
    
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    trainer.model.save_pretrained(final_checkpoint_dir)
    print('训练完成，已保存 final checkpoint')
else:
    print("训练已完成，跳过训练步骤，直接进行模型合并")

# 准备合并目录
codellama_merged_dir = os.path.join(output_dir, 'codellama_merged')
os.makedirs(codellama_merged_dir, exist_ok=True)

print('training process finished ...')

# merge model
del trainer
del base_model
del data_collator
del full_dataset
import gc
torch.cuda.empty_cache()
gc.collect()
gc.collect()

base_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

print('load model success ...')

if training_completed:
    peft_model_path = final_checkpoint_dir
else:
    peft_model_path = output_dir

model = PeftModel.from_pretrained(base_model, peft_model_path)
model = model.merge_and_unload()
print('merge model success ...')
model.save_pretrained(codellama_merged_dir, safe_serialization=True)

print('merge model saved success ...')

tokenizer = AutoTokenizer.from_pretrained(original_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(codellama_merged_dir)

wandb.finish()

