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

max_len = 2048 # replace with the max input length of your model, recommend no less than 2k

model_name = sys.argv[1] # 模型名称或路径，如 "codellama/CodeLlama-7b-Instruct-hf" 或 "Qwen/Qwen2.5-Coder-7B-Instruct"
original_model_name = model_name  # 保存原始模型名用于后续 tokenizer 加载

# 配置项：是否将 prompt 部分算入 loss（默认为 True，即算入 loss）
# 可以通过环境变量 INCLUDE_PROMPT_IN_LOSS 控制，或者添加命令行参数
include_prompt_in_loss = False
print(f"配置：是否将 prompt 算入 loss: {include_prompt_in_loss}")

# 智能加载模型：支持本地路径、HF 缓存、完整 HF 路径
if not os.path.exists(model_name):
    # 如果不是本地路径，尝试从 HF cache 加载
    # 将 HF 格式 (org/model-name) 转换为缓存目录格式 (models--org--model-name)
    cache_model_name = "models--" + model_name.replace("/", "--")
    local_cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/{cache_model_name}/snapshots")
    
    if os.path.exists(local_cache_path):
        # 获取最新的 snapshot
        snapshots = [d for d in os.listdir(local_cache_path) if os.path.isdir(os.path.join(local_cache_path, d))]
        if snapshots:
            # 使用最新的 snapshot（按修改时间排序）
            latest_snapshot = max(snapshots, key=lambda x: os.path.getmtime(os.path.join(local_cache_path, x)))
            model_name = os.path.join(local_cache_path, latest_snapshot)
            print(f"Loading model from local cache: {model_name}")
        else:
            print(f"Using HuggingFace model: {model_name}")
    else:
        print(f"Using HuggingFace model: {model_name} (will download if needed)")
else:
    print(f"Loading model from local path: {model_name}")

full_dataset = load_dataset("json", data_files=sys.argv[2], split="train") # replace with your dataset location

output_dir = 'models/' + sys.argv[3] # fine-tuned model location

# 检查是否存在 checkpoint 用于断点续训
resume_from_checkpoint = None
wandb_run_id = None

# 支持通过环境变量手动指定 wandb run ID
manual_wandb_id = os.environ.get('WANDB_RUN_ID', None)
if manual_wandb_id:
    wandb_run_id = manual_wandb_id
    print(f"使用手动指定的 wandb run ID: {wandb_run_id}")

if os.path.exists(output_dir):
    # 查找所有 checkpoint 目录
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # 按步数排序，选择最新的 checkpoint
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
        if os.path.exists(latest_checkpoint):
            resume_from_checkpoint = latest_checkpoint
            print(f"发现 checkpoint，将从 {resume_from_checkpoint} 恢复训练")
            
            # 如果没有手动指定，尝试从 checkpoint 中读取 wandb run ID
            if not wandb_run_id:
                # 1. 尝试从 trainer_state.json 读取
                trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
                if os.path.exists(trainer_state_file):
                    import json
                    try:
                        with open(trainer_state_file, 'r') as f:
                            trainer_state = json.load(f)
                            # wandb 会在 trainer_state 中保存 run_id
                            if 'log_history' in trainer_state and len(trainer_state['log_history']) > 0:
                                # 尝试从 log_history 中找到 wandb run_id
                                for log_entry in trainer_state['log_history']:
                                    if '_wandb' in log_entry or 'wandb_run_id' in log_entry:
                                        wandb_run_id = log_entry.get('wandb_run_id', None)
                                        if wandb_run_id:
                                            print(f"从 checkpoint 的 trainer_state.json 找到 wandb run ID: {wandb_run_id}")
                                            break
                    except Exception as e:
                        print(f"读取 trainer_state.json 时出错: {e}")
                
                # 2. 尝试从 wandb_run_id.txt 读取
                if not wandb_run_id:
                    wandb_id_file = os.path.join(output_dir, "wandb_run_id.txt")
                    if os.path.exists(wandb_id_file):
                        with open(wandb_id_file, 'r') as f:
                            wandb_run_id = f.read().strip()
                        print(f"从 wandb_run_id.txt 找到 wandb run ID: {wandb_run_id}")
                
                # 3. 尝试从 wandb 目录读取
                if not wandb_run_id:
                    wandb_dir = os.path.join(output_dir, "wandb")
                    if os.path.exists(wandb_dir):
                        # 查找最新的 wandb run 目录
                        run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
                        if run_dirs:
                            run_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x)), reverse=True)
                            latest_run_dir = run_dirs[0]
                            # 从目录名提取 run_id（格式: run-20231201_123456-abc123xyz）
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

# 初始化 wandb
if wandb_run_id:
    # 恢复已有的 wandb run
    wandb.init(
        project="morepair-single-training",
        name=sys.argv[3],
        id=wandb_run_id,
        resume="must",  # 必须恢复到指定的 run
        config={
            "model_name": model_name,
            "max_len": max_len,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 1,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "include_prompt_in_loss": include_prompt_in_loss,
        }
    )
else:
    # 创建新的 wandb run
    wandb.init(
        project="morepair-single-training",
        name=sys.argv[3],
        config={
            "model_name": model_name,
            "max_len": max_len,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 1,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "include_prompt_in_loss": include_prompt_in_loss,
        }
    )
    # 保存新的 wandb run ID
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
    quantization_config=bnb_config
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=False)

print(f"Dataset size: {len(full_dataset)}")
full_dataset = full_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# define single-task data collator (只保留预测任务)
class SingleTaskDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, include_prompt_in_loss=True):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.include_prompt_in_loss = include_prompt_in_loss
    
    def __call__(self, features, return_tensors=None):
        processed_features = []
        prompt_lengths = []  # 记录每个样本的 prompt 长度
        
        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            
            # find the split points, which is the eos_token_id, in input_ids
            split_indices = [i for i, x in enumerate(input_ids) if x == eos_token_id]

            # assure at least 1 split point
            if len(split_indices) < 2:
                print('data illegal, not enough split points!')
                sys.exit(0)
            
            # 只提取预测任务的数据（第一个分割点到第二个分割点之间）
            # prompt 部分：input_ids[:split_indices[0]]
            # response 部分：input_ids[split_indices[0]+1:split_indices[1]]
            prompt_ids = input_ids[:split_indices[0]]
            response_ids = input_ids[split_indices[0]+1:split_indices[1]]
            
            combined_input_ids = (prompt_ids + response_ids)[:max_len-1] + [2]
            combined_attention_mask = (attention_mask[:split_indices[0]] + attention_mask[split_indices[0]+1:split_indices[1]])[:max_len-1] + [1]
            
            processed_feature = {
                'input_ids': combined_input_ids,
                'attention_mask': combined_attention_mask
            }
            
            # 记录 prompt 长度用于后续处理
            actual_prompt_len = min(len(prompt_ids), len(combined_input_ids) - 1)
            prompt_lengths.append(actual_prompt_len)
            
            processed_features.append(processed_feature)
        
        # 先让父类处理 padding 和创建 labels
        batch = super().__call__(processed_features, return_tensors)
        
        # 如果不将 prompt 算入 loss，修改已经 padded 的 labels
        if not self.include_prompt_in_loss:
            # batch['labels'] 现在是一个 tensor，形状为 [batch_size, seq_len]
            for i, prompt_len in enumerate(prompt_lengths):
                # 将每个样本的 prompt 部分设置为 -100
                batch['labels'][i, :prompt_len] = -100
        
        return batch

# QLoRA parameters selection function
def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
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
            # last layer is not add to lora_module_names
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
# get peft model
base_model = get_peft_model(base_model, peft_config)

# trainer config
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing =True,
    prediction_loss_only=False,
    max_grad_norm= 0.3,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    save_strategy="steps",  # 定期保存 checkpoint
    save_steps=100,  # 每 500 步保存一次
    logging_steps=100,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="constant",
    warmup_ratio=0.05,
    remove_unused_columns = False,
    neftune_noise_alpha=5,
    # wandb 配置
    report_to="wandb",
    logging_first_step=True,
    logging_strategy="steps",
)

# data collator
data_collator = SingleTaskDataCollator(
    tokenizer=tokenizer, 
    mlm=False, 
    include_prompt_in_loss=include_prompt_in_loss
)

# 使用标准的 Trainer，自动计算 loss
trainer = Trainer(
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
    # 检查 final_checkpoint 是否包含必要的文件
    adapter_model_path = os.path.join(final_checkpoint_dir, "adapter_model.safetensors")
    adapter_config_path = os.path.join(final_checkpoint_dir, "adapter_config.json")
    
    if os.path.exists(adapter_model_path) and os.path.exists(adapter_config_path):
        print(f"发现已完成的训练 checkpoint: {final_checkpoint_dir}")
        training_completed = True
    else:
        print(f"final_checkpoint 存在但不完整，将重新训练")

if not training_completed:
    # start training (从 checkpoint 恢复或从头开始)
    if resume_from_checkpoint:
        print(f"从 checkpoint 恢复训练: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("从头开始训练")
        trainer.train()
    trainer.save_model(output_dir)
    
    # save final checkpoint
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

# 根据训练状态选择加载路径
if training_completed:
    # 如果训练已完成，从 final_checkpoint 加载
    peft_model_path = final_checkpoint_dir
else:
    # 如果刚完成训练，从 output_dir 加载
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

# 结束 wandb 记录
wandb.finish()

