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

model_name = sys.argv[1] # such as "CodeLlama-7b-Instruct-hf" the model name you want to train.
original_model_name = model_name  # 保存原始模型名用于后续 tokenizer 加载

# 处理本地缓存路径：如果模型名包含 '/' 但不是绝对路径，尝试从 HF cache 加载
if model_name == "codellama-7b" or model_name == "CodeLlama-7b-Instruct-hf":
    # 尝试从本地 cache 加载，如果不存在则使用原名称
    local_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots")
    if os.path.exists(local_cache_path):
        # 获取最新的 snapshot
        snapshots = os.listdir(local_cache_path)
        if snapshots:
            model_name = os.path.join(local_cache_path, snapshots[0])
            print(f"Loading model from local cache: {model_name}")
        else:
            model_name = "codellama/CodeLlama-7b-Instruct-hf"
    else:
        model_name = "codellama/CodeLlama-7b-Instruct-hf"
    original_model_name = "codellama/CodeLlama-7b-Instruct-hf"

full_dataset = load_dataset("json", data_files=sys.argv[2], split="train") # replace with your dataset location

output_dir = 'models/' + sys.argv[3] # fine-tuned model location

# 初始化 wandb
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
    }
)

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

print(f"Dataset size before tokenization: {len(full_dataset)}")
full_dataset = full_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Dataset size after tokenization: {len(full_dataset)}")

# define single-task data collator (只保留预测任务)
class SingleTaskDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        processed_features = []
        
        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            print(len(input_ids))

            # find the split points, which is the eos_token_id, in input_ids
            split_indices = [i for i, x in enumerate(input_ids) if x == eos_token_id]

            # assure at least 1 split point
            if len(split_indices) < 2:
                print('data illegal, not enough split points!')
                sys.exit(0)
            
            # 只提取预测任务的数据（第一个分割点到第二个分割点之间）
            processed_features.append({
                'input_ids': (input_ids[:split_indices[0]] + input_ids[split_indices[0]+1:split_indices[1]])[:max_len-1] + [2],
                'attention_mask': (attention_mask[:split_indices[0]] + attention_mask[split_indices[0]+1:split_indices[1]])[:max_len-1] + [1]
            })
        
        # use the base class's __call__ method to process the features
        return super().__call__(processed_features, return_tensors)

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
data_collator = SingleTaskDataCollator(tokenizer=tokenizer, mlm=False)

# 使用标准的 Trainer，自动计算 loss
trainer = Trainer(
    model=base_model,
    train_dataset=full_dataset,
    data_collator=data_collator,
    args=training_args,
)

print(f"Train dataset size in trainer: {len(trainer.train_dataset)}")
if trainer.eval_dataset is not None:
    print(f"Eval dataset size in trainer: {len(trainer.eval_dataset)}")
else:
    print("No eval dataset")

print(f"Max steps: {training_args.max_steps}")
print(f"Num train epochs: {training_args.num_train_epochs}")
print(f"Per device train batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"Expected steps per epoch: {len(trainer.train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

# start training
trainer.train()
trainer.save_model(output_dir)

# save final checkpoint
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
codellama_merged_dir = os.path.join(output_dir, 'codellama_merged')

os.makedirs(final_checkpoint_dir, exist_ok=True)
os.makedirs(codellama_merged_dir, exist_ok=True)

trainer.model.save_pretrained(final_checkpoint_dir)

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

model = PeftModel.from_pretrained(base_model, output_dir)
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

