#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import time
import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from typing import List, Dict

# 设置日志
logging.set_verbosity_info()


def main():
    # 定义固定参数
    model_path = "/home/liu01/projects/Programrepair/model"
    dataset_path = "/home/liu01/projects/Programrepair/data/trainset/llama_brief_llm.json"  # 你需要修改为你的数据集路径
    output_dir = "/home/liu01/projects/Programrepair/sft_model"
    max_seq_length = 2048  # 增加序列长度以适应较长的编程问题
    batch_size = 1  # 增大批量大小，加速训练
    gradient_accumulation_steps = 1  # 有足够显存，不需要梯度累积
    num_train_epochs = 3
    learning_rate = 3e-4  # 略微提高学习率
    lora_r = 16  # 增加LoRA秩，提高模型容量
    lora_alpha = 32  # 增加alpha值，与r成比例
    lora_dropout = 0.05

    # 显示基本训练信息
    print("=" * 50)
    print(f"Llama 3-8B LoRA微调 | 模型: {model_path} | 数据集: {dataset_path}")
    print(f"参数: 轮数={num_train_epochs}, 学习率={learning_rate}, 批量={batch_size}, LoRA秩={lora_r}")
    print("=" * 50)

    start_time = time.time()

    # 加载基础模型 (不使用量化)
    print(f"正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用bfloat16精度，更稳定
        device_map="auto",
        trust_remote_code=True,
    )
    model.tie_weights()
    model.config.use_cache = False

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 确保tokenizer知道正确的EOS token
    if tokenizer.pad_token is None or tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 配置LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            
            "v_proj",

        ],
    )

    # 加载数据集
    print(f"正在加载数据集...")
    try:
        dataset = load_dataset("json",data_files=dataset_path, keep_in_memory=True)

        # 使用第一个可用的split作为训练集
        if "train" not in dataset:
            first_key = list(dataset.keys())[0]
            dataset = dataset.rename_split(first_key, "train")

        # 处理数据集
        def preprocess_function(examples):
            inputs = []
            targets = []
            for example in examples['text']:
                # 通过分割[INST]和[/INST]，获取每个对话的输入和输出
                parts = example.split("[INST]")
                input_text = ""
                target_text = ""
                for part in parts:
                    if "[|INST]" in part:
                        # 获取到用户和模型的对话
                        conversation = part.split("[|INST]")[-1].strip()
                        if conversation:  # 确保有内容
                            input_text = conversation.split(" ")[0]  # 假设第一个部分是错误代码
                            target_text = " ".join(conversation.split(" ")[1:]) + "</s>"  # 假设剩下的是修复后的内容
                inputs.append(input_text)
                targets.append(target_text)
            return {"input_text": inputs, "output_text": targets}

        processed_dataset = dataset.map(preprocess_function)

        # 打印数据集信息
        print(f"数据集大小: {len(processed_dataset['train'])} 个样本")
        if len(processed_dataset["train"]) > 0:
            sample = processed_dataset["train"][0]
            print(f"样例格式: {list(sample.keys())}")

    except Exception as e:
        print(f"加载数据集出错: {e}")
        return

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,  # 关闭梯度检查点，有足够显存时可提高速度
        optim="adamw_torch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        bf16=True,  # 使用bfloat16混合精度训练
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        group_by_length=True,
        report_to="tensorboard",
        save_safetensors=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataloader_num_workers=4,  # 增加数据加载线程数
        ddp_find_unused_parameters=False,  # 提高多GPU训练效率
    )

    # 设置训练器
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=processed_dataset["train"],
        peft_config=peft_config,
    )

    # 输出训练摘要
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"模型总参数: {total_params:,} | 可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    # 计算训练步数
    batch_size_effective = batch_size * gradient_accumulation_steps
    steps_per_epoch = len(processed_dataset["train"]) // batch_size_effective + (
        1 if len(processed_dataset["train"]) % batch_size_effective != 0 else 0)
    total_steps = steps_per_epoch * num_train_epochs
    print(f"每轮训练步数: {steps_per_epoch} | 总训练步数: {total_steps}")
    print(f"使用 tensorboard --logdir={output_dir}/logs 查看训练进度")
    print("=" * 50)

    # 开始训练
    print("开始训练...")
    train_start = time.time()
    trainer.train()
    train_duration = time.time() - train_start

    # 保存模型
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 训练完成信息
    print("=" * 50)
    print(f"训练完成! 模型已保存到: {output_dir}")
    print(f"训练耗时: {str(datetime.timedelta(seconds=int(train_duration)))}")


if __name__ == "__main__":
    main()
