#!/usr/bin/env python3
"""测试 Seed-Coder 模型是否能正常工作"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'ByteDance-Seed/Seed-Coder-8B-Instruct'

print("=" * 60)
print("加载模型...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"\nBOS token: {repr(tokenizer.bos_token)}")
print(f"EOS token: {repr(tokenizer.eos_token)}")

# 测试1: 简单的Python代码生成
print("\n" + "=" * 60)
print("测试 1: 简单的 Python 函数")
print("=" * 60)

prompt1 = f"""{tokenizer.bos_token}system
You are an AI programming assistant.

{tokenizer.eos_token}{tokenizer.bos_token}user
Write a Python function to calculate fibonacci numbers.
{tokenizer.eos_token}{tokenizer.bos_token}assistant
"""

inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
# 移除 token_type_ids（模型不需要）
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
response1 = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n模型输出:\n{response1}")

# 测试2: 简单的Java bug修复
print("\n" + "=" * 60)
print("测试 2: 简单的 Java Bug 修复")
print("=" * 60)

prompt2 = f"""{tokenizer.bos_token}system
You are an AI programming assistant.

{tokenizer.eos_token}{tokenizer.bos_token}user
Fix this Java code:
```java
public int add(int a, int b) {{
    return a - b;  // Bug: should be a + b
}}
```
{tokenizer.eos_token}{tokenizer.bos_token}assistant
"""

inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
# 移除 token_type_ids（模型不需要）
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
response2 = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n模型输出:\n{response2}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

