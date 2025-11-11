# XCodeEval 自动程序修复 (APR) 工具

这是一个基于大语言模型的自动程序修复工具，支持多种开源本地模型，使用 vLLM 进行高效推理，能够自动修复代码中的bug。

## 文件结构

```
xcodeeval/
├── gen_apr.py          # 主要生成脚本 (支持 vLLM 本地模型)
├── eval_apr.py         # 评估脚本
├── get_result.py       # 结果统计和对比脚本
├── xcodeeval_merged.json  # 数据集文件
├── result/             # 生成结果目录
│   └── {model_name}/   # 按模型名分类的结果
│       ├── 0_1.0_C++.json
│       ├── 1_1.0_C++.json
│       ├── ...
│       └── eval_apr_val_execeval/  # 评估结果
│           └── GNU C++17.jsonl
├── cache/              # 统计结果缓存
│   ├── {model_name}_stats.json
│   └── ...
└── README.md           # 本说明文件
```

## 支持的模型

### 本地模型 (gen_apr.py)
- **Qwen 系列**: qwen3-8b, qwen3-4b, qwen3-8b-trained-noprompt, qwen3-4b-trained-parepair
- **LLaMA 系列**: llama3.1-8b, llama3.1-8b-nopro, llama3.1-8b-trained-parepair  
- **DeepSeek 系列**: deepseek-6.7b, deepseek-6.7b-nopro, deepseek-6.7b-parepair
- **StarCoder 系列**: starcoder-7b, starcoder-7b-nopro, starcoder-7b-par

## 安装依赖

```bash
# 安装基础依赖
pip install torch transformers datasets tqdm psutil

# 安装 vLLM (用于本地模型)
pip install vllm

# 安装其他依赖
pip install promptsource jsonlines requests
```

## 完整工作流程

### 1. 生成修复代码 (gen_apr.py)

#### 单进程模式
```bash
# 基本用法
python gen_apr.py --model-name qwen3-8b

# 完整参数
python gen_apr.py \
    --model-name qwen3-8b \
    --nsample 10 \
    --output-dir xcodeeval/result/qwen3-8b \
    --dataset-path xcodeeval/xcodeeval_merged.json \
    --dry-run 0
```

#### 多进程模式 (推荐用于大规模生成)
```bash
# 启动4个进程，分别使用GPU 0-3
python gen_apr.py qwen3-8b 4 0 0  # 进程0，GPU 0
python gen_apr.py qwen3-8b 4 1 1  # 进程1，GPU 1
python gen_apr.py qwen3-8b 4 2 2  # 进程2，GPU 2
python gen_apr.py qwen3-8b 4 3 3  # 进程3，GPU 3
```

#### 生成结果格式
每个样本生成一个JSON文件：
```json
{
  "model_response": {
    "choices": [
      {
        "message": {
          "content": "修复后的代码1"
        }
      },
      {
        "message": {
          "content": "修复后的代码2"
        }
      }
      // ... 共 nsample 个候选
    ],
    "prompt": "完整的输入提示词"
  },
  "source_data": {
    "src_uid": "样本ID",
    "lang_cluster": "C++",
    "bug_source_code": "原始有bug的代码",
    "prob_desc_description": "问题描述",
    // ... 其他元数据
  }
}
```

### 2. 评估修复效果 (eval_apr.py)

```bash
# 评估生成的修复代码
python eval_apr.py \
    --model-name qwen3-8b \
    --input-path xcodeeval/result
```

#### 评估结果格式
生成 `eval_apr_val_execeval/GNU C++17.jsonl` 文件：
```json
{
  "source_data": {
    "src_uid": "样本ID",
    "lang_cluster": "C++",
    "bug_source_code": "原始有bug的代码",
    "hidden_unit_tests": "[{\"input\":\"1 2\",\"output\":\"3\"}]"
  },
  "unit_test_results": [
    [
      {
        "input": "1 2",
        "output": ["3"],
        "result": "3",
        "exec_outcome": "PASSED"
      }
    ],
    [
      {
        "input": "1 2", 
        "output": ["3"],
        "result": "4",
        "exec_outcome": "WRONG_ANSWER"
      }
    ]
    // ... 对应每个候选修复的测试结果
  ]
}
```

#### 执行结果类型
- **PASSED**: 代码执行成功且输出正确
- **WRONG_ANSWER**: 代码执行成功但输出错误
- **TIME_LIMIT_EXCEEDED**: 执行超时
- **RUNTIME_ERROR**: 运行时错误
- **COMPILATION_ERROR**: 编译错误
- **MEMORY_LIMIT_EXCEEDED**: 内存超限

### 3. 统计分析结果 (get_result.py)

#### 单模型分析
```bash
# 分析单个模型（自动缓存）
python get_result.py --model qwen3-8b

# 强制重新计算
python get_result.py --model qwen3-8b --force-recalc
```

#### 模型对比
```bash
# 对比两个模型
python get_result.py --model qwen3-8b --compare llama3.1-8b

# 强制重新计算两个模型进行对比
python get_result.py --model qwen3-8b --compare llama3.1-8b --force-recalc
```

#### 输出指标

**Pass@k 成功率：**
- **Pass@1**: 在1个候选中至少有1个正确的概率
- **Pass@5**: 在5个候选中至少有1个正确的概率  
- **Pass@10**: 在10个候选中至少有1个正确的概率

**代码修改统计（仅针对通过测试的补丁）：**
- **Hunks per patch**: 每个补丁的diff块数量
- **Lines added/deleted per patch**: 每个补丁的增加/删除行数
- **Total changed lines per patch**: 每个补丁的总变化行数
- **Tokens added/deleted/changed per patch**: Token级别的变化统计
- **Edit distance per patch**: Levenshtein编辑距离
- **Edit similarity (%)**: 编辑相似度百分比
- **Normalized edit distance**: 归一化编辑距离
- **Code preserved ratio (%)**: 代码保留比例

**代码保留比例分布：**
- **Minimal change (>95% preserved)**: 最小修改
- **Moderate change (80-95% preserved)**: 中等修改  
- **Major change (<80% preserved)**: 大幅修改

#### 缓存结果格式
统计结果保存在 `xcodeeval/cache/{model_name}_stats.json`：
```json
{
  "pass_at_k": {
    "C++": {
      "pass@1": 0.4532,
      "pass@5": 0.7845,
      "pass@10": 0.8917
    }
  },
  "diff_stats": {
    "patch_count": 156,
    "total_hunks": 192,
    "total_added": 382,
    "total_deleted": 291,
    "total_changed_lines": 673,
    "total_added_tokens": 1926,
    "total_deleted_tokens": 1538,
    "total_changed_tokens": 3464,
    "total_edit_distance": 7124,
    "total_edit_similarity": 13608.36,
    "total_norm_edit_distance": 19.9224,
    "total_preserved": 14266.2,
    "preservation_distribution": {
      "high": 89,
      "medium": 52,
      "low": 15
    }
  },
  "timestamp": 1699718142.345,
  "model_name": "qwen3-8b"
}
```

## 参数说明

### gen_apr.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | qwen3-8b | 模型名称 |
| `--nsample` | 10 | 每个样本生成的候选数量 |
| `--output-dir` | xcodeeval/result/{model_name} | 输出目录 |
| `--dataset-path` | xcodeeval/xcodeeval_merged.json | 数据集路径 |
| `--dry-run` | 0 | 干运行模式 (0/1) |

### eval_apr.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | qwen3-8b | 模型名称 |
| `--input-path` | xcodeeval/result | 输入路径 |

### get_result.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | 要分析的模型名称 |
| `--compare` | None | 用于对比的第二个模型名称 |
| `--force-recalc` | False | 强制重新计算，忽略缓存 |

## 输出示例

### 单模型分析输出
```
qwen3-8b Results:
================================================================================
Pass@1: 45.32%
Pass@5: 78.45%
Pass@10: 89.17%
================================================================================

[PATCH MODIFICATION STATISTICS - PLAUSIBLE Patches]
================================================================================
Total PLAUSIBLE patches analyzed: 156
Metric                                   |         Average
--------------------------------------------------------------------------------
Hunks per patch                          |            1.23
Lines added per patch                    |            2.45
Lines deleted per patch                  |            1.87
Total changed lines per patch            |            4.32
Tokens added per patch                   |           12.34
Tokens deleted per patch                 |            9.87
Total changed tokens per patch           |           22.21
Edit distance per patch                  |           45.67
Edit similarity (%)                      |           87.23
Normalized edit distance                 |          0.1277
Code preserved ratio (%)                 |           91.45

Distribution of code preservation ratio:
  Minimal change   (>95% preserved):    89 patches ( 57.1%)
  Moderate change (80-95% preserved):   52 patches ( 33.3%)
  Major change     (<80% preserved):    15 patches (  9.6%)
================================================================================
```

### 模型对比输出
```
[COMPARISON RESULTS - qwen3-8b vs llama3.1-8b]
========================================================
Metric          |        qwen3-8b |     llama3.1-8b |            Diff |       Δ%
--------------------------------------------------------
Pass@1          |          45.32% |          42.18% |          +3.14% |   +6.93%
Pass@5          |          78.45% |          81.22% |          +2.77% |   +3.53%
Pass@10         |          89.17% |          87.65% |          -1.52% |   -1.70%

[DETAILED DIFF COMPARISON - PLAUSIBLE Patches]
========================================================
Metric                                   |        qwen3-8b |     llama3.1-8b |            Diff |       Δ%
--------------------------------------------------------
Patch count                              |             156 |             142 |               - |        -
Avg hunks                                |            1.23 |            1.45 |          +0.22 |  +17.89%
Avg added lines                          |            2.45 |            2.12 |          -0.33 |  -13.47%
...
```

## 工作流程

1. **数据加载**: 从 `xcodeeval_merged.json` 加载数据集
2. **模型加载**: 使用 vLLM 加载指定模型
3. **Prompt 构建**: 根据模板和模型格式构建提示词
4. **批量生成**: 每个样本同时生成 nsample 个修复候选
5. **结果保存**: 保存为结构化JSON文件
6. **评估**: 使用 eval_apr.py 评估修复效果
7. **统计分析**: 使用 get_result.py 计算Pass@k和详细统计

## 模型格式支持

工具自动为不同模型添加相应的对话格式：

- **Qwen**: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
- **LLaMA**: `<|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
- **DeepSeek**: `###Instruction\n...###response\n\n`
- **StarCoder**: 无特殊格式

## GPU 管理

### 自动GPU分配
- 多进程模式下自动分配GPU
- 支持GPU冲突检测和锁机制
- 自动清理GPU锁文件

### 内存优化
- GPU内存利用率: 90%
- 最大模型长度: 4096 tokens
- 最大生成长度: 1024 tokens

## 注意事项

1. **模型路径**: 确保模型文件存在于 `MODEL_CONFIGS` 中指定的路径
2. **GPU内存**: 确保GPU内存足够加载模型
3. **数据集**: 确保 `xcodeeval_merged.json` 文件存在
4. **并发控制**: 多进程模式下注意GPU资源分配
5. **干运行**: 使用 `--dry-run 1` 可以测试流程而不实际生成
6. **缓存管理**: 使用 `--force-recalc` 强制重新计算统计结果

