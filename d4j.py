import json
import os
import sys
import torch
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from peft import PeftModel
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_first_java_code(s: str) -> str:
    matches = re.findall(r'```java(.*?)```', s, re.DOTALL)
    return matches[0].strip() if matches else ""


BOF = '[INST]'
EOF = '[/INST]'

# 设置模型别名前缀对应的开头与结尾提示
MODEL_PROMPT_FORMATS = {
    'qwen': ('<|im_start|>user\n', '<|im_end|>'),
    'codellama': ('[INST]', '[/INST]'),
    'llama': ('[INST]', '[/INST]'),
    'mistral': ('[INST]', '[/INST]'),
    'starchat': ('<|system|>\n<|end|>\n<|user|>', '<|end|>\n<|assistant|>'),
}

# 模型配置：支持基础模型和训练后模型两种版本
MODEL_CONFIGS = {
    # Qwen3 模型
    'qwen3-8b': {
        'base_model': '/data1/czj/model/qwen3-8b',
        'adapter_path': None
    },
    'qwen3-8b-trained': {
        'base_model': '/data1/czj/model/qwen3-8b',
        'adapter_path': '/data1/czj/model/trained_model_qwen3'
    },

    # CodeLlama 模型
    'codellama-7b': {
        'base_model': '/data1/czj/model/CodeLlama-7b-Instruct',
        'adapter_path': None
    },
    'codellama-7b-trained': {
        'base_model': '/data1/czj/model/CodeLlama-7b-Instruct',
        'adapter_path': '/data1/czj/model/trained_model_llama'
    },

    'codellama-13b': {
        'base_model': '/data1/czj/model/codeLlama-13b-instruct',
        'adapter_path': None
    },
    'trained_model_codellama-v1': {
        'base_model': '/data1/czj/model/codeLlama-13b-instruct',
        'adapter_path': '/data1/czj/model/trained_model_codellama_13b-v1'
    },
    # Llama3.1 模型
    'llama3.1-8b': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': None
    },
    'llama3.1-8b-trained': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': '/data1/czj/model/trained_model_llama'
    },
    'llama3.1-8b-trained-v1': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': '/data1/czj/model/trained_model_llama-v1'
    },
    'sft-grpo-llama': {
        'base_model': '/data1/czj/model/Llama-3-8B-Instruct',
        'adapter_path': '/data1/czj/model/trained_sft_grpo_llama_3_8b_instruct/sft'
    },

}


def get_prompt_format(model_key):
    for key in MODEL_PROMPT_FORMATS:
        if model_key.startswith(key):
            return MODEL_PROMPT_FORMATS[key]
    # 默认格式
    return '[INST]', '[/INST]'


def load_model_with_adapter(model_config):
    """加载带LoRA适配器的模型"""
    base_model_path = model_config['base_model']
    adapter_path = model_config.get('adapter_path')

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 加载基础模型
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        quantization_config=bnb_config
    )

    # 如果有适配器路径，加载并合并LoRA适配器
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Original model type: {type(model).__name__}")

        # 合并LoRA权重到基础模型中
        model = model.merge_and_unload()
        print(f"Merged model type: {type(model).__name__}")
        print("LoRA weights merged into base model")

    return model, tokenizer


# 设置prompt格式
BOF, EOF = get_prompt_format(sys.argv[1])

# 模型加载
if sys.argv[1] in MODEL_CONFIGS:
    # 使用配置系统
    model_config = MODEL_CONFIGS[sys.argv[1]]
    model, tokenizer = load_model_with_adapter(model_config)
else:
    raise ValueError(f"Unknown model key: {sys.argv[1]}. Available models: {list(MODEL_CONFIGS.keys())}")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print('load model success ..', flush=True)


def cal(bug_id, code, title, description, filename):
    # 构造用于修复错误代码的提示 (Prompt)
    prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
    print(prompt, flush=True)
    # cnt = len(tokenizer.tokenize(prompt))
    # 计算 Prompt 的 token 数量
    # if cnt >= 1000:
    # print('prompt too long', bug_id, flush=True)
    # 删除 dataset/bug_id.json
    # if os.path.exists('/data1/czj/prorepair/defects4j/dataset/' + bug_id + '.json'):
    # os.remove('/data1/czj/prorepair/defects4j/dataset/' + bug_id + '.json')
    # return [None, None]  # 如果 Prompt 太长，删除相应的 JSON 文件并返回 None
    # max_d = cnt   # 设定生成文本的最大 token 长度
    while True:  # 无限循环，直到成功生成非空的修复代码
        output = pipe(prompt, max_new_tokens=1024, temperature=1.0, do_sample=True)  # 使用 Transformer 模型进行文本生成
        full_text = output[0]['generated_text']  # 获取模型生成的完整文本
        print(full_text)
        ret = extract_first_java_code(full_text.split('[/INST]')[1])  # 提取生成的 Java 代码部分，去除无关文本
        print('code:', ret, flush=True)
        if ret.strip() != '':
            break
    return [full_text, ret]


base_dir = '/data1/czj/prorepair/defects4j/dataset'
base_fix_dir = f'/data1/czj/prorepair/defects4j/results/{sys.argv[1]}'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.json'), reverse=True):  # 遍历 base_dir 目录下所有 .json 文件，并按文件名降序排序
    cnt += 1  # 计数器自增，用于分配任务
    if cnt % int(sys.argv[2]) != int(sys.argv[3]):  # 根据 sys.argv 传入的参数，判断当前文件是否属于当前进程需要处理的任务
        continue  # 如果不是当前进程要处理的文件，则跳过

    # 获取文件的完整路径
    full_path = str(file_path)
    print(full_path, flush=True)

    # 读取文件内容
    with open(full_path, 'r') as file:
        content = file.read()

    json_data = json.loads(content)
    result_data = json_data

    for e in range(int(sys.argv[-1])):
        # 创建固定的输出目录
        fix_dir = f'{base_fix_dir}/fixed{e}'
        os.makedirs(fix_dir, exist_ok=True)

        # 获取文件名
        file_name = os.path.basename(full_path)
        fix_name = os.path.join(fix_dir, file_name)
        print(f"Output path: {fix_name}", flush=True)
        if os.path.exists(fix_name) and os.path.exists(fix_name + '.log'):
            print('result exists ...')
            continue
        # cnt += 1
        # if cnt % int(sys.argv[1]) != int(sys.argv[2]):
        #    continue
        full, res = cal(file_name.split('.')[0], json_data['buggy'], json_data['issue_title'],
                        json_data['issue_description'], json_data['loc'])
        if full == None:  # 如果 full 为 None，则说明修复失败，跳过此轮
            continue
        result_data['fix'] = res  # 将修复后的代码存入 JSON 数据
        with open(fix_name, 'w') as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)
        with open(fix_name + '.log', 'w') as file:
            print(full, file=file)