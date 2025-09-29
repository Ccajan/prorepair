import os
import sys
import re
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def extract_first_java_code(s: str) -> str:
    matches = re.findall(r'```java(.*?)```', s, re.DOTALL)
    return matches[0].strip() if matches else ""

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
        'adapter_path': '/data1/czj/model/trained_model_codellama'
    },

    'codellama-13b': {
        'base_model': '/data1/czj/model/CodeLlama-13b-Instruct',
        'adapter_path': None
    },
    # Llama3.1 模型
    'llama3.1-8b': {
        'base_model': '/home/liu01/projects/Programrepair/model',
        'adapter_path': None
    },
    'llama3.1-8b-trained': {
        'base_model': '/home/liu01/projects/Programrepair/model',
        'adapter_path': '/data1/czj/model/trained_model_llama'
    }
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

def cal(code, filename):
    prompt = BOF + " This is an incorrect code(" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    max_d = 500
    while True:
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, temperature=1.0, do_sample=True)
        full_text = output[0]['generated_text']
        print(full_text)
        # 根据不同的EOF格式提取代码
        if EOF in full_text:
            ret = extract_first_java_code(full_text.split(EOF)[1])  # 提取模型生成的第一个 Java 代码块
        else:
            ret = extract_first_java_code(full_text)
        print('code:', ret, flush=True)
        if ret.strip() != '':
            break
        max_d = min(3000 - cnt, max_d + 500)
    return [full_text, ret]

base_dir = 'evalrepair-java/origin/'
result_base_dir = 'evalrepair-java-res/' + sys.argv[1] + '/'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.java'), reverse=True):
    cnt += 1  # 计数器自增，用于分配任务
    if len(sys.argv) >= 4 and cnt % int(sys.argv[2]) != int(sys.argv[3]):  # 多进程任务分配
        continue  # 如果不是当前进程要处理的文件，则跳过
        
    full_path = str(file_path)
    print(full_path, flush=True)

    with open(full_path, 'r') as file:
        content = file.read()

    print(content)

    # 获取生成版本数，默认10个
    num_generations = int(sys.argv[-1]) if len(sys.argv) >= 5 else 10

    for e in range(num_generations):
        file_name = os.path.basename(full_path)
        # 修复：构建正确的目录路径格式 fixed0/, fixed1/, fixed2/, ...
        fix_subdir = os.path.join(result_base_dir, f'fixed{e}')
        fix_name = os.path.join(fix_subdir, file_name)
        
        # 修复：自动创建目录（如果不存在）
        os.makedirs(fix_subdir, exist_ok=True)
        
        print(f"Output path: {fix_name}", flush=True)
        
        # 检查文件是否已存在，避免重复处理
        if os.path.exists(fix_name) and os.path.exists(fix_name + '.log'):
            print('result exists ...')
            continue
            
        full, res = cal(content, file_name)
        if full == None:
            continue
        with open(fix_name, 'w') as file:
            print(res, file=file)
        with open(fix_name + '.log', 'w') as file:
            print(full, file=file)