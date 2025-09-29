import os
import sys
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_first_cpp_code(s: str) -> str:
    matches = re.findall(r'```c\+\+(.*?)```', s, re.DOTALL)
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
        'adapter_path': '/home/liu01/projects/train_model/trained_model_qwen3_4b_curriculum'
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

def remove_comments_and_return_first_part(code):
    comments = re.findall(r'/\*(.*?)\*/', code, flags=re.DOTALL)  # 提取所有 /* ... */ 之间的内容
    first_comment = comments[0].strip() if comments else ''  # 获取第一个注释（如果存在）
    input_content_match = re.search(r'INPUT.*', first_comment, flags=re.DOTALL)  # 在第一个注释中查找包含 "INPUT" 的内容
    input_content = input_content_match.group(0) if input_content_match else ''   # 如果找到了 "INPUT"，提取相关内容，否则为空字符串
    code_without_comments = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)   # 删除所有 /* ... */ 之间的内容，得到无注释的代码
    return input_content, code_without_comments

def cal(code, filename):
    prompt_suffix = open('/data1/czj/prorepair/evalrepair-c++/prompt/' + filename, 'r', encoding='utf-8').read()   # 读取 `filename` 对应的 prompt 文件内容（可能包含修复代码的提示）
    _, prompt_suffix = remove_comments_and_return_first_part(prompt_suffix) # 使用 `remove_comments_and_return_first_part` 去除 prompt 代码中的注释
    prompt = BOF + " This is an incorrect code (" + filename + "):\n```c++\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```c++\n" + prompt_suffix
    # 构造输入给模型的 prompt，包含错误代码和修复请求
    print(prompt, flush=True)  # 打印生成的 prompt（用于调试）
    cnt = len(tokenizer.tokenize(prompt))  # 计算 prompt 转换为 tokens 之后的长度
    max_d = 500   # 设定最大生成长度的初始值

    while True: # 进入循环，直到模型生成的代码不为空
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, temperature=1.0, do_sample=True)# 调用 Transformer 语言模型进行文本生成
        full_text = output[0]['generated_text']  # 获取模型返回的完整文本
        print(full_text)  # 打印完整的生成文本（用于调试）
        # 根据不同的EOF格式提取代码
        if EOF in full_text:
            ret = extract_first_cpp_code(full_text.split(EOF)[1])  # 提取模型生成的第一个 C++ 代码块
        else:
            ret = extract_first_cpp_code(full_text)
        print('code:', ret, flush=True)  # 打印提取出的代码（用于调试）
        if ret.strip() != '':   # 如果提取出的代码不为空，则跳出循环
            break
        max_d = min(3000 - cnt, max_d + 500)  # 如果代码为空，则增加最大生成长度 `max_d`
    return [full_text, ret]



base_dir = '/data1/czj/prorepair/evalrepair-c++/buggy/' ##c++代码的基准目录
result_base_dir = '/data1/czj/prorepair/evalrepair-c++/result/' + sys.argv[1] + '/'  ##修复好后的代码的基础目录

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.cpp'), reverse=True):  # 遍历 base_dir 目录下所有 .cpp 文件，并按文件名倒序排序
    cnt += 1  # 计数器自增，用于分配任务
    if len(sys.argv) >= 4 and cnt % int(sys.argv[2]) != int(sys.argv[3]):  # 多进程任务分配
        continue  # 如果不是当前进程要处理的文件，则跳过
    
    full_path = str(file_path)  # 获取完整的文件路径并转换为字符串
    print(full_path, flush=True)  # 打印当前处理的文件路径，flush=True 确保立即输出

    with open(full_path, 'r', encoding='utf-8') as file:
        content = file.read()  # 读取文件内容

    print(content) # 打印读取到的 C++ 代码，方便调试

    # 获取生成版本数，默认10个
    num_generations = int(sys.argv[-1]) if len(sys.argv) >= 5 else 10
    
    for e in range(num_generations):
        file_name = os.path.basename(full_path)  # 获取文件名（不包含路径）
        # 修复：构建正确的目录路径格式 fixed0/, fixed1/, fixed2/, ...
        fix_subdir = os.path.join(result_base_dir, f'fixed{e}')
        fix_name = os.path.join(fix_subdir, file_name)  # 生成修复后代码的文件路径，存放在不同子目录 fixed0/ ~ fixed9/
        
        # 修复：自动创建目录（如果不存在）
        os.makedirs(fix_subdir, exist_ok=True)
        
        print(f"Output path: {fix_name}", flush=True) # 打印修复后文件的存储路径
        
        # 检查文件是否已存在，避免重复处理
        if os.path.exists(fix_name) and os.path.exists(fix_name + '.log'):
            print('result exists ...')
            continue
            
        full, res = cal(content, file_name)   # 调用 cal() 进行 C++ 代码修复
        if full == None:   # 如果修复失败（full == None），跳过当前修复版本
            continue
        with open(fix_name, 'w', encoding='utf-8') as file:   # 保存修复后的 C++ 代码到 fix_name
            print(res, file=file)
        with open(fix_name + '.log', 'w', encoding='utf-8') as file:  # 保存完整 AI 生成的日志信息到 fix_name.log
            print(full, file=file)