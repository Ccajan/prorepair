import subprocess
import os, glob
import sys
import json
import difflib
import re

# 确保必要的目录存在
os.makedirs('tmp', exist_ok=True)

def remove_java_comments(code):
    """移除Java代码中的多行注释/* */"""
    # 只移除多行注释 /* ... */
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除空行，只保留有内容的行
    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip():
            cleaned_lines.append(line.rstrip())
    
    return '\n'.join(cleaned_lines)

def calc_diff_stats(buggy, fix):
    """计算补丁的diff统计信息，比较错误代码和修复后的代码"""
    if not buggy or not fix:  # 处理空输入
        return {'added_lines': 0, 'deleted_lines': 0, 'preserved_ratio': 0.0}
    
    # 移除注释后再进行比较
    buggy_clean = remove_java_comments(buggy)
    fix_clean = remove_java_comments(fix)
    
    buggy_lines = buggy_clean.strip().splitlines()
    fix_lines = fix_clean.strip().splitlines()
    
    # 计算增删行数
    diff = list(difflib.ndiff(buggy_lines, fix_lines))
    added = sum(1 for line in diff if line.startswith('+ '))
    deleted = sum(1 for line in diff if line.startswith('- '))
    
    # 计算保留比例 (Code Consistency Rate) - 排除虚拟的结束块
    # CCR = r / k, 其中 k 是修复后代码的总行数，r 是保留的代码行数
    matcher = difflib.SequenceMatcher(None, buggy_lines, fix_lines)
    matching_blocks = matcher.get_matching_blocks()
    # 排除最后一个虚拟块 (len(a), len(b), 0)
    preserved = sum(block.size for block in matching_blocks[:-1])
    preserved_ratio = preserved / len(fix_lines) if fix_lines else 0.0
    
    return {
        'added_lines': added,
        'deleted_lines': deleted,
        'preserved_ratio': round(preserved_ratio * 100, 2)
    }

def print_diff_stats(diff_stats):
    """打印补丁的差异统计信息"""
    print(f"[DIFF STATS] Added: {diff_stats['added_lines']}, Deleted: {diff_stats['deleted_lines']}, Preserved: {diff_stats['preserved_ratio']}%")

def test(id, name, tag):
    # 获取绝对路径的 tmp 目录，避免系统 /tmp 空间不足
    tmp_dir = os.path.abspath('tmp')
    # 设置 TMPDIR 环境变量，避免编译器使用系统 /tmp
    command = f"export TMPDIR={tmp_dir} && cd evalrepair-java && bash test.sh {id} {name} {tag}"
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("Standard Output:", stdout.decode())
    print("Standard Error:", stderr.decode())
    print(process.returncode)
    return [process.returncode, str(stdout)]

def calc_pass_at_k(n, c, k):
    """计算单个bug的pass@k
    
    Args:
        n: 总补丁数（固定为10）
        c: 正确补丁数（通过测试的数量）
        k: 采样数量
    
    Returns:
        pass@k 的值 (0.0 到 1.0)
    """
    # n 现在固定为 10，k 最大为 10
    assert n == 10, f"Expected n=10, got n={n}"
    assert k <= 10, f"k should not exceed 10, got k={k}"
    
    if c == 0:
        return 0.0
    
    if n - c < k:  # 失败补丁数 < 采样数，必然成功
        return 1.0
    
    # pass@k = 1 - C(n-c, k) / C(n, k)
    return 1.0 - (comb(n - c, k) / comb(n, k))

def comb(n, k):
    """计算组合数 C(n, k) = n! / (k! * (n-k)!)"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    
    return result

# 存储每个bug的所有补丁结果
bug_results = {}  # {bug_name: [result0, result1, ..., result9]}

# 添加保留率统计
diff_stats = {
    'total_preserved': 0,
    'patch_count': 0
}

for id in range(10):
    directory_path = f"./evalrepair-java-res/{sys.argv[1]}/fixed{id}/"
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for file_path in sorted(glob.glob(os.path.join(directory_path, '*.java')), reverse=False):
            print(file_path)
            name = file_path.split('/')[-1].split('.')[0]
            
            # 初始化bug结果列表
            if name not in bug_results:
                bug_results[name] = []
            
            if sys.argv[-1] == "rejudge":
                ret, detail = test(id, name, sys.argv[1])
                
                # 计算保留率（如果测试成功）
                result_data = {'return_code': ret}
                if ret == 0:
                    try:
                        # 读取修复后的代码
                        with open(file_path, 'r', encoding='utf-8') as f:
                            fixed_code = f.read()
                        
                        # 读取原始buggy代码
                        buggy_file = f"./evalrepair-java-res/buggy/{name}.java"
                        if os.path.exists(buggy_file):
                            with open(buggy_file, 'r', encoding='utf-8') as f:
                                buggy_code = f.read()
                            
                            # 只计算保留率
                            current_diff = calc_diff_stats(buggy_code, fixed_code)
                            result_data['preserved_ratio'] = current_diff['preserved_ratio']
                        else:
                            result_data['preserved_ratio'] = 0.0
                    except Exception as e:
                        print(f"[ERROR] Failed to calculate diff stats for {name}: {e}")
                        result_data['preserved_ratio'] = 0.0
                else:
                    result_data['preserved_ratio'] = 0.0
                
                # 保存结果到 .result 文件（JSON格式）
                with open(file_path + '.result', 'w') as f:
                    f.write(json.dumps(result_data))
            else:
                # 读取 .result 文件（兼容旧格式和新格式）
                result_content = open(file_path + '.result', 'r').read().strip()
                try:
                    result_data = json.loads(result_content)
                    ret = result_data['return_code']
                except (json.JSONDecodeError, KeyError):
                    # 兼容旧格式（只有数字）
                    ret = int(result_content)
                    result_data = {'return_code': ret, 'preserved_ratio': 0.0}
            
            # 记录结果 (0表示成功, 非0表示失败)
            bug_results[name].append(ret)
            
            # 统计保留率信息（无论是rejudge还是读取文件）
            if ret == 0 and 'preserved_ratio' in result_data:
                ratio = result_data['preserved_ratio']
                print(f"[PRESERVED RATIO] {ratio}%")
                
                # 更新全局统计
                diff_stats['total_preserved'] += ratio
                diff_stats['patch_count'] += 1

# 计算 pass@k（只计算有完整10个候选补丁的bug）
pass1_sum = 0.0
pass5_sum = 0.0
pass10_sum = 0.0
valid_bugs = 0

for bug_name, results in bug_results.items():
    n = len(results)  # 总补丁数
    
    # 只计算有完整10个候选补丁的bug
    if n != 10:
        print(f"[WARNING] Bug {bug_name} has {n} patches, skipping (expected 10)")
        continue
    
    c = sum(1 for r in results if r == 0)  # 成功的补丁数
    valid_bugs += 1
    
    pass1_sum += calc_pass_at_k(n, c, 1)
    pass5_sum += calc_pass_at_k(n, c, 5)
    pass10_sum += calc_pass_at_k(n, c, 10)

pass1_rate = (pass1_sum / valid_bugs) * 100 if valid_bugs > 0 else 0
pass5_rate = (pass5_sum / valid_bugs) * 100 if valid_bugs > 0 else 0
pass10_rate = (pass10_sum / valid_bugs) * 100 if valid_bugs > 0 else 0

print("\n" + "="*80)
print("[SUCCESS RATE - Pass@k]")
print("="*80)
print(f"Total bugs analyzed: {valid_bugs} (with complete 10 patches)")
print(f"pass@1:  {pass1_rate:6.2f}%")
print(f"pass@5:  {pass5_rate:6.2f}%")
print(f"pass@10: {pass10_rate:6.2f}%")
print("="*80)

# 输出保留率统计信息
if diff_stats['patch_count'] > 0:
    patch_count = diff_stats['patch_count']
    avg_preserved = diff_stats['total_preserved'] / patch_count
    
    print("\n" + "="*80)
    print("[CODE PRESERVATION STATISTICS]")
    print("="*80)
    print(f"Total successful patches analyzed:      {patch_count}")
    print(f"Average code preserved (CCR):           {avg_preserved:.1f}%")
    print("="*80)
else:
    print("\n" + "="*80)
    print("[CODE PRESERVATION STATISTICS]")
    print("="*80)
    print("No successful patches found for preservation analysis")
    print("="*80)