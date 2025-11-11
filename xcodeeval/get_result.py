import os
from collections import defaultdict
import tqdm
import jsonlines
from typing import List, Union
import itertools
import numpy as np
import difflib
import re
import argparse
import json
import time

LANG_CLUSTER_TO_LANG_COMPILER = {
    "C++": "GNU C++17",
}

# 将路径设置为可配置
def get_paths(model_name):
    path = f'xcodeeval/result/{model_name}/'
    output_path = os.path.join(path, "eval_apr_val_execeval")
    return path, output_path

ks = [1, 5, 10]  # 只计算 Pass@1, Pass@5, Pass@10


def save_results(stats, model_name, output_dir="xcodeeval/cache"):
    """保存计算结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{model_name}_stats.json")
    
    # 将统计结果序列化
    result_data = {
        'pass_at_k': dict(stats.pass_at_k),
        'diff_stats': stats.diff_stats,
        'timestamp': time.time(),
        'model_name': model_name
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"[CACHE] Results saved to: {result_file}")
    return result_file


def load_cached_results(model_name, output_dir="xcodeeval/cache"):
    """加载缓存的结果"""
    result_file = os.path.join(output_dir, f"{model_name}_stats.json")
    
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # 重建 ModelStats 对象
            stats = ModelStats()
            stats.pass_at_k = defaultdict(dict, data['pass_at_k'])
            stats.diff_stats = data['diff_stats']
            
            timestamp = data.get('timestamp', 0)
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            print(f"[CACHE] Loaded cached results for {model_name} (computed at {time_str})")
            return stats
        except Exception as e:
            print(f"[WARNING] Failed to load cached results: {e}")
    
    return None


def count_tokens(text):
    """简单的token计数，使用空格和常见分隔符分割"""
    if not text:
        return 0
    # 按空格、标点等分割来近似统计token
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return len(tokens)


def levenshtein_distance(s1, s2):
    """计算两个字符串之间的Levenshtein编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # 插入、删除、替换的代价
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calc_diff_stats(buggy, fix):
    """
    计算详细的diff统计信息
    返回：
    - hunks: diff块的数量
    - added_lines: 新增行数
    - deleted_lines: 删除行数
    - total_changed_lines: 总变化行数 (added + deleted)
    - added_tokens: 新增的token数量
    - deleted_tokens: 删除的token数量
    - total_changed_tokens: 总变化token数量
    - edit_distance: Levenshtein编辑距离
    - edit_similarity: 编辑相似度 (%) - 基于编辑距离
    - norm_edit_distance: 归一化编辑距离 (0-1)
    - norm_edit_distance_pct: 归一化编辑距离百分比
    - buggy_length: buggy代码字符数
    - fix_length: fix代码字符数
    - preserved_ratio: 代码保留比例 (%)
    """
    if not buggy or not fix:
        return {
            'hunks': 0,
            'added_lines': 0,
            'deleted_lines': 0,
            'total_changed_lines': 0,
            'added_tokens': 0,
            'deleted_tokens': 0,
            'total_changed_tokens': 0,
            'edit_distance': 0,
            'edit_similarity': 100.0,
            'norm_edit_distance': 0.0,
            'norm_edit_distance_pct': 0.0,
            'buggy_length': 0,
            'fix_length': 0,
            'preserved_ratio': 0.0
        }
    
    buggy_lines = buggy.strip().splitlines()
    fix_lines = fix.strip().splitlines()
    
    # 生成unified diff来统计hunks
    diff_lines = list(difflib.unified_diff(buggy_lines, fix_lines, lineterm=''))
    hunks = sum(1 for line in diff_lines if line.startswith('@@'))
    
    # 使用ndiff统计增删行数和token数
    diff = list(difflib.ndiff(buggy_lines, fix_lines))
    added_lines = [line[2:] for line in diff if line.startswith('+ ')]
    deleted_lines = [line[2:] for line in diff if line.startswith('- ')]
    
    added_line_count = len(added_lines)
    deleted_line_count = len(deleted_lines)
    
    # 统计token
    added_tokens = sum(count_tokens(line) for line in added_lines)
    deleted_tokens = sum(count_tokens(line) for line in deleted_lines)
    
    # 计算编辑距离和相似度
    buggy_str = buggy.strip()
    fix_str = fix.strip()
    edit_dist = levenshtein_distance(buggy_str, fix_str)
    max_len = max(len(buggy_str), len(fix_str))
    edit_similarity = (1 - edit_dist / max_len) * 100 if max_len > 0 else 100.0
    norm_edit = (edit_dist / max_len) if max_len > 0 else 0.0
    norm_edit_pct = norm_edit * 100
    buggy_len_chars = len(buggy_str)
    fix_len_chars = len(fix_str)
    
    # 计算保留比例 (Code Consistency Rate)
    # CCR = r / k, 其中 k 是修复后代码的总行数，r 是保留的代码行数
    matcher = difflib.SequenceMatcher(None, buggy_lines, fix_lines)
    matching_blocks = matcher.get_matching_blocks()
    # 排除最后一个虚拟块 (len(a), len(b), 0)
    preserved = sum(block.size for block in matching_blocks[:-1])
    preserved_ratio = (preserved / len(fix_lines) * 100) if fix_lines else 0.0
    
    return {
        'hunks': hunks,
        'added_lines': added_line_count,
        'deleted_lines': deleted_line_count,
        'total_changed_lines': added_line_count + deleted_line_count,
        'added_tokens': added_tokens,
        'deleted_tokens': deleted_tokens,
        'total_changed_tokens': added_tokens + deleted_tokens,
        'edit_distance': edit_dist,
        'edit_similarity': round(edit_similarity, 2),
        'norm_edit_distance': round(norm_edit, 4),
        'norm_edit_distance_pct': round(norm_edit_pct, 2),
        'buggy_length': buggy_len_chars,
        'fix_length': fix_len_chars,
        'preserved_ratio': round(preserved_ratio, 2)
    }


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def get_execeval_out_file_name(output_path, compiler):
    return os.path.join(output_path, f"{compiler}.jsonl")


class ModelStats:
    def __init__(self):
        self.pass_at_k = defaultdict(dict)
        self.diff_stats = {
            'patch_count': 0,
            'total_hunks': 0,
            'total_added': 0,
            'total_deleted': 0,
            'total_changed_lines': 0,
            'total_added_tokens': 0,
            'total_deleted_tokens': 0,
            'total_changed_tokens': 0,
            'total_edit_distance': 0,
            'total_edit_similarity': 0,
            'total_norm_edit_distance': 0,
            'total_preserved': 0,
            'preservation_distribution': {
                'high': 0,    # >95%
                'medium': 0,  # 80-95%
                'low': 0      # <80%
            }
        }


def load_model_results(model_name, force_recalc=False):
    """加载单个模型的结果，支持缓存"""
    
    # 如果不强制重新计算，先尝试加载缓存
    if not force_recalc:
        cached_stats = load_cached_results(model_name)
        if cached_stats is not None:
            return cached_stats
    
    print(f"[COMPUTING] Calculating statistics for {model_name}...")
    path, output_path = get_paths(model_name)
    
    if not os.path.exists(output_path):
        print(f"[ERROR] Output path does not exist: {output_path}")
        return None
    
    stats = ModelStats()
    
    for lang, compiler in tqdm.tqdm(LANG_CLUSTER_TO_LANG_COMPILER.items(), desc=f"Loading {model_name}"):
        execeval_out_file = get_execeval_out_file_name(output_path, compiler)
        
        if not os.path.exists(execeval_out_file):
            print(f"[WARNING] File does not exist: {execeval_out_file}")
            continue
            
        results = defaultdict(list)
        
        with jsonlines.open(execeval_out_file) as jrp:
            for sample in jrp:
                src_uid = sample["source_data"]["src_uid"]
                task_id = f"{src_uid}|||{lang}"
                buggy_code = sample["source_data"].get("bug_source_code", "")
                
                for i, ut_res in enumerate(sample["unit_test_results"]):
                    if "error" in ut_res:
                        continue
                    
                    # 检查是否通过所有测试
                    passed = all(x["exec_outcome"] == "PASSED" for x in ut_res)
                    
                    # 如果通过测试，计算diff统计
                    if passed and buggy_code:
                        try:
                            # 获取对应的修复代码
                            choices = sample["model_response"]["choices"]
                            
                            if i < len(choices):
                                fix_code = choices[i]["message"]["content"]
                                diff_stat = calc_diff_stats(buggy_code, fix_code)
                                
                                # 更新统计
                                stats.diff_stats['patch_count'] += 1
                                stats.diff_stats['total_hunks'] += diff_stat['hunks']
                                stats.diff_stats['total_added'] += diff_stat['added_lines']
                                stats.diff_stats['total_deleted'] += diff_stat['deleted_lines']
                                stats.diff_stats['total_changed_lines'] += diff_stat['total_changed_lines']
                                stats.diff_stats['total_added_tokens'] += diff_stat['added_tokens']
                                stats.diff_stats['total_deleted_tokens'] += diff_stat['deleted_tokens']
                                stats.diff_stats['total_changed_tokens'] += diff_stat['total_changed_tokens']
                                stats.diff_stats['total_edit_distance'] += diff_stat['edit_distance']
                                stats.diff_stats['total_edit_similarity'] += diff_stat['edit_similarity']
                                stats.diff_stats['total_norm_edit_distance'] += diff_stat['norm_edit_distance']
                                stats.diff_stats['total_preserved'] += diff_stat['preserved_ratio']
                                
                                # 保留比例分布
                                ratio = diff_stat['preserved_ratio']
                                if ratio > 95:
                                    stats.diff_stats['preservation_distribution']['high'] += 1
                                elif ratio > 80:
                                    stats.diff_stats['preservation_distribution']['medium'] += 1
                                else:
                                    stats.diff_stats['preservation_distribution']['low'] += 1
                        except Exception as e:
                            print(f"Error calculating diff stats: {e}")
                    
                    results[task_id].append(ut_res)

        total, correct = [], []
        for result in results.values():
            passed = [
                all(x["exec_outcome"] == "PASSED" for x in ut_res) for ut_res in result
            ]
            total.append(len(passed))
            correct.append(sum(passed))
        total = np.array(total)
        correct = np.array(correct)

        stats.pass_at_k[lang] = {
            f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
            for k in ks
            if (total >= k).all()
        }
    
    # 保存计算结果
    save_results(stats, model_name)
    
    return stats


def print_single_model_results(stats, model_name):
    """打印单个模型的结果"""
    print(f"\n{model_name} Results:")
    print("=" * 80)
    
    # Pass@k 结果
    for k in ks:
        if f"pass@{k}" in stats.pass_at_k["C++"]:
            result = stats.pass_at_k["C++"][f"pass@{k}"] * 100
            print(f"Pass@{k}: {result:.2f}%")
        else:
            print(f"Pass@{k}: N/A (insufficient samples)")

    print("=" * 80)

    # 详细的diff统计信息
    if stats.diff_stats['patch_count'] > 0:
        print(f"\n[PATCH MODIFICATION STATISTICS - PLAUSIBLE Patches]")
        print(f"{'='*80}")
        print(f"Total PLAUSIBLE patches analyzed: {stats.diff_stats['patch_count']}")
        print(f"\n{'Metric':<40} | {'Average':>15}")
        print(f"{'-'*80}")
        print(f"{'Hunks per patch':<40} | {stats.diff_stats['total_hunks']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Lines added per patch':<40} | {stats.diff_stats['total_added']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Lines deleted per patch':<40} | {stats.diff_stats['total_deleted']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Total changed lines per patch':<40} | {stats.diff_stats['total_changed_lines']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Tokens added per patch':<40} | {stats.diff_stats['total_added_tokens']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Tokens deleted per patch':<40} | {stats.diff_stats['total_deleted_tokens']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Total changed tokens per patch':<40} | {stats.diff_stats['total_changed_tokens']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Edit distance per patch':<40} | {stats.diff_stats['total_edit_distance']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Edit similarity (%)':<40} | {stats.diff_stats['total_edit_similarity']/stats.diff_stats['patch_count']:>15.2f}")
        print(f"{'Normalized edit distance':<40} | {stats.diff_stats['total_norm_edit_distance']/stats.diff_stats['patch_count']:>15.4f}")
        print(f"{'Code preserved ratio (%)':<40} | {stats.diff_stats['total_preserved']/stats.diff_stats['patch_count']:>15.2f}")
        
        print(f"\n{'Distribution of code preservation ratio:':}")
        dist = stats.diff_stats['preservation_distribution']
        total_dist = sum(dist.values())
        if total_dist > 0:
            print(f"  Minimal change   (>95% preserved):   {dist['high']:3d} patches ({dist['high']/total_dist*100:5.1f}%)")
            print(f"  Moderate change (80-95% preserved):  {dist['medium']:3d} patches ({dist['medium']/total_dist*100:5.1f}%)")
            print(f"  Major change     (<80% preserved):   {dist['low']:3d} patches ({dist['low']/total_dist*100:5.1f}%)")
        print(f"{'='*80}")
    else:
        print("\n[WARNING] No PLAUSIBLE patches found for diff statistics")


def print_comparison_results(stats1, stats2, model1_name, model2_name):
    """打印两个模型的对比结果"""
    print(f"\n[COMPARISON RESULTS - {model1_name} vs {model2_name}]")
    print("=" * 120)
    
    # Pass@k 对比
    print(f"{'Metric':<15} | {model1_name:>15} | {model2_name:>15} | {'Diff':>15} | {'Δ%':>10}")
    print("-" * 120)
    
    for k in ks:
        if f"pass@{k}" in stats1.pass_at_k["C++"] and f"pass@{k}" in stats2.pass_at_k["C++"]:
            val1 = stats1.pass_at_k["C++"][f"pass@{k}"] * 100
            val2 = stats2.pass_at_k["C++"][f"pass@{k}"] * 100
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            color = '\033[92m' if diff > 0 else '\033[91m' if diff < 0 else '\033[0m'
            print(f"Pass@{k:<10} | {val1:>14.2f}% | {val2:>14.2f}% | {color}{diff:>+14.2f}%\033[0m | {color}{pct_change:>+9.2f}%\033[0m")
    
    print("=" * 120)
    
    # 详细diff统计对比
    print_detailed_diff_comparison(stats1, stats2, model1_name, model2_name)


def print_detailed_diff_comparison(stats1, stats2, model1_name, model2_name):
    """打印两个模型的详细diff统计对比"""
    
    print(f"\n[DETAILED DIFF COMPARISON - PLAUSIBLE Patches]")
    print("=" * 120)
    
    diff_stats1 = stats1.diff_stats
    diff_stats2 = stats2.diff_stats
    
    if diff_stats1['patch_count'] == 0 or diff_stats2['patch_count'] == 0:
        print("[WARNING] One or both models have no PLAUSIBLE patches to compare")
        return
    
    print(f"{'Metric':<40} | {model1_name:>15} | {model2_name:>15} | {'Diff':>15} | {'Δ%':>10}")
    print("-" * 120)
    
    # 定义指标
    metrics = [
        ('Patch count', 'patch_count', False),
        ('Avg hunks', 'total_hunks', True),
        ('Avg added lines', 'total_added', True),
        ('Avg deleted lines', 'total_deleted', True),
        ('Avg total changed lines', 'total_changed_lines', True),
        ('Avg added tokens', 'total_added_tokens', True),
        ('Avg deleted tokens', 'total_deleted_tokens', True),
        ('Avg total changed tokens', 'total_changed_tokens', True),
        ('Avg edit distance', 'total_edit_distance', True),
        ('Avg edit similarity (%)', 'total_edit_similarity', True),
        ('Avg normalized edit distance', 'total_norm_edit_distance', True),
        ('Avg preserved ratio (%)', 'total_preserved', True)
    ]
    
    for metric_name, key, show_diff in metrics:
        if key == 'patch_count':
            val1 = diff_stats1[key]
            val2 = diff_stats2[key]
            print(f"{metric_name:<40} | {val1:>15.0f} | {val2:>15.0f} | {'-':>15} | {'-':>10}")
        else:
            # 计算平均值
            val1 = diff_stats1[key] / diff_stats1['patch_count']
            val2 = diff_stats2[key] / diff_stats2['patch_count']
            
            if show_diff:
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else 0
                
                color = '\033[92m' if diff > 0 else '\033[91m' if diff < 0 else '\033[0m'
                
                print(f"{metric_name:<40} | {val1:>15.2f} | {val2:>15.2f} | {color}{diff:>+15.2f}\033[0m | {color}{pct_change:>+9.2f}%\033[0m")
            else:
                print(f"{metric_name:<40} | {val1:>15.2f} | {val2:>15.2f} | {'-':>15} | {'-':>10}")
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description='Calculate Pass@k and diff statistics for code repair models')
    parser.add_argument('--model', type=str, required=True, help='Model name to analyze')
    parser.add_argument('--compare', type=str, default=None, help='Second model name for comparison')
    parser.add_argument('--force-recalc', action='store_true', help='Force recalculation even if cached results exist')
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        print(f"\n[COMPARE MODE] Comparing {args.model} vs {args.compare}")
        if args.force_recalc:
            print("[FORCE RECALC] Will recalculate both models")
        
        stats1 = load_model_results(args.model, force_recalc=args.force_recalc)
        stats2 = load_model_results(args.compare, force_recalc=args.force_recalc)
        
        if stats1 is None or stats2 is None:
            print("[ERROR] Failed to load one or both models")
            return
        
        print_comparison_results(stats1, stats2, args.model, args.compare)
        
    else:
        # 单模型模式
        print(f"\n[SINGLE MODEL MODE] Analyzing {args.model}")
        if args.force_recalc:
            print("[FORCE RECALC] Will recalculate statistics")
        
        stats = load_model_results(args.model, force_recalc=args.force_recalc)
        
        if stats is None:
            print("[ERROR] Failed to load model results")
            return
        
        print_single_model_results(stats, args.model)


if __name__ == "__main__":
    main()
