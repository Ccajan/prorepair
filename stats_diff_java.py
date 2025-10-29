#!/usr/bin/env python3
"""
Java Benchmark 差异统计脚本 - 统计patch的hunks、lines、tokens等详细指标
用法：
  python stats_diff_java.py -m model_id                          # 统计单个模型
  python stats_diff_java.py -m model_id --compare_with model_id2 # 对比两个模型
"""

import os
import json
import glob
import argparse
import difflib
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def count_tokens(text):
    """简单的token计数，使用空格和常见分隔符分割"""
    if not text:
        return 0
    import re
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
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calc_detailed_diff_stats(buggy, fix):
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
    }


def load_results_with_diff(model_id, base_path="./evalrepair-java-res"):
    """
    加载Java验证结果并计算diff统计
    返回：(bug_results, stats_summary, diff_stats_list)
    """
    results_base = os.path.join(base_path, model_id)
    buggy_base = os.path.join(base_path, "buggy")
    
    bug_results = {}  # {bug_name: [result0, result1, ...]}
    diff_stats_list = []
    
    print(f"[INFO] Loading results for {model_id}...")
    print(f"[INFO] Results path: {results_base}")
    
    # 遍历fixed0-fixed9目录
    for id in range(10):
        directory_path = os.path.join(results_base, f"fixed{id}")
        if not os.path.exists(directory_path):
            continue
        
        java_files = sorted(glob.glob(os.path.join(directory_path, '*.java')))
        print(f"[INFO] Processing fixed{id}: {len(java_files)} files")
        
        for file_path in tqdm(java_files, desc=f"Processing fixed{id}", disable=not HAS_TQDM):
            name = os.path.basename(file_path).replace('.java', '')
            
            # 初始化bug结果
            if name not in bug_results:
                bug_results[name] = []
            
            # 读取结果
            result_file = file_path + '.result'
            if not os.path.exists(result_file):
                continue
            
            try:
                with open(result_file, 'r') as f:
                    ret = int(f.read().strip())
                bug_results[name].append(ret)
                
                # 如果是成功的补丁，计算diff统计
                if ret == 0:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        fixed_code = f.read()
                    
                    buggy_file = os.path.join(buggy_base, f"{name}.java")
                    if os.path.exists(buggy_file):
                        with open(buggy_file, 'r', encoding='utf-8') as f:
                            buggy_code = f.read()
                        
                        diff_stats = calc_detailed_diff_stats(buggy_code, fixed_code)
                        diff_stats['bug_name'] = name
                        diff_stats['patch_id'] = id
                        diff_stats_list.append(diff_stats)
            except Exception as e:
                if HAS_TQDM:
                    tqdm.write(f"[WARNING] Failed to process {file_path}: {e}")
                else:
                    print(f"[WARNING] Failed to process {file_path}: {e}")
    
    # 汇总统计
    stats_summary = aggregate_stats(diff_stats_list)
    
    return bug_results, stats_summary, diff_stats_list


def aggregate_stats(diff_stats_list):
    """汇总统计信息"""
    if not diff_stats_list:
        return None
    
    n = len(diff_stats_list)
    
    return {
        'patch_count': n,
        'avg_hunks': sum(s['hunks'] for s in diff_stats_list) / n,
        'avg_added_lines': sum(s['added_lines'] for s in diff_stats_list) / n,
        'avg_deleted_lines': sum(s['deleted_lines'] for s in diff_stats_list) / n,
        'avg_total_changed_lines': sum(s['total_changed_lines'] for s in diff_stats_list) / n,
        'avg_added_tokens': sum(s['added_tokens'] for s in diff_stats_list) / n,
        'avg_deleted_tokens': sum(s['deleted_tokens'] for s in diff_stats_list) / n,
        'avg_total_changed_tokens': sum(s['total_changed_tokens'] for s in diff_stats_list) / n,
        'avg_edit_distance': sum(s['edit_distance'] for s in diff_stats_list) / n,
        'avg_edit_similarity': sum(s['edit_similarity'] for s in diff_stats_list) / n,
        'avg_norm_edit_distance': sum(s['norm_edit_distance'] for s in diff_stats_list) / n,
        'detail_list': diff_stats_list
    }


def print_single_model_stats(model_id, stats):
    """打印单个模型的统计结果"""
    if not stats:
        print(f"[ERROR] No successful patches found for {model_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"[DIFF STATISTICS] {model_id}")
    print(f"{'='*80}")
    print(f"Total successful patches analyzed: {stats['patch_count']}")
    print(f"\n{'Metric':<35} | {'Average':<15}")
    print(f"{'-'*80}")
    print(f"{'Hunks per patch':<35} | {stats['avg_hunks']:<15.2f}")
    print(f"{'Added lines per patch':<35} | {stats['avg_added_lines']:<15.2f}")
    print(f"{'Deleted lines per patch':<35} | {stats['avg_deleted_lines']:<15.2f}")
    print(f"{'Total changed lines per patch':<35} | {stats['avg_total_changed_lines']:<15.2f}")
    print(f"{'Added tokens per patch':<35} | {stats['avg_added_tokens']:<15.2f}")
    print(f"{'Deleted tokens per patch':<35} | {stats['avg_deleted_tokens']:<15.2f}")
    print(f"{'Total changed tokens per patch':<35} | {stats['avg_total_changed_tokens']:<15.2f}")
    print(f"{'Edit distance per patch':<35} | {stats['avg_edit_distance']:<15.2f}")
    print(f"{'Normalized edit distance':<35} | {stats['avg_norm_edit_distance']:<15.4f}")
    print(f"{'Edit similarity':<35} | {stats['avg_edit_similarity']:<15.2f}%")
    print(f"{'='*80}")


def print_comparison_stats(model_id1, model_id2, stats1, stats2):
    """打印两个模型的对比统计"""
    if not stats1 or not stats2:
        print("[ERROR] Cannot compare: one or both models have no successful patches")
        return
    
    print(f"\n{'='*110}")
    print(f"[DIFF STATISTICS COMPARISON] {model_id1} vs {model_id2}")
    print(f"{'='*110}")
    print(f"{'Metric':<35} | {model_id1:>20} | {model_id2:>20} | {'Diff':>15} | {'Δ%':>10}")
    print(f"{'-'*110}")
    
    metrics = [
        ('Patch count', 'patch_count', False, False),
        ('Avg hunks', 'avg_hunks', True, False),
        ('Avg added lines', 'avg_added_lines', True, False),
        ('Avg deleted lines', 'avg_deleted_lines', True, False),
        ('Avg total changed lines', 'avg_total_changed_lines', True, False),
        ('Avg added tokens', 'avg_added_tokens', True, False),
        ('Avg deleted tokens', 'avg_deleted_tokens', True, False),
        ('Avg total changed tokens', 'avg_total_changed_tokens', True, False),
        ('Avg edit distance', 'avg_edit_distance', True, False),
        ('Avg norm edit distance', 'avg_norm_edit_distance', True, False),
        ('Avg edit similarity (%)', 'avg_edit_similarity', True, True),
    ]
    
    for metric_name, key, show_diff, higher_is_better in metrics:
        val1 = stats1[key]
        val2 = stats2[key]
        
        if show_diff:
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            is_good = (diff > 0) if higher_is_better else (diff < 0)
            color = '\033[92m' if is_good else '\033[91m' if diff != 0 else '\033[0m'
            
            if key == 'avg_norm_edit_distance':
                print(f"{metric_name:<35} | {val1:>20.4f} | {val2:>20.4f} | {color}{diff:>+15.4f}\033[0m | {color}{pct_change:>+9.2f}%\033[0m")
            else:
                print(f"{metric_name:<35} | {val1:>20.2f} | {val2:>20.2f} | {color}{diff:>+15.2f}\033[0m | {color}{pct_change:>+9.2f}%\033[0m")
        else:
            print(f"{metric_name:<35} | {val1:>20.0f} | {val2:>20.0f} | {'-':>15} | {'-':>10}")
    
    print(f"{'='*110}")
    
    # 打印详细分布
    print_distribution_comparison(model_id1, model_id2, stats1, stats2)


def print_distribution_comparison(model_id1, model_id2, stats1, stats2):
    """打印分布对比"""
    from statistics import median, stdev
    
    print(f"\n[DISTRIBUTION COMPARISON]")
    print(f"{'-'*100}")
    
    metrics = [
        ('Hunks', 'hunks'),
        ('Total changed lines', 'total_changed_lines'),
        ('Total changed tokens', 'total_changed_tokens'),
        ('Edit distance', 'edit_distance'),
        ('Edit similarity', 'edit_similarity'),
    ]
    
    for metric_name, key in metrics:
        print(f"\n{metric_name} distribution:")
        values1 = [s[key] for s in stats1['detail_list']]
        values2 = [s[key] for s in stats2['detail_list']]
        
        if values1 and values2:
            print(f"  {model_id1:20}: min={min(values1):8.1f}, max={max(values1):8.1f}, median={median(values1):8.1f}, stdev={stdev(values1) if len(values1) > 1 else 0:8.1f}")
            print(f"  {model_id2:20}: min={min(values2):8.1f}, max={max(values2):8.1f}, median={median(values2):8.1f}, stdev={stdev(values2) if len(values2) > 1 else 0:8.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='统计Java benchmark patch的详细diff指标（hunks、lines、tokens）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 统计单个模型
  python stats_diff_java.py -m model_name
  
  # 对比两个模型
  python stats_diff_java.py -m model1 --compare_with model2
  
  # 指定自定义路径
  python stats_diff_java.py -m model1 --base_path ./evalrepair-java-res
        """
    )
    parser.add_argument('-m', '--model_id', type=str, required=True, help='主模型ID')
    parser.add_argument('--compare_with', type=str, default=None, help='对比另一个模型的结果')
    parser.add_argument('--base_path', type=str, default='./evalrepair-java-res', 
                        help='Java benchmark基础路径')
    
    args = parser.parse_args()
    
    if not HAS_TQDM:
        print("[提示] 安装 tqdm 可以显示进度条: pip install tqdm\n")
    
    if args.compare_with:
        # 对比模式
        print(f"\n[对比模式] 对比 {args.model_id} 和 {args.compare_with}")
        
        bug_results1, stats1, details1 = load_results_with_diff(args.model_id, args.base_path)
        bug_results2, stats2, details2 = load_results_with_diff(args.compare_with, args.base_path)
        
        # 找共同的bug
        common_bugs = set(bug_results1.keys()) & set(bug_results2.keys())
        print(f"\n[INFO] 共同bug数量: {len(common_bugs)}")
        
        # 重新计算只包含共同bug的统计
        details1_filtered = [d for d in details1 if d['bug_name'] in common_bugs]
        details2_filtered = [d for d in details2 if d['bug_name'] in common_bugs]
        
        if details1_filtered and details2_filtered:
            stats1_filtered = aggregate_stats(details1_filtered)
            stats2_filtered = aggregate_stats(details2_filtered)
            print_comparison_stats(args.model_id, args.compare_with, stats1_filtered, stats2_filtered)
        else:
            print("[ERROR] 没有足够的共同成功patches进行对比")
    else:
        # 单模型统计模式
        print(f"\n[统计模式] 统计 {args.model_id}")
        
        bug_results, stats, details = load_results_with_diff(args.model_id, args.base_path)
        
        if stats:
            print_single_model_stats(args.model_id, stats)
            
            # 打印详细的分布信息
            if details:
                from statistics import median, stdev
                
                print(f"\n[DETAILED DISTRIBUTION]")
                print(f"{'-'*80}")
                
                hunks = [s['hunks'] for s in details]
                lines = [s['total_changed_lines'] for s in details]
                tokens = [s['total_changed_tokens'] for s in details]
                edit_dist = [s['edit_distance'] for s in details]
                edit_sim = [s['edit_similarity'] for s in details]
                print(f"Hunks:         min={min(hunks):8.1f}, max={max(hunks):8.1f}, median={median(hunks):8.1f}, stdev={stdev(hunks) if len(hunks) > 1 else 0:8.1f}")
                print(f"Lines:         min={min(lines):8.1f}, max={max(lines):8.1f}, median={median(lines):8.1f}, stdev={stdev(lines) if len(lines) > 1 else 0:8.1f}")
                print(f"Tokens:        min={min(tokens):8.1f}, max={max(tokens):8.1f}, median={median(tokens):8.1f}, stdev={stdev(tokens) if len(tokens) > 1 else 0:8.1f}")
                print(f"Edit Dist:     min={min(edit_dist):8.1f}, max={max(edit_dist):8.1f}, median={median(edit_dist):8.1f}, stdev={stdev(edit_dist) if len(edit_dist) > 1 else 0:8.1f}")
                print(f"Edit Sim(%):   min={min(edit_sim):8.1f}, max={max(edit_sim):8.1f}, median={median(edit_sim):8.1f}, stdev={stdev(edit_sim) if len(edit_sim) > 1 else 0:8.1f}")
                print(f"{'='*80}")
        else:
            print(f"[ERROR] 没有找到成功的patches")


if __name__ == '__main__':
    main()



