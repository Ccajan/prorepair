#!/usr/bin/env python3
"""
差异统计脚本 - 统计patch的hunks、lines、tokens等详细指标
用法：
  python stats_diff.py -m model_id                          # 统计单个模型
  python stats_diff.py -m model_id --compare_with model_id2 # 对比两个模型
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
    # 如果没有tqdm，定义一个简单的替代
    def tqdm(iterable, **kwargs):
        return iterable


def count_tokens(text):
    """简单的token计数，使用空格和常见分隔符分割"""
    if not text:
        return 0
    # 按空格、标点等分割来近似统计token
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
            # 插入、删除、替换的代价
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
    
    # 计算保留比例
    matcher = difflib.SequenceMatcher(None, buggy_lines, fix_lines)
    preserved = sum(block.size for block in matcher.get_matching_blocks())
    preserved_ratio = (preserved / len(buggy_lines) * 100) if buggy_lines else 0.0
    
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


def load_buggy_code(bug_id):
    """从dataset加载buggy代码"""
    dataset_file = os.path.join('defects4j/dataset', f'{bug_id}.json')
    try:
        with open(dataset_file) as f:
            dataset_info = json.load(f)
            return dataset_info.get('buggy', '')
    except Exception as e:
        print(f"[WARNING] Failed to load buggy code for {bug_id}: {e}")
        return None


def load_results_with_diff(model_id, min_patches=1):
    """
    加载验证结果并计算diff统计
    返回：(results_dict, stats_dict)
    """
    results_dir = os.path.join('defects4j/results', model_id)
    bug_results = {}
    diff_stats_list = []
    
    print(f"[INFO] Loading results for {model_id}...")
    
    # 优先从 *-validated.jsonl 加载
    validated_files = list(glob.glob(os.path.join(results_dir, '*-validated.jsonl')))
    
    if validated_files:
        print(f"[INFO] Found {len(validated_files)} validated files")
        for validated_file in tqdm(validated_files, desc="Processing validated files", disable=not HAS_TQDM):
            bug_id = os.path.basename(validated_file).replace('-validated.jsonl', '')
            try:
                with open(validated_file, 'r') as f:
                    results = json.load(f)
                
                if len(results) < min_patches:
                    continue
                
                bug_results[bug_id] = results
                
                # 加载buggy代码
                buggy_code = load_buggy_code(bug_id)
                if not buggy_code:
                    continue
                
                # 为每个PLAUSIBLE patch计算diff统计
                for patch in results:
                    if patch.get('patch_status') == 'PLAUSIBLE':
                        diff_stats = calc_detailed_diff_stats(buggy_code, patch.get('patch_code', ''))
                        diff_stats['bug_id'] = bug_id
                        diff_stats_list.append(diff_stats)
                        
            except Exception as e:
                if HAS_TQDM:
                    tqdm.write(f"[WARNING] Failed to process {validated_file}: {e}")
                else:
                    print(f"[WARNING] Failed to process {validated_file}: {e}")
    
    # 如果没有validated文件，尝试从.result文件加载
    if not bug_results:
        print(f"[INFO] No validated files found, trying .result files...")
        fixed_dirs = sorted(glob.glob(os.path.join(results_dir, 'fixed*')))
        
        # 收集所有.result文件
        all_result_files = []
        for fixed_dir in fixed_dirs:
            if not os.path.isdir(fixed_dir):
                continue
            all_result_files.extend(glob.glob(os.path.join(fixed_dir, '*.json.result')))
        
        print(f"[INFO] Found {len(all_result_files)} result files across {len(fixed_dirs)} directories")
        
        # 使用进度条加载所有result文件
        for result_file in tqdm(all_result_files, desc="Loading result files", disable=not HAS_TQDM):
            bug_id = os.path.basename(result_file).replace('.json.result', '')
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                if bug_id not in bug_results:
                    bug_results[bug_id] = results
                else:
                    bug_results[bug_id].extend(results)
                    
            except Exception as e:
                if HAS_TQDM:
                    tqdm.write(f"[WARNING] Failed to load {result_file}: {e}")
                else:
                    print(f"[WARNING] Failed to load {result_file}: {e}")
        
        # 处理从.result文件加载的结果
        print(f"[INFO] Processing {len(bug_results)} bugs...")
        bugs_to_process = list(bug_results.items())
        for bug_id, results in tqdm(bugs_to_process, desc="Calculating diff stats", disable=not HAS_TQDM):
            if len(results) < min_patches:
                del bug_results[bug_id]
                continue
            
            buggy_code = load_buggy_code(bug_id)
            if not buggy_code:
                continue
            
            for patch in results:
                if patch.get('patch_status') == 'PLAUSIBLE':
                    diff_stats = calc_detailed_diff_stats(buggy_code, patch.get('patch_code', ''))
                    diff_stats['bug_id'] = bug_id
                    diff_stats_list.append(diff_stats)
    
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
        'avg_preserved_ratio': sum(s['preserved_ratio'] for s in diff_stats_list) / n,
        'detail_list': diff_stats_list
    }


def print_single_model_stats(model_id, stats):
    """打印单个模型的统计结果"""
    if not stats:
        print(f"[ERROR] No PLAUSIBLE patches found for {model_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"[DIFF STATISTICS] {model_id}")
    print(f"{'='*80}")
    print(f"Total PLAUSIBLE patches analyzed: {stats['patch_count']}")
    print(f"\n{'Metric':<30} | {'Average':<15}")
    print(f"{'-'*80}")
    print(f"{'Hunks per patch':<30} | {stats['avg_hunks']:<15.2f}")
    print(f"{'Added lines per patch':<30} | {stats['avg_added_lines']:<15.2f}")
    print(f"{'Deleted lines per patch':<30} | {stats['avg_deleted_lines']:<15.2f}")
    print(f"{'Total changed lines per patch':<30} | {stats['avg_total_changed_lines']:<15.2f}")
    print(f"{'Added tokens per patch':<30} | {stats['avg_added_tokens']:<15.2f}")
    print(f"{'Deleted tokens per patch':<30} | {stats['avg_deleted_tokens']:<15.2f}")
    print(f"{'Total changed tokens per patch':<30} | {stats['avg_total_changed_tokens']:<15.2f}")
    print(f"{'Edit distance per patch':<30} | {stats['avg_edit_distance']:<15.2f}")
    print(f"{'Edit similarity':<30} | {stats['avg_edit_similarity']:<15.2f}%")
    print(f"{'Code preserved ratio':<30} | {stats['avg_preserved_ratio']:<15.2f}%")
    print(f"{'='*80}")


def print_comparison_stats(model_id1, model_id2, stats1, stats2):
    """打印两个模型的对比统计"""
    if not stats1 or not stats2:
        print("[ERROR] Cannot compare: one or both models have no PLAUSIBLE patches")
        return
    
    print(f"\n{'='*100}")
    print(f"[DIFF STATISTICS COMPARISON] {model_id1} vs {model_id2}")
    print(f"{'='*100}")
    print(f"{'Metric':<35} | {model_id1:>15} | {model_id2:>15} | {'Diff':>15} | {'Δ%':>10}")
    print(f"{'-'*100}")
    
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
        ('Avg edit similarity (%)', 'avg_edit_similarity', True, True),
        ('Avg preserved ratio (%)', 'avg_preserved_ratio', True, True)
    ]
    
    for metric_name, key, show_diff, higher_is_better in metrics:
        val1 = stats1[key]
        val2 = stats2[key]
        
        if show_diff:
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            # higher_is_better=True: 增加是好的（如相似度）
            # higher_is_better=False: 减少是好的（如编辑距离、变化行数等）
            is_good = (diff > 0) if higher_is_better else (diff < 0)
            color = '\033[92m' if is_good else '\033[91m' if diff != 0 else '\033[0m'
            
            print(f"{metric_name:<35} | {val1:>15.2f} | {val2:>15.2f} | {color}{diff:>+15.2f}\033[0m | {color}{pct_change:>+9.2f}%\033[0m")
        else:
            print(f"{metric_name:<35} | {val1:>15.0f} | {val2:>15.0f} | {'-':>15} | {'-':>10}")
    
    print(f"{'='*100}")
    
    # 打印详细分布
    print_distribution_comparison(model_id1, model_id2, stats1, stats2)


def print_distribution_comparison(model_id1, model_id2, stats1, stats2):
    """打印分布对比"""
    print(f"\n[DISTRIBUTION COMPARISON]")
    print(f"{'-'*100}")
    
    # Hunks分布
    print(f"\nHunks distribution:")
    hunks1 = [s['hunks'] for s in stats1['detail_list']]
    hunks2 = [s['hunks'] for s in stats2['detail_list']]
    print_metric_distribution('Hunks', model_id1, hunks1, model_id2, hunks2)
    
    # Total changed lines分布
    print(f"\nTotal changed lines distribution:")
    lines1 = [s['total_changed_lines'] for s in stats1['detail_list']]
    lines2 = [s['total_changed_lines'] for s in stats2['detail_list']]
    print_metric_distribution('Lines', model_id1, lines1, model_id2, lines2)
    
    # Total changed tokens分布
    print(f"\nTotal changed tokens distribution:")
    tokens1 = [s['total_changed_tokens'] for s in stats1['detail_list']]
    tokens2 = [s['total_changed_tokens'] for s in stats2['detail_list']]
    print_metric_distribution('Tokens', model_id1, tokens1, model_id2, tokens2)
    
    # Edit distance分布
    print(f"\nEdit distance distribution:")
    edit_dist1 = [s['edit_distance'] for s in stats1['detail_list']]
    edit_dist2 = [s['edit_distance'] for s in stats2['detail_list']]
    print_metric_distribution('EditDist', model_id1, edit_dist1, model_id2, edit_dist2)
    
    # Edit similarity分布
    print(f"\nEdit similarity distribution:")
    edit_sim1 = [s['edit_similarity'] for s in stats1['detail_list']]
    edit_sim2 = [s['edit_similarity'] for s in stats2['detail_list']]
    print_metric_distribution('EditSim', model_id1, edit_sim1, model_id2, edit_sim2)


def print_metric_distribution(metric_name, model_id1, values1, model_id2, values2):
    """打印指标的分布统计"""
    from statistics import median, stdev
    
    if not values1 or not values2:
        return
    
    print(f"  {model_id1:15}: min={min(values1):6.1f}, max={max(values1):6.1f}, median={median(values1):6.1f}, stdev={stdev(values1) if len(values1) > 1 else 0:6.1f}")
    print(f"  {model_id2:15}: min={min(values2):6.1f}, max={max(values2):6.1f}, median={median(values2):6.1f}, stdev={stdev(values2) if len(values2) > 1 else 0:6.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='统计patch的详细diff指标（hunks、lines、tokens）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 统计单个模型
  python stats_diff.py -m deepseek-r1-distill-qwen-32b
  
  # 对比两个模型
  python stats_diff.py -m model1 --compare_with model2
  
  # 对比时只考虑有10个以上patch的bug
  python stats_diff.py -m model1 --compare_with model2 --min_patches 10
        """
    )
    parser.add_argument('-m', '--model_id', type=str, required=True, help='主模型ID')
    parser.add_argument('--compare_with', type=str, default=None, help='对比另一个模型的结果')
    parser.add_argument('--min_patches', type=int, default=1, help='最小patch数量要求（默认1）')
    
    args = parser.parse_args()
    
    # 提示安装tqdm以获得进度条
    if not HAS_TQDM:
        print("[提示] 安装 tqdm 可以显示进度条: pip install tqdm\n")
    
    if args.compare_with:
        # 对比模式
        print(f"\n[对比模式] 对比 {args.model_id} 和 {args.compare_with}")
        print(f"[筛选条件] 只统计有 >={args.min_patches} 个patch的bug\n")
        
        bug_results1, stats1, details1 = load_results_with_diff(args.model_id, args.min_patches)
        bug_results2, stats2, details2 = load_results_with_diff(args.compare_with, args.min_patches)
        
        # 只统计共同的bug
        common_bugs = set(bug_results1.keys()) & set(bug_results2.keys())
        print(f"\n[INFO] 共同bug数量: {len(common_bugs)}")
        
        # 重新计算只包含共同bug的统计
        details1_filtered = [d for d in details1 if d['bug_id'] in common_bugs]
        details2_filtered = [d for d in details2 if d['bug_id'] in common_bugs]
        
        if details1_filtered and details2_filtered:
            stats1_filtered = aggregate_stats(details1_filtered)
            stats2_filtered = aggregate_stats(details2_filtered)
            print_comparison_stats(args.model_id, args.compare_with, stats1_filtered, stats2_filtered)
        else:
            print("[ERROR] 没有足够的共同PLAUSIBLE patches进行对比")
    else:
        # 单模型统计模式
        print(f"\n[统计模式] 统计 {args.model_id}")
        print(f"[筛选条件] 只统计有 >={args.min_patches} 个patch的bug\n")
        
        bug_results, stats, details = load_results_with_diff(args.model_id, args.min_patches)
        
        if stats:
            print_single_model_stats(args.model_id, stats)
            
            # 打印详细的分布信息
            if details:
                print(f"\n[DETAILED DISTRIBUTION]")
                print(f"{'-'*80}")
                from statistics import median, stdev
                
                hunks = [s['hunks'] for s in details]
                lines = [s['total_changed_lines'] for s in details]
                tokens = [s['total_changed_tokens'] for s in details]
                edit_dist = [s['edit_distance'] for s in details]
                edit_sim = [s['edit_similarity'] for s in details]
                
                print(f"Hunks:       min={min(hunks):8.1f}, max={max(hunks):8.1f}, median={median(hunks):8.1f}, stdev={stdev(hunks) if len(hunks) > 1 else 0:8.1f}")
                print(f"Lines:       min={min(lines):8.1f}, max={max(lines):8.1f}, median={median(lines):8.1f}, stdev={stdev(lines) if len(lines) > 1 else 0:8.1f}")
                print(f"Tokens:      min={min(tokens):8.1f}, max={max(tokens):8.1f}, median={median(tokens):8.1f}, stdev={stdev(tokens) if len(tokens) > 1 else 0:8.1f}")
                print(f"Edit Dist:   min={min(edit_dist):8.1f}, max={max(edit_dist):8.1f}, median={median(edit_dist):8.1f}, stdev={stdev(edit_dist) if len(edit_dist) > 1 else 0:8.1f}")
                print(f"Edit Sim(%): min={min(edit_sim):8.1f}, max={max(edit_sim):8.1f}, median={median(edit_sim):8.1f}, stdev={stdev(edit_sim) if len(edit_sim) > 1 else 0:8.1f}")
                print(f"{'='*80}")
        else:
            print(f"[ERROR] 没有找到PLAUSIBLE patches")


if __name__ == '__main__':
    main()

