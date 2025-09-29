import subprocess
import os, glob
import sys
import json
import difflib

def calc_diff_stats(buggy, fix):
    """计算补丁的diff统计信息，比较错误代码和修复后的代码"""
    if not buggy or not fix:  # 处理空输入
        return {'added_lines': 0, 'deleted_lines': 0, 'preserved_ratio': 0.0}
    
    buggy_lines = buggy.strip().splitlines()
    fix_lines = fix.strip().splitlines()
    
    # 计算增删行数
    diff = list(difflib.ndiff(buggy_lines, fix_lines))
    added = sum(1 for line in diff if line.startswith('+ '))
    deleted = sum(1 for line in diff if line.startswith('- '))
    
    # 计算保留比例 - 改进：排除虚拟的结束块
    matcher = difflib.SequenceMatcher(None, buggy_lines, fix_lines)
    matching_blocks = matcher.get_matching_blocks()
    # 排除最后一个虚拟块 (len(a), len(b), 0)
    preserved = sum(block.size for block in matching_blocks[:-1])
    preserved_ratio = preserved / len(buggy_lines) if buggy_lines else 0.0
    
    return {
        'added_lines': added,
        'deleted_lines': deleted,
        'preserved_ratio': round(preserved_ratio * 100, 2)
    }

def print_diff_stats(diff_stats):
    """打印补丁的差异统计信息"""
    print(f"[DIFF STATS] Added: {diff_stats['added_lines']}, Deleted: {diff_stats['deleted_lines']}, Preserved: {diff_stats['preserved_ratio']}%")

def test(id, name, tag):
    command = "cd evalrepair-java && bash test.sh " + str(id) + " " + name + " " + tag
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("Standard Output:", stdout.decode())
    print("Standard Error:", stderr.decode())
    print(process.returncode)
    return [process.returncode, str(stdout)]

ac = {}
ac5 = {}
ac1 = {}
bug = 0

ss = 0

# 添加diff统计
diff_stats = {
    'total_added': 0,
    'total_deleted': 0,
    'total_preserved': 0,
    'patch_count': 0,
    'preservation_distribution': {
        'high': 0,    # >95%
        'medium': 0,  # 80-95%
        'low': 0      # <80%
    }
}

for id in range(10):
    directory_path = f"./evalrepair-java-res/{sys.argv[1]}/fixed{id}/"
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for file_path in sorted(glob.glob(os.path.join(directory_path, '*.java')), reverse=False):
            print(file_path)
            name = file_path.split('/')[-1].split('.')[0]
            if sys.argv[-1] == "rejudge":
                ret, detail = test(id, name, sys.argv[1])
            else:
                ret = (int)(open(file_path + '.result', 'r').read())
            if ret == 0:
                ac[name] = ac.get(name, 0) + 1
                if id < 5:
                    ac5[name] = ac5.get(name, 0) + 1
                if id == 0:
                    ac1[name] = ac1.get(name, 0) + 1
                
                # 添加diff统计（只对成功修复的代码）
                try:
                    # 读取修复后的代码
                    with open(file_path, 'r', encoding='utf-8') as f:
                        fixed_code = f.read()
                    
                    # 读取原始buggy代码
                    buggy_file = f"./evalrepair-java-res/buggy/{name}.java"
                    if os.path.exists(buggy_file):
                        with open(buggy_file, 'r', encoding='utf-8') as f:
                            buggy_code = f.read()
                        
                        # 计算diff统计
                        current_diff = calc_diff_stats(buggy_code, fixed_code)
                        print_diff_stats(current_diff)
                        
                        # 更新全局统计
                        diff_stats['total_added'] += current_diff['added_lines']
                        diff_stats['total_deleted'] += current_diff['deleted_lines']
                        diff_stats['total_preserved'] += current_diff['preserved_ratio']
                        diff_stats['patch_count'] += 1
                        
                        # 更新保留率分布
                        ratio = current_diff['preserved_ratio']
                        if ratio > 95:
                            diff_stats['preservation_distribution']['high'] += 1
                        elif ratio > 80:
                            diff_stats['preservation_distribution']['medium'] += 1
                        else:
                            diff_stats['preservation_distribution']['low'] += 1
                    else:
                        print(f"[WARNING] Buggy file not found: {buggy_file}")
                except Exception as e:
                    print(f"[ERROR] Failed to calculate diff stats for {name}: {e}")

print('TOP-10:', len(ac) / 163 * 100, 'TOP-5:', len(ac5) / 163 * 100, 'TOP-1:', len(ac1) / 163 * 100)

# 输出diff统计信息
if diff_stats['patch_count'] > 0:
    patch_count = diff_stats['patch_count']
    dist = diff_stats['preservation_distribution']
    
    print("\n[PATCH MODIFICATION STATISTICS]")
    print(f"- Total successful patches analyzed:    {patch_count}")
    print(f"- Average lines added per patch:        {diff_stats['total_added']/patch_count:.1f}")
    print(f"- Average lines deleted per patch:      {diff_stats['total_deleted']/patch_count:.1f}")
    print(f"- Average code preserved:               {diff_stats['total_preserved']/patch_count:.1f}%")
    
    print("\nDistribution of code preservation ratio:")
    total = sum(dist.values())
    if total > 0:
        print(f"- Minimal change   (>95% preserved):   {dist['high']:3d} patches ({dist['high']/total*100:5.1f}%)")
        print(f"- Moderate change (80-95% preserved):  {dist['medium']:3d} patches ({dist['medium']/total*100:5.1f}%)")
        print(f"- Major change     (<80% preserved):   {dist['low']:3d} patches ({dist['low']/total*100:5.1f}%)")
else:
    print("\n[PATCH MODIFICATION STATISTICS]")
    print("- No successful patches found for diff analysis")
