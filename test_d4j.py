import os
import sys
import time
import json
import shutil
import random
import psutil
import difflib
import argparse
import threading
import traceback
import subprocess
import multiprocessing
from pathlib import Path
import concurrent.futures as cf
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import glob

ROOT_PATH = '/tmp/playground/'
DEFECTS4J_PATH = '/home/barty/research/defects4j/framework/bin/defects4j'
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))  # 保存工作区根目录的绝对路径

# 设置 Perl 环境变量
import os as _os
_perl5lib = f"/home/barty/perl5/lib/perl5{':' + _os.environ['PERL5LIB'] if 'PERL5LIB' in _os.environ else ''}"
_os.environ['PERL5LIB'] = _perl5lib
_os.environ['PATH'] = f"{_os.path.dirname(DEFECTS4J_PATH)}:{_os.environ.get('PATH', '')}"

def clean_tmp_folder(tmp_dir):
    if os.path.isdir(tmp_dir) and tmp_dir.startswith(ROOT_PATH):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


def strip_lines(lines):
    return [line.strip() for line in lines]


def encoding_check(encoding_check_file_path):
    if not os.path.exists(encoding_check_file_path):
        print(f"[ERROR] File does not exist: {encoding_check_file_path}")
        return 'utf-8', None  # 返回默认编码和None作为内容
        
    file_content = None
    encoding_mode = 'utf-8'
    try:
        with open(encoding_check_file_path, 'r', encoding=encoding_mode) as f:
            file_content = f.read()
    except UnicodeDecodeError:
        encoding_mode = 'ISO-8859-1'
        with open(encoding_check_file_path, 'r', encoding=encoding_mode) as f:
            file_content = f.read()
    except Exception as e:
        print(f"[ERROR] read encoding_check FAILURE: {e}")
        return 'utf-8', None  # 返回默认编码和None作为内容
    return encoding_mode, file_content

def guess_source_dir(project_dir):
    """
    枚举可能的源码目录结构并匹配 org/ 开头的包路径。
    返回第一个匹配成功的目录（相对 project_dir）。
    """
    candidates = [
        "src",               # Lang 等
        "src/java",          # Cli
        "src/main/java",     # 一些 Gradle 项目
        "source",            # Chart
        "java",              # 极个别
        "",                  # 有可能 org 就直接在根目录
    ]
    for candidate in candidates:
        full_path = os.path.join(project_dir, candidate, "org")
        if os.path.exists(full_path):
            return candidate
    return None
def checkout_defects4j_project(current_bug, project_dir):
    project, bug_id = current_bug.split('-')
    command = f"{DEFECTS4J_PATH} checkout -p {project} -v {bug_id}b -w {project_dir}"
    print('[CHECKOUT]', command)

    # 执行 checkout 命令
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = p.communicate()

    if p.returncode != 0:
        print(f"[ERROR] Checkout failed with return code {p.returncode}")
        print(f"[ERROR] stdout: {stdout}")
        print(f"[ERROR] stderr: {stderr}")
        return False

    if not os.path.exists(project_dir):
        print(f"[ERROR] Project directory does not exist after checkout: {project_dir}")
        return False

    # 自动猜测源码目录
    src_dir_name = guess_source_dir(project_dir)
    if not src_dir_name:
        print(f"[ERROR] Could not find a valid source directory under {project_dir}")
        return False

    print(f"[SUCCESS] Checked out to {project_dir}, source dir: {src_dir_name}")
    return True


def monitor_memory(pid, interval, stop_event, max_memory_event):
    max_memory = 0
    try:
        main_proc = psutil.Process(pid)
        while not stop_event.is_set():
            procs = [main_proc] + main_proc.children(recursive=True)
            total_memory_usage = sum(proc.memory_info().rss for proc in procs if proc.is_running())
            max_memory = max(max_memory, total_memory_usage)
            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass
    max_memory_event[0] = max_memory / (1024 ** 3)


def command_with_timeout(cmd, timeout=90):
    max_memory_event = [None]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_memory, args=(process.pid, 1, stop_event, max_memory_event))
    try:
        monitor_thread.start()
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        ps_process = psutil.Process(process.pid)
        procs_kill = [ps_process] + ps_process.children(recursive=True)
        for proc in procs_kill:
            proc.kill()
        return 'TIMEOUT', 'TIMEOUT'
    finally:
        stop_event.set()
        monitor_thread.join()
        max_memory_usage = max_memory_event[0]
        if max_memory_usage and max_memory_usage > 6:
            print(f'[WARNING] MEMORY OCCUPIED {max_memory_usage:.2f} GB -- {cmd}')
    return stdout, stderr


def defects4j_test_suite(project_dir, timeout=1000):
    os.chdir(project_dir)
    out, err = command_with_timeout([DEFECTS4J_PATH, "test", "-r"], timeout)
    if "Compilation failed" in str(out):
        print("[FAIL] Compile tests for ", project_dir)
    return out, err


def defects4j_export_trigger(project_dir, timeout=90):
    os.chdir(project_dir)
    out, err = command_with_timeout([DEFECTS4J_PATH, "export", "-p", "tests.trigger"], timeout)
    return out, err


def defects4j_export_relevant(project_dir, timeout=90):
    os.chdir(project_dir)
    out, err = command_with_timeout([DEFECTS4J_PATH, "export", "-p", "tests.relevant"], timeout)
    return out, err


def defects4j_test_one(project_dir, test_case, timeout=100):
    os.chdir(project_dir)
    out, err = command_with_timeout([DEFECTS4J_PATH, "test", "-t", test_case], timeout)
    return out, err


def extract_d4j_result(err, out, val_stage):
    err_str, out_str = str(err), str(out)
    if 'TIMEOUT' in err_str or 'TIMEOUT' in out_str:
        correctness = 'TRIGGER_TIMEOUT' if val_stage == 'trigger' else 'RELEVANT_TIMEOUT'
    elif 'FAIL' in err_str or 'FAIL' in out_str:
        correctness = 'UNCOMPILABLE'
    elif "Failing tests: 0" in out_str:
        correctness = 'PLAUSIBLE'
    else:
        correctness = 'TRIGGER_ERROR' if val_stage == 'trigger' else 'RELEVANT_ERROR'
    return correctness




class ValTime:
    def __init__(self, val_start_time):
        self.val_start_timestamp = val_start_time
        
        self.val_init_time = 0
        self.val_overall_time = 0

        self.val_trigger_time = 0
        self.curr_trigger_time = 0
        self.trigger_start_timestamp = 0

        self.val_relevant_time = 0
        self.curr_relevant_time = 0
        self.relevant_start_timestamp = 0

        self.curr_overall_time = 0

    def set_init_time(self, init_timestamp):
        self.val_init_time = init_timestamp - self.val_start_timestamp

    def set_trigger_start_timestamp(self, trigger_start_timestamp):
        self.trigger_start_timestamp = trigger_start_timestamp
    
    def set_relevant_start_timestamp(self, relevant_start_timestamp):
        self.relevant_start_timestamp = relevant_start_timestamp

    def set_trigger_end_time(self, trigger_end_timestamp):
        self.curr_trigger_time = trigger_end_timestamp - self.trigger_start_timestamp
        self.val_trigger_time += self.curr_trigger_time
    
    def set_relevant_end_time(self, relevant_end_timestamp):
        self.curr_relevant_time = relevant_end_timestamp - self.relevant_start_timestamp
        self.val_relevant_time += self.curr_relevant_time
    
    def get_curr_overall_time(self):
        self.curr_overall_time = self.curr_trigger_time + self.curr_relevant_time
        return int(self.curr_overall_time)
    
    def set_overall_time(self, end_timestamp):
        self.val_overall_time = end_timestamp - self.val_start_timestamp
    
    def get_relevant_time(self):
        return int()
    
    def print_validation_time_info(self, curr_bug):
        print(f"[TIME INFO] PREPARE  = {int(self.val_init_time)}s")
        print(f"[TIME INFO] TRIGGER  = {int(self.val_trigger_time)}s")
        if self.val_relevant_time > 2:
            print(f"[TIME INFO] RELEVANT = {int(self.val_relevant_time)}s")
        print(f'[TIME INFO] TOTAL {curr_bug} -- {int(int(self.val_overall_time))}s')
        print('=' * 100)




class ValInfo():
    def __init__(self, candidate_patch, model_id):
        print(f"[DEBUG] Initializing ValInfo with: {candidate_patch[1].keys()}")
        self.unvrf_patches = candidate_patch
        self.curr_bug = candidate_patch[0]
        patch_info = candidate_patch[1]
        self.patches = patch_info['patches']
        self.model_id = model_id
        self.patch_info = {
            'loc': patch_info['loc'],
            'start': patch_info['start'],
            'end': patch_info['end'],
            'buggy': patch_info['buggy']  # 添加buggy字段
        }
        
        # 初始化其他属性
        self.patch_id = 0
        self.validated_result = []
        self.overall_patch_status = 'failure'

        # 按顺序调用初始化函数
        self.init_buggy_project()
        self.init_bug_status_info()
        self.init_extract_project_info()

    def init_buggy_project(self):
        self.validation_path = ROOT_PATH
        self.proj_dir = os.path.join(self.validation_path, self.curr_bug)
        clean_tmp_folder(self.proj_dir)

        self.val_result_path = os.path.join('defects4j/results/', self.model_id)
        checkout_defects4j_project(self.curr_bug, self.proj_dir)

    def init_extract_project_info(self):
        self.buggy_file_path = os.path.join(self.proj_dir, self.patch_info['loc'])
        self.encoding_mode, self.original_buggy_file_content = encoding_check(self.buggy_file_path)
        
        # 如果文件内容为None，说明文件不存在或读取失败
        if self.original_buggy_file_content is None:
            print(f"[ERROR] Failed to read or find file: {self.buggy_file_path}")
            return False
        
        self.backup_buggy_file_path = f'{self.buggy_file_path}.llm4apr_backup'
        try:
            shutil.copyfile(self.buggy_file_path, self.backup_buggy_file_path)
        except Exception as e:
            print(f"[ERROR] Failed to create backup file: {e}")
            return False
        
        return True
    
    
    def check_init_success(self):
        return len(self.failed_test_cases) > 0
    
    
    def patch_id_counter(self):
        self.patch_id += 1


    def update_patch_val_result(self, patch_validation_info):
        self.validated_result.append(patch_validation_info)

    
    def save_validation_results(self, done=False):
        if not done and len(self.validated_result) % 10 != 0:
            return
        filename = str(self.curr_bug) + '-validated.jsonl'
        log_file = os.path.join(self.val_result_path, filename)
        if not os.path.exists(self.val_result_path):
            os.makedirs(self.val_result_path, exist_ok=True)
        try:   
            with open(log_file, "w") as f: 
                json.dump(self.validated_result, f, indent=2)
        except Exception as e:
            print('[ERROR] write_results_to_file: ', e)  


    def init_bug_status_info(self):
        """初始化bug的测试状态信息"""
        print(f"[DEBUG] Initializing bug status for {self.curr_bug}")
        
        # 获取触发bug的测试用例
        out, err = defects4j_export_trigger(self.proj_dir)
        self.trigger_tests = []
        if out:
            self.trigger_tests = [line.strip() for line in str(out).split('\n') if line.strip()]
        
        # 获取相关的测试用例
        out, err = defects4j_export_relevant(self.proj_dir)
        self.relevant_tests = []
        if out:
            self.relevant_tests = [line.strip() for line in str(out).split('\n') if line.strip()]
        
        # 运行初始测试套件，确认bug状态
        init_out, init_err = defects4j_test_suite(self.proj_dir)
        self.failed_test_cases = []
        
        print(f"[DEBUG] Test suite output: {str(init_out)[:200]}")  # 打印前200个字符
        print(f"[DEBUG] Test suite error: {str(init_err)[:200]}")
        
        if init_out:
            self.failed_test_cases = [test.strip() for test in str(init_out).split(' - ')[1:]]
        
        print(f"[DEBUG] Found {len(self.trigger_tests)} trigger tests and {len(self.relevant_tests)} relevant tests")
        print(f"[DEBUG] Found {len(self.failed_test_cases)} failed test cases")



        
class PatchValidation():
    def __init__(self, patch_code):
        self.patch_code = patch_code
        self.patch_status = 'UNVERIFIED'
        self.failing_test = {
            'TRIGGER' : [],
            'RELEVANT' : [],
            'TIMEOUT' : [],
        }
        self.patch_val_info = {}

    def apply_patch(self, bug_info, proj_dir, encoding_mode):
        bug_path = bug_info['loc']
        start_loc = bug_info['start']
        end_loc = bug_info['end']
        patch = self.patch_code.strip()
        buggy_full_path = os.path.join(proj_dir, bug_path)        
        with open(buggy_full_path, 'r', encoding=encoding_mode) as file:
            orig_buggy_code = file.readlines()
        with open(buggy_full_path, 'w', encoding=encoding_mode, errors='ignore') as file:
            patched = False
            for idx, line in enumerate(orig_buggy_code):
                if start_loc - 1 <= idx <= end_loc -1:
                    if not patched:
                        file.write(patch)
                        patched = True
                else:
                    file.write(line)
            assert patched, f'[ERROR] [ASSERT FAILURE] insert_fix_into_src not pateced'

    
    def trigger_test_validation(self, trigger_tests, proj_dir):
        for trigger in trigger_tests:
            if self.patch_status == 'UNVERIFIED' or self.patch_status == 'PLAUSIBLE':
                out, err = defects4j_test_one(proj_dir, trigger)
                self.patch_status = extract_d4j_result(err, out, 'trigger')
                if self.patch_status == 'TRIGGER_ERROR': 
                    self.failing_test['TRIGGER'].append(trigger)
                elif self.patch_status == 'TRIGGER_TIMEOUT':
                    self.failing_test['TIMEOUT'].append(trigger)


    def relevant_test_validation(self, proj_dir):
        if self.patch_status != 'PLAUSIBLE':
            return
        out, err = defects4j_test_suite(proj_dir)
        self.patch_status = extract_d4j_result(err, out, 'relevant')
        self.failing_test['RELEVANT'] = [test_case.strip() for test_case in str(out).split(' - ')[1:]]
    
    
    def print_curr_patch_status(self, curr_bug, curr_overall_time):
        status_color = {
            'PLAUSIBLE': '\033[92m',  # 绿色
            'UNCOMPILABLE': '\033[91m',  # 红色
            'TRIGGER_ERROR': '\033[93m',  # 黄色
            'TRIGGER_TIMEOUT': '\033[93m',
            'RELEVANT_ERROR': '\033[93m',
            'RELEVANT_TIMEOUT': '\033[93m'
        }
        end_color = '\033[0m'
        
        color = status_color.get(self.patch_status, '')
        status_line = f'[PATCH STATUS] | {curr_bug:20} | {color}{self.patch_status:16}{end_color} | {curr_overall_time:4}s  |'
        print(status_line)
        
        # 构建日志消息
        log_messages = [status_line.replace(color, '').replace(end_color, '')]
        
        if self.patch_status == 'PLAUSIBLE':
            msg = f'[SUCCESS] Patch {curr_bug} passed all tests! 🎉'
            print(msg)
            log_messages.append(msg)
        elif self.patch_status == 'UNCOMPILABLE':
            msg = f'[FAILED] Patch {curr_bug} failed to compile ❌'
            print(msg)
            log_messages.append(msg)
        elif 'TIMEOUT' in self.patch_status:
            msg = f'[TIMEOUT] Patch {curr_bug} timed out ⏰'
            print(msg)
            log_messages.append(msg)
        elif 'ERROR' in self.patch_status:
            if self.failing_test['TRIGGER']:
                msg = f'[FAILED] Failed trigger tests: {", ".join(self.failing_test["TRIGGER"])} ❌'
                print(msg)
                log_messages.append(msg)
            if self.failing_test['RELEVANT']:
                msg = f'[FAILED] Failed relevant tests: {", ".join(self.failing_test["RELEVANT"])} ❌'
                print(msg)
                log_messages.append(msg)
        
        separator = '-' * 100
        print(separator)
        log_messages.append(separator)
        
        return '\n'.join(log_messages)
        
        
    def recover_buggy_file(self, backup_buggy_file_path, orig_file_content, patch_id, encoding_mode, proj_dir):
        if '.llm4apr_backup' not in backup_buggy_file_path:
            print(f'[ERROR] .llm4apr_backup not in backup_file')
            return
        
        recover_buggy_path = backup_buggy_file_path.replace('.llm4apr_backup', '')
        patched_backup_file_path = f'{recover_buggy_path}_{patch_id}_{self.patch_status}'
        
        # 添加文件存在性检查
        if not os.path.exists(recover_buggy_path):
            print(f'[WARNING] Source file not found: {recover_buggy_path}')
            return
        
        try:
            # 尝试移动文件
            if os.path.exists(recover_buggy_path):
                shutil.move(recover_buggy_path, patched_backup_file_path)
            
            # 复制备份文件
            if os.path.exists(backup_buggy_file_path):
                shutil.copyfile(backup_buggy_file_path, recover_buggy_path)
                
                # 验证文件内容
                with open(recover_buggy_path, 'r', encoding=encoding_mode) as f:
                    file_content = f.read()
                    if orig_file_content != file_content:
                        print(f'[ERROR] File content mismatch after recovery')
                        return
                    
                # 清理编译文件
                if proj_dir.startswith(ROOT_PATH):
                    rm_class_filename = os.path.basename(recover_buggy_path).replace('.java', '.class')
                    root_dir = Path(proj_dir)
                    for file in root_dir.rglob(rm_class_filename):
                        try:
                            file.unlink()
                        except Exception as e:
                            print(f'[WARNING] Failed to remove class file: {e}')
                else:
                    print(f'[ERROR] Invalid project directory: {proj_dir}')
                
        except Exception as e:
            print(f'[ERROR] Failed to recover file: {str(e)}')
            traceback.print_exc()

    def summarize_patch_info(self, bug_name):
        self.patch_val_info = {
            'patch_code': self.patch_code, 
            'patch_status': self.patch_status, 
            'failing_tests': self.failing_test,
            'val_cnt' : 1,
            'bug_name' : bug_name,
            'diff_stats': None  # 先设为 None，在外部设置
        }
        return self.patch_val_info
        



def get_result_paths(fixed_dir, json_file):
    # 确保使用绝对路径，防止 os.chdir 导致路径错误
    if not os.path.isabs(fixed_dir):
        fixed_dir = os.path.join(WORKSPACE_ROOT, fixed_dir)
    base_path = os.path.join(fixed_dir, json_file)
    log_path = f"{base_path}.judgelog"
    result_path = f"{base_path}.result"
    return log_path, result_path

def load_cached_result(log_path, result_path):
    """尝试加载缓存的验证结果"""
    if os.path.exists(result_path) and os.path.exists(log_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
        with open(log_path, 'r') as f:
            log = f.read()
        print(f"[CACHE] Found cached validation result")
        return result, log
    return None, None

def save_validation_result(log_path, result_path, results, log_content):
    """保存验证结果和日志"""
    # 确保目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    print(f"[DEBUG] Saving results to:")
    print(f"[DEBUG] - Log: {log_path}")
    print(f"[DEBUG] - Result: {result_path}")
    
    try:
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[DEBUG] Successfully saved result file")
    except Exception as e:
        print(f"[ERROR] Failed to save result file: {str(e)}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Result path exists: {os.path.exists(os.path.dirname(result_path))}")
        raise
        
    try:
        with open(log_path, 'w') as f:
            f.write(log_content)
        print(f"[DEBUG] Successfully saved log file")
    except Exception as e:
        print(f"[ERROR] Failed to save log file: {str(e)}")
        raise

def validate_patches_per_bug(candidate_patch, model_id):
    bug_name, patch_info = candidate_patch
    patches = patch_info['patches']
    total_patches = len(patches)
    
    print(f"\n{'='*50}")
    print(f"[VALIDATING] {bug_name} - Testing {total_patches} patches")
    print(f"{'='*50}")
    
    # 打印调试信息
    print(f"[DEBUG] Original JSON path: {patch_info['original_json']}")
    print(f"[DEBUG] Directory: {os.path.dirname(patch_info['original_json'])}")
    
    # 构建日志字符串
    validation_log = []
    validation_log.append(f"Validating {bug_name} with {total_patches} patches\n")
    
    # 检查是否有缓存的结果
    log_path, result_path = get_result_paths(os.path.dirname(patch_info['original_json']), f"{bug_name}.json")
    
    # 打印最终路径
    print(f"[DEBUG] Final paths:")
    print(f"[DEBUG] - Log: {log_path}")
    print(f"[DEBUG] - Result: {result_path}")
    cached_result, cached_log = load_cached_result(log_path, result_path)
    if cached_result is not None:
        print(cached_log)
        return cached_result
    
    val_time = ValTime(time.time())
    val_info = ValInfo(candidate_patch, model_id)
    if not val_info.check_init_success():
        print(f"[ERROR] Initialization failed for {bug_name} - failed_test_cases is empty!")
        print(f"[ERROR] Trigger tests: {len(val_info.trigger_tests)}, Relevant tests: {len(val_info.relevant_tests)}")
        # 保存错误信息
        error_result = [{
            'patch_code': 'N/A',
            'patch_status': 'INIT_FAILED',
            'failing_tests': {'TRIGGER': [], 'RELEVANT': [], 'TIMEOUT': []},
            'val_cnt': 0,
            'bug_name': bug_name,
            'diff_stats': None,
            'error': 'Failed to initialize: no failing test cases detected'
        }]
        save_validation_result(log_path, result_path, error_result, f"Initialization failed for {bug_name}\n")
        return error_result
    val_time.set_init_time(time.time())
    
    patch_results = []
    for i, curr_patch_code in enumerate(val_info.patches, 1):
        patch_log = f"\n[TESTING] Patch {i}/{total_patches} for {bug_name}\n"
        print(patch_log)
        validation_log.append(patch_log)
        
        val_info.patch_id_counter()
        patch_val = PatchValidation(curr_patch_code)
        
        patch_val.apply_patch(val_info.patch_info, val_info.proj_dir, val_info.encoding_mode)
        
        val_time.set_trigger_start_timestamp(time.time())
        patch_val.trigger_test_validation(val_info.trigger_tests, val_info.proj_dir)
        val_time.set_trigger_end_time(time.time())

        val_time.set_relevant_start_timestamp(time.time())
        patch_val.relevant_test_validation(val_info.proj_dir)
        val_time.set_relevant_end_time(time.time())    

        status_log = patch_val.print_curr_patch_status(val_info.curr_bug, val_time.get_curr_overall_time())
        validation_log.append(status_log)
        
        patch_val.recover_buggy_file(val_info.backup_buggy_file_path, val_info.original_buggy_file_content, \
                                     val_info.patch_id, val_info.encoding_mode, val_info.proj_dir)

        curr_patch_summary = patch_val.summarize_patch_info(val_info.curr_bug)
        # 只在patch通过测试时计算diff统计
        if curr_patch_summary['patch_status'] == 'PLAUSIBLE':
            curr_patch_summary['diff_stats'] = calc_diff_stats(
                val_info.patch_info['buggy'], 
                curr_patch_summary['patch_code']
            )
            print_diff_stats(curr_patch_summary['diff_stats'])  # 打印差异统计
        print(f"[DEBUG] Patch validation result: {curr_patch_summary['patch_status']}")  # 调试信息
        patch_results.append(curr_patch_summary)
        val_info.update_patch_val_result(curr_patch_summary)
        val_info.save_validation_results()
    
    # 保存验证结果和日志
    save_validation_result(log_path, result_path, patch_results, '\n'.join(validation_log))
    return patch_results
    

class ValidationStats:
    def __init__(self):
        self.total_bugs = 0
        self.bug_results = {}
        self.diff_stats = {
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

    def update(self, bug_id, patch_results):
        """更新bug的验证结果和diff统计"""
        self.total_bugs += 1
        self.bug_results[bug_id] = patch_results
        if patch_results:
            self.update_diff_stats(patch_results)

    def update_diff_stats(self, patch_results):
        """更新diff统计信息，只统计通过测试的patch（状态为PLAUSIBLE）"""
        for patch in patch_results:
            if patch.get('patch_status') == 'PLAUSIBLE':
                print(f"[DEBUG] Found PLAUSIBLE patch")
                if 'diff_stats' in patch:
                    stats = patch['diff_stats']
                    print(f"[DEBUG] Diff stats found: added={stats.get('added_lines', 0)}, deleted={stats.get('deleted_lines', 0)}, preserved={stats.get('preserved_ratio', 0)}%")
                    self.diff_stats['total_added'] += stats.get('added_lines', 0)
                    self.diff_stats['total_deleted'] += stats.get('deleted_lines', 0)
                    self.diff_stats['total_preserved'] += stats.get('preserved_ratio', 0)
                    self.diff_stats['patch_count'] += 1
                else:
                    print(f"[DEBUG] PLAUSIBLE patch found but no diff_stats available")
                
                ratio = stats.get('preserved_ratio', 0)
                if ratio > 95:
                    self.diff_stats['preservation_distribution']['high'] += 1
                elif ratio > 80:
                    self.diff_stats['preservation_distribution']['medium'] += 1
                else:
                    self.diff_stats['preservation_distribution']['low'] += 1

    def get_success_rate(self):
        """计算Top-1、Top-5和Top-10的成功率"""
        if self.total_bugs == 0:
            return 0.0, 0.0, 0.0  # 确保返回三个值
        
        top1_success = 0
        top5_success = 0
        top10_success = 0
        
        print(f"\n[DEBUG] 计算成功率详情:")
        print(f"[DEBUG] 总bug数量: {self.total_bugs}")
        
        for bug_id, patches in self.bug_results.items():
            if patches is None:
                continue
            
            success_position = None
            for i, p in enumerate(patches, 1):
                if p is not None and p['patch_status'] == 'PLAUSIBLE':
                    success_position = i
                    break
                
            if success_position is not None:
                if success_position == 1:
                    top1_success += 1
                if success_position <= 5:
                    top5_success += 1
                if success_position <= 10:
                    top10_success += 1
        
        print(f"[DEBUG] Top-1成功数: {top1_success}")
        print(f"[DEBUG] Top-5成功数: {top5_success}")
        print(f"[DEBUG] Top-10成功数: {top10_success}")
        
        top1_rate = (top1_success / self.total_bugs) * 100
        top5_rate = (top5_success / self.total_bugs) * 100
        top10_rate = (top10_success / self.total_bugs) * 100
        
        return top1_rate, top5_rate, top10_rate  # 确保这行一定会执行

def load_previous_results(model_id):
    results_dir = f'defects4j/results/{model_id}'
    previous_results = {}
    
    bug_dates = {}
    try:
        with open('defects4j/time.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                bug_id, date = list(data.items())[0]
                bug_dates[bug_id] = date
    except Exception as e:
        print(f"[WARNING] Failed to load time.jsonl: {e}")
    
    status_summary = []
    
    for json_file in glob.glob(os.path.join(results_dir, '*-validated.jsonl')):
        try:
            bug_id = os.path.basename(json_file).replace('-validated.jsonl', '')
            with open(json_file, 'r') as f:
                results = json.load(f)
                if results:
                    previous_results[bug_id] = results
                    
                    plausible_found = False
                    for idx, patch in enumerate(results, 1):
                        if patch['patch_status'] == 'PLAUSIBLE':
                            plausible_found = True
                            status_summary.append({
                                'bug_id': bug_id,
                                'date': bug_dates.get(bug_id, 'N/A'),
                                'status': 'PLAUSIBLE',
                                'position': idx,
                                'total_patches': len(results)
                            })
                            break
                    
                    if not plausible_found:
                        status_summary.append({
                            'bug_id': bug_id,
                            'date': bug_dates.get(bug_id, 'N/A'),
                            'status': 'FAILED',
                            'position': None,
                            'total_patches': len(results)
                        })
        except Exception as e:
            print(f"[WARNING] Failed to load previous results from {json_file}: {e}")
    
    print("\n[PREVIOUS VALIDATION SUMMARY]")
    print("=" * 100)
    print(f"{'Bug ID':15} | {'Date':10} | {'Status':10} | {'Position':10} | {'Total Patches':15}")
    print("-" * 100)
    
    for item in sorted(status_summary, key=lambda x: x['bug_id']):
        position_str = f"{item['position']}/{item['total_patches']}" if item['position'] else "N/A"
        status_color = '\033[92m' if item['status'] == 'PLAUSIBLE' else '\033[91m'
        print(f"{item['bug_id']:15} | {item['date']:10} | {status_color}{item['status']:10}\033[0m | {position_str:10} | {item['total_patches']:15}")
    
    print("=" * 100)
    print(f"Total previously validated bugs: {len(previous_results)}")
    print(f"Successfully fixed bugs: {len([x for x in status_summary if x['status'] == 'PLAUSIBLE'])}")
    print()
    
    return previous_results

def validate_defects4j(model_id, n_generations):
    stats = ValidationStats()
    candidate_patches = {}
    
    previous_results = load_previous_results(model_id)
    print(f"[INFO] Loaded {len(previous_results)} previously validated bugs")
    
    for bug_id, results in previous_results.items():
        stats.update(bug_id, results)
    top1, top5, top10 = stats.get_success_rate()
    print("\n[FINAL SUCCESS RATE]")
    print(f"Top-1:  {top1:.2f}%")
    print(f"Top-5:  {top5:.2f}%")
    print(f"Top-10: {top10:.2f}%")
    
    print(f"\n[DEBUG] Total patch count for diff stats: {stats.diff_stats['patch_count']}")  # 添加调试信息
    
    if stats.diff_stats['patch_count'] > 0:
        patch_count = stats.diff_stats['patch_count']
        dist = stats.diff_stats['preservation_distribution']
        
        print("\n[PATCH MODIFICATION STATISTICS]")
        print(f"- Average lines added per patch:    {stats.diff_stats['total_added']/patch_count:.1f}")
        print(f"- Average lines deleted per patch:  {stats.diff_stats['total_deleted']/patch_count:.1f}")
        print(f"- Average code preserved:           {stats.diff_stats['total_preserved']/patch_count:.1f}%")
        
        print("\nDistribution of code preservation ratio:")
        total = sum(dist.values())
        if total > 0:
            print(f"- Minimal change   (>95% preserved):   {dist['high']:3d} patches ({dist['high']/total*100:5.1f}%)")
            print(f"- Moderate change (80-95% preserved):  {dist['medium']:3d} patches ({dist['medium']/total*100:5.1f}%)")
            print(f"- Major change     (<80% preserved):   {dist['low']:3d} patches ({dist['low']/total*100:5.1f}%)")
    
    for i in range(n_generations):
        fix_dir = os.path.join('defects4j/results', str(model_id), f'fixed{i}')
        if not os.path.exists(fix_dir):
            print(f"Warning: {fix_dir} does not exist")
            continue
            
        for json_file in glob.glob(os.path.join(fix_dir, '*.json')):
            if json_file.endswith('.log'):
                continue
                
            bug_id = os.path.basename(json_file).replace('.json', '')
            if bug_id in previous_results:
                continue
                
            if bug_id not in candidate_patches:
                candidate_patches[bug_id] = {
                    'patches': [],
                    'original_json': json_file
                }
                
                # 从dataset目录读取buggy代码
                dataset_file = os.path.join('defects4j/dataset', f'{bug_id}.json')
                try:
                    with open(dataset_file) as df:
                        dataset_info = json.load(df)
                        candidate_patches[bug_id]['buggy'] = dataset_info['buggy']
                except Exception as e:
                    print(f"[WARNING] Failed to load dataset file for {bug_id}: {e}")
                    continue
                
            with open(json_file) as f:
                patch_info = json.load(f)
                if 'fix' in patch_info:
                    candidate_patches[bug_id]['patches'].append(patch_info['fix'])
                    candidate_patches[bug_id]['loc'] = patch_info['loc']
                    candidate_patches[bug_id]['start'] = patch_info['start']
                    candidate_patches[bug_id]['end'] = patch_info['end']
                    candidate_patches[bug_id]['buggy'] = candidate_patches[bug_id].get('buggy', '')  # 确保buggy代码被传递
    
    filtered_candidates = {}
    skipped_bugs = []
    for bug_id, patch_info in candidate_patches.items():
        if len(patch_info['patches']) >= 10:
            filtered_candidates[bug_id] = patch_info
        else:
            skipped_bugs.append((bug_id, len(patch_info['patches'])))
    
    if skipped_bugs:
        print("\n[SKIPPED BUGS] (insufficient patches)")
        print(f"{'Bug ID':20} | {'Patch Count':12}")
        print("-" * 35)
        for bug_id, count in sorted(skipped_bugs):
            print(f"{bug_id:20} | {count:12}")
        print(f"\nTotal skipped: {len(skipped_bugs)} bugs")
    
    remaining_bugs = len(filtered_candidates)
    print(f"\n[INFO] Found {remaining_bugs} new bugs with 10+ patches to validate")
    print(f"[INFO] Total bugs (including previous): {len(previous_results) + remaining_bugs}")
    
    if remaining_bugs == 0:
        print("[INFO] No new bugs to validate")
        return

    max_workers = min(multiprocessing.cpu_count(), 4)
    print(f"[INFO] Using {max_workers} workers for parallel validation")
    
    validated_count = 0
    total_count = len(filtered_candidates)
    results_lock = threading.Lock()
    
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_bug = {
            executor.submit(validate_patches_per_bug, (bug_name, patch_info), model_id): bug_name 
            for bug_name, patch_info in sorted(filtered_candidates.items())
        }
        
        for future in cf.as_completed(future_to_bug):
            bug_name = future_to_bug[future]
            try:
                results = future.result()
                
                with results_lock:
                    validated_count += 1
                    stats.update(bug_name, results)
                    
                    top1, top5, top10 = stats.get_success_rate()
                    print(f"\n[CURRENT PROGRESS] Validated {validated_count}/{total_count} bugs")
                    print(f"[CURRENT SUCCESS RATE] Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}%")
                    print("=" * 100)
                    
            except Exception as e:
                print(f"Exception when validating {bug_name}: {str(e)}")
                traceback.print_exc()
    
    top1, top5, top10 = stats.get_success_rate()
    print("\n[FINAL SUCCESS RATE]")
    print(f"Top-1:  {top1:.2f}%")
    print(f"Top-5:  {top5:.2f}%")
    print(f"Top-10: {top10:.2f}%")
    print("=" * 100)
    
    # 添加diff统计输出
    if stats.diff_stats['patch_count'] > 0:
        patch_count = stats.diff_stats['patch_count']
        dist = stats.diff_stats['preservation_distribution']
        
        print("\n[PATCH MODIFICATION STATISTICS]")
        print(f"- Average lines added per patch:    {stats.diff_stats['total_added']/patch_count:.1f}")
        print(f"- Average lines deleted per patch:  {stats.diff_stats['total_deleted']/patch_count:.1f}")
        print(f"- Average code preserved:           {stats.diff_stats['total_preserved']/patch_count:.1f}%")
        
        print("\nDistribution of code preservation ratio:")
        total = sum(dist.values())
        if total > 0:
            print(f"- Minimal change   (>95% preserved):   {dist['high']:3d} patches ({dist['high']/total*100:5.1f}%)")
            print(f"- Moderate change (80-95% preserved):  {dist['medium']:3d} patches ({dist['medium']/total*100:5.1f}%)")
            print(f"- Major change     (<80% preserved):   {dist['low']:3d} patches ({dist['low']/total*100:5.1f}%)")
    print("=" * 100)
    
    print('[END VALIDATION]')
    sys.stdout.flush()
    time.sleep(3)
    return

@contextmanager
def log_or_print(log_mode, log_path):
    if log_mode:
        with open(log_path, 'a') as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
            yield
    else:
        yield

def shuffle_validated_patches(candidate_patches):
    items = list(candidate_patches.items())
    random.shuffle(items)
    shuffled_patches = {key: value for key, value in items}
    return shuffled_patches

def load_and_compare_results(model_id1, model_id2, min_patches=1):
    bug_dates = {}
    try:
        with open('time.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                bug_id, date = list(data.items())[0]
                bug_dates[bug_id] = date
    except Exception as e:
        print(f"[WARNING] Failed to load time.jsonl: {e}")

    results1 = load_previous_results(model_id1)
    results2 = load_previous_results(model_id2)
    
    filtered_results1 = {bug_id: results for bug_id, results in results1.items() 
                        if len(results) >= min_patches}
    filtered_results2 = {bug_id: results for bug_id, results in results2.items() if len(results) >= min_patches}
    
    bug_lengths = {}
    for bug_id in filtered_results1.keys() | filtered_results2.keys():
        json_file = os.path.join('results', model_id1, 'fixed0', f'{bug_id}.json')
        try:
            with open(json_file, 'r') as f:
                patch_info = json.load(f)
                total_length = len(patch_info.get('title', '')) + len(patch_info.get('description', '')) + len(patch_info.get('buggy', ''))
                bug_lengths[bug_id] = total_length
        except Exception as e:
            print(f"[WARNING] Failed to load patch info for {bug_id}: {e}")
    
    all_bugs = sorted(set(filtered_results1.keys()) | set(filtered_results2.keys()))
    common_bugs = set(filtered_results1.keys()) & set(filtered_results2.keys())
    
    print("\n[VALIDATION COMPARISON SUMMARY]")
    print("=" * 145)
    print(f"{'Bug ID':20} | {'Date':10} | {'Length':8} | {model_id1:^15} | {model_id2:^15} | {'Notes':20}")
    print("-" * 145)
    
    for bug_id in all_bugs:
        date = bug_dates.get(bug_id, 'N/A')
        length = bug_lengths.get(bug_id, 'N/A')
        if length != 'N/A':
            length = f"{length:,}"
        
        status1 = "N/A"
        position1 = ""
        if bug_id in filtered_results1:
            results = filtered_results1[bug_id]
            for idx, patch in enumerate(results, 1):
                if patch['patch_status'] == 'PLAUSIBLE':
                    status1 = f"PLAUSIBLE({idx})"
                    break
            if status1 == "N/A":
                status1 = "FAILED"
        
        status2 = "N/A"
        position2 = ""
        if bug_id in filtered_results2:
            results = filtered_results2[bug_id]
            for idx, patch in enumerate(results, 1):
                if patch['patch_status'] == 'PLAUSIBLE':
                    status2 = f"PLAUSIBLE({idx})"
                    break
            if status2 == "N/A":
                status2 = "FAILED"
        
        notes = ""
        if status1 == "N/A" and status2 != "N/A":
            notes = f"Only in {model_id2}"
        elif status1 != "N/A" and status2 == "N/A":
            notes = f"Only in {model_id1}"
        elif 'PLAUSIBLE' in status1 and 'PLAUSIBLE' in status2:
            notes = "Fixed by both"
        elif 'PLAUSIBLE' in status1:
            notes = f"Only fixed by {model_id1}"
        elif 'PLAUSIBLE' in status2:
            notes = f"Only fixed by {model_id2}"
        
        status1_color = '\033[92m' if 'PLAUSIBLE' in status1 else '\033[91m' if status1 == 'FAILED' else '\033[0m'
        status2_color = '\033[92m' if 'PLAUSIBLE' in status2 else '\033[91m' if status2 == 'FAILED' else '\033[0m'
        
        print(f"{bug_id:20} | {date:10} | {length:>8} | {status1_color}{status1:^15}\033[0m | {status2_color}{status2:^15}\033[0m | {notes:20}")
    
    print("=" * 145)
    
    fixed_bugs1 = {bug_id for bug_id, results in filtered_results1.items() 
                  if any(patch['patch_status'] == 'PLAUSIBLE' for patch in results)}
    fixed_bugs2 = {bug_id for bug_id, results in filtered_results2.items() 
                  if any(patch['patch_status'] == 'PLAUSIBLE' for patch in results)}
    
    common_fixed_bugs = fixed_bugs1 & fixed_bugs2
    only_fixed_by_1 = fixed_bugs1 - fixed_bugs2
    only_fixed_by_2 = fixed_bugs2 - fixed_bugs1
    
    print(f"\n[DATA COVERAGE]")
    print(f"Model {model_id1} total bugs with 10 patches: {len(filtered_results1)}")
    print(f"Model {model_id2} total bugs with 10 patches: {len(filtered_results2)}")
    print(f"Common bugs with 10 patches: {len(common_bugs)}")
    
    print(f"\n[FIX STATISTICS]")
    print(f"Model {model_id1} fixed total: {len(fixed_bugs1)}")
    print(f"Model {model_id2} fixed total: {len(fixed_bugs2)}")
    print(f"Fixed by both models: {len(common_fixed_bugs)}")
    print(f"Only fixed by {model_id1}: {len(only_fixed_by_1)} {sorted(only_fixed_by_1)}")
    print(f"Only fixed by {model_id2}: {len(only_fixed_by_2)} {sorted(only_fixed_by_2)}")
    
    filtered_stats1, filtered_stats2 = ValidationStats(), ValidationStats()
    
    for bug_id in common_bugs:
        if bug_id in filtered_results1:
            filtered_stats1.update(bug_id, filtered_results1[bug_id])
        if bug_id in filtered_results2:
            filtered_stats2.update(bug_id, filtered_results2[bug_id])
    
    return filtered_stats1, filtered_stats2

def print_comparison_results(stats1, stats2, model_id1, model_id2):
    print("\n[COMPARISON RESULTS]")
    print("=" * 80)
    print(f"{'Metric':15} | {model_id1:>10} | {model_id2:>10} | {'Diff':>10}")
    print("-" * 80)
    
    top1_1, top5_1, top10_1 = stats1.get_success_rate()
    top1_2, top5_2, top10_2 = stats2.get_success_rate()
    
    metrics = [
        ("Top-1", top1_1, top1_2),
        ("Top-5", top5_1, top5_2),
        ("Top-10", top10_1, top10_2)
    ]
    
    for metric_name, val1, val2 in metrics:
        diff = val2 - val1
        diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
        color = '\033[92m' if diff > 0 else '\033[91m' if diff < 0 else '\033[0m'
        print(f"{metric_name:15} | {val1:>9.2f}% | {val2:>9.2f}% | {color}{diff_str:>10}\033[0m")
    
    print("=" * 80)
    print(f"Total bugs compared: {stats1.total_bugs}")

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
    
    # 计算保留比例
    matcher = difflib.SequenceMatcher(None, buggy_lines, fix_lines)
    preserved = sum(block.size for block in matcher.get_matching_blocks())
    preserved_ratio = preserved / len(buggy_lines) if buggy_lines else 0.0
    
    return {
        'added_lines': added,
        'deleted_lines': deleted,
        'preserved_ratio': round(preserved_ratio * 100, 2)
    }

def print_diff_stats(diff_stats):
    """打印补丁的差异统计信息"""
    print(f"[DIFF STATS]   | Added: {diff_stats['added_lines']}, Deleted: {diff_stats['deleted_lines']}, Preserved: {diff_stats['preserved_ratio']}%")

def recalculate_diff_stats(model_id):
    """重新计算所有已验证patch的diff统计"""
    print(f"[INFO] 重新计算 {model_id} 的diff统计")
    results_dir = os.path.join('defects4j/results', model_id)
    
    for validated_file in glob.glob(os.path.join(results_dir, '*-validated.jsonl')):
        bug_id = os.path.basename(validated_file).replace('-validated.jsonl', '')
        print(f"\n[处理] {bug_id}")
        
        try:
            # 读取dataset中的buggy代码
            dataset_file = os.path.join('defects4j/dataset', f'{bug_id}.json')
            with open(dataset_file) as df:
                dataset_info = json.load(df)
                buggy_code = dataset_info['buggy']
            
            # 读取验证结果
            with open(validated_file, 'r') as f:
                results = json.load(f)
            
            # 重新计算每个PLAUSIBLE patch的diff
            modified = False
            for patch in results:
                if patch['patch_status'] == 'PLAUSIBLE':
                    patch['diff_stats'] = calc_diff_stats(buggy_code, patch['patch_code'])
                    print_diff_stats(patch['diff_stats'])
                    modified = True
            
            # 如果有修改，保存更新后的结果
            if modified:
                with open(validated_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"[已更新] {bug_id}")
            else:
                print(f"[跳过] {bug_id} - 没有PLAUSIBLE的patch")
                
        except Exception as e:
            print(f"[错误] 处理 {bug_id} 时出错: {e}")
            traceback.print_exc()
    
    print("\n[完成] diff统计重新计算完成")

if __name__ == '__main__':
    start_val_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_id', type=str, required=True)
    parser.add_argument('-n', '--n_generations', type=int, default=1)
    parser.add_argument('--recalc_diff', action='store_true', help='只重新计算diff统计，不重新运行测试')
    args = parser.parse_args()
    
    if args.recalc_diff:
        recalculate_diff_stats(args.model_id)
    else:
        validate_defects4j(args.model_id, args.n_generations)
