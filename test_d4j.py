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
import math

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
        # 直接尝试创建目录，避免竞态条件
        try:
            os.makedirs(self.val_result_path, exist_ok=True)
        except FileExistsError:
            pass  # 目录已存在，继续
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
    # 确保目录存在，使用 try-except 避免竞态条件
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except (FileExistsError, FileNotFoundError):
        pass
    try:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
    except (FileExistsError, FileNotFoundError):
        pass
    
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
        
        # 如果补丁代码为None（占位符，表示缺失），直接标记为失败
        if curr_patch_code is None:
            missing_log = f"[MISSING] Patch {i} is missing (LLM output incomplete)\n"
            print(missing_log)
            validation_log.append(missing_log)
            
            # 创建一个失败的补丁记录
            missing_patch_info = {
                'patch_code': '/* MISSING PATCH */',
                'patch_status': 'MISSING_PATCH',
                'failing_tests': {'TRIGGER': [], 'RELEVANT': [], 'TIMEOUT': []},
                'val_cnt': 1,
                'bug_name': bug_name,
                'diff_stats': None
            }
            patch_results.append(missing_patch_info)
            val_info.update_patch_val_result(missing_patch_info)
            val_info.save_validation_results()
            
            status_line = f"[PATCH STATUS] | {bug_name:20} | \033[91mMISSING_PATCH\033[0m    | {0:4}s  |"
            print(status_line)
            print('-' * 100)
            validation_log.append(status_line)
            validation_log.append('-' * 100)
            continue
        
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
        # 只对 PLAUSIBLE 补丁计算详细的 diff 统计（节省计算资源）
        if curr_patch_summary['patch_status'] == 'PLAUSIBLE':
            curr_patch_summary['diff_stats'] = calc_diff_stats(
                val_info.patch_info['buggy'],
                curr_patch_summary['patch_code']
            )
            print_diff_stats(curr_patch_summary['diff_stats'])
        else:
            # 非 PLAUSIBLE 补丁不计算详细统计
            curr_patch_summary['diff_stats'] = None
        patch_results.append(curr_patch_summary)
        val_info.update_patch_val_result(curr_patch_summary)
        val_info.save_validation_results()
    
    # 确保所有结果都被保存到 validated.jsonl
    val_info.save_validation_results(done=True)
    
    # 保存验证结果和日志
    save_validation_result(log_path, result_path, patch_results, '\n'.join(validation_log))
    return patch_results
    

class ValidationStats:
    def __init__(self):
        self.total_bugs = 0
        self.bug_results = {}
        # PLAUSIBLE 补丁的详细统计（只统计可行补丁）
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

    def update(self, bug_id, patch_results):
        """更新bug的验证结果和diff统计"""
        self.total_bugs += 1
        self.bug_results[bug_id] = patch_results
        if patch_results:
            self.update_diff_stats(patch_results)

    def _update_single_diff_stats(self, stats_dict, diff_stats):
        """更新单个统计字典
        
        Args:
            stats_dict: 要更新的统计字典
            diff_stats: 补丁的diff统计信息
        """
        stats_dict['total_hunks'] += diff_stats.get('hunks', 0)
        stats_dict['total_added'] += diff_stats.get('added_lines', 0)
        stats_dict['total_deleted'] += diff_stats.get('deleted_lines', 0)
        stats_dict['total_changed_lines'] += diff_stats.get('total_changed_lines', 0)
        stats_dict['total_added_tokens'] += diff_stats.get('added_tokens', 0)
        stats_dict['total_deleted_tokens'] += diff_stats.get('deleted_tokens', 0)
        stats_dict['total_changed_tokens'] += diff_stats.get('total_changed_tokens', 0)
        stats_dict['total_edit_distance'] += diff_stats.get('edit_distance', 0)
        stats_dict['total_edit_similarity'] += diff_stats.get('edit_similarity', 0)
        stats_dict['total_norm_edit_distance'] += diff_stats.get('norm_edit_distance', 0)
        stats_dict['total_preserved'] += diff_stats.get('preserved_ratio', 0)
        stats_dict['patch_count'] += 1
        
        ratio = diff_stats.get('preserved_ratio', 0)
        if ratio > 95:
            stats_dict['preservation_distribution']['high'] += 1
        elif ratio > 80:
            stats_dict['preservation_distribution']['medium'] += 1
        else:
            stats_dict['preservation_distribution']['low'] += 1
    
    def update_diff_stats(self, patch_results):
        """更新diff统计信息，只统计PLAUSIBLE补丁"""
        for patch in patch_results:
            # 只统计 PLAUSIBLE 补丁
            if patch.get('patch_status') == 'PLAUSIBLE':
                if 'diff_stats' in patch and patch['diff_stats']:
                    self._update_single_diff_stats(self.diff_stats, patch['diff_stats'])
                else:
                    print(f"[WARNING] PLAUSIBLE patch missing diff_stats")

    def get_success_rate(self):
        """计算pass@1、pass@5和pass@10的成功率（只计算有完整10个候选补丁的bug）
        
        使用无偏估计公式：pass@k = 1 - C(n-c, k) / C(n, k)
        其中 n=总补丁数, c=正确补丁数, k=采样数
        """
        if self.total_bugs == 0:
            return 0.0, 0.0, 0.0
        
        pass1_sum = 0.0
        pass5_sum = 0.0
        pass10_sum = 0.0
        valid_bugs = 0
        
        for bug_id, patches in self.bug_results.items():
            if patches is None:
                continue
            
            # 统计总补丁数和正确补丁数
            n = len(patches)
            
            # 只计算有完整10个候选补丁的bug
            if n != 10:
                continue
            
            c = sum(1 for p in patches if p is not None and p.get('patch_status') == 'PLAUSIBLE')
            valid_bugs += 1
            
            # 计算 pass@k
            pass1_sum += self._calc_pass_at_k(n, c, 1)
            pass5_sum += self._calc_pass_at_k(n, c, 5)
            pass10_sum += self._calc_pass_at_k(n, c, 10)
        
        pass1_rate = (pass1_sum / valid_bugs) * 100 if valid_bugs > 0 else 0
        pass5_rate = (pass5_sum / valid_bugs) * 100 if valid_bugs > 0 else 0
        pass10_rate = (pass10_sum / valid_bugs) * 100 if valid_bugs > 0 else 0
        
        return pass1_rate, pass5_rate, pass10_rate
    
    def _calc_pass_at_k(self, n, c, k):
        """计算单个bug的pass@k
        
        Args:
            n: 总补丁数（固定为10）
            c: 正确补丁数（PLAUSIBLE的数量）
            k: 采样数量
        
        Returns:
            pass@k 的值 (0.0 到 1.0)
        """
        # n 现在固定为 10，k 最大为 10
        assert n == 10, f"Expected n=10, got n={n}"
        assert k <= 10, f"k should not exceed 10, got k={k}"
        
        if c == 0:
            # 没有正确补丁
            return 0.0
        
        if n - c < k:
            # 错误补丁数少于k，必然能抽到正确的
            return 1.0
        
        # pass@k = 1 - C(n-c, k) / C(n, k)
        return 1.0 - (self._comb(n - c, k) / self._comb(n, k))
    
    def _comb(self, n, k):
        """计算组合数 C(n, k) = n! / (k! * (n-k)!)
        
        使用优化算法避免大数阶乘溢出
        """
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        # 优化：C(n, k) = C(n, n-k)，选择较小的k
        k = min(k, n - k)
        
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        
        return result
    
    def get_pass_at_k(self):
        """别名方法，调用 get_success_rate() 以保持兼容性"""
        return self.get_success_rate()

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
    
    # 从 fixed0 目录的 .result 文件加载结果
    fixed0_dir = os.path.join(results_dir, 'fixed0')
    
    if not os.path.exists(fixed0_dir):
        print(f"[WARNING] fixed0 directory does not exist: {fixed0_dir}")
        return previous_results
    
    result_files = glob.glob(os.path.join(fixed0_dir, '*.json.result'))
    
    for result_file in result_files:
        try:
            filename = os.path.basename(result_file)
            bug_id = filename.replace('.json.result', '')
            
            with open(result_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load previous results from {result_file}: {e}")
            continue
        
        # 加载所有bug的结果（不限制补丁数量）
        if results and len(results) > 0:
            previous_results[bug_id] = results
            
            plausible_found = False
            for idx, patch in enumerate(results, 1):
                if patch.get('patch_status') == 'PLAUSIBLE':
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
            
            # 如果补丁数不是10，记录警告
            if len(results) != 10:
                print(f"[INFO] {bug_id} has {len(results)} patches (not 10, will exclude from pass@k calculation)")
    
    print("\n[PREVIOUS VALIDATION SUMMARY]")
    print("=" * 100)
    print(f"{'Bug ID':15} | {'Date':10} | {'Status':10} | {'Position':10} | {'Total Patches':15}")
    print("-" * 100)
    
    for item in sorted(status_summary, key=lambda x: x['bug_id']):
        position_str = f"{item['position']}/{item['total_patches']}" if item['position'] else "N/A"
        status_color = '\033[92m' if item['status'] == 'PLAUSIBLE' else '\033[91m'
        print(f"{item['bug_id']:15} | {item['date']:10} | {status_color}{item['status']:10}\033[0m | {position_str:10} | {item['total_patches']:15}")
    
    print("=" * 100)
    
    # 统计信息
    total_bugs = len(previous_results)
    bugs_with_10_patches = len([bug_id for bug_id, results in previous_results.items() if len(results) == 10])
    bugs_incomplete = total_bugs - bugs_with_10_patches
    successfully_fixed = len([x for x in status_summary if x['status'] == 'PLAUSIBLE'])
    successfully_fixed_10 = len([x for x in status_summary if x['status'] == 'PLAUSIBLE' and x['total_patches'] == 10])
    
    print(f"Total previously validated bugs: {total_bugs}")
    print(f"  - With exactly 10 patches (used in pass@k): {bugs_with_10_patches}")
    if bugs_incomplete > 0:
        print(f"  - With <10 patches (excluded from pass@k): {bugs_incomplete}")
    print(f"Successfully fixed bugs: {successfully_fixed}")
    if successfully_fixed != successfully_fixed_10:
        print(f"  - Among 10-patch bugs: {successfully_fixed_10}")
        print(f"  - Among incomplete bugs: {successfully_fixed - successfully_fixed_10}")
    print()
    
    return previous_results

def validate_defects4j(model_id, n_generations):
    stats = ValidationStats()
    candidate_patches = {}
    
    # 检查dataset中有多少个bug
    dataset_dir = 'defects4j/dataset'
    dataset_bugs = set()
    if os.path.exists(dataset_dir):
        for json_file in glob.glob(os.path.join(dataset_dir, '*.json')):
            bug_id = os.path.basename(json_file).replace('.json', '')
            dataset_bugs.add(bug_id)
        print(f"[INFO] Found {len(dataset_bugs)} bugs in dataset")
    
    previous_results = load_previous_results(model_id)
    print(f"[INFO] Loaded {len(previous_results)} previously validated bugs")
    
    # 找出缺失的bug，并为它们创建空结果（计入分母）
    if dataset_bugs:
        validated_bugs = set(previous_results.keys())
        missing_bugs = dataset_bugs - validated_bugs
        if missing_bugs:
            print(f"[WARNING] {len(missing_bugs)} bugs in dataset but not validated:")
            for bug_id in sorted(missing_bugs):
                print(f"  - {bug_id}")
                # 为缺失的bug创建空结果（10个失败的占位符）
                previous_results[bug_id] = [
                    {
                        'patch_code': '/* NO LLM OUTPUT */',
                        'patch_status': 'NO_OUTPUT',
                        'failing_tests': {'TRIGGER': [], 'RELEVANT': [], 'TIMEOUT': []},
                        'val_cnt': 0,
                        'bug_name': bug_id,
                        'diff_stats': None
                    }
                ] * 10  # 创建10个相同的占位符
    
    for bug_id, results in previous_results.items():
        stats.update(bug_id, results)
    
    # 统计参与pass@k计算的bug数量
    bugs_in_passk = sum(1 for bug_id, results in stats.bug_results.items() if len(results) == 10)
    bugs_total = len(stats.bug_results)
    bugs_excluded = bugs_total - bugs_in_passk
    
    pass1, pass5, pass10 = stats.get_pass_at_k()
    print("\n[PREVIOUS RESULTS - Pass@k]")
    print(f"Total bugs loaded: {bugs_total}")
    if bugs_excluded > 0:
        print(f"  - Used in pass@k (exactly 10 patches): {bugs_in_passk}")
        print(f"  - Excluded from pass@k (<10 patches): {bugs_excluded}")
    print(f"pass@1:  {pass1:.2f}%  (based on {bugs_in_passk} bugs)")
    print(f"pass@5:  {pass5:.2f}%")
    print(f"pass@10: {pass10:.2f}%")
    
    # 打印 PLAUSIBLE 补丁的详细统计信息
    print_diff_statistics_summary(stats.diff_stats, "PLAUSIBLE Patches")
    
    # 首先收集所有可能的bug_id（从所有fixed目录）
    all_bug_ids = set()
    for i in range(n_generations):
        fix_dir = os.path.join('defects4j/results', str(model_id), f'fixed{i}')
        if os.path.exists(fix_dir):
            for json_file in glob.glob(os.path.join(fix_dir, '*.json')):
                if not json_file.endswith('.log') and not json_file.endswith('.result') and not json_file.endswith('.judgelog'):
                    bug_id = os.path.basename(json_file).replace('.json', '')
                    if bug_id not in previous_results:
                        all_bug_ids.add(bug_id)
    
    # 对每个bug_id，确保从fixed0-fixed9都收集到补丁（缺失的用占位符）
    for bug_id in all_bug_ids:
        # 初始化bug的基本信息
        if bug_id not in candidate_patches:
            candidate_patches[bug_id] = {
                'patches': [],
                'patch_files': [],  # 记录每个补丁的来源文件
                'original_json': None,
                'loc': None,
                'start': None,
                'end': None,
                'buggy': ''
            }
            
            # 从dataset目录读取buggy代码和位置信息
            dataset_file = os.path.join('defects4j/dataset', f'{bug_id}.json')
            try:
                with open(dataset_file) as df:
                    dataset_info = json.load(df)
                    candidate_patches[bug_id]['buggy'] = dataset_info.get('buggy', '')
                    candidate_patches[bug_id]['loc'] = dataset_info.get('loc', '')
                    candidate_patches[bug_id]['start'] = dataset_info.get('start', 0)
                    candidate_patches[bug_id]['end'] = dataset_info.get('end', 0)
            except Exception as e:
                print(f"[WARNING] Failed to load dataset file for {bug_id}: {e}")
        
        # 遍历所有fixed目录，收集或创建占位补丁
        for i in range(n_generations):
            fix_dir = os.path.join('defects4j/results', str(model_id), f'fixed{i}')
            json_file = os.path.join(fix_dir, f'{bug_id}.json')
            
            patch_added = False
            if os.path.exists(json_file):
                try:
                    with open(json_file) as f:
                        patch_info = json.load(f)
                        
                        # 更新位置信息（如果之前没有从dataset获取到）
                        if candidate_patches[bug_id]['loc'] is None and 'loc' in patch_info:
                            candidate_patches[bug_id]['loc'] = patch_info['loc']
                            candidate_patches[bug_id]['start'] = patch_info.get('start', 0)
                            candidate_patches[bug_id]['end'] = patch_info.get('end', 0)
                        
                        if candidate_patches[bug_id]['original_json'] is None:
                            candidate_patches[bug_id]['original_json'] = json_file
                        
                        # 如果有fix字段且不为空，添加真实补丁
                        if 'fix' in patch_info and patch_info['fix'] and patch_info['fix'].strip():
                            candidate_patches[bug_id]['patches'].append(patch_info['fix'])
                            candidate_patches[bug_id]['patch_files'].append(json_file)
                            patch_added = True
                        else:
                            # fix字段缺失或为空，添加占位补丁
                            print(f"[WARNING] {bug_id} in {fix_dir}: 'fix' field missing or empty")
                except Exception as e:
                    print(f"[WARNING] Failed to load {json_file}: {e}")
            else:
                print(f"[WARNING] {bug_id}: Missing file in {fix_dir}")
            
            # 如果没有添加真实补丁，添加占位补丁（表示缺失或无效）
            if not patch_added:
                candidate_patches[bug_id]['patches'].append(None)  # None表示缺失的补丁
                candidate_patches[bug_id]['patch_files'].append(None)
    
    # 现在所有bug都应该有10个补丁（真实的或None占位符）
    filtered_candidates = {}
    skipped_bugs = []
    incomplete_bugs = []  # 记录有缺失补丁的bug
    
    for bug_id, patch_info in candidate_patches.items():
        num_patches = len(patch_info['patches'])
        
        if num_patches == 10:
            # 检查是否有None占位符
            none_count = sum(1 for p in patch_info['patches'] if p is None)
            if none_count > 0:
                incomplete_bugs.append((bug_id, none_count))
            
            # 只要有基本信息就可以验证
            if patch_info['loc'] and patch_info['buggy']:
                filtered_candidates[bug_id] = patch_info
            else:
                skipped_bugs.append((bug_id, num_patches, "Missing loc or buggy info"))
        else:
            skipped_bugs.append((bug_id, num_patches, f"Expected 10, got {num_patches}"))
    
    if incomplete_bugs:
        print("\n[INCOMPLETE BUGS] (some patches missing but will be tested)")
        print(f"{'Bug ID':20} | {'Missing Patches':16}")
        print("-" * 40)
        for bug_id, missing_count in sorted(incomplete_bugs):
            print(f"{bug_id:20} | {missing_count:16}")
        print(f"\nTotal incomplete: {len(incomplete_bugs)} bugs (will mark missing patches as FAILED)")
    
    if skipped_bugs:
        print("\n[SKIPPED BUGS] (cannot be validated)")
        print(f"{'Bug ID':20} | {'Patch Count':12} | {'Reason':30}")
        print("-" * 70)
        for item in sorted(skipped_bugs):
            if len(item) == 3:
                bug_id, count, reason = item
                print(f"{bug_id:20} | {count:12} | {reason:30}")
            else:
                bug_id, count = item
                print(f"{bug_id:20} | {count:12} | {'Unknown':30}")
        print(f"\nTotal skipped: {len(skipped_bugs)} bugs")
    
    remaining_bugs = len(filtered_candidates)
    print(f"\n[INFO] Found {remaining_bugs} new bugs with exactly 10 patches to validate")
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
                    pass1, pass5, pass10 = stats.get_pass_at_k()
                    print(f"[CURRENT SUCCESS RATE] Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}%")
                    print(f"[CURRENT PASS@K] pass@1: {pass1:.2f}% | pass@5: {pass5:.2f}% | pass@10: {pass10:.2f}%")
                    print("=" * 100)
                    
            except Exception as e:
                print(f"Exception when validating {bug_name}: {str(e)}")
                traceback.print_exc()
    
    pass1, pass5, pass10 = stats.get_pass_at_k()
    
    # 统计参与pass@k计算的bug数量
    bugs_in_passk = sum(1 for bug_id, results in stats.bug_results.items() if len(results) == 10)
    bugs_total = len(stats.bug_results)
    bugs_excluded = bugs_total - bugs_in_passk
    
    print("\n[FINAL SUCCESS RATE - Pass@k]")
    print("=" * 80)
    print(f"Total bugs validated: {bugs_total}")
    if bugs_excluded > 0:
        print(f"  - Used in pass@k (exactly 10 patches): {bugs_in_passk}")
        print(f"  - Excluded from pass@k (<10 patches): {bugs_excluded}")
    print("-" * 80)
    print(f"pass@1:  {pass1:6.2f}%  (based on {bugs_in_passk} bugs with 10 patches)")
    print(f"pass@5:  {pass5:6.2f}%")
    print(f"pass@10: {pass10:6.2f}%")
    print("=" * 80)
    
    # 打印 PLAUSIBLE 补丁的详细统计信息
    print_diff_statistics_summary(stats.diff_stats, "PLAUSIBLE Patches")
    
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
        with open('defects4j/time.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                bug_id, date = list(data.items())[0]
                bug_dates[bug_id] = date
    except Exception as e:
        print(f"[WARNING] Failed to load time.jsonl: {e}")

    # 加载dataset中所有的bug
    dataset_dir = 'defects4j/dataset'
    dataset_bugs = set()
    if os.path.exists(dataset_dir):
        for json_file in glob.glob(os.path.join(dataset_dir, '*.json')):
            bug_id = os.path.basename(json_file).replace('.json', '')
            dataset_bugs.add(bug_id)
    
    results1 = load_previous_results(model_id1)
    results2 = load_previous_results(model_id2)
    
    # 为缺失的bug添加NO_OUTPUT占位符（确保所有dataset中的bug都在分母中）
    no_output_placeholder = [
        {
            'patch_code': '/* NO LLM OUTPUT */',
            'patch_status': 'NO_OUTPUT',
            'failing_tests': {'TRIGGER': [], 'RELEVANT': [], 'TIMEOUT': []},
            'val_cnt': 0,
            'bug_name': '',
            'diff_stats': None
        }
    ] * 10
    
    for bug_id in dataset_bugs:
        if bug_id not in results1:
            results1[bug_id] = no_output_placeholder.copy()
        if bug_id not in results2:
            results2[bug_id] = no_output_placeholder.copy()
    
    filtered_results1 = {bug_id: results for bug_id, results in results1.items() 
                        if len(results) >= min_patches}
    filtered_results2 = {bug_id: results for bug_id, results in results2.items() if len(results) >= min_patches}
    
    bug_lengths = {}
    for bug_id in filtered_results1.keys() | filtered_results2.keys():
        # 尝试从dataset获取bug信息
        json_file = os.path.join('defects4j/dataset', f'{bug_id}.json')
        try:
            with open(json_file, 'r') as f:
                patch_info = json.load(f)
                total_length = len(patch_info.get('issue_title', '')) + len(patch_info.get('issue_description', '')) + len(patch_info.get('buggy', ''))
                bug_lengths[bug_id] = total_length
        except Exception as e:
            # 如果dataset中没有，尝试从结果目录获取
            json_file = os.path.join('defects4j/results', model_id1, 'fixed0', f'{bug_id}.json')
            try:
                with open(json_file, 'r') as f:
                    patch_info = json.load(f)
                    total_length = len(patch_info.get('issue_title', '')) + len(patch_info.get('issue_description', '')) + len(patch_info.get('buggy', ''))
                    bug_lengths[bug_id] = total_length
            except Exception as e2:
                print(f"[WARNING] Failed to load patch info for {bug_id}: {e2}")
    
    # 包含所有有任一模型输出的bug（不要求共同）
    all_bugs = sorted(set(filtered_results1.keys()) | set(filtered_results2.keys()))
    common_bugs = set(filtered_results1.keys()) & set(filtered_results2.keys())
    
    print("\n[VALIDATION COMPARISON SUMMARY]")
    print("=" * 145)
    print(f"Total bugs: {len(all_bugs)} (Common: {len(common_bugs)}, Only {model_id1}: {len(set(filtered_results1.keys()) - set(filtered_results2.keys()))}, Only {model_id2}: {len(set(filtered_results2.keys()) - set(filtered_results1.keys()))})")
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
            # 检查是否是NO_OUTPUT
            if results and results[0].get('patch_status') == 'NO_OUTPUT':
                status1 = "NO_OUTPUT"
            else:
                for idx, patch in enumerate(results, 1):
                    if patch.get('patch_status') == 'PLAUSIBLE':
                        status1 = f"PLAUSIBLE({idx})"
                        break
                if status1 == "N/A":
                    status1 = "FAILED"
        
        status2 = "N/A"
        position2 = ""
        if bug_id in filtered_results2:
            results = filtered_results2[bug_id]
            # 检查是否是NO_OUTPUT
            if results and results[0].get('patch_status') == 'NO_OUTPUT':
                status2 = "NO_OUTPUT"
            else:
                for idx, patch in enumerate(results, 1):
                    if patch.get('patch_status') == 'PLAUSIBLE':
                        status2 = f"PLAUSIBLE({idx})"
                        break
                if status2 == "N/A":
                    status2 = "FAILED"
        
        notes = ""
        if status1 == "NO_OUTPUT" and status2 == "NO_OUTPUT":
            notes = "Both no output"
        elif status1 == "NO_OUTPUT":
            notes = f"Only {model_id2} output"
        elif status2 == "NO_OUTPUT":
            notes = f"Only {model_id1} output"
        elif 'PLAUSIBLE' in status1 and 'PLAUSIBLE' in status2:
            notes = "Fixed by both"
        elif 'PLAUSIBLE' in status1:
            notes = f"Only fixed by {model_id1}"
        elif 'PLAUSIBLE' in status2:
            notes = f"Only fixed by {model_id2}"
        
        status1_color = '\033[92m' if 'PLAUSIBLE' in status1 else '\033[91m' if status1 == 'FAILED' else '\033[93m' if status1 == 'NO_OUTPUT' else '\033[0m'
        status2_color = '\033[92m' if 'PLAUSIBLE' in status2 else '\033[91m' if status2 == 'FAILED' else '\033[93m' if status2 == 'NO_OUTPUT' else '\033[0m'
        
        print(f"{bug_id:20} | {date:10} | {length:>8} | {status1_color}{status1:^15}\033[0m | {status2_color}{status2:^15}\033[0m | {notes:20}")
    
    print("=" * 145)
    
    # 排除NO_OUTPUT的bug进行统计
    bugs_with_output1 = {bug_id for bug_id, results in filtered_results1.items()
                        if not (results and results[0].get('patch_status') == 'NO_OUTPUT')}
    bugs_with_output2 = {bug_id for bug_id, results in filtered_results2.items()
                        if not (results and results[0].get('patch_status') == 'NO_OUTPUT')}
    
    fixed_bugs1 = {bug_id for bug_id, results in filtered_results1.items() 
                  if any(patch.get('patch_status') == 'PLAUSIBLE' for patch in results)}
    fixed_bugs2 = {bug_id for bug_id, results in filtered_results2.items() 
                  if any(patch.get('patch_status') == 'PLAUSIBLE' for patch in results)}
    
    common_fixed_bugs = fixed_bugs1 & fixed_bugs2
    only_fixed_by_1 = fixed_bugs1 - fixed_bugs2
    only_fixed_by_2 = fixed_bugs2 - fixed_bugs1
    
    print(f"\n[DATA COVERAGE]")
    print(f"Model {model_id1} bugs with LLM output: {len(bugs_with_output1)}")
    print(f"Model {model_id2} bugs with LLM output: {len(bugs_with_output2)}")
    print(f"Common bugs (both have output): {len(bugs_with_output1 & bugs_with_output2)}")
    print(f"Total bugs in dataset: {len(dataset_bugs)}")
    
    print(f"\n[FIX STATISTICS]")
    print(f"Model {model_id1} fixed total: {len(fixed_bugs1)}")
    print(f"Model {model_id2} fixed total: {len(fixed_bugs2)}")
    print(f"Fixed by both models: {len(common_fixed_bugs)}")
    print(f"Only fixed by {model_id1}: {len(only_fixed_by_1)} {sorted(only_fixed_by_1)}")
    print(f"Only fixed by {model_id2}: {len(only_fixed_by_2)} {sorted(only_fixed_by_2)}")
    
    filtered_stats1, filtered_stats2 = ValidationStats(), ValidationStats()
    
    # 统计所有有输出的bug（不只是common的）
    for bug_id in all_bugs:
        if bug_id in filtered_results1 and bug_id in bugs_with_output1:
            filtered_stats1.update(bug_id, filtered_results1[bug_id])
        if bug_id in filtered_results2 and bug_id in bugs_with_output2:
            filtered_stats2.update(bug_id, filtered_results2[bug_id])
    
    return filtered_stats1, filtered_stats2

def print_detailed_diff_comparison(stats1, stats2, model_id1, model_id2):
    """打印两个模型的详细diff统计对比"""
    
    # 对比 PLAUSIBLE 补丁的统计
    print(f"\n[DETAILED DIFF COMPARISON - PLAUSIBLE Patches]")
    print("=" * 120)
    
    diff_stats1 = stats1.diff_stats
    diff_stats2 = stats2.diff_stats
    
    if diff_stats1['patch_count'] == 0 or diff_stats2['patch_count'] == 0:
        print("[WARNING] One or both models have no PLAUSIBLE patches to compare")
        return
    
    print(f"{'Metric':<40} | {model_id1:>15} | {model_id2:>15} | {'Diff':>15} | {'Δ%':>10}")
    print("-" * 120)
    
    # 定义指标：(显示名称, 字段名, 是否显示差异)
    metrics = [
        ('Patch count', 'patch_count', False),
        ('Avg hunks', 'total_hunks', True),
        ('Avg added lines', 'total_added', True),
        ('Avg deleted lines', 'total_deleted', True),
        ('ATCL', 'total_changed_lines', True),
        ('Avg added tokens', 'total_added_tokens', True),
        ('Avg deleted tokens', 'total_deleted_tokens', True),
        ('ATCT', 'total_changed_tokens', True),
        ('AED', 'total_edit_distance', True),
        ('Avg edit similarity (%)', 'total_edit_similarity', True),
        ('Avg normalized edit distance', 'total_norm_edit_distance', True),
        ('CCR (%)', 'total_preserved', True)
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
                
                # 简化颜色逻辑：增加用绿色，减少用红色
                if diff > 0:
                    color = '\033[92m'  # 绿色
                elif diff < 0:
                    color = '\033[91m'  # 红色
                else:
                    color = '\033[0m'   # 无颜色
                
                print(f"{metric_name:<40} | {val1:>15.2f} | {val2:>15.2f} | {color}{diff:>+15.2f}\033[0m | {color}{pct_change:>+9.2f}%\033[0m")
            else:
                print(f"{metric_name:<40} | {val1:>15.2f} | {val2:>15.2f} | {'-':>15} | {'-':>10}")
    
    print("=" * 120)

def print_comparison_results(stats1, stats2, model_id1, model_id2):
    # 先打印详细的diff统计对比
    print_detailed_diff_comparison(stats1, stats2, model_id1, model_id2)
    
    # 再打印 Pass@k 对比
    print("\n[COMPARISON RESULTS - Pass@k]")
    print("=" * 80)
    print(f"{'Metric':15} | {model_id1:>15} | {model_id2:>15} | {'Diff':>10}")
    print("-" * 80)
    
    pass1_1, pass5_1, pass10_1 = stats1.get_pass_at_k()
    pass1_2, pass5_2, pass10_2 = stats2.get_pass_at_k()
    
    metrics = [
        ("pass@1", pass1_1, pass1_2),
        ("pass@5", pass5_1, pass5_2),
        ("pass@10", pass10_1, pass10_2)
    ]
    
    for metric_name, val1, val2 in metrics:
        diff = val2 - val1
        diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
        color = '\033[92m' if diff > 0 else '\033[91m' if diff < 0 else '\033[0m'
        print(f"{metric_name:15} | {val1:>14.2f}% | {val2:>14.2f}% | {color}{diff_str:>10}\033[0m")
    
    print("=" * 80)
    print(f"Total bugs compared: {stats1.total_bugs}")

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
    - preserved_ratio: CCR代码一致性比率 (%)
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

def print_diff_stats(diff_stats):
    """打印补丁的差异统计信息"""
    print(f"[DIFF STATS]   | Added: {diff_stats['added_lines']}, Deleted: {diff_stats['deleted_lines']}, CCR: {diff_stats['preserved_ratio']}%")

def print_diff_statistics_summary(stats_dict, title):
    """打印diff统计汇总信息（只针对PLAUSIBLE补丁）
    
    Args:
        stats_dict: 包含统计信息的字典
        title: 统计标题（如 "PLAUSIBLE Patches"）
    """
    patch_count = stats_dict['patch_count']
    if patch_count == 0:
        return
    
    dist = stats_dict['preservation_distribution']
    
    print(f"\n[PATCH MODIFICATION STATISTICS - {title}]")
    print(f"{'='*80}")
    print(f"Total patches analyzed: {patch_count}")
    print(f"\n{'Metric':<40} | {'Average':>15}")
    print(f"{'-'*80}")
    print(f"{'Hunks per patch':<40} | {stats_dict['total_hunks']/patch_count:>15.2f}")
    print(f"{'Lines added per patch':<40} | {stats_dict['total_added']/patch_count:>15.2f}")
    print(f"{'Lines deleted per patch':<40} | {stats_dict['total_deleted']/patch_count:>15.2f}")
    print(f"{'ATCL per patch':<40} | {stats_dict['total_changed_lines']/patch_count:>15.2f}")
    print(f"{'Tokens added per patch':<40} | {stats_dict['total_added_tokens']/patch_count:>15.2f}")
    print(f"{'Tokens deleted per patch':<40} | {stats_dict['total_deleted_tokens']/patch_count:>15.2f}")
    print(f"{'ATCT per patch':<40} | {stats_dict['total_changed_tokens']/patch_count:>15.2f}")
    print(f"{'AED per patch':<40} | {stats_dict['total_edit_distance']/patch_count:>15.2f}")
    print(f"{'Edit similarity (%)':<40} | {stats_dict['total_edit_similarity']/patch_count:>15.2f}")
    print(f"{'Normalized edit distance':<40} | {stats_dict['total_norm_edit_distance']/patch_count:>15.4f}")
    print(f"{'CCR (%)':<40} | {stats_dict['total_preserved']/patch_count:>15.2f}")
    
    print(f"\n{'Distribution of CCR:':}")
    total = sum(dist.values())
    if total > 0:
        print(f"  Minimal change   (>95% preserved):   {dist['high']:3d} patches ({dist['high']/total*100:5.1f}%)")
        print(f"  Moderate change (80-95% preserved):  {dist['medium']:3d} patches ({dist['medium']/total*100:5.1f}%)")
        print(f"  Major change     (<80% preserved):   {dist['low']:3d} patches ({dist['low']/total*100:5.1f}%)")
    print(f"{'='*80}")

def recalculate_diff_stats(model_id):
    """重新计算所有已验证patch的diff统计，保存到.result文件"""
    print(f"[INFO] 重新计算 {model_id} 的统计信息")
    results_dir = os.path.join('defects4j/results', model_id)
    fixed0_dir = os.path.join(results_dir, 'fixed0')
    
    if not os.path.exists(fixed0_dir):
        print(f"[ERROR] fixed0目录不存在: {fixed0_dir}")
        return
    
    # 统计信息
    total_files = 0
    updated_files = 0
    skipped_files = 0
    
    # 扫描 fixed0 目录下的所有 .result 文件
    result_files = glob.glob(os.path.join(fixed0_dir, '*.json.result'))
    
    for result_file in result_files:
        total_files += 1
        # 提取 bug_id (例如: Chart-1.json.result -> Chart-1)
        filename = os.path.basename(result_file)
        bug_id = filename.replace('.json.result', '')
        
        try:
            # 读取dataset中的buggy代码
            dataset_file = os.path.join('defects4j/dataset', f'{bug_id}.json')
            if not os.path.exists(dataset_file):
                print(f"[警告] Dataset文件不存在: {dataset_file}")
                skipped_files += 1
                continue
            
            with open(dataset_file) as df:
                dataset_info = json.load(df)
                buggy_code = dataset_info['buggy']
            
            # 读取验证结果
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # 只对 PLAUSIBLE 补丁重新计算统计信息
            plausible_count = 0
            total_patches = len(results)
            
            for idx, patch in enumerate(results, 1):
                # 只对 PLAUSIBLE 补丁计算详细的 diff 统计
                if patch['patch_status'] == 'PLAUSIBLE':
                    patch['diff_stats'] = calc_diff_stats(buggy_code, patch['patch_code'])
                    plausible_count += 1
                else:
                    # 非 PLAUSIBLE 补丁不计算详细统计
                    patch['diff_stats'] = None
            
            # 保存更新后的结果
            with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            status_msg = f"[更新] {bug_id} - {total_patches} 个补丁"
            if plausible_count > 0:
                status_msg += f" (其中 {plausible_count} 个PLAUSIBLE)"
            print(status_msg)
            updated_files += 1
                
        except Exception as e:
            print(f"[错误] 处理 {result_file} 时出错: {e}")
            traceback.print_exc()
            skipped_files += 1
    
    print(f"\n[完成] 统计信息重新计算完成")
    print(f"- 总文件数: {total_files}")
    print(f"- 已更新: {updated_files}")
    print(f"- 已跳过: {skipped_files}")

if __name__ == '__main__':
    start_val_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_id', type=str, required=True)
    parser.add_argument('-n', '--n_generations', type=int, default=1)
    parser.add_argument('--recalc', action='store_true', help='重新计算diff统计和Pass@k，不重新运行测试')
    parser.add_argument('--compare', type=str, default=None, help='对比另一个模型的结果')
    parser.add_argument('--min_patches', type=int, default=10, help='对比时最小patch数量要求（默认10）')
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        print(f"\n[COMPARE MODE] Comparing {args.model_id} vs {args.compare}")
        print(f"[FILTER] Only comparing bugs with >={args.min_patches} patches\n")
        
        stats1, stats2 = load_and_compare_results(args.model_id, args.compare, args.min_patches)
        print_comparison_results(stats1, stats2, args.model_id, args.compare)
        
    elif args.recalc:
        # 重新计算diff统计和Pass@k
        print(f"[INFO] 重新计算 {args.model_id} 的diff统计和Pass@k")
        
        # 1. 重新计算diff统计
        recalculate_diff_stats(args.model_id)
        
        # 2. 重新计算Pass@k
        stats = ValidationStats()
        previous_results = load_previous_results(args.model_id)
        print(f"\n[INFO] Loaded {len(previous_results)} bugs with exactly 10 patches")
        
        for bug_id, results in previous_results.items():
            stats.update(bug_id, results)
        
        pass1, pass5, pass10 = stats.get_pass_at_k()
        print("\n[RECALCULATED SUCCESS RATE - Pass@k]")
        print("=" * 80)
        print(f"pass@1:  {pass1:6.2f}%")
        print(f"pass@5:  {pass5:6.2f}%")
        print(f"pass@10: {pass10:6.2f}%")
        print("=" * 80)
        
        # 打印 PLAUSIBLE 补丁的详细统计信息
        print_diff_statistics_summary(stats.diff_stats, "PLAUSIBLE Patches")
    else:
        validate_defects4j(args.model_id, args.n_generations)
