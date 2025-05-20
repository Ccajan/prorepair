import os
import json
import subprocess
import difflib
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class D4JEvaluator:
    def __init__(self, model_path: str = "path/to/llama3-8b", device: str = "cuda", defects4j_home: str = None):
        self.device = device

        if defects4j_home is None:
            defects4j_home = os.environ.get("DEFECTS4J_HOME")
            if defects4j_home is None:
                raise ValueError("DEFECTS4J_HOME is not set in environment or passed as argument.")

        self.defects4j_cmd = os.path.join(defects4j_home, "framework", "bin", "defects4j")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

    def extract_bug_pairs(self, project: str, bug_id: str, work_dir: str) -> Dict:
        """从 Defects4J 提取 buggy/fixed 代码对"""
        buggy_dir = os.path.join(work_dir, f"{project}_{bug_id}_buggy")
        fixed_dir = os.path.join(work_dir, f"{project}_{bug_id}_fixed")

        os.makedirs(buggy_dir, exist_ok=True)
        os.makedirs(fixed_dir, exist_ok=True)

        # Checkout buggy and fixed versions
        subprocess.run(f"{self.defects4j_cmd} checkout -p {project} -v {bug_id}b -w {buggy_dir}",
                       shell=True, check=True)
        subprocess.run(f"{self.defects4j_cmd} checkout -p {project} -v {bug_id}f -w {fixed_dir}",
                       shell=True, check=True)

        # Extract metadata
        result = subprocess.run(f"{self.defects4j_cmd} info -p {project} -b {bug_id}",
                                shell=True, capture_output=True, text=True)

        modified_file, bug_desc, failing_tests = None, "", []
        in_modified_section = False
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("List of modified sources:"):
                in_modified_section = True
                continue
            if in_modified_section and line.startswith("- "):
                class_name = line[2:].strip()
                modified_file = os.path.join("src", class_name.replace('.', '/') + ".java")
            elif "Bug report:" in line:
                bug_desc = line.split(":", 1)[1].strip()
            elif "Failing tests:" in line:
                failing_tests = line.split(":", 1)[1].strip().split(',')

        if not modified_file:
            raise ValueError(f"Could not find modified file for {project}-{bug_id}")

        # 构造完整路径并检查文件是否存在
        buggy_path = os.path.join(buggy_dir, modified_file)
        fixed_path = os.path.join(fixed_dir, modified_file)

        if not os.path.isfile(buggy_path):
            raise FileNotFoundError(f"Buggy file does not exist: {buggy_path}")
        if not os.path.isfile(fixed_path):
            raise FileNotFoundError(f"Fixed file does not exist: {fixed_path}")

        with open(buggy_path, "r", encoding="utf-8") as f:
            buggy_code = f.read()
        with open(fixed_path, "r", encoding="utf-8") as f:
            fixed_code = f.read()

        return {
            "project": project,
            "bug_id": bug_id,
            "buggy_dir": buggy_dir,
            "fixed_dir": fixed_dir,
            "modified_file": modified_file,
            "buggy_code": buggy_code,
            "fixed_code": fixed_code,
            "bug_desc": bug_desc,
            "failing_tests": failing_tests
        }

    def construct_prompt(self, bug_info: Dict) -> str:
        """构造LLaMA模型的输入提示"""
        return f""""You are a code repair expert. Your task is to fix the given buggy Java code 
        with the minimal necessary change to make it correct.

Bug Description:
{bug_info['bug_desc']}

Buggy Code:
```java
{bug_info['buggy_code']}
```

Please provide the fixed code. Only output the fixed code without any explanation:"""

    def generate_repair(self, prompt: str) -> str:
        """使用LLaMA模型生成修复代码"""
        # print(f"\n[Debug] Prompt length (chars): {len(prompt)}")
        # print(f"[Debug] Prompt preview:\n{'=' * 40}\n{prompt[:1000]}\n{'=' * 40}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        repair = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        repair = repair.split("Please provide the fixed code:")[-1].strip()
        return repair.replace("```java", "").replace("```", "").strip()

    def apply_repair(self, bug_info: Dict, repaired_code: str):
        """将修复应用到buggy版本"""
        with open(os.path.join(bug_info["buggy_dir"], bug_info["modified_file"]), "w", encoding="utf-8") as f:
            f.write(repaired_code)

    def verify_repair(self, bug_info: Dict) -> bool:
        """验证修复是否正确"""
        try:
            compile_cmd = f"cd {bug_info['buggy_dir']} && {self.defects4j_cmd} compile"
            if subprocess.run(compile_cmd, shell=True).returncode != 0:
                print(f"Compilation failed for {bug_info['project']}-{bug_info['bug_id']}")
                return False

            for test in bug_info["failing_tests"]:
                test_cmd = f"cd {bug_info['buggy_dir']} && {self.defects4j_cmd} test -t {test.strip()}"
                if subprocess.run(test_cmd, shell=True).returncode != 0:
                    print(f"Test {test} failed for {bug_info['project']}-{bug_info['bug_id']}")
                    return False
            return True
        except Exception as e:
            print(f"Error verifying repair for {bug_info['project']}-{bug_info['bug_id']}: {str(e)}")
            return False

    def compute_diff_stats(self, original: str, repaired: str) -> Dict:
        """计算差异统计"""
        diff = list(difflib.unified_diff(
            original.splitlines(), repaired.splitlines(), lineterm=''
        ))
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        return {
            "additions": additions,
            "deletions": deletions,
            "total_changes": additions + deletions
        }

    def evaluate(self, projects: List[str], work_dir: str):
        """运行评估流程"""
        results = []
        for project in projects:
            info_cmd = f"{self.defects4j_cmd} info -p {project}"
            result = subprocess.run(info_cmd, shell=True, capture_output=True, text=True)

            num_bugs = 0
            for line in result.stdout.splitlines():
                if "Number of bugs:" in line:
                    num_bugs = int(line.split(":")[1].strip())
                    break

            for bug_id in tqdm(range(1, num_bugs + 1), desc=f"评估 {project}"):
                try:
                    bug_info = self.extract_bug_pairs(project, str(bug_id), work_dir)
                    prompt = self.construct_prompt(bug_info)
                    repaired_code = self.generate_repair(prompt)
                    self.apply_repair(bug_info, repaired_code)
                    is_correct = self.verify_repair(bug_info)
                    diff_stats = self.compute_diff_stats(bug_info["buggy_code"], repaired_code)
                    results.append({
                        "project": project,
                        "bug_id": bug_id,
                        "is_correct": is_correct,
                        "diff_stats": diff_stats
                    })
                except Exception as e:
                    print(f"Error processing {project}-{bug_id}: {str(e)}")
                    continue

        # 输出汇总报告
        total_bugs = len(results)
        correct_repairs = sum(1 for r in results if r["is_correct"])
        total_changes = sum(r["diff_stats"]["total_changes"] for r in results)

        print("\n=== Evaluation Report ===")
        print(f"Total bugs evaluated: {total_bugs}")
        print(f"Correct repairs: {correct_repairs}")
        print(f"Repair accuracy: {correct_repairs / total_bugs:.2%}")
        print(f"Average changes per repair: {total_changes / total_bugs:.2f} lines")

        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def main():
    MODEL_PATH = "/home/liu01/projects/Programrepair/model"
    WORK_DIR = "/home/liu01/projects/Programrepair/d4j_work"
    DEFECTS4J_HOME = "/home/liu01/projects/Programrepair/data/defects4j"

    os.environ["DEFECTS4J_HOME"] = DEFECTS4J_HOME
    os.makedirs(WORK_DIR, exist_ok=True)

    evaluator = D4JEvaluator(model_path=MODEL_PATH, defects4j_home=DEFECTS4J_HOME)
    projects = ["Chart", "Closure", "Lang", "Math", "Mockito", "Time"]
    evaluator.evaluate(projects, WORK_DIR)


if __name__ == "__main__":
    main()