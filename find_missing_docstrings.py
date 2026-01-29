#!/usr/bin/env python3
"""
查找项目中没有文档字符串的函数定义
"""

import os
import re

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 要检查的目录
CHECK_DIRS = [
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "tests"),
    PROJECT_ROOT,
]

# 要排除的目录
EXCLUDE_DIRS = [
    "__pycache__",
    ".pytest_cache",
    "cache",
    "logs",
    "models",
    "auto_examples_python",
]

# 函数定义的正则表达式
FUNCTION_PATTERN = re.compile(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(.*\):\s*$")

# 类定义的正则表达式
CLASS_PATTERN = re.compile(r"^\s*class\s+([a-zA-Z_]\w*)\s*\(.*\):\s*$")


def is_excluded(file_path):
    """检查文件是否应该被排除"""
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in file_path:
            return True
    return False


def has_docstring(lines, start_line):
    """检查函数是否有文档字符串"""
    for i in range(start_line + 1, len(lines)):
        line = lines[i].strip()
        if line:
            # 检查是否是文档字符串
            if line.startswith('"""') or line.startswith("'''"):
                return True
            # 如果遇到非空白行且不是文档字符串，则没有文档字符串
            else:
                return False
    return False


def find_missing_docstrings():
    """查找没有文档字符串的函数"""
    missing_docstrings = []

    for check_dir in CHECK_DIRS:
        for root, dirs, files in os.walk(check_dir):
            # 过滤掉排除的目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if is_excluded(file_path):
                        continue

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except UnicodeDecodeError:
                        continue

                    for i, line in enumerate(lines):
                        # 检查函数定义
                        match = FUNCTION_PATTERN.match(line)
                        if match:
                            function_name = match.group(1)
                            # 跳过特殊函数
                            if function_name.startswith(
                                "__"
                            ) and function_name.endswith("__"):
                                continue
                            # 检查是否有文档字符串
                            if not has_docstring(lines, i):
                                missing_docstrings.append(
                                    {
                                        "file": file_path,
                                        "line": i + 1,
                                        "function": function_name,
                                    }
                                )

                        # 检查类定义
                        match = CLASS_PATTERN.match(line)
                        if match:
                            class_name = match.group(1)
                            # 检查是否有文档字符串
                            if not has_docstring(lines, i):
                                missing_docstrings.append(
                                    {
                                        "file": file_path,
                                        "line": i + 1,
                                        "class": class_name,
                                    }
                                )

    return missing_docstrings


def main():
    """主函数"""
    missing = find_missing_docstrings()

    if missing:
        print(f"找到 {len(missing)} 个没有文档字符串的函数或类：")
        for item in missing[:50]:  # 只显示前50个
            if "function" in item:
                print(f"{item['file']}:{item['line']}: 函数 {item['function']}")
            else:
                print(f"{item['file']}:{item['line']}: 类 {item['class']}")
    else:
        print("所有函数和类都有文档字符串！")


if __name__ == "__main__":
    main()
