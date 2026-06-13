#!/usr/bin/env python3

import ast
import subprocess
import sys

MAX_LINES = 20

def get_staged_python_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        f.strip()
        for f in result.stdout.splitlines()
        if f.endswith(".py")
    ]


def check_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=filename)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "end_lineno"):
                length = node.end_lineno - node.lineno + 1
                if length > MAX_LINES:
                    violations.append(
                        (
                            node.name,
                            node.lineno,
                            length,
                        )
                    )
    return violations

def main():
    failed = False
    for file in get_staged_python_files():
        violations = check_file(file)
        for name, line, length in violations:
            failed = True
            print(
                f"{file}:{line}: "
                f"function '{name}' is {length} lines "
                f"(max {MAX_LINES})"
            )
    if failed:
        print("\nCommit rejected.")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()