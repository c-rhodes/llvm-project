#!/usr/bin/env python3
"""Find duplicate LLVM IR test functions after normalization.

This utility is intentionally LLVM-IR-specific. It scans textual `.ll` files,
handles `split-file`-style multi-module containers, normalizes each module with
`opt -passes=strip,normalize`, then groups functions whose normalized bodies are
identical.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


SPLIT_FILE_MARKER_RE = re.compile(r"^.\-\-\- (.+)$")
GLOBAL_VALUE_RE = re.compile(r"@(?:[-a-zA-Z$._0-9]+|\"(?:[^\"\\]|\\.)+\")")
FUNCTION_NAME_RE = re.compile(r"^\s*define\b.*?(" + GLOBAL_VALUE_RE.pattern + r")\s*\(")
ATTRIBUTE_GROUP_RE = re.compile(r"#(\d+)\b")
ATTRIBUTE_DEFINITION_RE = re.compile(r"^attributes #(\d+) = \{(.*)\}$")


@dataclass(frozen=True)
class ModuleSlice:
    path: str
    part_name: str | None
    text: str


@dataclass(frozen=True)
class FunctionInstance:
    location: str
    normalized_body: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find duplicate LLVM IR test functions after `opt "
            "-passes=strip,normalize` canonicalization."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="LLVM IR test files or directories to scan",
    )
    parser.add_argument(
        "--opt-binary",
        default="opt",
        help="Path to the `opt` binary to use (default: opt from PATH)",
    )
    parser.add_argument(
        "--min-group-size",
        type=positive_int,
        default=2,
        help="Only print groups with at least this many functions (default: 2)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=1,
        help="Number of files to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped files and normalization failures to stderr",
    )
    return parser.parse_args()


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def discover_ir_files(paths: Iterable[str]) -> list[str]:
    files: list[str] = []
    for path in paths:
        normalized = os.path.normpath(path)
        if os.path.isdir(normalized):
            for dirpath, _, filenames in os.walk(normalized):
                for filename in sorted(f for f in filenames if f.endswith(".ll")):
                    files.append(os.path.join(dirpath, filename))
        elif os.path.isfile(normalized):
            if normalized.endswith(".ll"):
                files.append(normalized)
        else:
            raise FileNotFoundError(f"input path does not exist: {path}")

    # Keep output deterministic and avoid duplicate work when the same file was
    # named multiple times.
    return sorted(dict.fromkeys(files))


def split_modules(path: str, text: str) -> list[ModuleSlice]:
    parts: list[ModuleSlice] = []
    current_name: str | None = None
    current_lines: list[str] = []
    saw_marker = False

    for line in text.splitlines(keepends=True):
        match = SPLIT_FILE_MARKER_RE.match(line)
        if match:
            saw_marker = True
            if current_name is not None:
                parts.append(
                    ModuleSlice(path=path, part_name=current_name, text="".join(current_lines))
                )
            current_name = match.group(1).strip()
            current_lines = []
            continue

        if current_name is not None:
            current_lines.append(line)

    if not saw_marker:
        return [ModuleSlice(path=path, part_name=None, text=text)]

    if current_name is not None:
        parts.append(ModuleSlice(path=path, part_name=current_name, text="".join(current_lines)))

    return [part for part in parts if part.part_name and part.part_name.endswith(".ll")]


def normalize_module(module: ModuleSlice, opt_binary: str) -> str:
    command = [opt_binary, "-S", "-passes=strip,normalize", "-o", "-", "-"]
    process = subprocess.run(
        command,
        input=module.text,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        part = f"[{module.part_name}]" if module.part_name else ""
        raise RuntimeError(f"{module.path}{part}: {process.stderr.strip()}")
    return process.stdout


def parse_attribute_groups(module_text: str) -> dict[str, str]:
    groups = {}
    for line in module_text.splitlines():
        match = ATTRIBUTE_DEFINITION_RE.match(line)
        if match:
            groups[match.group(1)] = " ".join(match.group(2).split())
    return groups


def extract_functions(module_text: str) -> list[tuple[str, str]]:
    functions: list[tuple[str, str]] = []
    lines = module_text.splitlines()
    current_lines: list[str] | None = None
    current_name: str | None = None

    for line in lines:
        if current_lines is None:
            match = FUNCTION_NAME_RE.match(line)
            if not match:
                continue
            current_lines = [line]
            current_name = match.group(1)
            continue

        current_lines.append(line)
        if line.strip() == "}":
            assert current_name is not None
            functions.append((current_name, "\n".join(current_lines)))
            current_lines = None
            current_name = None

    return functions


def canonicalize_function(function_name: str, function_text: str, attr_groups: dict[str, str]) -> str:
    def replace_attr_group(match: re.Match[str]) -> str:
        group = attr_groups.get(match.group(1))
        if group is None:
            return match.group(0)
        return f"#attr({group})"

    def replace_global_value(match: re.Match[str]) -> str:
        if match.group(0) == function_name:
            return "@__func__"
        return match.group(0)

    normalized = ATTRIBUTE_GROUP_RE.sub(replace_attr_group, function_text)
    normalized = GLOBAL_VALUE_RE.sub(replace_global_value, normalized)
    return normalized.strip()


def collect_function_instances(
    files: Iterable[str], opt_binary: str, verbose: bool
) -> list[FunctionInstance]:
    instances: list[FunctionInstance] = []
    for path in files:
        instances.extend(collect_function_instances_for_file(path, opt_binary, verbose))
    return instances


def collect_function_instances_for_file(
    path: str, opt_binary: str, verbose: bool
) -> list[FunctionInstance]:
    instances: list[FunctionInstance] = []
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    for module in split_modules(path, text):
        try:
            normalized_module = normalize_module(module, opt_binary)
        except RuntimeError as exc:
            if verbose:
                print(f"warning: skipping {exc}", file=sys.stderr)
            continue

        attr_groups = parse_attribute_groups(normalized_module)
        for function_name, function_text in extract_functions(normalized_module):
            label = module.path
            if module.part_name is not None:
                label += f"[{module.part_name}]"
            label += f"::{function_name.lstrip('@')}"
            instances.append(
                FunctionInstance(
                    location=label,
                    normalized_body=canonicalize_function(
                        function_name, function_text, attr_groups
                    ),
                )
            )

    return instances


def collect_function_instances_parallel(
    files: list[str], opt_binary: str, verbose: bool, jobs: int
) -> list[FunctionInstance]:
    if jobs == 1:
        return collect_function_instances(files, opt_binary, verbose)

    instances: list[FunctionInstance] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(collect_function_instances_for_file, path, opt_binary, verbose)
            for path in files
        ]
        for future in futures:
            instances.extend(future.result())
    return instances


def group_duplicates(
    functions: Iterable[FunctionInstance], min_group_size: int
) -> list[list[FunctionInstance]]:
    groups: dict[str, list[FunctionInstance]] = collections.defaultdict(list)
    for function in functions:
        groups[function.normalized_body].append(function)

    duplicate_groups = [
        sorted(group, key=lambda function: function.location)
        for group in groups.values()
        if len(group) >= min_group_size
    ]
    duplicate_groups.sort(key=lambda group: [function.location for function in group])
    return duplicate_groups


def main() -> int:
    args = parse_args()
    try:
        files = discover_ir_files(args.paths)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    functions = collect_function_instances_parallel(
        files, args.opt_binary, args.verbose, args.jobs
    )
    groups = group_duplicates(functions, args.min_group_size)

    print(
        f"Found {len(groups)} duplicate groups across "
        f"{sum(len(group) for group in groups)} functions."
    )
    for i, group in enumerate(groups, 1):
        print(f"Group {i}: {len(group)} functions")
        for function in group:
            print(f"  {function.location}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
