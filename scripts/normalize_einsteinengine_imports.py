#!/usr/bin/env python3

# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
#
# This file is part of the Einstein Engine (EinsteinEngine).
#
# EinsteinEngine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EinsteinEngine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PACKAGE_NAME = "EinsteinEngine"
DEFAULT_SKIP_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "venv",
}


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    text: str


def build_line_offsets(src: str) -> list[int]:
    offsets = [0]
    for idx, ch in enumerate(src):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def position_to_offset(line_offsets: Sequence[int], lineno: int, col_offset: int) -> int:
    return line_offsets[lineno - 1] + col_offset


def gather_python_files(root: Path, skip_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".pyi"}:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def is_init_file(path: Path) -> bool:
    return path.name in {"__init__.py", "__init__.pyi"}


def path_to_module(path: Path, base: Path) -> str | None:
    try:
        rel = path.relative_to(base)
    except ValueError:
        return None

    parts = list(rel.parts)
    if not parts:
        return None

    parts[-1] = Path(parts[-1]).stem
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def build_local_modules(package_root: Path) -> set[str]:
    modules: set[str] = {PACKAGE_NAME}
    if not package_root.is_dir():
        return modules

    for file_path in package_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix not in {".py", ".pyi"}:
            continue
        module_suffix = path_to_module(file_path, package_root)
        if module_suffix is None:
            continue
        modules.add(f"{PACKAGE_NAME}.{module_suffix}")
    return modules


def is_known_module(module: str, local_modules: set[str]) -> bool:
    if module in local_modules:
        return True
    module_prefix = f"{module}."
    return any(candidate.startswith(module_prefix) for candidate in local_modules)


def aliases_to_text(aliases: Sequence[ast.alias]) -> str:
    pieces: list[str] = []
    for alias in aliases:
        if alias.asname:
            pieces.append(f"{alias.name} as {alias.asname}")
        else:
            pieces.append(alias.name)
    return ", ".join(pieces)


def module_context_for_file(file_path: Path, repo_root: Path) -> tuple[str | None, bool]:
    module_name = path_to_module(file_path, repo_root)
    if module_name is None:
        return None, False
    return module_name, is_init_file(file_path)


def package_context(module_name: str, file_is_package: bool) -> str:
    if file_is_package:
        return module_name
    split = module_name.split(".")
    return ".".join(split[:-1])


def resolve_relative_module(
    file_path: Path,
    repo_root: Path,
    level: int,
    module: str | None,
) -> str | None:
    if level <= 0:
        return module

    module_name, file_is_package = module_context_for_file(file_path, repo_root)
    if module_name is None:
        return None

    context = package_context(module_name, file_is_package)
    context_parts = [part for part in context.split(".") if part]

    ascends = level - 1
    if ascends > len(context_parts):
        return None

    base_parts = context_parts[: len(context_parts) - ascends]
    module_parts = [part for part in (module or "").split(".") if part]
    resolved_parts = [*base_parts, *module_parts]
    if not resolved_parts:
        return None
    return ".".join(resolved_parts)


def build_import_text(node: ast.Import, local_modules: set[str]) -> str | None:
    changed = False
    rewritten: list[ast.alias] = []
    for alias in node.names:
        new_name = alias.name
        if not new_name.startswith(f"{PACKAGE_NAME}.") and new_name != PACKAGE_NAME:
            candidate = f"{PACKAGE_NAME}.{new_name}"
            if is_known_module(candidate, local_modules):
                new_name = candidate
                changed = True
        rewritten.append(ast.alias(name=new_name, asname=alias.asname))

    if not changed:
        return None
    return f"import {aliases_to_text(rewritten)}"


def build_import_from_text(
    node: ast.ImportFrom,
    file_path: Path,
    repo_root: Path,
    local_modules: set[str],
) -> str | None:
    target_module: str | None = None
    if node.level > 0:
        resolved = resolve_relative_module(file_path, repo_root, node.level, node.module)
        if resolved is None or not resolved.startswith(PACKAGE_NAME):
            return None
        target_module = resolved
    else:
        if node.module is None:
            return None
        if node.module == PACKAGE_NAME or node.module.startswith(f"{PACKAGE_NAME}."):
            return None
        candidate = f"{PACKAGE_NAME}.{node.module}"
        if is_known_module(candidate, local_modules):
            target_module = candidate

    if target_module is None:
        return None

    return f"from {target_module} import {aliases_to_text(node.names)}"


def rewrite_file(
    file_path: Path,
    repo_root: Path,
    local_modules: set[str],
    *,
    write: bool,
) -> tuple[bool, str]:
    src = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False, "skipped (syntax error)"

    line_offsets = build_line_offsets(src)
    replacements: list[Replacement] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            replacement_body = build_import_text(node, local_modules)
        elif isinstance(node, ast.ImportFrom):
            replacement_body = build_import_from_text(node, file_path, repo_root, local_modules)
        else:
            continue

        if replacement_body is None:
            continue
        if node.end_lineno is None or node.end_col_offset is None:
            continue

        start = position_to_offset(line_offsets, node.lineno, node.col_offset)
        end = position_to_offset(line_offsets, node.end_lineno, node.end_col_offset)
        indent = src[line_offsets[node.lineno - 1] : start]
        replacements.append(Replacement(start=start, end=end, text=f"{indent}{replacement_body}"))

    if not replacements:
        return False, "unchanged"

    new_src = src
    for repl in sorted(replacements, key=lambda item: item.start, reverse=True):
        new_src = new_src[: repl.start] + repl.text + new_src[repl.end :]

    if new_src == src:
        return False, "unchanged"

    if write:
        file_path.write_text(new_src, encoding="utf-8")
    return True, "rewritten"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize imports that refer to the EinsteinEngine codebase so that "
            "they use fully-qualified paths beginning with 'EinsteinEngine'."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan (default: current working directory).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: report files that would change, but do not rewrite them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only summary output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    package_root = repo_root / PACKAGE_NAME
    local_modules = build_local_modules(package_root)

    files = gather_python_files(repo_root, DEFAULT_SKIP_DIRS)
    changed_files: list[Path] = []

    for file_path in files:
        changed, status = rewrite_file(file_path, repo_root, local_modules, write=not args.check)
        if changed:
            changed_files.append(file_path)
        if not args.quiet and status != "unchanged":
            print(f"{status}: {file_path.relative_to(repo_root)}")

    if args.check:
        if changed_files:
            print(f"{len(changed_files)} file(s) would be rewritten.")
            return 1
        print("No import normalization changes needed.")
        return 0

    print(f"Rewrote {len(changed_files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
