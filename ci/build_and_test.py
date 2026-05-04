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

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


class CIError(RuntimeError):
    """Error with an explicit process exit code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TestEntry:
    """One test entry from test.json (thorn/test_name)."""

    thorn: str
    test_name: str
    data_path: Path
    par_path: Path

    @property
    def pair(self) -> str:
        return f"{self.thorn}/{self.test_name}"


@dataclass(frozen=True)
class CIConfig:
    """Resolved, convention-derived configuration for one test recipe."""

    recipe_group_name: str
    arrangement_name: str
    suite_name: str
    source_makefile: Path
    source_arrangement_dir: Path
    test_root_dir: Path
    tests: Tuple[str, ...]
    extra_thorns: Tuple[str, ...]
    extra_parfiles: Tuple[Path, ...]


def fail(message: str, code: int) -> None:
    raise CIError(message, code)


def require_env(name: str, code: int) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        fail(f"{name} is not set", code)
    return value


def resolve_config_path(repo_root: Path) -> Path:
    raw = require_env("CI_TEST_CONFIG", 5)
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root / p
    p = p.resolve()
    if not p.is_file():
        fail(f"CI test config is not readable: '{p}'", 6)
    return p


def resolve_cactus_paths() -> Tuple[Path, Path]:
    thornlist = Path(require_env("THORNLIST", 1)).resolve()
    if not thornlist.is_file():
        fail("THORNLIST is not readable", 2)

    cactus_dir_env = os.environ.get("CACTUS_DIR", "").strip()
    cactus_dir = Path(cactus_dir_env).resolve() if cactus_dir_env else thornlist.parent.parent.resolve()

    print(f"CACTUS_DIR: {cactus_dir}")

    if not (cactus_dir / "arrangements").is_dir():
        fail(f"Cannot find '{cactus_dir}/arrangements'", 3)
    if not (cactus_dir / "simfactory" / "etc" / "defs.local.ini").is_file():
        fail(f"Cannot find '{cactus_dir}/simfactory/etc/defs.local.ini'", 4)

    return thornlist, cactus_dir


def load_config(config_path: Path, repo_root: Path) -> CIConfig:
    config_dir = config_path.parent
    with config_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # The config lives at recipes/<recipe_group>/test/test.json.
    # Recipe groups own recipes/test data; arrangements own thorns.
    recipe_group_name = config_dir.parent.name
    arrangement_name = str(raw.get("arrangement_name", recipe_group_name))
    if not arrangement_name:
        fail(f"Could not derive arrangement_name from {config_path}", 6)

    suite_name = str(raw.get("suite_name", arrangement_name.lower()))
    if not suite_name:
        fail(f"suite_name must be non-empty in {config_path}", 6)

    recipes_dir = repo_root / "recipes" / recipe_group_name

    raw_tests = raw.get("tests")
    if not isinstance(raw_tests, list):
        fail(f"'tests' must be a list in {config_path}", 6)
    tests: List[str] = []
    for t in raw_tests:
        t_str = str(t).strip()
        if not t_str:
            fail(f"'tests' entries must be non-empty in {config_path}", 6)
        tests.append(t_str)
    if not tests:
        fail(f"'tests' must contain at least one entry in {config_path}", 6)

    raw_extra_thorns = raw.get("extra_thorns", [])
    if not isinstance(raw_extra_thorns, list):
        fail(f"'extra_thorns' must be a list in {config_path}", 6)
    extra_thorns: List[str] = []
    for thorn in raw_extra_thorns:
        thorn_str = str(thorn)
        if not thorn_str:
            fail(f"extra_thorns entries must be non-empty in {config_path}", 6)
        extra_thorns.append(thorn_str)

    raw_extra_parfiles = raw.get("extra_parfiles", [])
    if not isinstance(raw_extra_parfiles, list):
        fail(f"'extra_parfiles' must be a list in {config_path}", 6)
    extra_parfiles: List[Path] = []
    for p in raw_extra_parfiles:
        p_str = str(p)
        if Path(p_str).is_absolute():
            abs_path = Path(p_str)
        elif p_str.startswith("recipes/"):
            abs_path = repo_root / p_str
        else:
            abs_path = recipes_dir / p_str
        extra_parfiles.append(abs_path.resolve())

    return CIConfig(
        recipe_group_name=recipe_group_name,
        arrangement_name=arrangement_name,
        suite_name=suite_name,
        source_makefile=(recipes_dir / "Makefile").resolve(),
        source_arrangement_dir=(repo_root / "generated" / arrangement_name).resolve(),
        test_root_dir=(recipes_dir / "test").resolve(),
        tests=tuple(tests),
        extra_thorns=tuple(extra_thorns),
        extra_parfiles=tuple(extra_parfiles),
    )


def parse_tests(cfg: CIConfig) -> List[TestEntry]:
    if not cfg.source_makefile.is_file():
        fail(f"Cannot read source makefile: '{cfg.source_makefile}'", 7)
    if not cfg.test_root_dir.is_dir():
        fail(f"Cannot find test root dir: '{cfg.test_root_dir}'", 10)
    entries = list(cfg.tests)

    tests: List[TestEntry] = []
    for entry in entries:
        if "/" not in entry:
            fail(f"Invalid tests entry '{entry}'. Expected '<thorn>/<test_name>'.", 12)
        thorn_name, test_name = entry.split("/", 1)
        if not thorn_name or not test_name:
            fail(f"Invalid tests entry '{entry}'. Thorn or test name is empty.", 13)
        if "/" in thorn_name or "/" in test_name:
            fail(f"Invalid tests entry '{entry}'. Nested paths are not supported.", 14)

        data_path = cfg.test_root_dir / thorn_name / test_name
        par_path = cfg.test_root_dir / thorn_name / f"{test_name}.par"
        if not data_path.exists():
            fail(f"Missing test data path: '{data_path}'", 15)
        if not par_path.is_file():
            fail(f"Missing test parfile: '{par_path}'", 16)

        tests.append(
            TestEntry(
                thorn=thorn_name,
                test_name=test_name,
                data_path=data_path.resolve(),
                par_path=par_path.resolve(),
            )
        )

    for par_path in cfg.extra_parfiles:
        if not par_path.is_file():
            fail(f"Missing extra parfile: '{par_path}'", 17)

    return tests


def run_checked(cmd: Sequence[str], cwd: Path, env: Dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_and_tee(cmd: Sequence[str], cwd: Path, log_file: Path, env: Dict[str, str] | None = None) -> None:
    """
    Run command, mirror combined stdout/stderr to both console and log file.
    This replaces shell `|& tee ...` in a deterministic/readable way.
    """
    with log_file.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            out.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def ensure_symlink(link_path: Path, target: Path, code_symlink: int, code_mismatch: int) -> None:
    """
    Ensure `link_path` is a symlink that resolves to `target`.
    If absent, create it. If present but wrong, fail.
    """
    if not link_path.exists() and not link_path.is_symlink():
        link_path.symlink_to(target)

    if not link_path.is_symlink():
        fail(f"'{link_path}' is not a symlink", code_symlink)

    if link_path.resolve() != target.resolve():
        fail(f"Bad symlink: '{link_path}'", code_mismatch)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def compute_build_jobs() -> int:
    env_jobs = os.environ.get("CI_BUILD_JOBS") or os.environ.get("COTTONMOUTH_BUILD_JOBS")
    if env_jobs:
        try:
            parsed = int(env_jobs)
        except ValueError:
            fail("CI_BUILD_JOBS must be a positive integer", 20)
        if parsed < 1:
            fail("CI_BUILD_JOBS must be a positive integer", 20)
        return parsed

    cpus = os.cpu_count() or 1
    return max(1, cpus // 4)


def read_env_with_fallback(primary: str, legacy: str, default: str) -> str:
    value = os.environ.get(primary, "").strip()
    if value:
        return value
    legacy_value = os.environ.get(legacy, "").strip()
    if legacy_value:
        return legacy_value
    return default


def parse_failed_count(run_out_text: str) -> int:
    matches = re.findall(r"Number\s+failed\s*->\s*(\d+)", run_out_text)
    if len(matches) != 1:
        fail(f"Expected exactly one 'Number failed -> N' line, found {len(matches)}", 22)
    return int(matches[0])


def parse_last_tests_passed_block(run_out_text: str) -> List[str]:
    """
    Parse the final `Tests passed:` block into `thorn/test_name` pairs.
    A line is expected to look like:
      <spaces>test_name (from ThornName)
    """
    tests_header = re.compile(r"^\s*Tests passed:\s*$")
    passed_line = re.compile(r"^\s+([A-Za-z0-9_-]+)\s+\(from\s+([A-Za-z0-9_-]+)\)\s*$")

    last_block: List[str] = []
    current_block: List[str] = []
    capturing = False

    for line in run_out_text.splitlines():
        if tests_header.match(line):
            if capturing:
                last_block = current_block
            capturing = True
            current_block = []
            continue

        if not capturing:
            continue

        m = passed_line.match(line)
        if m:
            test_name = m.group(1)
            thorn_name = m.group(2)
            current_block.append(f"{thorn_name}/{test_name}")
            continue

        if line.strip() == "":
            continue

        last_block = current_block
        capturing = False
        current_block = []

    if capturing:
        last_block = current_block

    return last_block


def main() -> int:
    repo_root = Path.cwd().resolve()
    thornlist, cactus_dir = resolve_cactus_paths()
    cfg = load_config(resolve_config_path(repo_root), repo_root)
    tests = parse_tests(cfg)

    # 1) Generate all required thorns by invoking the arrangement Makefile.
    run_checked(["make", "-j4", "-f", str(cfg.source_makefile)], cwd=repo_root)
    if not cfg.source_arrangement_dir.is_dir():
        fail(f"Cannot find source arrangement dir: '{cfg.source_arrangement_dir}'", 8)

    # 2) Symlink arrangement + per-test assets into Cactus tree.
    arrangement_link = cactus_dir / "arrangements" / cfg.arrangement_name
    ensure_symlink(arrangement_link, cfg.source_arrangement_dir, 18, 19)

    for t in tests:
        target_test_dir = cactus_dir / "arrangements" / cfg.arrangement_name / t.thorn / "test"
        target_test_dir.mkdir(parents=True, exist_ok=True)
        ensure_symlink(target_test_dir / t.test_name, t.data_path, 18, 19)
        ensure_symlink(target_test_dir / f"{t.test_name}.par", t.par_path, 18, 19)
        # Include test.ccl file in each thorn. Prefer a thorn-local test.ccl
        # next to the parfiles, and fall back to a shared test.ccl under test root.
        thorn_local_test_ccl = t.par_path.parent / "test.ccl"
        shared_test_ccl = cfg.test_root_dir / "test.ccl"
        if thorn_local_test_ccl.is_file():
            test_ccl_source = thorn_local_test_ccl
        elif shared_test_ccl.is_file():
            test_ccl_source = shared_test_ccl
        else:
            fail(
                f"Missing test.ccl for thorn '{t.thorn}'. "
                f"Expected '{thorn_local_test_ccl}' or '{shared_test_ccl}'.",
                27,
            )
        ensure_symlink(target_test_dir / "test.ccl", test_ccl_source.resolve(), 18, 19)

    # Build thornlist used by SimFactory build.
    pre_thornlist = cactus_dir / f".pre_{cfg.suite_name}.th"
    suite_thornlist = cactus_dir / f"{cfg.suite_name}.th"
    pre_content = thornlist.read_text(encoding="utf-8")
    owning_thorns = dedupe_keep_order(t.thorn for t in tests)
    with pre_thornlist.open("w", encoding="utf-8") as f:
        f.write(pre_content)
        if pre_content and not pre_content.endswith("\n"):
            f.write("\n")
        for thorn in owning_thorns:
            f.write(f"{cfg.arrangement_name}/{thorn}\n")
        for thorn in cfg.extra_thorns:
            f.write(f"{cfg.arrangement_name}/{thorn}\n")

    parfiles: List[str] = [str(t.par_path) for t in tests] + [str(p) for p in cfg.extra_parfiles]
    run_checked(
        ["perl", "./utils/Scripts/MakeThornList", "-o", suite_thornlist.name, "--master", pre_thornlist.name, *parfiles],
        cwd=cactus_dir,
    )

    # 3) Build Cactus.
    build_jobs = compute_build_jobs()
    sim_config = read_env_with_fallback("CI_SIM_CONFIG", "COTTONMOUTH_SIM_CONFIG", "sim")
    if not sim_config:
        fail("CI_SIM_CONFIG must be non-empty", 21)

    run_and_tee(
        ["./simfactory/bin/sim", "build", sim_config, f"-j{build_jobs}", "--thornlist", suite_thornlist.name],
        cwd=cactus_dir,
        log_file=cactus_dir / "make.out",
    )

    # 4) Run the test suite.
    testsuite_thorns = dedupe_keep_order(t.thorn for t in tests)
    testsuite_run_processors = read_env_with_fallback(
        "CI_TESTSUITE_RUN_PROCESSORS", "COTTONMOUTH_TESTSUITE_RUN_PROCESSORS", "2"
    )
    testsuite_run_tests = os.environ.get("CI_TESTSUITE_RUN_TESTS", "").strip()
    if not testsuite_run_tests:
        testsuite_run_tests = os.environ.get("COTTONMOUTH_TESTSUITE_RUN_TESTS", "").strip()
    if not testsuite_run_tests:
        testsuite_run_tests = " ".join(testsuite_thorns)

    simulations_dir = Path(os.environ.get("HOME", "")) / "simulations" / f"{sim_config}-testsuite"
    if simulations_dir.exists():
        shutil.rmtree(simulations_dir)

    run_env = os.environ.copy()
    run_env.setdefault("OMP_NUM_THREADS", "4")
    run_env.setdefault("OMP_PLACES", "cores")
    run_env.setdefault("OMP_PROC_BIND", "close")
    run_env["CCTK_TESTSUITE_RUN_PROCESSORS"] = testsuite_run_processors
    run_env["CCTK_TESTSUITE_RUN_TESTS"] = testsuite_run_tests

    run_and_tee(
        ["make", f"{sim_config}-testsuite", "PROMPT=no"],
        cwd=cactus_dir,
        log_file=cactus_dir / "run.out",
        env=run_env,
    )

    # 5) Parse and validate output.
    run_out_text = (cactus_dir / "run.out").read_text(encoding="utf-8", errors="replace")
    failed_count = parse_failed_count(run_out_text)
    if failed_count != 0:
        fail(f"Testsuite reported failures: {failed_count}", 24)

    passed_pairs = parse_last_tests_passed_block(run_out_text)
    if not passed_pairs:
        fail("Could not find any tests under a 'Tests passed:' block in run.out", 25)

    expected_pairs = [t.pair for t in tests]
    if sorted(expected_pairs) != sorted(passed_pairs):
        print("Mismatch between expected tests and reported passed tests", file=sys.stderr)
        print("Expected tests (thorn/test):", file=sys.stderr)
        for p in expected_pairs:
            print(f"  {p}", file=sys.stderr)
        print("Reported passed tests (thorn/test):", file=sys.stderr)
        for p in passed_pairs:
            print(f"  {p}", file=sys.stderr)
        fail("Passed-test set mismatch", 26)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        cmd_str = " ".join(str(x) for x in e.cmd)
        print(f"Command failed with exit code {e.returncode}: {cmd_str}", file=sys.stderr)
        raise SystemExit(e.returncode)
    except CIError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(e.code)
