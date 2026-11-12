#!/usr/bin/env bash
# Runs the AMReX backend's compile and differential gates (docs, section 9, items 6 and 7) in WORK_DIR:
#   unit_tests/amrex/run_gate.sh WORK_DIR AMREX_HOME PYTHON [MAKEVAR=VALUE ...]
# e.g. CPU: run_gate.sh gate-cpu ~/amrex python3 COMP=gnu
#      CUDA: run_gate.sh gate-cuda ~/amrex python3 USE_CUDA=TRUE CUDA_ARCH=80
# Generates the toy headers (needs EE's Python dependencies; PYTHONPATH must reach EinsteinEngine), builds main.cpp
# with AMReX (MAKE_JOBS, default 8), runs it, and compares with the 50-digit reference.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$1 amrex=$2 python=$3; shift 3
mkdir -p "$work/gen" "$work/run"
work=$(cd "$work" && pwd)
"$python" "$here/gate.py" generate "$work/gen"
make -C "$here" -j"${MAKE_JOBS:-8}" AMREX_HOME="$amrex" GEN_DIR="$work/gen" TMP_BUILD_DIR="$work/tmp" \
  USE_STOIC_QUOTES=FALSE "$@" > "$work/build.log" 2>&1 || { echo "gate: build FAILED"; tail -40 "$work/build.log"; exit 1; }
exe=$(make -s -C "$here" AMREX_HOME="$amrex" GEN_DIR="$work/gen" "$@" print-executable | sed -n 's/^executable is //p')
mv "$here/$exe" "$work/"
"$work/$exe" "$work/run" > "$work/run.log" 2>&1
"$python" "$here/gate.py" compare "$work/run"
