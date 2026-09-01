#!/bin/bash

# Copyright (C) 2025-2026 Max Morris and other Einstein Engine contributors.
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

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            rm -rf "$SCRIPT_DIR/.mypy_cache/"
            shift
        ;;
        -n|--normalize)
            python "$SCRIPT_DIR/scripts/normalize_einsteinengine_imports.py"
            shift
        ;;
    esac
done

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "$SCRIPT_DIR/venv does not exist. Please set up your venv."
    exit 2
fi

cd "$SCRIPT_DIR"

. ./venv/bin/activate

# Typecheck results depend on the exact versions of mypy and of typed
# dependencies (e.g. sympy 1.12.1 ships py.typed; later releases do not),
# so refuse to run against a venv that has drifted from the pins.
python - "$SCRIPT_DIR/requirements.txt" <<'EOF'
import re, sys
from importlib import metadata

drift = []
with open(sys.argv[1]) as f:
    for line in f:
        m = re.fullmatch(r'([A-Za-z0-9._-]+)==([^\s;#]+)', line.split('#')[0].strip())
        if not m:
            continue
        name, pinned = m.groups()
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            drift.append(f"  {name}: pinned {pinned}, but not installed")
            continue
        if installed != pinned:
            drift.append(f"  {name}: pinned {pinned}, but {installed} is installed")

if drift:
    print("The venv has drifted from the versions pinned in requirements.txt:")
    print("\n".join(drift))
    print("Run './venv/bin/pip install -r requirements.txt', then re-run this script.")
    sys.exit(3)
EOF

echo "Checking EinsteinEngine..."
mypy 

echo "Checking recipes..."
mypy recipes

echo "Type checks passed!"
