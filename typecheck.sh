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

if [ "$1" == "-c" ] || [ "$1" == "--clean" ]; then
    rm -rf "$SCRIPT_DIR/.mypy_cache/"
fi

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "$SCRIPT_DIR/venv does not exist. Please set up your venv."
    exit 2
fi

cd "$SCRIPT_DIR"

. ./venv/bin/activate

echo "Checking EinsteinEngine..."
mypy 

echo "Checking recipes..."
mypy recipes

echo "Type checks passed!"
