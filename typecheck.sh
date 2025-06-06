#!/bin/bash

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

echo "Checking EmitCactus..."
mypy 

echo "Checking recipes..."
mypy recipes

echo "Type checks passed!"
