#!/bin/bash

set -e

if [ ! -d "venv" ]; then
    echo "./venv does not exist. Please set up your venv."
    exit 2
fi

. ./venv/bin/activate

echo "Checking EmitCactus..."
mypy 

echo "Checking recipes..."
mypy recipes

echo "Type checks passed!"
