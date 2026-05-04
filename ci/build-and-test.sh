#!/bin/bash

set -euo pipefail
exec python3 "$(dirname "$0")/build_and_test.py"
