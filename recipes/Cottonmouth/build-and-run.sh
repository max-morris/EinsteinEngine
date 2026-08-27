#!/bin/bash

# Copyright (C) 2025-2026 Lucas Timotheo Sanches, Max Morris, and other Einstein Engine contributors.
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

set -euo pipefail

# Build and Run
if [ "${THORNLIST}" = "" ]
then
    echo "THORNLIST is not set" >&2
    exit 1
fi
THORNLIST=$(realpath "$THORNLIST")
if [ ! -r "${THORNLIST}" ]
then
    echo "THORNLIST is not readable" >&2
    exit 2
fi
CACTUS_DIR=$(dirname $(dirname "${THORNLIST}"))
echo "CACTUS_DIR: $CACTUS_DIR"
if [ ! -d "${CACTUS_DIR}/arrangements" ]
then
    echo "Cannot find '${CACTUS_DIR}/arrangements'" >&2
    exit 3
fi
if [ ! -r "${CACTUS_DIR}/simfactory/etc/defs.local.ini" ]
then
    echo "Cannot find '${CACTUS_DIR}/simfactory/etc/defs.local.ini'" >&2
    exit 4
fi
EINSTEIN_ENGINE_DIR="$PWD"
make -j4 -f recipes/Cottonmouth/Makefile
GENERATED_COTTONMOUTH_DIR="$PWD/generated/Cottonmouth"
if [ ! -d "$GENERATED_COTTONMOUTH_DIR" ]
then
    echo "Cannot find generated arrangement dir '$GENERATED_COTTONMOUTH_DIR'" >&2
    exit 5
fi
if [ ! -L "$CACTUS_DIR/arrangements/Cottonmouth" ]
then
    ln -s "$GENERATED_COTTONMOUTH_DIR" "$CACTUS_DIR/arrangements/Cottonmouth"
fi
if [ ! -L "$CACTUS_DIR/arrangements/Cottonmouth" ]
then
    echo "'$CACTUS_DIR/arrangements/Cottonmouth' is not a symlink"
    exit 6
fi
P1=$(realpath "$CACTUS_DIR/arrangements/Cottonmouth")
P2=$(realpath "$GENERATED_COTTONMOUTH_DIR")
if [ "$P1" != "$P2" ]
then
    echo "Bad symlink: '$P1' != '$P2'" >& 2
    exit 7
fi
cd "$CACTUS_DIR"
cat "$THORNLIST" > .pre_cottonmouth.th
echo Cottonmouth/CottonmouthBSSNOK4m >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthGaugeWaveID >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthLinearWaveID >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthZ4c4m >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthBosonStarID >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthBS4m >> .pre_cottonmouth.th

set -e

parfiles=(
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/linear_wave_bssnok.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/linear_wave_z4c.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/mag_TOV_bssnok.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/mag_TOV_z4c.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/qc0_bssnok.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/qc0_z4c.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBS4m/boson_star.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/apples_with_apples/gauge_wave_z4c.par"
)

perl ./utils/Scripts/MakeThornList -o cottonmouth.th --master .pre_cottonmouth.th "${parfiles[@]}"

if command -v nproc >/dev/null 2>&1
then
    CPUS=$(nproc --all)
else
    CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
fi

DEFAULT_BUILD_JOBS=$(($CPUS / 4))
if [ "$DEFAULT_BUILD_JOBS" -lt 1 ]
then
    DEFAULT_BUILD_JOBS=1
fi

BUILD_JOBS=${COTTONMOUTH_BUILD_JOBS:-$DEFAULT_BUILD_JOBS}
if ! [[ "$BUILD_JOBS" =~ ^[0-9]+$ ]] || [ "$BUILD_JOBS" -lt 1 ]
then
    echo "COTTONMOUTH_BUILD_JOBS must be a positive integer" >&2
    exit 8
fi

./simfactory/bin/sim build cottonmouth -j"$BUILD_JOBS" --thornlist cottonmouth.th |& tee make.out

TARGET_TEST_DIR_BSSNOK=arrangements/Cottonmouth/CottonmouthBSSNOK4m/test
TARGET_TEST_DIR_Z4c=arrangements/Cottonmouth/CottonmouthZ4c4m/test
TARGET_TEST_DIR_BS=arrangements/Cottonmouth/CottonmouthBS4m/test

bssnok_test_data=(
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/linear_wave_bssnok"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/linear_wave_bssnok.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/qc0_bssnok"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/qc0_bssnok.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/mag_TOV_bssnok"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBSSNOK4m/mag_TOV_bssnok.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/test.ccl"
)

z4c_test_dirs=(
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/linear_wave_z4c"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/linear_wave_z4c.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/mag_TOV_z4c"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/mag_TOV_z4c.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/qc0_z4c"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthZ4c4m/qc0_z4c.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/test.ccl"
)

bs_test_data=(
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBS4m/boson_star"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/CottonmouthBS4m/boson_star.par"
  "$EINSTEIN_ENGINE_DIR/recipes/Cottonmouth/test/test.ccl"
)

if [ ! -d $TARGET_TEST_DIR_BSSNOK ]
then
    mkdir $TARGET_TEST_DIR_BSSNOK
    for test_dir in "${bssnok_test_data[@]}"; do
        ln -s $test_dir $TARGET_TEST_DIR_BSSNOK
    done
fi

if [ ! -d $TARGET_TEST_DIR_Z4c ]
then
    mkdir $TARGET_TEST_DIR_Z4c
    for test_dir in "${z4c_test_dirs[@]}"; do
        ln -s $test_dir $TARGET_TEST_DIR_Z4c
    done
fi

if [ ! -d $TARGET_TEST_DIR_BS ]
then
    mkdir $TARGET_TEST_DIR_BS
    for test_dir in "${bs_test_data[@]}"; do
        ln -s $test_dir $TARGET_TEST_DIR_BS
    done
fi

if [ -d $HOME/simulations/cottonmouth-testsuite ]; then
    rm -r $HOME/simulations/cottonmouth-testsuite
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

TESTSUITE_RUN_PROCESSORS=${COTTONMOUTH_TESTSUITE_RUN_PROCESSORS:-2}
TESTSUITE_RUN_TESTS=${COTTONMOUTH_TESTSUITE_RUN_TESTS:-"CottonmouthBSSNOK4m CottonmouthZ4c4m CottonmouthBS4m"}

make \
    cottonmouth-testsuite \
    PROMPT=no \
    CCTK_TESTSUITE_RUN_PROCESSORS="$TESTSUITE_RUN_PROCESSORS" \
    CCTK_TESTSUITE_RUN_TESTS="$TESTSUITE_RUN_TESTS" \
    |& tee run.out

mapfile -t failed_lines < <(grep -E 'Number[[:space:]]+failed[[:space:]]*->[[:space:]]*[0-9]+' run.out || true)
if [ "${#failed_lines[@]}" -ne 1 ]
then
    echo "Expected exactly one 'Number failed -> N' line, found ${#failed_lines[@]}" >&2
    exit 9
fi

FAILED_COUNT=$(printf '%s\n' "${failed_lines[0]}" | sed -E 's/.*Number[[:space:]]+failed[[:space:]]*->[[:space:]]*([0-9]+).*/\1/')
if ! [[ "$FAILED_COUNT" =~ ^[0-9]+$ ]]
then
    echo "Could not parse failure count from testsuite output" >&2
    exit 10
fi

if [ "$FAILED_COUNT" -ne 0 ]
then
    echo "Testsuite reported failures: $FAILED_COUNT" >&2
    exit 11
fi
