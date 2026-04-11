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

set -e

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
EMIT_CACTUS_DIR="$PWD"
make -j4 -f recipes/Cottonmouth/Makefile
if [ ! -L "$CACTUS_DIR/arrangements/Cottonmouth" ]
then
    ln -s "$PWD/Cottonmouth" "$CACTUS_DIR/arrangements/Cottonmouth" 
fi
if [ ! -L "$CACTUS_DIR/arrangements/Cottonmouth" ]
then
    echo "'$CACTUS_DIR/arrangements/Cottonmouth' is not a symlink"
    exit 6
fi
P1=$(realpath "$CACTUS_DIR/arrangements/Cottonmouth")
P2=$(realpath "Cottonmouth")
if [ "$P1" != "$P2" ]
then
    echo "Bad symlink: '$CACTUS_DIR/arrangements/Cottonmouth'"
    exit 7
fi
cd "$CACTUS_DIR"
cat "$THORNLIST" > .pre_cottonmouth.th
echo Cottonmouth/CottonmouthBSSNOK >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthDiagLinearWaveID >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthLinearWaveID >> .pre_cottonmouth.th
echo Cottonmouth/CottonmouthZ4c >> .pre_cottonmouth.th

set -e

parfiles=(
  "$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/linear_wave.par"
  "$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/qc0.par"
  "$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/mag_TOV.par"
  "$EMIT_CACTUS_DIR/recipes/Cottonmouth/apples_with_apples/linear_wave_z4c.par"
)

perl ./utils/Scripts/MakeThornList -o cottonmouth.th --master .pre_cottonmouth.th "${parfiles[@]}"

CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build cottonmouth -j$(($CPUS / 4)) --thornlist cottonmouth.th |& tee make.out

SOURCE_TEST_DIR=$EMIT_CACTUS_DIR/recipes/Cottonmouth/test
TARGET_TEST_DIR=arrangements/Cottonmouth/CottonmouthBSSNOK/test

if [ ! -d $TARGET_TEST_DIR ]
then
    ln -s $SOURCE_TEST_DIR $TARGET_TEST_DIR
fi

export OMP_NUM_THREADS=4
make cottonmouth-testsuite PROMPT=no CCTK_TESTSUITE_RUN_PROCESSORS=1 CCTK_TESTSUITE_RUN_TESTS=CottonmouthBSSNOK |& tee run.out
