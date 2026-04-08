#!/bin/bash

# Copyright (C) 2024-2026 Lucas Timotheo Sanches, Max Morris, and other Einstein Engine contributors.
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
echo "python3 recipies/non_flat_waveeqn/non_flat_waveeqn.py"
set -e
python3 recipes/non_flat_waveeqn/non_flat_waveeqn.py
set +e
if [ ! -r "./TestEinsteinEngine/NonFlatWaveEqn/interface.ccl" ]
then
    echo "Cannot find './TestEinsteinEngine/NonFlatWaveEqn/interface.ccl" >&2
    exit 5
fi
ln -s "$PWD/TestEinsteinEngine" "$CACTUS_DIR/arrangements/TestEinsteinEngine" 2>/dev/null
if [ ! -L "$CACTUS_DIR/arrangements/TestEinsteinEngine" ]
then
    echo "'$CACTUS_DIR/arrangements/TestEinsteinEngine' is not a symlink"
    exit 6
fi
P1=$(realpath "$CACTUS_DIR/arrangements/TestEinsteinEngine")
P2=$(realpath "TestEinsteinEngine")
if [ "$P1" != "$P2" ]
then
    echo "Bad symlink: '$CACTUS_DIR/arrangements/TestEinsteinEngine'"
    exit 7
fi
cd "$CACTUS_DIR"
cat "$THORNLIST" > .pre_non_flat_waveeqn.th
echo TestEinsteinEngine/NonFlatWaveEqn >> .pre_non_flat_waveeqn.th
echo TestEinsteinEngine/ZeroTest >> .pre_non_flat_waveeqn.th

set -e

perl ./utils/Scripts/MakeThornList -o non_flat_waveeqn.th --master .pre_non_flat_waveeqn.th "$EMIT_CACTUS_DIR/recipes/non_flat_waveeqn/non_flat_waveeqn.par"
CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build non_flat_waveeqn -j$(($CPUS / 4)) --thornlist non_flat_waveeqn.th |& tee make.out
rm -fr ~/simulations/non_flat_waveeqn
./simfactory/bin/sim create-run non_flat_waveeqn --config non_flat_waveeqn --parfile "$EMIT_CACTUS_DIR/recipes/non_flat_waveeqn/non_flat_waveeqn.par" --procs 2 --ppn-used 2 --num-thread 1 |& tee run.out

set +e

OUTFILE=$(./simfactory/bin/sim get-output-dir non_flat_waveeqn)/non_flat_waveeqn.out
ERRFILE=$(./simfactory/bin/sim get-output-dir non_flat_waveeqn)/non_flat_waveeqn.err
############
echo "OUTPUT FILE IS: ${OUTFILE}"
echo "ERROR FILE IS: ${ERRFILE}"
############
if [ ! -r "$OUTFILE" ]
then
    echo "TEST FAILED no output"
    exit 8
fi
############
if grep 'MPI_ABORT was invoked on rank' "${ERRFILE}"
then
    echo "TEST RUN DIED UNEXPECTEDLY"
    exit 11
fi
############
if grep 'ERROR from host' "${ERRFILE}"
then
    echo "TEST RUN DIED UNEXPECTEDLY"
    exit 11
fi
############
N=$(grep '::ZERO TEST RAN' ${OUTFILE}|wc -l)
echo "ZERO TESTS THAT RAN: ${N}"
EXPECTED=1578
if [ "$N" != "${EXPECTED}" ]
then
    echo "ZERO TEST FAILURE: Expected ${EXPECTED}, got ${N}"
    exit 10
fi
############
if grep ::ERROR:: $OUTFILE
then
    echo "TEST FAILED tolerances not satisfied"
    exit 9
else
    echo "TEST PASSED"
    exit 0
fi
