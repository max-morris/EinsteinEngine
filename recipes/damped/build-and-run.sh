#!/bin/bash
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
echo "python3 recipies/damped/damped.py"
set -e
python3 recipes/damped/damped.py
set +e
if [ ! -r "./TestEmitCactus/Damped/interface.ccl" ]
then
    echo "Cannot find './TestEmitCactus/Damped/interface.ccl" >&2
    exit 5
fi
ln -s "$PWD/TestEmitCactus" "$CACTUS_DIR/arrangements/TestEmitCactus" 2>/dev/null
if [ ! -L "$CACTUS_DIR/arrangements/TestEmitCactus" ]
then
    echo "'$CACTUS_DIR/arrangements/TestEmitCactus' is not a symlink"
    exit 6
fi
P1=$(realpath "$CACTUS_DIR/arrangements/TestEmitCactus")
P2=$(realpath "TestEmitCactus")
if [ "$P1" != "$P2" ]
then
    echo "Bad symlink: '$CACTUS_DIR/arrangements/TestEmitCactus'"
    exit 7
fi
cd "$CACTUS_DIR"
cat "$THORNLIST" > .pre_damped.th
echo TestEmitCactus/Damped >> .pre_damped.th
echo TestEmitCactus/ZeroTest >> .pre_damped.th

set -e

perl ./utils/Scripts/MakeThornList -o damped.th --master .pre_damped.th "$EMIT_CACTUS_DIR/recipes/damped/damped.par"
CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
#perl -p -i -e 's{ExternalLibraries/openPMD}{# ExternalLibraries/openPMD}' damped.th
#perl -p -i -e 's{ExternalLibraries/ADIOS2}{# ExternalLibraries/ADIOS2}' damped.th
./simfactory/bin/sim build damped -j$(($CPUS / 4)) --thornlist damped.th |& tee make.out
rm -fr ~/simulations/damped
./simfactory/bin/sim create-run damped --config damped --parfile "$EMIT_CACTUS_DIR/recipes/damped/damped.par" --procs 2 --ppn-used 2 --num-thread 1 |& tee run.out

set +e

OUTFILE=$(./simfactory/bin/sim get-output-dir damped)/damped.out
ERRFILE=$(./simfactory/bin/sim get-output-dir damped)/damped.err
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
EXPECTED=642
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
