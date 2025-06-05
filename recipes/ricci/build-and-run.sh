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
echo "python3 recipies/ricci/testric.py"
set -e
python3 recipes/ricci/testric.py
set +e
if [ ! -r "./TestEmitCactus/Ricci/interface.ccl" ]
then
    echo "Cannot find './TestEmitCactus/Ricci/interface.ccl" >&2
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
cat "$THORNLIST" > .pre_ricci.th
echo TestEmitCactus/Ricci >> .pre_ricci.th
echo TestEmitCactus/ZeroTest >> .pre_ricci.th

set -e

perl ./utils/Scripts/MakeThornList -o ricci.th --master .pre_ricci.th "$EMIT_CACTUS_DIR/recipes/ricci/testric.par"
CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build ricci -j$(($CPUS / 4)) --thornlist ricci.th |& tee make.out
rm -fr ~/simulations/ricci
./simfactory/bin/sim create-run ricci --config ricci --parfile "$EMIT_CACTUS_DIR/recipes/ricci/testric.par" --procs 2 --ppn-used 2 --num-thread 1 |& tee run.out

set +e

OUTFILE=$(./simfactory/bin/sim get-output-dir ricci)/ricci.out
ERRFILE=$(./simfactory/bin/sim get-output-dir ricci)/ricci.err
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
if [ "$N" != 2 ]
then
    echo "ZERO TEST FAILURE"
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
