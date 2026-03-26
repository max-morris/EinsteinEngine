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
echo "python3 recipies/newradx_weq/newradx_weq.py"
set -e
python3 recipes/newradx_weq/newradx_weq.py
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
cat "$THORNLIST" > .pre_newradx_weq.th
echo TestEinsteinEngine/NewRadXWeq >> .pre_newradx_weq.th

set -e

perl ./utils/Scripts/MakeThornList -o newradx_weq.th --master .pre_newradx_weq.th "$EMIT_CACTUS_DIR/recipes/newradx_weq/newradx_weq.par"
CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build newradx -j$(($CPUS / 4)) --thornlist newradx_weq.th |& tee make.out
exit 0 
rm -fr ~/simulations/newradx_weq
./simfactory/bin/sim create-run newradx_weq --config newradx_weq --parfile "$EMIT_CACTUS_DIR/recipes/newradx_weq/newradx_weq.par" --procs 2 --ppn-used 2 --num-thread 1 |& tee run.out

set +e

OUTFILE=$(./simfactory/bin/sim get-output-dir newradx_weq)/newradx_weq.out
ERRFILE=$(./simfactory/bin/sim get-output-dir newradx_weq)/newradx_weq.err
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
