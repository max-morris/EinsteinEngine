#!/bin/bash
set -ex

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

### EDIT THIS PER SCRIPT

TEST_SCRIPT_NAME="abcd"
TEST_SCRIPT_DIR="testing_recipes"
THORN_DIR="Test/Abcd"

TEST_SCRIPT="${TEST_SCRIPT_DIR}/${TEST_SCRIPT_NAME}/${TEST_SCRIPT_NAME}.py"
PAR_FILE="${TEST_SCRIPT_DIR}/${TEST_SCRIPT_NAME}/${TEST_SCRIPT_NAME}.par"
ARRANGEMENT_DIR=$(dirname $THORN_DIR)

### EDIT THIS PER SCRIPT

echo "python3 ${TEST_SCRIPT}"
set -e
python3 ${TEST_SCRIPT}
set +e
if [ ! -r "${THORN_DIR}/interface.ccl" ]
then
    echo "Cannot find '${THORN_DIR}/interface.ccl" >&2
    exit 5
fi
ln -s "${EMIT_CACTUS_DIR}/${ARRANGEMENT_DIR}" "${CACTUS_DIR}/arrangements/${ARRANGEMENT_DIR}" 2>/dev/null
if [ ! -L "${CACTUS_DIR}/arrangements/${ARRANGEMENT_DIR}" ]
then
    echo "'${CACTUS_DIR}/arrangements/${ARRANGEMENT_DIR}'  is not a symlink"
    exit 6
fi
P1=$(realpath "${CACTUS_DIR}/arrangements/${ARRANGEMENT_DIR}")
P2=$(realpath "${ARRANGEMENT_DIR}")
if [ "$P1" != "$P2" ]
then
    echo "Bad symlink: '${CACTUS_DIR}/arrangements/${ARRANGEMENT_DIR}'"
    exit 7
fi
cd "$CACTUS_DIR"
cat "$THORNLIST" > .pre_${TEST_SCRIPT_NAME}.th
echo "${THORN_DIR}" >> .pre_${TEST_SCRIPT_NAME}.th

set -e

perl ./utils/Scripts/MakeThornList -o ${TEST_SCRIPT_NAME}.th --master .pre_${TEST_SCRIPT_NAME}.th "${EMIT_CACTUS_DIR}/${PAR_FILE}"
CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build ${TEST_SCRIPT_NAME} -j$(($CPUS / 4)) --thornlist ${TEST_SCRIPT_NAME}.th |& tee make.out
rm -fr ~/simulations/${TEST_SCRIPT_NAME}
./simfactory/bin/sim create-run ${TEST_SCRIPT_NAME} --config ${TEST_SCRIPT_NAME} --parfile "$EMIT_CACTUS_DIR/${PAR_FILE}" --procs 2 --ppn-used 2 --num-thread 1 |& tee run.out

set +e

OUTFILE=$(./simfactory/bin/sim get-output-dir ${TEST_SCRIPT_NAME})/${TEST_SCRIPT_NAME}.out
ERRFILE=$(./simfactory/bin/sim get-output-dir ${TEST_SCRIPT_NAME})/${TEST_SCRIPT_NAME}.err
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
