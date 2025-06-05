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
make -f recipes/Cottonmouth/Makefile
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
cat "$THORNLIST" > .pre_bssn.th
echo Cottonmouth/CottonmouthBSSNOK >> .pre_bssn.th
echo Cottonmouth/CottonmouthDiagLinearWaveID >> .pre_bssn.th
echo Cottonmouth/CottonmouthKerrSchildID >> .pre_bssn.th
echo Cottonmouth/CottonmouthLinearWaveID >> .pre_bssn.th
echo Cottonmouth/CottonmouthTestBSSNOK >> .pre_bssn.th

set -e

PAR_FILE1="$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/kerr_schild_id.par"
PAR_FILE2="$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/kerr_schild.par"
PAR_FILE3="$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/linear_wave.par"
PAR_FILE4="$EMIT_CACTUS_DIR/recipes/Cottonmouth/test/qc0.par"
TEST_NAME=bssn_test

perl ./utils/Scripts/MakeThornList -o bssn.th --master .pre_bssn.th "$PAR_FILE1" "$PAR_FILE2" "$PAR_FILE3" "$PAR_FILE4"

CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build bssn -j$(($CPUS / 4)) --thornlist bssn.th |& tee make.out

if [ ! -d arrangements/Cottonmouth/CottonmouthTestBSSNOK/test ]
then
    ln -s "$EMIT_CACTUS_DIR/recipes/Cottonmouth/test" "arrangements/Cottonmouth/CottonmouthTestBSSNOK"
fi

export OMP_NUM_THREADS=1
make bssn-testsuite PROMPT=no CCTK_TESTSUITE_RUN_PROCESSORS=1 CCTK_TESTSUITE_RUN_TESTS=CottonmouthTestBSSNOK
