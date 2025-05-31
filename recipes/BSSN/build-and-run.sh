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
make -f recipes/BSSN/bssn.mk
if [ ! -L "$CACTUS_DIR/arrangements/PyBSSN" ]
then
    ln -s "$PWD/PyBSSN" "$CACTUS_DIR/arrangements/PyBSSN" 
fi
if [ ! -L "$CACTUS_DIR/arrangements/PyBSSN" ]
then
    echo "'$CACTUS_DIR/arrangements/PyBSSN' is not a symlink"
    exit 6
fi
P1=$(realpath "$CACTUS_DIR/arrangements/PyBSSN")
P2=$(realpath "PyBSSN")
if [ "$P1" != "$P2" ]
then
    echo "Bad symlink: '$CACTUS_DIR/arrangements/PyBSSN'"
    exit 7
fi
cd "$CACTUS_DIR"
cat "$THORNLIST" > .pre_bssn.th
echo PyBSSN/BSSN >> .pre_bssn.th
echo PyBSSN/KerrSchildID >> .pre_bssn.th
echo PyBSSN/LinearWaveID >> .pre_bssn.th
echo PyBSSN/DiagonalLinearWaveID >> .pre_bssn.th
echo PyBSSN/TestBSSN >> .pre_bssn.th

set -e

PAR_FILE="$EMIT_CACTUS_DIR/recipes/BSSN/parfiles/test_kerr_schild.par"
TEST_NAME=bssn_test

if [ ! -r bssn.th ]
then
    perl ./utils/Scripts/MakeThornList -o bssn.th --master .pre_bssn.th "$PAR_FILE"
fi

CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
./simfactory/bin/sim build bssn -j$(($CPUS / 4)) --thornlist bssn.th |& tee make.out

if [ ! -d arrangements/PyBSSN/TestBSSN/test ]
then
    ln -s "$EMIT_CACTUS_DIR/recipes/BSSN/test" "arrangements/PyBSSN/TestBSSN/test"
fi

export OMP_NUM_THREADS=1
make bssn-testsuite PROMPT=no CCTK_TESTSUITE_RUN_PROCESSORS=1 CCTK_TESTSUITE_RUN_TESTS=TestBSSN
