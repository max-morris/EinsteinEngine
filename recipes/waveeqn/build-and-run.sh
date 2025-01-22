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
echo "python3 recipies/waveeqn/waveeqn.py"
set -e
python3 recipes/waveeqn/waveeqn.py
set +e
if [ ! -r "./TestEmitCactus/WaveEqn/interface.ccl" ]
then
    echo "Cannot find './TestEmitCactus/WaveEqn/interface.ccl" >&2
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
cat "$THORNLIST" > .pre_waveeqn.th
echo TestEmitCactus/WaveEqn >> .pre_waveeqn.th
echo TestEmitCactus/ZeroTest >> .pre_waveeqn.th

set -e

perl ./utils/Scripts/MakeThornList -o waveeqn.th --master .pre_waveeqn.th "$EMIT_CACTUS_DIR/recipes/waveeqn/waveeqn.par"
CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
perl -p -i -e 's{ExternalLibraries/openPMD}{# ExternalLibraries/openPMD}' waveeqn.th
perl -p -i -e 's{ExternalLibraries/ADIOS2}{# ExternalLibraries/ADIOS2}' waveeqn.th
./simfactory/bin/sim build waveeqn -j$(($CPUS / 4)) --thornlist waveeqn.th |& tee make.out
rm -fr ~/simulations/waveeqn
./simfactory/bin/sim create-run waveeqn --config waveeqn --parfile "$EMIT_CACTUS_DIR/recipes/waveeqn/waveeqn.par" --procs 2 --ppn-used 2 --num-thread 1 |& tee run.out

set +e

OUTFILE=$(./simfactory/bin/sim get-output-dir waveeqn)/waveeqn.out
echo "OUTPUT FILE IS: ${OUTFILE}"
if [ ! -r "$OUTFILE" ]
then
    echo "TEST FAILED no output"
    exit 8
fi
if ! grep "::ZERO TEST RAN::" $OUTFILE > /dev/null
then
    echo "ZERO TEST DID NOT RUN"
    exit 10
fi
if grep ::ERROR:: $OUTFILE
then
    echo "TEST FAILED tolerances not satisfied"
    exit 9
else
    echo "TEST PASSED"
fi
