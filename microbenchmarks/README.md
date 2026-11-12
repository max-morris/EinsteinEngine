# microbenchmarks — measured costs for `sympy_complexity.py` weights

This directory times the operations the CarpetX emitter generates
(`cpp_carpetx_visitor.py`) and turns those timings into integer weights for
`EinsteinEngine/generators/sympy_complexity.py`.

The default is `DEFAULT_WEIGHTS` in that module, the hand-written model with
one add as the unit: `Pow` 15, integer powers `max(2, int(log2|p|))`, grid
`Symbol` 10 vs local 1, stencil center 10 / x 40 / y-z 100, and every
transcendental 15. Weights are loaded on the first `get_weights()` call, not
when `sympy_complexity` is imported.
`EE_COMPLEXITY_WEIGHTS=/path/to/profile.json` selects another profile; a
missing or unreadable file is fatal at that call.

Measured results are never committed: they describe one machine at one time
(its compiler, driver and modules), and they go stale. The tuner measures its
target instead (below).

The generic cost model charges every non-integer power, `sqrt` and `cbrt`
included, as `pow_default` plus both arguments. F90 emits those powers as
`pow`. CarpetX emits `x**(1/2)` and `x**(1/3)` as `sqrt()` and `cbrt()`, and
only `CppCarpetXComplexityVisitor` charges those as the function weight plus
the base. A transcendental the profile does not list costs the same as `sin`.
A profile that lists it, as a fitted profile does for `tan` and `erf`,
uses the listed weight. `Abs` has no surcharge.

## Backends

CPU, CUDA, and ROCm are three builds of the same sources. One binary is only
one of them. Do not pass `-DEE_ENABLE_CUDA=ON` and `-DEE_ENABLE_HIP=ON`
together, and do not pass `--use_fast_math` or `-ffast-math`.
`-DEE_VREAL_FLOAT=ON` selects `float` for `vreal`; the default is `double`.

The host build is ordinary C++ and does not launch a kernel. The CUDA and
ROCm builds compile `src/bench_*.cpp` with nvcc or hipcc, launch each case as
a device kernel, and time that kernel with events. Copies and the checksum
reduction stay outside the timer. The JSON schema is the same for all three,
so the fitter does not care which binary produced the file.

| Backend | CMake | Make |
|---|---|---|
| CPU | `cmake -S microbenchmarks -B microbenchmarks/build -DCMAKE_BUILD_TYPE=Release` | `make -C microbenchmarks` |
| CUDA | same, plus `-DEE_ENABLE_CUDA=ON` into `build-cuda`. Default arch is sm_80 (`-DCMAKE_CUDA_ARCHITECTURES=80`) | `make -C microbenchmarks cuda CUDA_ARCH=80` |
| ROCm | same, plus `-DEE_ENABLE_HIP=ON` into `build-hip`. Pass `-DCMAKE_HIP_ARCHITECTURES=gfx908` (MI100), `gfx90a` (MI200), or `gfx942` (MI300) | `make -C microbenchmarks hip HIP_ARCH=gfx908` |

`make rocm` is the same target as `make hip`. Each Make target overwrites
`microbenchmarks/ee_microbench`. Off a GPU node the CMake ROCm default is
gfx908; if `amdgpu-arch` sees a GPU, that architecture is used instead.

```bash
cmake --build microbenchmarks/build
./microbenchmarks/build/ee_microbench --quick
./microbenchmarks/build/ee_microbench --json /tmp/ee_bench.json
python -m EinsteinEngine.tuning.fit_weights /tmp/ee_bench.json
```

Use `build-cuda` or `build-hip` the same way. `--json -` sends the human
table to stderr and pure JSON to stdout, so this pipe works:

```bash
./microbenchmarks/build/ee_microbench --quick --json - 2>/dev/null \
  | python -m EinsteinEngine.tuning.fit_weights -
```

`--group arith,pow` selects cases. `--rounds R`, `--repeats K` and `--size N`
override the number of rounds, the repeats per round and the stream length. `--quick` is a smoke test: do not fit a
profile from it. `unit_tests/test_fit_weights.py` checks the fitter without
running a benchmark.

A full host run takes under a minute; a full GPU run takes several minutes.

## Running from the tuner

`run.sh cpu|cuda|hip [ee_microbench args...]` builds `ee_microbench` for one
target and writes `results.json` next to itself (in `SLURM_SUBMIT_DIR` when
run by sbatch, which runs a copy of the script). `remote_tuner`'s
`--microbench-command` runs it on the tuning target, directly or through
sbatch, once per checkpoint, and records the results and the fitted profile in
`<checkpoint>.microbench.json`; see "Measuring the cost model on the target"
in `docs/TUNING-QUICKSTART.md`. Build trees and `results.json` are ignored by
git and are not synced to the target.

## What is timed

Each case has two flavors where that split means something:

* `*_stream`: `c[i] = f(a[i], b[i])` over a large array. This is the CarpetX
  `grid.loop_*_device` body, which is the context the weights rank.
* `*_reg`: a chained `x = f(x, k)` with the result kept live. This is ALU
  throughput with no global memory in the inner loop. On a GPU it is many
  independent per-thread chains, so the number is device throughput, not
  single-thread latency.

The suite runs in rounds: every case is timed a few times in each round, after
one untimed warm-up pass over the suite. Reported fields are each round's
minimum ns/element (`ns_rounds`), their minimum and median, GOPS, and a
checksum. The fitter divides each case by add within the same round and takes
the median over rounds, so a clock that changes between rounds cancels out. A
non-finite checksum aborts. Host builds keep values live with `asm volatile`.
Device builds do the same inside the kernel and repeat each streamed math
functor so the store cannot hide the arithmetic: 256 times per element, or
32 with `--quick`. The reported ns/element is per repetition. One math op per
element is HBM-bound on these GPUs and cannot separate libm from a load.
Memory kernels stay at one evaluation per interior point and are
informational: the fitter does not turn them into stencil weights.

Host and device numbers are not comparable. A host stream does one operation
per element and is memory- and loop-bound for cheap ops. A device streamed
math kernel does 256 repetitions per element and is compute-bound, and each
repetition also pays a perturbation FMA, a finiteness check, and an xor into
the sink. The device arith and transcendental groups each include
`identity_stream`, the same harness with a functor that returns its input:
arith's times the two-input kernel, transcendental's the one-input kernel.
The fitter subtracts the row's own group's identity (arith's for the other
groups, which use the two-input kernel) and floors the difference at 0.
Subtracting can otherwise go negative from noise.

The harness times scalar `vreal` (`double`, or `float` with
`-DEE_VREAL_FLOAT=ON`). It does not time `Arith::simd<CCTK_REAL>`, whose
relative costs differ.

The host binary pins itself to the CPU it is already running on
(`sched_setaffinity`). It does not change the CPU governor; that needs
privileges the process does not take; the rounds stand in for it. Host
defaults are 2^20 stream elements, 10 rounds of 3 repeats, and a 128^3 grid
(64^3 with `--quick`). Device defaults are 2^27 stream elements (2^24 with
`--quick`), 5 rounds of 3 repeats, and a 512^3 grid (256^3 with `--quick`),
so launch overhead is not the result. `--quick` is 2 rounds of 2 repeats with
no warm-up pass. `--quick` is applied first, then an explicit `--rounds`,
`--repeats` or `--size` overrides it (`--repeats 30 --quick` keeps 30).

`mul_reg` and `div_reg` reset the accumulator on every repeat. A carried
multiply by `1.000002` overflows after about 20 passes, and a carried divide
drifts into the subnormals. Integer powers use the same repeated-squaring
loop as `Arith::pown`, and the pow group times `x**-1` and `x**-2` as well
as the positive exponents.

Device memory launches step threads along x, so a center load and an x, y, or
z neighbor load are each coalesced. That will not reproduce the
10/40/100 stencil spread, and it cannot see a ghost exchange.

`EinsteinEngine.tuning.fit_weights` writes a schema_version 1 profile. Fitted slots use

```text
weight(op) = max(1, floor(adjusted_ns / adjusted_add + 0.5))
```

that is, whole adds, rounded half up, and at least one. Fractions of an add
would mostly be run-to-run noise. `sub`,
`div`, `neg`, `const`, `cmp_lt`, and `if_else` are recorded and are not
weights, because the cost model has no slot for them. The grid symbol and
stencil weights keep the default model's values. Rows are keyed by `(group, name)`, so `pow/sqrt_stream`
cannot overwrite `transcendental/sqrt_stream`. A min/median spread above 25%
is a warning. The script does not probe the machine.

## Layout

| Path | Role |
|---|---|
| `common/ee_bench.hpp` | Host harness: timing, stats, JSON and table output. Never runs on device. |
| `common/kernels.hpp` | Scalar kernels shared by the CPU, CUDA, and ROCm drivers: `pow2` / `pown` / `sqrt` / `cbrt` / `pow`, `sin`…`erf`, `if_else`, grid index math. |
| `common/ee_device.hpp` | CUDA and ROCm launch and event timing. The CPU build does not include it. |
| `src/bench_arith.cpp` | `Add`, `Mul`, and the derived `sub`, `div`, and unary `neg`. |
| `src/bench_pow.cpp` | `Pow`: `x**2,3,4,8,16`, `x**-1`, `x**-2`, `sqrt`, `cbrt`, generic `pow(x,y)`, `x**2.5`. |
| `src/bench_transcendental.cpp` | `sin`, `cos`, `exp`, `log`, `sqrt`, `cbrt`, plus `tan`, `sinh`, `cosh`, `tanh`, `erf`. |
| `src/bench_memory.cpp` | `Symbol` grid vs local, `Atom` baseline, stencil center / x / y / z. |
| `src/bench_branch.cpp` | `Relational` compare, and `Piecewise` as `if_else` plus a real branch. |
| `src/main.cpp` | Driver: rounds, `--quick`, `--rounds`, `--repeats`, `--size`, `--json`, `--group`. |
| `run.sh` | Build for one target and write `results.json`; what the tuner runs. |
| `EinsteinEngine/tuning/fit_weights.py` | Results JSON to a schema_version 1 profile. |
