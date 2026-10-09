# microbenchmarks — measured costs for `sympy_complexity.py` weights

This directory times the operations the CarpetX emitter generates
(`cpp_carpetx_visitor.py`) and turns those timings into integer weights for
`EinsteinEngine/generators/sympy_complexity.py`.

The default profile, `guestimates.json`, is still the hand-written model:
`Pow` 1500, integer powers `max(200, 100*int(log2|p|))` (and 200 when
`|p| <= 1`), grid `Symbol` 1000 vs local 100, stencil center 1000 / x 4000 /
y-z 10000, and `sin/cos/exp/log/sqrt/cbrt` 1500. Every profile scales its
positive weights so the smallest is 100. `transcendental_default` stays 0.
That file is the only profile in the installed package. Measured profiles
live in `microbenchmarks/profiles/`. Nothing selects one unless
`EE_COMPLEXITY_WEIGHTS` is a path to it. The profile is loaded on the first
`get_weights()` call, not when `sympy_complexity` is imported.

The generic cost model charges every non-integer power, `sqrt` and `cbrt`
included, as `pow_default` plus both arguments. F90 emits those powers as
`pow`. CarpetX emits `x**(1/2)` and `x**(1/3)` as `sqrt()` and `cbrt()`, and
only `CppCarpetXComplexityVisitor` charges those as the function weight plus
the base. A transcendental the profile does not list costs the same as `sin`.
A profile that lists it, as the measured profiles do for `tan` and `erf`,
uses the listed weight. `Abs` has no surcharge.

Those weights are about 100 times the old integer model. `promote_threshold()`
and `retain_threshold()` take absolute complexity counts, so a threshold
written against the old scale keeps about 100 times fewer candidates.
Percentile and rank strategies do not use that absolute scale.

`sin`, `cos`, `exp`, `log`, `sqrt`, and `cbrt` used to add 0: the old check
compared a call with the function classes and was always false. The default
profile now adds 1500, and the same 1500 applies to the other transcendentals
`sympywrap` exports (`tan`, `cot`, `atan`, `erf`, and the rest) even though
the file does not list them. That changes promotion and ordering for
expressions that contain those calls. A full CarpetX Z4c regeneration was not
run for this change; the cost delta is pinned in
`unit_tests/test_sympy_complexity_weights.py`.

## Backends

CPU, CUDA, and ROCm are three builds of the same sources. One binary is only
one of them. Do not pass `-DEE_ENABLE_CUDA=ON` and `-DEE_ENABLE_HIP=ON`
together, and do not pass `--use_fast_math` or `-ffast-math`.
`-DEE_VREAL_FLOAT=ON` selects `float` for `vreal`; the default is `double`.

The host build is ordinary C++ and does not launch a kernel. The CUDA and
ROCm builds compile `src/bench_*.cpp` with nvcc or hipcc, launch each case as
a device kernel, and time that kernel with events. Copies and the checksum
reduction stay outside the timer. The JSON schema is the same for all three,
so `fit_weights.py` does not care which binary produced the file.

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
python3 microbenchmarks/scripts/fit_weights.py /tmp/ee_bench.json
```

Use `build-cuda` or `build-hip` the same way. `--json -` sends the human
table to stderr and pure JSON to stdout, so this pipe works:

```bash
./microbenchmarks/build/ee_microbench --quick --json - 2>/dev/null \
  | python3 microbenchmarks/scripts/fit_weights.py -
```

`--group arith,pow` selects cases. `--repeats K` and `--size N` override the
repeat count and the stream length. `--quick` is a smoke test: do not fit a
profile from it. `python3 microbenchmarks/scripts/test_fit_weights.py` checks
the fitter without running a benchmark.

A full host run is about 30 seconds. A full GPU run is longer; the A100 80GB
PCIe device run was about two minutes.

## What is timed

Each case has two flavors where that split means something:

* `*_stream`: `c[i] = f(a[i], b[i])` over a large array. This is the CarpetX
  `grid.loop_*_device` body, which is the context the weights rank.
* `*_reg`: a chained `x = f(x, k)` with the result kept live. This is ALU
  throughput with no global memory in the inner loop. On a GPU it is many
  independent per-thread chains, so the number is device throughput, not
  single-thread latency.

Reported fields are min and median ns/element, GOPS, and a checksum. A
non-finite checksum aborts. Host builds keep values live with `asm volatile`.
Device builds do the same inside the kernel and repeat each streamed math
functor so the store cannot hide the arithmetic: 256 times per element, or
32 with `--quick`. The reported ns/element is per repetition. One math op per
element is HBM-bound on these GPUs and cannot separate libm from a load.
Memory kernels stay at one evaluation per interior point and are
informational: `fit_weights.py` does not turn them into stencil weights.

Host and device numbers are not comparable. A host stream does one operation
per element and is memory- and loop-bound for cheap ops. A device streamed
math kernel does 256 repetitions per element and is compute-bound, and each
repetition also pays a perturbation FMA, a finiteness check, and an xor into
the sink. The device arith group includes `identity_stream`, the same harness
with a functor that returns its input. `fit_weights.py` subtracts that timing
and floors the difference at 0. Subtracting can otherwise go negative from
noise.

The harness times scalar `vreal` (`double`, or `float` with
`-DEE_VREAL_FLOAT=ON`). It does not time `Arith::simd<CCTK_REAL>`, whose
relative costs differ.

The host binary pins itself to the CPU it is already running on
(`sched_setaffinity`). It does not change the CPU governor; that needs
privileges the process does not take. Host defaults are 2^20 stream elements,
30 repeats, and a 128^3 grid (64^3 with `--quick`). Device defaults are 2^27
stream elements (2^24 with `--quick`), 15 repeats, and a 512^3 grid (256^3
with `--quick`), so launch overhead is not the result. `--size` still
overrides the stream length. `--quick` is applied first, then an explicit
`--repeats` or `--size` overrides it (`--repeats 30 --quick` keeps 30).

`mul_reg` and `div_reg` reset the accumulator on every repeat. A carried
multiply by `1.000002` overflows after about 20 passes, and a carried divide
drifts into the subnormals. Integer powers use the same repeated-squaring
loop as `Arith::pown`, and the pow group times `x**-1` and `x**-2` as well
as the positive exponents.

Device memory launches step threads along x, so a center load and an x, y, or
z neighbor load are each coalesced. That will not reproduce the host
1000/4000/10000 stencil spread, and it cannot see a ghost exchange.

`fit_weights.py` writes a schema_version 1 profile. Fitted slots use

```text
weight(op) = max(100, round(100 * adjusted_ns / adjusted_add))
```

with one adjusted add rounding to 100. A ratio below one add can still store
a value between 100 and 199; only a ratio below 0.5 clamps to 100. `sub`,
`div`, `neg`, `const`, `cmp_lt`, and `if_else` are recorded and are not
weights, because the cost model has no slot for them. Stencil weights stay
1000/4000/10000. Rows are keyed by `(group, name)`, so `pow/sqrt_stream`
cannot overwrite `transcendental/sqrt_stream`. A min/median spread above 25%
is a warning. The script does not probe the machine.

## Profiles

`EinsteinEngine/generators/sympy_complexity.py` loads
`guestimates.json` from the package on the first `get_weights()` call.
Override with `EE_COMPLEXITY_WEIGHTS=/path/to/profile.json`. A missing or
unreadable file is fatal at that call, not at import. Measured profiles are
files under `microbenchmarks/profiles/`; they are not package data, and there
is no name lookup. Author a schema_version 1 file (or let `fit_weights.py`
write the draft). `amd-ryzen-ai-9-hx-pro-370.json` is the pattern, including
`measurements.decisions`.

| Profile | What was measured |
|---|---|
| `guestimates.json` (package) | Nothing. Hand-written default. Smallest positive weight is 100. |
| `amd-ryzen-ai-9-hx-pro-370.json` | Host loops. ALU and libm weights are measured. Stencil weights stay 1000/4000/10000 because a single-node loop cannot see a ghost exchange. |
| `amd-epyc-7763.json` | Host loops on an AMD EPYC 7763 (`add_stream` 1.861 ns). Not A100 weights: that node has an A100, and this run did not use it. Stencil weights stay 1000/4000/10000 for the same reason as the Ryzen profile. `pow_default` is 700. |
| `intel-xeon-gold-5118.json` | Host loops on rostam1, the login node (`add_stream` 2.1836 ns). No GPU on this machine. Stencil weights stay 1000/4000/10000. `pow_default` is 1100. `pow_integer_floor` stays 200: integer powers are at or below one add. |
| `nvidia-a100-80gb-pcie.json` | Device kernels on an NVIDIA A100 80GB PCIe, sm_80 (`add_stream` 0.000909 ns per repetition). The device ratios are untrusted until the run is repeated with `identity_stream`: per-repetition harness overhead squeezed cheap ops toward one add, and `pown<16>` was 16 sequential multiplies rather than `Arith::pown`. `pow_default` remains the recorded 2100. `pow_integer_floor` is the shared policy 200. `symbol_grid` and the stencil weights are the shared policy 1000/4000/10000; the flat coalesced timings are kept in the file and are not weights. |

Do not reuse a host profile on a GPU, or the A100 profile on an AMD GPU.
An MI100 (`gfx908`) build runs; there is no full MI100 profile yet.

## Layout

| Path | Role |
|---|---|
| `common/ee_bench.hpp` | Host harness: timing, stats, JSON and table output. Never runs on device. |
| `common/kernels.hpp` | Scalar kernels shared by the CPU, CUDA, and ROCm drivers: `pow2` / `pown` / `sqrt` / `cbrt` / `pow`, `sin`…`erf`, `if_else`, grid index math. |
| `common/ee_device.hpp` | CUDA and ROCm launch and event timing. The CPU build does not include it. |
| `src/bench_arith.cpp` | `Add`, `Mul`, and the derived `sub`, `div`, and unary `neg`. |
| `src/bench_pow.cpp` | `Pow`: `x**2,3,4,8,16`, `sqrt`, `cbrt`, generic `pow(x,y)`, `x**2.5`. |
| `src/bench_transcendental.cpp` | `sin`, `cos`, `exp`, `log`, `sqrt`, `cbrt`, plus `tan`, `sinh`, `cosh`, `tanh`, `erf`. |
| `src/bench_memory.cpp` | `Symbol` grid vs local, `Atom` baseline, stencil center / x / y / z. |
| `src/bench_branch.cpp` | `Relational` compare, and `Piecewise` as `if_else` plus a real branch. |
| `src/main.cpp` | Driver: `--quick`, `--repeats`, `--size`, `--json`, `--group`. |
| `scripts/fit_weights.py` | Results JSON to a schema_version 1 profile. |
| `profiles/` | Measured profiles. Not installed with the package. Pass one with `EE_COMPLEXITY_WEIGHTS`. |
