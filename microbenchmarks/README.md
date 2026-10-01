# microbenchmarks — measured costs for `sympy_complexity.py` weights

This directory times the operations the CarpetX emitter generates
(`cpp_carpetx_visitor.py`) and turns those timings into integer weights for
`EinsteinEngine/generators/sympy_complexity.py`.

The default profile, `guestimates.json`, is still the hand-written model:
`Pow` 1500, integer powers `max(200, round(100*log2|p|))`, grid `Symbol` 1000
vs local 100, stencil center 1000 / x 4000 / y-z 10000, and
`sin/cos/exp/log/sqrt/cbrt` 1500. Every profile scales its positive weights
so the smallest is 100. `transcendental_default` stays 0.
Measured profiles sit next to it. Nothing selects them unless
`EE_COMPLEXITY_WEIGHTS` points at one.

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
Memory kernels stay at one evaluation per interior point.

Host defaults are 2^20 stream elements, 15 repeats, and a 128^3 grid (64^3
with `--quick`). Device defaults are 2^27 stream elements (2^24 with
`--quick`) and a 512^3 grid (256^3 with `--quick`), so launch overhead is not
the result. `--size` still overrides the stream length.

Device memory launches step threads along x, so a center load and an x, y, or
z neighbor load are each coalesced. That will not reproduce the host
1000/4000/10000 stencil spread, and it cannot see a ghost exchange.

Suggested weight, printed by `fit_weights.py`:

```text
weight(op) = 100 * max(1, round(ns_op / ns_add))
```

with `add_stream` rounding to 100. A ratio below half an add clamps to 100.
Those integers are a draft. Record in the profile why any of them differs
from the measurement, especially when the guestimate encodes a cost this
loop cannot see.

## Profiles

`EinsteinEngine/generators/sympy_complexity.py` loads
`EinsteinEngine/generators/complexity_weights/*.json` at startup. The default
is `guestimates.json`. Override with
`EE_COMPLEXITY_WEIGHTS=/path/to/profile.json`. A missing or unreadable file
is fatal. `available_profiles()` globs every `*.json` in that directory, so
a raw `ee_microbench` results dump placed there is a broken profile. Author a
schema_version 1 file; `amd-ryzen-ai-9-hx-pro-370.json` is the pattern,
including `measurements.decisions`.

| Profile | What was measured |
|---|---|
| `guestimates.json` | Nothing. Hand-written default. Smallest positive weight is 100. |
| `amd-ryzen-ai-9-hx-pro-370.json` | Host loops. ALU and libm weights are measured. Stencil weights stay 1000/4000/10000 because a single-node loop cannot see a ghost exchange. |
| `amd-epyc-7763.json` | Host loops on an AMD EPYC 7763 (`add_stream` 1.861 ns). Not A100 weights: that node has an A100, and this run did not use it. Stencil weights stay 1000/4000/10000 for the same reason as the Ryzen profile. `pow_default` is 700. |
| `intel-xeon-gold-5118.json` | Host loops on rostam1, the login node (`add_stream` 2.1836 ns). No GPU on this machine. Stencil weights stay 1000/4000/10000. `pow_default` is 1100. `pow_integer_floor` stays 200: integer powers are at or below one add. |
| `nvidia-a100-80gb-pcie.json` | Device kernels on an NVIDIA A100 80GB PCIe, sm_80 (`add_stream` 0.000909 ns per repetition). `pow_default` is 2100. `pow_integer_floor` is 100 because `x**2` measures as one add and `x**2,3,4,8,16` track `log2(N)`, not a flat floor of 200. `symbol_grid` and every stencil weight are 1200: the coalesced launch is flat to about 1%, so 4000/10000 were not copied. |

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
| `scripts/fit_weights.py` | Results JSON to suggested integer weights. |
