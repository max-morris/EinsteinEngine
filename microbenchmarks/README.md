# microbenchmarks — measured costs for `sympy_complexity.py` weights

The integers in `EinsteinEngine/generators/sympy_complexity.py`
(`Add`/`Mul` summation, `Pow` default 15 / `max(2, log2|p|)` for integer powers,
grid `Symbol` 10 vs local 1, `stencil` center 10 / x 40 / y-z 100,
`sin/cos/exp/log/sqrt/cbrt` 15) are guestimates. This directory measures them
with small portable C++ benchmarks whose kernels mirror exactly what the
CarpetX emitter generates (`cpp_carpetx_visitor.py`).

## Layout

| Path | What it measures (complexity node) |
|---|---|
| `common/ee_bench.hpp` | Host-only harness: timing, stats, JSON/table output. Never runs on device. |
| `common/kernels.hpp` | Device-compatible kernels: `pow2`/`pown`/`sqrt`/`cbrt`/`pow`, `sin`…`erf`, `if_else`, grid index math. Shared by CPU and GPU drivers. |
| `src/bench_arith.cpp` | `Add`, `Mul` (+ derived `sub`, `div`, unary `neg`) |
| `src/bench_pow.cpp` | `Pow`: `x**2,3,4,8,16`, `sqrt`, `cbrt`, generic `pow(x,y)`, `x**2.5` |
| `src/bench_transcendental.cpp` | `sin cos exp log sqrt cbrt` (=15) + `tan sinh cosh tanh erf` |
| `src/bench_memory.cpp` | `Symbol` grid vs local, `Atom` baseline, `stencil` center/x/y/z (10/40/100) |
| `src/bench_branch.cpp` | `Relational` compare, `Piecewise` `if_else` ternary + branching form |
| `src/main.cpp` | Driver: `--quick --repeats K --size N --json f --group g1,g2` |
| `scripts/fit_weights.py` | JSON → suggested integer weights (`Add`=1 reference) |

## Build & run (CPU, this machine)

```bash
# cmake path
cmake -S microbenchmarks -B microbenchmarks/build -DCMAKE_BUILD_TYPE=Release
cmake --build microbenchmarks/build
./microbenchmarks/build/ee_microbench --quick

# or fallback without cmake
make -C microbenchmarks
./microbenchmarks/ee_microbench --quick
```

Full run (~30 s) and weight fitting:

```bash
./microbenchmarks/build/ee_microbench --json /tmp/ee_bench.json
python3 microbenchmarks/scripts/fit_weights.py /tmp/ee_bench.json
# pipe-friendly: human table goes to stderr, pure JSON to stdout
./microbenchmarks/build/ee_microbench --quick --json - 2>/dev/null \
  | python3 microbenchmarks/scripts/fit_weights.py -
```

Per-group smoke: `./ee_microbench --quick --group arith,pow`.

Self-test (no benchmark run needed):
`python3 microbenchmarks/scripts/test_fit_weights.py`.

## Feeding results back into EinsteinEngine

`EinsteinEngine/generators/sympy_complexity.py` loads its cost model from JSON
at startup (`EinsteinEngine/generators/complexity_weights/*.json`, default
`guestimates.json`, override with `EE_COMPLEXITY_WEIGHTS=/path/to/profile.json`).
Use `fit_weights.py` output to author a new machine profile next to
`guestimates.json` and `amd-ryzen-ai-9-hx-pro-370.json` (see the latter's
`measurements.decisions` for how measured numbers were mapped to integers).

## Method

Each case has two flavors where meaningful:

* `*_stream`: `c[i] = f(a[i], b[i])` over a large array — mimics a CarpetX
  `grid.loop_*_device` body, which is the context the weights rank/split for.
* `*_reg`: chained `x = f(x, k)` accumulation — isolates ALU throughput from
  DRAM bandwidth.

Reported: min/median ns/element over repeats, GOPS, checksum. The checksum is
printed and validated finite — a non-finite result aborts (fatal, never ignored)
and anti-optimization uses `asm volatile` barriers plus result consumption.

Normalized weight: `round(ns_op / ns_add)` with `Add`=1, matching the additive
complexity model (`Add`/`Mul` sum children; leaves cost 1).

## GPU portability (for a GPU machine)

* Measured code lives in `common/kernels.hpp` as `EE_HD_INLINE`
  (`__host__ __device__` under `__CUDACC__`/`__HIPCC__`, plain `inline` on CPU).
* Timing/allocation/JSON stay in `ee_bench.hpp` (host-only, never in kernels).
* Each `bench_*.cpp` documents the `__global__` launch skeleton reusing the same
  functor; GPU timing should use CUDA/HIP events around the kernel, keeping the
  same JSON schema so `fit_weights.py` works unchanged.
* CMake options: `-DEE_ENABLE_CUDA=ON`, `-DEE_ENABLE_HIP=ON`,
  `-DEE_VREAL_FLOAT=ON` (if the target uses `float` for `vreal`).
* Re-measure per device/dtype: cache, coalescing, and libm costs differ between
  CPU, CUDA, and HIP targets — do not reuse CPU weights on GPUs.
