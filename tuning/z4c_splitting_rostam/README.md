# Z4c RHS splitting tuner — rostam

A tuning run for the Z4c RHS loop splitting, set up for rostam. Everything
runs locally (`--remote-host localhost`); the trials go through Slurm on the
A100 nodes.

See `docs/TUNING-QUICKSTART.md` for how the framework itself works. This
directory is the rostam-specific instantiation of it; `tuning/z4c_splitting_qbd/`
is the original sample, hardcoded to a different user and host.

## What gets tuned

`recipes/Cottonmouth/Z4c.py` calls `get_tuning_param` for
`auto_hard_split_predicate` and `auto_soft_split_predicate` (Z4c.py:842). Those
predicates are consulted once per `add_eqn()` call on `fun_z4c_rhs`, with the
running count `1..N`, and decide whether to break the RHS loop there:

- `0` — no split
- `1` — soft split, retaining a tuned percentile of the intermediates
- `2` — hard split

The recipe makes exactly **13** such calls, so `tuner.py` uses
`n_vars=13`. (The qbd sample says 15; the extra two indices are never queried,
so they just cost four wasted search dimensions.)

That gives 13 categorical choices plus, for each index chosen as a soft split,
one continuous retain-percentile — 26 in_params, resolved conditionally.

## Configuration

Every path and Slurm setting lives in `config.sh`; the other scripts source it.
The defaults:

| Setting | Value | Why |
|---|---|---|
| Cactus | `/work/sbrandt/etk/Cactus` | |
| Config | `sim-a100` | A real SimFactory configuration built for sm_70+sm_80, with all four Cottonmouth thorns in its ThornList |
| Machine | `cactus-a100` | Must be passed explicitly; see below |
| Queue | `cuda-A100-amd` | Homogeneous hardware; see below |
| Geometry | 1 rank × 16 threads, 1 A100 | MPI kept out of the measurement; see below |
| Staging dir | `${CACTUS_DIR}/repos/CottonmouthTuning` | Must be inside the Cactus tree; see below |
| Simulation | `z4c-tune` | Deleted and recreated every trial |

Four of these were arrived at the hard way, by running the thing. Each note is
a failure that actually happened.

**Config `sim-a100`, not `simx`.** `simx` is the newest build and looks like the
obvious pick, but it was built with raw `make`, not simfactory, so it has no
`properties.ini` and `sim build` refuses it outright:

> Error: Configuration simx has no properties.ini file.

`sim-a100`, `sim-v100` and `simcc` are real simfactory configurations.

**Machine `cactus-a100` must be passed explicitly.** `sim-a100` belongs to the
simfactory machine `cactus-a100` (optionlist `cactus-cuda-mpich.cfg`,
`MPI_DIR=/opt/mpi/mpich`), not to `rostam`. `cactus-a100.ini` builds through
`singularity exec docker://stevenrbrandt/cactus-cuda make`, so every `sim`
invocation needs `--machine`, or simfactory picks `rostam` and builds and
launches the wrong way. `${CACTUS_DIR}/run-hpct-a100.sh` is the working
reference for all of this.

**The config's RunScript and SubmitScript are not to be overwritten.**
`run-queue.sh` copies `simfactory/mdb/runscripts/rostam.run` over its config's
`RunScript`, and copying that pattern here is destructive: `configs/sim-a100`'s
`RunScript` is the `cactus-a100` one, carrying the singularity invocation, the
`/work` autofs bind, `--mpi=pmi2` and `UCX_TLS=^posix`, and its `SubmitScript`
carries `--gpus-per-task`/`--gres`. `build-and-submit.sh` deliberately does not
touch either.

**The staging dir must live inside `${CACTUS_DIR}`.** `remote_feedback.py`
rsyncs with `--delete`, so it cannot point at `repos/Cottonmouth` — that is a
git checkout, and the recipe only generates `CottonmouthZ4c4m`, so `.git` and
the other three thorns would be deleted. But it also cannot live just anywhere:
the build runs inside singularity with no explicit `--bind`, so only `$PWD`
(the Cactus dir) is visible, and `/work` is an autofs indirect map. A staging
dir elsewhere under `/work` is simply absent when CST runs:

> Missing file /work/.../arrangements/Cottonmouth/CottonmouthZ4c4m/param.ccl

So it is `repos/CottonmouthTuning`, a sibling of the checkout, inside the tree.
`setup-arrangement.sh` retargets the single symlink
`arrangements/Cottonmouth/CottonmouthZ4c4m` at it, using a *relative* target
like the other three links so it resolves identically inside and outside the
container. The ThornList still reads `Cottonmouth/CottonmouthZ4c4m` and needs
no edit; the checkout is never touched.

**Why `cuda-A100-amd` and not `cuda-A100`.** The broader partition spans nasrin
(2× A100-80GB PCIe, 128 cores) and toranj (4× A100-40GB PCIe, 64 cores). Which
node Slurm happened to give you would move the measured time by more than many
of the splits do, and the optimizer is fitting a surrogate to exactly these
numbers — hardware noise costs real trials. `cuda-A100-amd` is just the nasrin
pair, so every trial is measured on identical hardware.

**Why one rank.** Two ranks on a node abort in UCX before the first iteration:

> cma_ep.c:81 process_vm_readv(...) returned -1: Operation not permitted

Each rank runs in its own singularity instance, so they are in separate PID
namespaces and CMA's cross-process reads are denied. `cactus-a100.run` already
sets `UCX_TLS=^posix` for the sibling shared-memory failure, but that just
falls back to CMA, which is what fails here. Forcing tcp
(`UCX_TLS=^posix,cma`) would work but puts a slow intra-node transport inside
the thing being measured. One rank sidesteps it and is the better benchmark
anyway: the split being tuned changes a local kernel, so taking MPI out of the
objective removes a source of run-to-run variance rather than averaging over
it.

Note also that simfactory rejects `--ppn` above the machine's declared `ppn`
(max 32 here), so the full-node 128/64 shape mentioned in `run-hpct-a100.sh`'s
comments is not reachable without editing `cactus-a100.ini`.

## Status

Verified end to end on 2026-09-18: generation → rsync → incremental rebuild →
Slurm submit on `cuda-A100-amd` → timing readback.

- Unsplit baseline (the checkout's thorn, as seeded by `setup-arrangement.sh`):
  **5.789 s** in `ODESolvers::Solve::rhs`.
- First random split trial: **6.729 s** — worse, as you would expect from an
  arbitrary split, and exactly the signal the search needs.

`split_tuning_checkpt.jsonl` holds that one real trial. It was measured with
the final configuration, so it is legitimate data and the search will resume
from it; delete the file if you would rather start clean.

A trial costs roughly: ~5 min recipe regeneration, an incremental rebuild of
`CottonmouthZ4c4m` only, then a ~15 s job plus queue wait.

## The benchmark

`gauge_wave_z4c_bench.par`, derived from
`recipes/Cottonmouth/apples_with_apples/gauge_wave_z4c.par`, with three
deliberate differences — it is a benchmark, not a convergence test:

1. **Cubic grid** (128³ by default) instead of 50×8×8. The gauge wave depends
   only on x and t (`gauge_wave.py:129`), so extending y and z is physically
   free and gives the RHS kernel a realistically shaped block. With
   `ghost_size = 3`, an 8-cell direction is almost entirely ghost zone and
   would not measure what the splitting costs in production.
2. **Fixed small `itlast`** (200), so a trial is minutes rather than the
   200000 iterations the apples-with-apples run does.
3. **TimerReport on**, output and poisoning off.

Grid size and iteration count are the two knobs at the top of the parfile.
At the defaults (128³, 200 iterations, RK4) the job runs ~15 s wall and spends
~5.8 s in the RHS, which is enough signal to rank splits without making the
queue wait dominate.

> One caveat worth keeping in mind when you read the results: the gauge wave is
> a smooth, single-level, periodic problem. A split tuned on it is tuned for
> that kernel shape. If the target is production binary-puncture runs, confirm
> the winner against `recipes/Cottonmouth/parfiles/qc0-z4c.par` before adopting it.

## The objective

`timings.sh` prints the total seconds in `ODESolvers::Solve::rhs`, averaged over
ranks, read from `AllTimersReadable.txt`. That timer is used rather than a
per-function one because hard-splitting renames and multiplies the generated
`z4c_rhs` schedule entries, so no per-function timer name is stable across
trials. Iteration count is fixed by the parfile, so totals are comparable.

If the run crashed or produced no timers, `timings.sh` exits nonzero; the
framework scores that trial `-inf` and the search avoids that region. This is
also what happens when a split is degenerate (empty loop, loop with no outputs).

## Prerequisites

EinsteinEngine needs Python >= 3.13, but rostam's default `python` is 3.9. A venv
built from `/opt/apps/python/3.13.2` is already set up at the repo root
(`venv/`, gitignored) with `requirements.txt` and the package installed;
`config.sh` picks it up automatically, so the scripts do not need it activated.

To rebuild it:

```bash
cd ../..
/opt/apps/python/3.13.2/bin/python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip install .
```

Export `EE_PYTHON` to point the scripts at a different interpreter.

## Usage

One-time, before the first run:

```bash
./setup-arrangement.sh
```

Then:

```bash
./run-tuning.sh                                        # 10 warmup + 20 guided
./run-tuning.sh --iterations 50                        # search longer
./run-tuning.sh --warmup-iterations 0 --iterations 1   # single-trial smoke test
```

The checkpoint (`split_tuning_checkpt.jsonl`, gitignored) is the source of
truth. Rerunning replays it, so you can stop and restart freely or raise the
iteration count later.

To bake the winner into generated code under `./Cottonmouth/` (local only, no
build or submit):

```bash
./generate-best.sh
```

To see progress:

```bash
PYTHONPATH=../.. python -m EinsteinEngine.tuning.plot_tuning split_tuning_checkpt.jsonl
```

When finished:

```bash
./restore-arrangement.sh
```

## Files

| File | Role |
|---|---|
| `config.sh` | All paths and Slurm settings; sourced by the rest |
| `tuner.py` | Supplies the `Tuner` (`get_tuner()`), read by `remote_tuner` |
| `gauge_wave_z4c_bench.par` | The benchmark |
| `setup-arrangement.sh` | One-time: stage dir + retarget the arrangement symlink |
| `restore-arrangement.sh` | Undo the above |
| `build-and-submit.sh` | `--remote-command`: build, then `sim create-submit` |
| `timings.sh` | `--remote-timing-command`: prints the one number to minimize |
| `run-tuning.sh` | Drives `EinsteinEngine.tuning.remote_tuner` |
| `generate-best.sh` | Drives `EinsteinEngine.tuning.generate_best` |

---

### Baking a tuned Z4c without running a search

The result checkpoints are committed, so another machine can generate the tuned
code directly:

    ./generate-best.sh              # best known: 4.933 s at 128^3

That writes generated code under `./Cottonmouth/` and runs nothing. Pair a
checkpoint only with the tuner that produced it -- the tuner supplies the
equation ordering the trials were measured under:

| checkpoint | tuner | best |
|---|---|---|
| `split_tuning_fixed_order.jsonl` | `tuner_fixed_order.py` | **4.933 s** @128³ (default) |
| `split_tuning_checkpt.jsonl` | `tuner.py` | 5.332 s @128³, recipe order |
| `split_tuning_256.jsonl` | `tuner_fixed_order.py` | 39.505 s @256³ |

    TUNER_FILE=tuner.py CHECKPOINT_FILE=split_tuning_checkpt.jsonl ./generate-best.sh

Note `generate-best.sh` needs no Cactus, no Slurm and no `config.sh` edits --
it only regenerates the recipe locally. `setup-arrangement.sh` and the search
scripts are the parts that need porting.

## Porting to another machine

Everything machine-specific lives in `config.sh`. Nothing else should need
editing. The values below are for rostam; replace them:

| setting | what it is |
|---|---|
| `CACTUS_DIR` | the Cactus installation the thorn is built in |
| `SIM_BASEDIR` | where simfactory writes simulations (`basedir` in the machine's ini) |
| `STAGE_DIR` | scratch dir the tuner rsyncs each trial's generated thorn into. **Must be inside `CACTUS_DIR`** if the build runs in a container that only binds `$PWD` — see the note in `config.sh`. Must NOT be a git checkout: the rsync uses `--delete`. |
| `ARRANGEMENT_LINK*` | the `arrangements/<arr>/<thorn>` symlink retargeted at `STAGE_DIR`, and its original target |
| `CONFIG` / `MACHINE` | simfactory configuration and machine names. The configuration must be a real simfactory one (it needs `properties.ini`); a config built with plain `make` will be refused. |
| `QUEUE` | Slurm partition for the timed runs. Prefer a **homogeneous** partition: on a mixed one the node you draw can change the measurement by more than the thing being tuned. |
| `PPN` / `NUM_THREADS` | node shape. simfactory rejects `--ppn` above the machine's declared `ppn`. |
| `PARFILE_DIR` | where the benchmark parfile is installed. Must be **outside** `STAGE_DIR`, which is wiped by the `--delete` rsync each trial. |
| `PYTHON` | defaults to the repo venv; override with `EE_PYTHON`. |

Also check:

- **`build-and-submit.sh`** does not copy the machine's generic runscript over
  the configuration's own. On rostam the `cactus-a100` config carries a
  singularity invocation, a `/work` bind, `--mpi=pmi2` and `UCX_TLS` settings
  that a generic runscript would destroy. Whether your machine needs the same
  care depends on its configuration.
- **`timings.sh`** greps `ODESolvers::Solve::rhs` out of
  `AllTimersReadable.txt`. Any benchmark works as long as this prints one
  number; all format-specific parsing is meant to live in this script.
- **`submit-search.sh`** runs the driver as a Slurm job so neither the sympy
  generation nor the build lands on a login node. It assumes a job may submit
  jobs (the driver `sbatch`es its own timed runs and polls `squeue`). Verify
  that is permitted before relying on it.
- The benchmark parfiles assume a GPU build (`CarpetX::max_tile_size_*` is set
  large to disable tiling). Adjust for CPU.

### First run on a new machine

    ./setup-arrangement.sh                                      # once; undo with restore-arrangement.sh
    ./submit-search.sh tuner_fixed_order.py smoke.jsonl --warmup 1 --iterations 0
    # check logs/search-<jobid>.out, then:
    ./submit-search.sh tuner_fixed_order.py mysearch.jsonl --warmup 10 --iterations 50

`tuner_fixed_order.py` reads `ordertest/derived_order.json`, which is committed.
That order was derived on the Z4c recipe and is machine-independent, so it does
not need regenerating — but `ordertest/run_derive.sh` will redo it if the recipe
changes.
