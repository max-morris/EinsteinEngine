# Z4c RHS splitting: tuning results

Benchmark: `gauge_wave_z4c_bench.par` — gauge wave, 128³ cells, 200 iterations,
RK4, single MPI rank on one A100 (nasrin, `cuda-A100-amd`). The objective is
total seconds in `ODESolvers::Solve::rhs`, read from `AllTimersReadable.txt`.

Measurement noise is **sd ≈ 0.007 s (0.13%)**, established from repeat runs of
identical configurations. Differences below ~0.5% are not meaningful from a
single measurement.

## Headline (128³)

| configuration | best | validated mean | vs control | vs unsplit |
|---|---|---|---|---|
| Unsplit (single kernel) | — | **8.201** (3 reps) | −53.8% | — |
| Release thorn (`repos/Cottonmouth`) | — | 5.789 | −8.6% | +29.4% |
| **Control**: recipe order + split search | 5.332 | **5.332** (5 reps) | — | +35.0% |
| Order + splits searched together (39 knobs) | 5.403 | — | −1.3% | +34.1% |
| **Derived order + split search** | 4.918 | **4.933** (3 reps) | **+7.5%** | **+39.8%** |

**Best known configuration: 4.933 s** at 128³ (39.707 s at 256³), a 7.5% improvement on the previous best
and 39.8% faster than not splitting at all.

## The winning configuration

Base equation order, derived (not searched) by running the RHS unsplit under
bare `prioritize_rare_symbols` and ranking the 13 author-level `add_eqn` calls
by the median position of their scalar equations:

    9, 2, 1, 3, 4, 6, 11, 8, 5, 10, 13, 7, 12

Splits, at positions **in that derived order** (`tuner_fixed_order.py`,
checkpoint `split_tuning_fixed_order.jsonl`):

| position | action | retain |
|---|---|---|
| 1 | hard | — |
| 2 | soft | 0.0231 |
| 3 | soft | 0.6347 |
| 8 | hard | — |
| 13 | soft | 0.0073 |

All other positions: no split. 60 trials, 1 failed, 26/59 valid trials beat the
control.

## What worked, and what did not

**Supplying a good order beats searching for one.** The 39-knob run searched
splits and order together and finished 1.3% *worse* than the control, while
pinning a derived order and searching only splits finished 7.5% *better*. The
larger space had strictly more freedom and did worse — 13 extra continuous
dimensions cost more in search efficiency than they returned on a 60-trial
budget.

**Deriving from the composition is a no-op.** `Z4c.py` bakes with
`cartesian_product(insertion_order(exclude_synthetic), prioritize_rare_symbols)`,
and `cartesian_product` gives the first function first claim on every symbol.
So `insertion_order` claims all 13 author equations and `prioritize_rare_symbols`
only ever orders the CSE temporaries. Deriving an order from the composition
returns the recipe order unchanged (verified: identity permutation). The useful
order comes from **bare** `prioritize_rare_symbols`.

**Caveat on the derived order.** Equations 1, 3, 4 and 6 have near-coincident
scalar ranges (830–897, 831–898, 833–900, 836–901; medians 871.5, 872.5, 874.5,
875.5). Their relative ranking is decided by 1–4 positions out of 1061 and is
effectively arbitrary. The well-determined parts are 9 and 2 early, 7 and 12
late — that is likely where the gain comes from.

**Order and splits interact.** A permutation that is harmless in one loop
becomes invalid once a split falls between an equation and its dependency
(`_calc_tile_temps` asserts). Only **0.18%** of random permutations of these 13
equations respect the dependency graph, so `auto_order_key` treats its floats as
priorities in a topological sort rather than raw sort keys; every draw is then
valid and no trial is wasted.

### The in-flight weight sweep (`minimize_inflight`)

Searched `k` in `score = k·(RHS symbols already in flight) + rarity`, one full
split search per `k`:

| k | best | vs control |
|---|---|---|
| 0 (≡ `prioritize_rare_symbols`) | 5.332 | — |
| 0.25 | 5.301 | +0.6% |
| 0.5 | 5.558 | −4.2% |
| 1.0 | 6.211 (26 trials, stopped) | −16.5% |

Only k=0.25 improved, marginally and unvalidated. Larger k is monotonically
worse. Superseded by the derived-order result above.

### Cutting the globally sorted list

Sorting all ~1061 post-CSE equations and cutting at liveness minima was
**strictly harmful** — monotonic in the number of cuts, best at zero cuts
(8.195 s, i.e. unsplit). See `docs/DESIGN-cut-after-sort.md` on the
`tuning-experimental` branch. The mechanism was verified correct (`n_cuts=0`
reproduced the 8.201 s unsplit baseline to 0.07%); the premise was wrong.

## Scaling to 256³

Both configurations measured on a 256³ grid (8× the volume), same node and
iteration count. Unsplit is 3 reps; the tuned split is 1 run.

| configuration | 128³ | 8× linear | actual 256³ | vs linear |
|---|---|---|---|---|
| Unsplit | 8.201 | 65.61 | **66.003** | +0.6% |
| Tuned split | 4.933 | 39.46 | **39.707** | +0.6% |

**The split advantage transfers exactly.** It is 39.8% faster than unsplit at
128³ and 39.8% faster at 256³, and both configurations sit 0.6% above perfect
linear scaling. A split tuned at 128³ does not degrade at 8× the volume.

This was worth checking rather than assuming: splitting trades register
pressure for memory traffic (each value crossing a boundary becomes a tile
temporary that is written and re-read), and at 8× the working set the caches
are far less effective, so it was plausible that the extra traffic would
dominate and make splitting a net loss at scale. It does not.

Note that comparing the split at 256³ against 8× the split at 128³ shows only
that *that* configuration scales linearly — it says nothing about whether
splitting is still the right choice at that size. The unsplit measurement at
256³ is what settles it.

## Reproducing

    ./setup-arrangement.sh                                   # once
    ./submit-search.sh tuner_fixed_order.py <checkpoint>.jsonl
    ./validate-after.sh tuner_fixed_order.py <checkpoint>.jsonl 3

Searches run on a medusa allocation (`submit-search.sh`), never the login node;
timed runs are submitted from there to `cuda-A100-amd`.
