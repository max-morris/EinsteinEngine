# Tuner quick-start

`EinsteinEngine.tuning.remote_tuner` runs a checkpointed Bayesian search over a
recipe's parameters. Each trial regenerates the recipe with a candidate set of
values, ships the generated code to a remote machine, builds and runs it under
Slurm, reads back a timing, and feeds that timing to the optimizer. Over many
trials it homes in on the parameters that make the run fastest.

You point it at three things:

| Piece            | What it is                                                        |
|------------------|-------------------------------------------------------------------|
| **Recipe**       | An ordinary EinsteinEngine recipe that reads a few tunable knobs.     |
| **Tuner file**   | A standalone `.py` that hands the optimizer an `Experiment`.      |
| **Checkpoint**   | A JSON-lines log of every trial; resumed automatically on rerun.  |

The optimizer *maximizes*, and the objective is `-timing_value` (the single
number your timing command prints), so a lower reported time wins.

A complete, working example lives in `tuning/z4c_splitting_qbd/` (recipe
`recipes/Cottonmouth/Z4c.py`, tuner `tuning/z4c_splitting_qbd/tuner.py`, plus the
`run-tuning.sh` / `generate-best.sh` wrappers). This guide walks through the
smallest possible version of that, then covers the richer domain features.

---

## 1. Make the recipe read tunable knobs

A recipe exposes a knob by calling `get_tuning_param(name, default)`. Outside of
a tuning run the default is used, so the recipe still runs normally on its own.
During a tuning run the tuner must provide the name, or `get_tuning_param`
raises (this catches typos). A knob that a tuner may leave out is read with
`get_optional_tuning_param(name, default)`, which returns the default whenever
the tuner does not provide it.

```python
from EinsteinEngine import SplitLocus, get_optional_tuning_param, get_tuning_param, maximize_symbol_reuse

fun = mod.create_function(
    "my_rhs",
    rhs_group,
    # These come from the tuner during a search; None (the default) otherwise.
    auto_hard_split_predicate=get_tuning_param('auto_hard_split_predicate', None),
    auto_soft_split_predicate=get_tuning_param('auto_soft_split_predicate', None),
    # Where in the bake the predicates are evaluated (see section 5).
    auto_split_locus=get_optional_tuning_param('auto_split_locus', SplitLocus.Early),
)

mod.bake(
    ordering_fn=maximize_symbol_reuse,
    # Optional ordering functions, one per locus (section 5). The defaults keep the usual bake. They are passed
    # for the tuned function only, since an add_eqn_order([...]) refers to that function's add_eqn calls.
    functions={"my_rhs": {
        "early_ordering_fn": get_optional_tuning_param('early_ordering_fn', None),
        "pre_population_ordering_fn": get_optional_tuning_param('pre_population_ordering_fn', None),
        "ordering_fn": get_optional_tuning_param('ordering_fn', maximize_symbol_reuse),
    }},
)
```

The `name` here must match an **out_param** name declared in the tuner (below).
The split tuners of section 5 provide all six of these knobs (the ordering
functions only when you give them one), and probing checks that the recipe reads
every knob that changes the number of split positions. See
`recipes/Cottonmouth/Z4c.py` for the real usage.

---

## 2. Write a tuner file

The tuner file is executed by `remote_tuner` (via `runpy`, *not* as `__main__`).
It must expose a `Tuner` instance in one of two ways:

- define `get_tuner() -> Tuner`, or
- assign a module-level variable named `tuner`.

A `Tuner` implements `get_experiment()`, which builds an `Experiment`. (A tuner
whose search space depends on the recipe also overrides `probe_targets()` and
`get_probed_experiment(probe)`; see section 5.) The `Experiment` has two halves:

- **in_params** — the search space the optimizer samples (`add_in_param`).
- **out_params** — functions that turn a sampled point into the recipe-facing
  values the recipe reads via `get_tuning_param` (`add_out_param`).

Here is a minimal tuner with **simple domains** — each in_param is just a
`(lo, hi)` tuple:

```python
from typing import Any

from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.tuning import Tuner


class MyTuner(Tuner):
    def get_experiment(self) -> Experiment:
        e = Experiment()

        # in_params: the raw knobs the optimizer searches.
        #   (lo, hi) with two ints  -> integer search in [lo, hi]
        #   (lo, hi) with any float -> continuous search in [lo, hi]
        e.add_in_param('block_size', (16, 256))          # int in [16, 256]
        e.add_in_param('unroll_factor', (1, 8))          # int in [1, 8]
        e.add_in_param('threshold', (0.0, 1.0))          # float in [0, 1]

        # out_params: map the sampled point to what the recipe consumes.
        # The mapping receives a dict of the realized in_param values.
        e.add_out_param('tile', lambda p: (int(p['block_size']), int(p['unroll_factor'])))
        e.add_out_param('threshold', lambda p: p['threshold'])

        return e


def get_tuner() -> Tuner:
    return MyTuner()
```

Notes:

- The recipe reads `get_tuning_param('tile', ...)` and
  `get_tuning_param('threshold', ...)` — those names are the out_param names.
- An out_param mapping that returns `None` is **omitted** for that trial. The
  recipe must then read that knob with `get_optional_tuning_param`, which falls
  back to its default; `get_tuning_param` raises for an omitted knob. This is
  how you make a knob conditionally present.
- in_param and out_param names are independent; you can have several out_params
  derived from the same in_params, as the Z4c example does.

---

## 3. Run the search

Invoke the module with the recipe and tuner paths, plus where to put the
generated code and how to reach the remote machine:

```bash
PYTHONPATH="$REPO_ROOT" python -m EinsteinEngine.tuning.remote_tuner \
    "$REPO_ROOT/recipes/Cottonmouth/Z4c.py" \
    ./tuner.py \
    --local-path   /path/to/local/generated/ \
    --remote-host  qbd \
    --remote-path  /home/you/project/Cottonmouth/ \
    --remote-cactus-path /home/you/project/Cactus/ \
    --remote-command './build.sh && ./run-all.sh' \
    --remote-timing-command ./timings.sh \
    --checkpoint-file ./my_tuning.jsonl \
    --warmup-iterations 10 \
    --iterations 20
```

The cleanest way to keep this reproducible is to copy `run-tuning.sh` from
`tuning/z4c_splitting_qbd/` and edit the paths; `"$@"` at the end lets you pass extra
flags through (e.g. `./run-tuning.sh --iterations 50`).

What each trial does, in order (see `remote_feedback.py:do_remote_run`):

1. Regenerate the recipe with the trial's parameters into `--local-path`.
2. `rsync --delete` `--local-path` → `--remote-path`.
3. Run `--remote-command` under `--remote-cactus-path`; parse the Slurm job id
   from a line matching `Submit finished, job id is <N>`.
4. Poll `squeue` every 60 s until the job leaves the queue.
5. Run `--remote-timing-command`; its stdout must be a **single number**, which
   is taken as the value to optimize for. All format-specific parsing lives in
   the timing script, so any run layout can be supported by editing that script
   alone. The sample `timings.sh` extracts the RHS solve time.
6. Record `{"target": -timing_value, "params": {...}}` to the checkpoint.

A trial that throws (e.g. the recipe produced a degenerate split) is scored
`-inf` and discarded, so the search simply avoids that region.

### Iterations and the budget

`--warmup-iterations` are random-ish exploration; `--iterations` are the guided
trials after that. The optimizer runs until the checkpoint holds
`warmup + iterations` **completed** trials total. Trials rejected by a parameter
constraint (below) do not count against the budget.

### Resuming

The checkpoint is the source of truth. Point a rerun at the same
`--checkpoint-file` and every recorded trial is replayed into the optimizer
before it continues — so you can stop and restart freely, or bump the iteration
count to search longer. You'll see `Resumed from checkpoint: N observations
loaded`.

### Running locally (no remote)

Pass `--remote-host localhost` to skip `ssh`/`scp` entirely: the recipe is
generated, `rsync`'d to a local `--remote-path`, and the build/run/timing
commands execute through your local shell. Useful for smoke-testing the loop
before pointing it at a cluster.

---

## 4. Generate the best code

Once you've searched, bake the winning parameters into generated code with the
companion module. It reads the checkpoint, finds the max-target entry, pins those
in_params, and runs the recipe **once, locally** (no build/run):

```bash
PYTHONPATH="$REPO_ROOT" python -m EinsteinEngine.tuning.generate_best \
    "$REPO_ROOT/recipes/Cottonmouth/Z4c.py" \
    ./tuner.py \
    --checkpoint-file ./my_tuning.jsonl
```

Use the same tuner file — it supplies the same `Experiment` that maps the stored
in_params back to recipe-facing out_params. See `generate-best.sh` in the sample.
For a tuner that probes the recipe (section 5), `generate_best` probes the recipe,
and the result must match the probe recorded next to the checkpoint, if there is
one. The best entry is checked against the recorded probe first, so a mismatched
checkpoint fails before the (slow) probe runs.

To eyeball progress, `plot_tuning.py` renders the checkpoint's target history.
Pass `--tuner` to plot reparameterized params as their values. For a probing
tuner, `--recipe` probes the recipe (checked against the probe file, as above);
without it the probe file next to the checkpoint is used, and with neither it
warns and plots the raw coordinates.

Both `generate_best` and resuming a search **refuse** a checkpoint entry that
does not fit the `Experiment` exactly, rather than silently misreading it: one
with params the `Experiment` does not declare, one missing a param that the
`Experiment` declares and whose condition holds, or one with a coordinate
outside the range its domain gives it (e.g. `add_distinct_sorted` positions that
are not increasing). A stored param whose condition does not hold is ignored.
That usually means the
checkpoint was recorded against a different tuner or recipe (see "Old
checkpoints" in section 5).

---

## 5. Split tuners, loci and probing

`EinsteinEngine.tuning.tune_splitting` has two ready-made tuners for a
function's auto split predicates. Both provide the out_params
`auto_hard_split_predicate`, `auto_soft_split_predicate` and `auto_split_locus`,
plus whichever of `early_ordering_fn`, `pre_population_ordering_fn` and
`ordering_fn` you pass them. The recipe must read all of them (section 1).

### Loci

The **split locus** says at which point of the bake a function evaluates its
predicates, and so what a position means. Positions are **0-based**: the
predicates are called with `0..N-1`, positions run across the function's loops,
and position `i` means "split after element `i`". A **cut** is a hard or soft
split at one position. The hard predicate is asked first; the soft one only when
hard returns False. Manual `split_loop()` / `soft_split()` calls still apply, and
no locus reorders equations across them.

| `SplitLocus`     | Elements                                                               | Order of the elements                                                                                   | Z4c N |
|------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------|
| `Early`          | `add_eqn` calls (a tensor equation's components stay together)          | `early_ordering_fn`, applied to groups; without it, recipe order                                        | 13    |
| `PrePopulation`  | scalar equations and pull-out temporaries, before CSE temporaries exist | the pre-CSE bake order: `pre_population_ordering_fn`, or `ordering_fn` when that is None               | 64    |
| `PostPopulation` | scalar equations and temporaries after global CSE                       | `ordering_fn` (the post-CSE order)                                                                      | 1061  |

With two params per position, `CombinatorialSplitTuner` is only practical at the `Early` locus; use
`CutPositionSplitTuner` at the others.

### Probing

A split tuner needs `N` before it can declare its search space, and `N`
depends on the recipe, the locus and the ordering functions. Rather than
hardcoding it, the tuner names the out_params to **probe** in
`probe_targets()`, and the fixed out_params that change `N` (the locus and the
ordering functions) in `probe_params()`. Before the search starts, the recipe is
run with a placeholder predicate that never splits, and with the
`probe_params()`; while probing, every other `get_tuning_param` returns its
default. The engine reports how many positions each placeholder is evaluated at,
and `N` is passed to `get_probed_experiment(probe)` as
`probe['auto_hard_split_predicate']`.

Every `probe_targets()` and `probe_params()` name must also be an out_param of
the `Experiment` that `get_probed_experiment` returns (a `probe_params()` value
with a mapping that returns it unchanged), so that the trials see what the probe
counted; `build_experiment` refuses a tuner whose `Experiment` lacks one.

What the probe does to the recipe:

- The recipe is stopped at the end of the first `bake` after which every
  placeholder has been reported. Everything before that point still runs: a
  recipe that bakes and generates several thorns emits the code of each thorn it
  generates before then (into the working directory).
- If a placeholder is never reported, nothing stops the recipe. It runs to
  completion, emitting all of its code, and the probe then fails.

A probe fails with a clear error if the recipe:

- never reads one of the probed or `probe_params()` names (a recipe that ignored
  the locus or an ordering function would be probed, and tuned, against the
  wrong positions). Only the reads made before the probe stops the recipe are
  seen, so a name read only after that bake (e.g. for a second thorn) counts as
  never read;
- never evaluates a placeholder (it is not passed to `create_function`, or the
  function never reaches its locus);
- shares one predicate between functions with different `N` (give each
  function its own out_param).

### The probe file

The probe is recorded next to the checkpoint in `<checkpoint>.probe.json`,
because the checkpoint's coordinates only mean something against it:

```json
{
  "counts": {"auto_hard_split_predicate": 13},
  "probe_params": {"auto_split_locus": "SplitLocus.Early",
                   "early_ordering_fn": "add_eqn_order([8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11])"}
}
```

`probe_params` describes each `probe_params()` value in a form that is stable
across runs (see `probe.describe_param`): an enum by name; a function by its
qualified name, and a lambda or nested function also by its defaults and the
contents of its closure (so `cartesian_product(a, b)` and
`cartesian_product(b, a)` differ); a
`functools.cache` wrapper by what it wraps; a `functools.partial` by its
function, arguments and keywords; lists, tuples, dicts and sets element by
element (sets sorted); and the functions made by `add_eqn_order` /
`add_eqn_key_order` by their order or key.

So the check detects a changed number of positions, and a change to the tuner's
locus or ordering functions as far as their descriptions show it. It does not
see everything:

- A function's code and the globals it reads are not described (nor the
  defaults of a module-level function), so two lambdas in one scope with the
  same defaults and closure look the same, and editing a function goes
  unnoticed. A bound method is described by its function,
  plus its instance only if the instance's class defines `__repr__`; any other
  object by its repr.
- Only the tuner's `probe_params()` are recorded. A change on the recipe side
  (e.g. its default ordering function) is detected only if it changes the
  number of positions.

How the probe file is used:

- `remote_tuner` probes the recipe every time it starts. If the probe file
  exists and does not match (different counts, or different probe param
  descriptions), it refuses to resume and says what changed, since the stored
  coordinates would be misinterpreted: start a new checkpoint file, or restore
  the recipe and tuner. A mismatch is not an error while the checkpoint has no
  entries.
- It writes the probe file when it is missing (or stale, for an empty
  checkpoint), but only after checking that every checkpoint entry loads against
  the new `Experiment`, with every coordinate in the range its domain gives it.
- `generate_best` and `plot_tuning --recipe` probe too, and refuse a mismatch in
  the same way, but never write the probe file. Without a recipe to probe, the
  probe file's counts are used, and its probe params are still compared.

`tuning/.gitignore` ignores `**/*.jsonl*`, which covers both checkpoints and
their `.jsonl.probe.json` files. If you commit a checkpoint (`git add -f`), commit
its probe file too; without it, `generate_best` still works (it probes), but the
check that the recipe and tuner match the checkpoint is lost.

To write a probing tuner of your own, override `probe_targets()` (and
`probe_params()` if needed) and `get_probed_experiment`:

```python
from collections.abc import Collection

from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.probe import ProbeResult
from EinsteinEngine.tuning.tuning import Tuner


class MyTuner(Tuner):
    def probe_targets(self) -> Collection[str]:
        return ('auto_hard_split_predicate',)

    def get_experiment(self) -> Experiment:
        raise RuntimeError("MyTuner probes the recipe; build it with tuning.build_experiment.")

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        n = probe['auto_hard_split_predicate']
        ...  # declare in_params over the n positions, and an out_param 'auto_hard_split_predicate'
```

`EinsteinEngine.tuning.probe.probe_recipe(recipe, names, extra_params)` runs a
probe directly, e.g. to see `N` at each locus.

### `CombinatorialSplitTuner`

```python
CombinatorialSplitTuner(n_vars=None, *, locus=SplitLocus.Early,
                        early_ordering_fn=None, pre_population_ordering_fn=None, ordering_fn=None)
```

Decides every position independently. Params, for `i` in `0..N-1`:
`split_i` (0: none, 1: soft, 2: hard) and `soft_retain_percentile_i` (only
when `split_i` is 1). `N` is probed unless `n_vars` is given; the predicates
never split at `i >= N`. It warns when `N` is above 64.

### `CutPositionSplitTuner`

```python
CutPositionSplitTuner(n_cuts, *, locus, early_ordering_fn=None, pre_population_ordering_fn=None,
                      ordering_fn=None, allow_unused_cuts=True)
```

Places `n_cuts` cuts among the `N` probed positions, so the number of params
does not grow with `N`. Params, for `j` in `0..n_cuts-1`: `cut_position_j`
(strictly increasing, via `add_distinct_sorted`), `cut_kind_j` (0: unused, only
searched when `allow_unused_cuts`; 1: soft; 2: hard) and
`cut_retain_percentile_j` (only for soft cuts). Use it at the
`PrePopulation` and `PostPopulation` loci.

### Samples

- `tuning/z4c_splitting_qbd/`: `CombinatorialSplitTuner()` at the early locus
  in recipe order.
- `tuning/z4c_derived_order/`: the same, with the `add_eqn` groups in a fixed
  order given by `add_eqn_order([...])` (0-based `add_eqn` call indices).

Each has a `tuner.py` and `run-tuning.sh` / `generate-best.sh` wrappers.

### Old checkpoints

A split checkpoint recorded before this change is refused, for one or both of
two reasons, and the error names the undeclared params:

- Split params used to be 1-based (`split_1..N`), so `split_N` is undeclared.
- It may have more positions than the recipe has now. The old Z4c tuner declared
  15 positions, but the Z4c RHS has 13 `add_eqn` calls; the positions after the
  last call were never queried, so they can be dropped.

Convert it once, into a new file:

```bash
python scripts/shift_checkpoint_indices.py old_checkpt.jsonl new_checkpt.jsonl
python scripts/shift_checkpoint_indices.py old_checkpt.jsonl new_checkpt.jsonl --n-positions 13
```

Only `split_<int>` and `soft_retain_percentile_<int>` keys are shifted; other
keys are copied unchanged. `--n-positions N` drops the (shifted) split keys with
an index of `N` or more, and reports how many of the dropped keys were splits.
The script refuses an already 0-based file, and an output file or output probe
file that exists, and warns if it found no split keys to shift.

---

## 6. Advanced domains

The `(lo, hi)` tuple is shorthand for `Interval(lo, hi)`. When a plain interval
isn't the right search space, `add_in_param` also accepts a `Domain` object or a
*resolver* callable. Everything below reparameterizes the search so that **every
point the optimizer proposes is valid** — no reject-and-retry — which keeps the
sampler's coordinate space dense and well-behaved.

Import the domain types from `EinsteinEngine.tuning.experiment`:

```python
from EinsteinEngine.tuning.experiment import Discrete, Interval, Union
```

### `Discrete` — an explicit set of allowed values

Values are sorted and de-duplicated; the optimizer searches an index into them,
so adjacent indices map to adjacent values.

```python
e.add_in_param('tile', Discrete([32, 64, 128, 256]))   # only these four
```

### `Union` — a domain with gaps

A discontinuous space built from disjoint inclusive intervals.

```python
# Integer union enumerates every value; the gap (4) is never produced.
e.add_in_param('n', Union([(0, 3), (5, 6)]))            # 0,1,2,3,5,6

# Float union samples uniformly across the union and skips the gap.
e.add_in_param('w', Union([(0.0, 1.0), (10.0, 11.0)]))
```

### Conditions — sample a param only sometimes

Pass `condition=` a predicate over the values realized *earlier* in the same
trial. If it returns `False`, the param is skipped entirely for that trial (and
is absent from the checkpoint entry). in_params are resolved in declaration
order, so a condition can read any param declared before it.

```python
e.add_in_param('use_soft', (0, 1))
# Only search the percentile when use_soft was sampled as 1.
e.add_in_param('soft_percentile', (0.0, 1.0),
               condition=lambda p: p['use_soft'] == 1)
```

This is exactly the pattern the Z4c tuner uses to search a soft-split percentile
only for positions it decided to soft-split (`CombinatorialSplitTuner` in `tune_splitting.py`).

### Constraints — reject a value outright

Pass `constraint=` a predicate over the *single* value. A violating value raises
`InfeasibleParamError`, the optimizer prunes that trial, and it does **not**
consume the iteration budget.

```python
e.add_in_param('n', (1, 10), constraint=lambda v: v % 2 == 1)   # odd only
```

Prefer a `Discrete`/`Union`/resolver domain over a constraint when you can
express the feasible set directly — a constraint that rejects most of the space
can burn through the internal attempt cap (100× the remaining budget) and stop
early with a warning. Reserve constraints for genuinely irreducible predicates.

### Dynamic domains — a domain that depends on earlier picks

Instead of a fixed domain, pass a callable `realized -> domain`. It is evaluated
per trial with the values already chosen, so the domain can narrow itself. This
is how you express "distinct values" without rejection.

```python
# 'hi' must always exceed 'lo': shrink hi's interval using lo's realized value.
e.add_in_param('lo', (0, 50))
e.add_in_param('hi', lambda p: Interval(int(p['lo']) + 1, 100))
```

Two ready-made helpers build common dynamic patterns and return the generated
names:

```python
# k strictly-increasing ints in [lo, hi] (distinct, and collapses permutation
# symmetry since order is fixed). Use when the params are interchangeable.
names = e.add_distinct_sorted('cut', k=3, bounds=(1, 20))

# k distinct values drawn from a pool without replacement (order is meaningful,
# i.e. the params are distinguishable roles).
names = e.add_distinct_choice('slot', k=3, pool=[10, 20, 30, 40])

e.add_out_param('cuts', lambda p: tuple(p[n] for n in names))
```

### What gets checkpointed

The checkpoint stores the raw **coordinates** the sampler searched, not the
mapped values (for a plain `Interval` they're identical). On resume the
coordinates are replayed through the `Experiment` so dynamic domains recover
their exact per-trial ranges. `plot_tuning.py` uses `reconstruct_values` to turn
those coordinates back into human-meaningful values (e.g. the actual number on
the far side of a `Union` gap).

---

## Reference

### `remote_tuner` CLI

| Argument                   | Default                        | Meaning                                             |
|----------------------------|--------------------------------|-----------------------------------------------------|
| `recipe` (positional)      | —                              | Path to the EinsteinEngine recipe.                      |
| `tuner` (positional)       | —                              | Path to the tuner `.py`.                            |
| `--local-path`             | —                              | Local dir the recipe generates into.               |
| `--remote-host`            | —                              | SSH host, or `localhost` to run locally.            |
| `--remote-path`            | —                              | Destination dir for the generated code.            |
| `--remote-cactus-path`     | —                              | Cactus install dir the commands run under.         |
| `--remote-command`         | `./build.sh && ./run-all.sh`   | Build + submit; must print the Slurm job id.        |
| `--remote-timing-command`  | `./timings.sh`                 | Prints a single number to optimize for.             |
| `--checkpoint-file`        | `split_tuning_checkpt.jsonl`   | Trial log; resumed automatically. A probing tuner also writes `<file>.probe.json`. |
| `--warmup-iterations`      | `10`                           | Exploration trials.                                 |
| `--iterations`             | `20`                           | Guided trials after warmup.                         |

### `Experiment` API (`EinsteinEngine.tuning.experiment`)

- `add_in_param(name, domain, condition=Always, constraint=Unconstrained)`
  where `domain` is a `(lo, hi)` tuple, a `Domain` (`Interval` / `Discrete` /
  `Union`), or a `realized -> domain` resolver.
- `add_out_param(name, mapping)` — `mapping: dict -> value | None`; `None` omits
  the arg, so the recipe must read it with `get_optional_tuning_param`.
- `add_distinct_sorted(prefix, k, bounds)` → list of names (increasing ints).
- `add_distinct_choice(prefix, k, pool)` → list of names (distinct pool picks).

### `Tuner` API (`EinsteinEngine.tuning.tuning`)

- `get_experiment() -> Experiment` (abstract).
- `get_probed_experiment(probe: ProbeResult) -> Experiment` — what
  `build_experiment` calls, with the counts for `probe_targets()` (empty when
  there are none). The default calls `get_experiment()`.
- `probe_targets() -> Collection[str]` — out_params to probe (default none).
- `probe_params() -> dict[str, Any]` — fixed out_params provided while probing
  and recorded in the probe file (default none). Each of these names, and each
  probe target, must also be an out_param of the probed `Experiment`.
- `build_experiment(tuner, recipe, checkpoint_file, *, record_probe=False)` —
  probes and handles `<checkpoint>.probe.json` as described in section 5;
  `remote_tuner` passes `record_probe=True`.
- `get_tuning_param(name, default)` / `get_optional_tuning_param(name, default)`.
