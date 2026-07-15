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

The optimizer *maximizes*, and the objective is `-rhs_time`, so lower RHS time
wins.

A complete, working example lives in `tuning/z4c_splitting/` (recipe
`recipes/Cottonmouth/Z4c.py`, tuner `tuning/z4c_splitting/tuner.py`, plus the
`run-tuning.sh` / `generate-best.sh` wrappers). This guide walks through the
smallest possible version of that, then covers the richer domain features.

---

## 1. Make the recipe read tunable knobs

A recipe exposes a knob by calling `get_tuning_param(name, default)`. Outside of
a tuning run the default is used, so the recipe still runs normally on its own.

```python
from EinsteinEngine import get_tuning_param

fun = mod.create_function(
    "my_rhs",
    rhs_group,
    # These come from the tuner during a search; None (the default) otherwise.
    auto_hard_split_predicate=get_tuning_param('auto_hard_split_predicate', None),
    auto_soft_split_predicate=get_tuning_param('auto_soft_split_predicate', None),
)
```

The `name` here must match an **out_param** name declared in the tuner (below).
See `recipes/Cottonmouth/Z4c.py:842` for the real usage.

---

## 2. Write a tuner file

The tuner file is executed by `remote_tuner` (via `runpy`, *not* as `__main__`).
It must expose a `Tuner` instance in one of two ways:

- define `get_tuner() -> Tuner`, or
- assign a module-level variable named `tuner`.

A `Tuner` implements a single method, `get_experiment()`, which builds an
`Experiment`. The `Experiment` has two halves:

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
- An out_param mapping that returns `None` is **omitted** for that trial, so the
  recipe falls back to its `get_tuning_param` default. This is how you make a
  knob conditionally present.
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
`tuning/z4c_splitting/` and edit the paths; `"$@"` at the end lets you pass extra
flags through (e.g. `./run-tuning.sh --iterations 50`).

What each trial does, in order (see `remote_feedback.py:do_remote_run`):

1. Regenerate the recipe with the trial's parameters into `--local-path`.
2. `rsync --delete` `--local-path` → `--remote-path`.
3. Run `--remote-command` under `--remote-cactus-path`; parse the Slurm job id
   from a line matching `Submit finished, job id is <N>`.
4. Poll `squeue` every 60 s until the job leaves the queue.
5. Run `--remote-timing-command`; the **second** numeric row's total-time column
   is taken as the RHS time (the `run-all.sh`/`timings.sh` in the sample produce
   exactly this layout).
6. Record `{"target": -rhs_time, "params": {...}}` to the checkpoint.

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

To eyeball progress, `plot_tuning.py` renders the checkpoint's target history.

---

## 5. Advanced domains

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
only for variables it decided to soft-split (`tune_splitting.py:55`).

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
| `--remote-timing-command`  | `./timings.sh`                 | Prints the timing table.                            |
| `--checkpoint-file`        | `split_tuning_checkpt.jsonl`   | Trial log; resumed automatically.                   |
| `--warmup-iterations`      | `10`                           | Exploration trials.                                 |
| `--iterations`             | `20`                           | Guided trials after warmup.                         |

### `Experiment` API (`EinsteinEngine.tuning.experiment`)

- `add_in_param(name, domain, condition=Always, constraint=Unconstrained)`
  where `domain` is a `(lo, hi)` tuple, a `Domain` (`Interval` / `Discrete` /
  `Union`), or a `realized -> domain` resolver.
- `add_out_param(name, mapping)` — `mapping: dict -> value | None`; `None` omits
  the arg so the recipe uses its `get_tuning_param` default.
- `add_distinct_sorted(prefix, k, bounds)` → list of names (increasing ints).
- `add_distinct_choice(prefix, k, pool)` → list of names (distinct pool picks).
