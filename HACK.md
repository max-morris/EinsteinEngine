# hack/z4c-zach-mixed-precision

The branch the mixed-precision GPU benchmarks are generated from. It is
the union of two branches that are kept separate for review:

- `z4c/zach-trick`, the head of pull request #100: the metric-offset trick
  in the Z4c recipe (`gt_m1`, the evolved `gt - delta`, the f64
  determinant enforcement, `as_f16/as_f32/as_f64` numeric conversions in
  the DSL behind the `enable_cctk_real2` generator option). It improves
  the roundoff characteristics of the f32 and f16 runs, which is why the
  benchmarks want it, but it has no dedicated test yet and the existing
  regression tests fail against it until their reference output is
  regenerated.
- `feature/z4c-mixed-precision`: the tuning framework (`feature/run-feedback`
  merged), the `precision_policy` tuning hook with `kernel_manifest.json`,
  the `--instrument-ranges` f64 range dump for CPU builds, and the
  `--remote-result-file` copy-back in `remote_feedback`. This is the
  reviewable, test-covered side.

Neither parent is complete for the benchmark on its own, and the trick is
not ready to merge into the feature branch, so this branch carries both.
It is a hack branch: nothing lands here first, and it is never the base of
a pull request. The only commits of its own are merges and this file.

## Updating

Both parents move. Bring the branch up to date by merging them, never by
committing directly:

```
git checkout hack/z4c-zach-mixed-precision
git merge origin/z4c/zach-trick
git merge origin/feature/z4c-mixed-precision
```

The conflicts the first merge produced, and how they were resolved, so a
later one can follow the same rule:

- `EinsteinEngine/__init__.py`: union of both sides' imports and `__all__`
  (`get_tuning_param`, `cartesian_product` from the feature branch;
  `as_f16/as_f32/as_f64`, `finite_difference_stencil` and the indices
  import from the PR).
- `EinsteinEngine/generators/cpp_carpetx_generator.py`: both generator
  options, `enable_cctk_real2` and `instrument_ranges`.
- `requirements.txt`, `setup.py`: the feature branch's pins (scipy).
- `dsl_frontend.py`, `unit_tests/test_use_indices.py`,
  `recipes/osdiv/osdiv.py`: the PR side, which carries the
  `indexes`-to-`indices` rename from master.

## Generating from it

The venv installs EinsteinEngine in editable mode, pointing at the
`~/src/EmitCactus` checkout. A recipe run from a worktree or another
checkout imports the engine of whatever branch `~/src/EmitCactus` has
checked out, silently. Either check this branch out there before
generating, or force the import path:

```
cd <emission root>
PYTHONPATH=<checkout of this branch> \
  <checkout>/venv/bin/python <checkout>/recipes/Cottonmouth/Z4c.py [--instrument-ranges] [--precision-policy p.json]
```

The thorn is written to `<emission root>/Cottonmouth/CottonmouthZ4c4m`
with `kernel_manifest.json` next to `src/`. `--instrument-ranges` is a
CPU-build option (the range tables live in loop bodies that the device
path cannot reduce); `--precision-policy` is consumed and hashed only, the
width rules that read it come later. Under the tuning driver both come
from the tuning parameters instead (`precision_policy`,
`instrument_ranges`); the command-line flags never reach the recipe there.

## Where it is used

- The GPU kit (`~/gpu-test-kit`, copied to qbd) symlinks
  `arrangements/Cottonmouth/CottonmouthZ4c4m` to the rsynced emission of
  this branch for the `mp-z4c` harness config (`kit.sh harness-prep`).
- `kit.sh trial` trials and the stage-A sweep generate from this branch on
  the workstation and stamp the thornlist with the policy hash.
- The CPU range dump of plan step 0 is this branch with
  `--instrument-ranges`, built in a CPU config.

## Status at the first merge (2026-09-28)

- `unit_tests`: 11 passed; mypy clean on the engine and the recipe.
- Generation is deterministic (two runs are byte-identical). The default
  emission is not byte-identical to PR 100's, for two reasons that both
  come from the feature side and merged without a conflict:
  1. The RHS split. PR 100's recipe calls `fun_z4c_rhs.soft_split()`; the
     feature branch has that line commented out and leaves splitting to
     the tuner's `auto_hard_split_predicate` / `auto_soft_split_predicate`
     tuning parameters, which are `None` outside the tuner. So this branch
     emits the RHS as 2 loops where PR 100 emits 6, with the `AtTF` and
     `Rchi` intermediates kept as temporaries instead of stored grid
     functions (129 stores over the thorn against 147). This is a
     benchmark-relevant choice: the plan's stage-A base is "splits at the
     current Z4c setting", and on this branch that setting is "no split
     unless the tuner says so". Re-enable the line if the manual split is
     the intended baseline.
  2. Temporary order and numbering differ in the other affected files
     from the feature branch's equation-ordering changes; the CCL files,
     the stored variables and the expressions are the same.
  With `--instrument-ranges`, eight source files carry range probes (1129
  in the RHS) and the schedule gains the dump functions.
- Cottonmouth regression tests: failing on the trick side until regenerated.
- The Zach trick itself: no dedicated test; validated only through the QC0
  harness comparison against the f64 control.
