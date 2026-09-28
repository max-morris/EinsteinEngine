#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools
import json
import os
import traceback
import typing
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from EinsteinEngine.common.util import pprint
from EinsteinEngine.tuning.checkpointed_optimizer import CheckpointedOptimizer, read_checkpoint
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.tuning.probe import EMPTY_PROBE, ProbeResult, active_params, describe_param, note_read, \
    probe_recipe
from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs, do_remote_run

_tuning_params: dict[str, Any] | None = None

def get_tuning_param[T](param_name: str, default: T) -> T:
    """The value of a tuner out-param, or ``default`` when not tuning.

    While tuning, a name the tuner does not provide is an error (it usually means a typo, or a tuner out-param that
    returned None for this trial); use ``get_optional_tuning_param`` for knobs a tuner may leave out. While probing
    (see ``probe.probe_recipe``), names the probe does not provide return ``default``.
    """
    if (params := active_params()) is not None:
        note_read(param_name)
        return typing.cast(T, params.get(param_name, default))
    if _tuning_params is None:
        return default
    if param_name not in _tuning_params:
        raise RuntimeError(f"Tuning parameter {param_name} not found.")
    return typing.cast(T, _tuning_params[param_name])

def get_optional_tuning_param[T](param_name: str, default: T) -> T:
    """Like ``get_tuning_param``, but returns ``default`` whenever the tuner does not provide ``param_name``."""
    if (params := active_params()) is not None:
        note_read(param_name)
    else:
        params = _tuning_params
    return default if params is None else typing.cast(T, params.get(param_name, default))

class Tuner(ABC):
    def probe_targets(self) -> Collection[str]:
        """Out-param names of split predicates whose number of positions the Experiment depends on.

        When non-empty, ``build_experiment`` probes the recipe for them and passes the counts to
        ``get_probed_experiment``.
        """
        return ()

    def probe_params(self) -> dict[str, Any]:
        """Fixed recipe-facing out-params that change the probed counts (e.g. the split locus), provided while probing.

        They are also recorded, by description, in the probe file next to the checkpoint (see ``build_experiment``).
        Each of these names, and each of ``probe_targets()``, must also be an out-param of the Experiment (with the
        same fixed value, for these), so that the trials see what the probe counted; ``build_experiment`` checks the
        names.
        """
        return {}

    @abstractmethod
    def get_experiment(self) -> Experiment:
        """Build the Experiment."""
        ...

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        """Build the Experiment, given the counts for ``probe_targets()`` (empty when there are none).

        This is what ``build_experiment`` calls. The default ignores the probe and calls ``get_experiment()``; a tuner
        whose search space depends on the probe overrides it.
        """
        return self.get_experiment()


def probe_sidecar_path(checkpoint_file: str) -> str:
    """The file next to a checkpoint that records the probe its coordinates were recorded against."""
    return f'{checkpoint_file}.probe.json'


@dataclass(frozen=True)
class ProbeRecord:
    """The contents of a probe sidecar (``<checkpoint>.probe.json``): what a checkpoint's coordinates were recorded
    against.

    - ``counts``: the number of split positions of each probed predicate (``"counts"`` in the file).
    - ``probe_params``: a description (``probe.describe_param``) of each of the tuner's ``probe_params()``, i.e. the
      locus and the ordering functions (``"probe_params"``).
    - ``derived_orders``: for each function whose early ordering function is ``rank_by_post_population(fn)``, the
      0-based add_eqn order the engine derived for it from its trial bake while probing (``"derived_orders"``, written
      only when there is one). The description of ``rank_by_post_population(fn)`` in ``probe_params`` names only the
      rule; the order it yields also depends on the recipe (its equations, the other functions, the bake options), so
      it is recorded separately. A split coordinate means "cut after the i-th group in this order", so the same
      coordinates under a different order are different cuts.
    """
    counts: ProbeResult
    probe_params: Mapping[str, str]
    derived_orders: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def to_json(self) -> str:
        obj: dict[str, Any] = {'counts': dict(self.counts), 'probe_params': dict(self.probe_params)}
        if self.derived_orders:
            obj['derived_orders'] = {name: list(order) for name, order in sorted(self.derived_orders.items())}
        return json.dumps(obj, indent=2) + '\n'

    @staticmethod
    def from_json(text: str) -> 'ProbeRecord':
        obj = json.loads(text)
        if not isinstance(obj, dict) or not isinstance(obj.get('probe_params'), dict) or \
                not all(isinstance(v, str) for v in obj['probe_params'].values()):
            raise ValueError(f'Expected a JSON object of the form {{"counts": {{...}}, "probe_params": {{...}}}}, '
                             f'got {text!r}')
        derived = obj.get('derived_orders', {})
        if not isinstance(derived, dict) or not all(
                isinstance(order, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in order)
                for order in derived.values()):
            raise ValueError(f'Expected "derived_orders" to map function names to lists of add_eqn indices, got '
                             f'{derived!r}')
        return ProbeRecord(ProbeResult.from_counts(obj.get('counts')), dict(obj['probe_params']),
                           {name: tuple(order) for name, order in derived.items()})

    def differences(self, current: 'ProbeRecord') -> list[str]:
        """How ``current`` differs from this (recorded) record, one phrase per difference."""
        changes: list[str] = []
        derived_was = {name: list(order) for name, order in self.derived_orders.items()}
        derived_now = {name: list(order) for name, order in current.derived_orders.items()}
        for kind, was, now in (('the number of split positions for', self.counts, current.counts),
                               ('the probe param', self.probe_params, current.probe_params),
                               ('the add_eqn order rank_by_post_population derived for function',
                                derived_was, derived_now)):
            for name in sorted(set(was) | set(now)):
                if was.get(name) != now.get(name):
                    changes.append(f"{kind} {name!r} was {_or_absent(was.get(name))} and is now "
                                   f"{_or_absent(now.get(name))}")
        return changes


def _or_absent(value: object) -> str:
    return 'absent' if value is None else repr(value)


def _count_checkpoint_entries(checkpoint_file: str) -> int:
    if not os.path.exists(checkpoint_file):
        return 0
    with open(checkpoint_file) as fh:
        return sum(1 for line in fh if line.strip())


def build_experiment(tuner: Tuner, recipe: str | None, checkpoint_file: str, *,
                     record_probe: bool = False) -> Experiment:
    """Build ``tuner``'s Experiment, probing ``recipe`` if the tuner has probe targets.

    A checkpoint's coordinates are only meaningful against the probe they were recorded with, so the probe is recorded
    in a sidecar next to the checkpoint (see ``probe_sidecar_path`` and ``ProbeRecord``), together with a description
    of the tuner's ``probe_params()`` (the locus and ordering functions) and any add_eqn order the engine derived with
    ``rank_by_post_population`` while probing.

    - With ``recipe``, the recipe is always probed. Without it, the sidecar's counts and derived orders are used (so
      it must exist).
    - Every probe target and ``probe_params()`` name must be an out-param of the Experiment.
    - If a sidecar exists and does not match (different counts, probe params that describe differently, see
      ``probe.describe_param`` for what a description covers, or a different derived order), this raises, naming
      what changed, unless the checkpoint has no entries yet.
    - ``record_probe=True`` (remote_tuner) writes the sidecar when it is missing or (for an empty checkpoint) stale,
      but only after checking that every checkpoint entry loads against the new Experiment.
    """
    targets = list(tuner.probe_targets())
    if not targets:
        return tuner.get_probed_experiment(EMPTY_PROBE)

    sidecar = probe_sidecar_path(checkpoint_file)
    recorded: ProbeRecord | None = None
    if os.path.exists(sidecar):
        with open(sidecar) as fh:
            try:
                recorded = ProbeRecord.from_json(fh.read())
            except ValueError as e:
                raise RuntimeError(f"The probe file {sidecar} (next to checkpoint {checkpoint_file}) is not valid: "
                                   f"{e}") from e

    params = tuner.probe_params()
    if recipe is not None:
        pprint(f"Probing {recipe} for {targets}...")
        counts = probe_recipe(recipe, targets, params)
        derived_orders = counts.derived_orders
        pprint(f"Probe result: {dict(counts)}" +
               (f", derived add_eqn orders {({name: list(order) for name, order in derived_orders.items()})}"
                if derived_orders else ""))
    elif recorded is not None:
        counts = recorded.counts
        derived_orders = recorded.derived_orders
        if set(counts) != set(targets):
            raise RuntimeError(
                f"The probe file {sidecar} records {sorted(counts)}, but the tuner probes {sorted(targets)}.")
    else:
        raise RuntimeError(
            f"The tuner probes the recipe for {targets}, but there is no probe file {sidecar} and no recipe was "
            f"given to probe.")
    current = ProbeRecord(counts, {name: describe_param(value) for name, value in sorted(params.items())},
                          dict(derived_orders))

    if recorded is not None and (changes := recorded.differences(current)):
        message = (f"The probe recorded for checkpoint {checkpoint_file} (in {sidecar}) does not match the recipe and "
                   f"tuner: {'; '.join(changes)}.")
        if _count_checkpoint_entries(checkpoint_file) > 0:
            raise RuntimeError(f"{message} The checkpoint's coordinates would be misinterpreted. Use a new checkpoint "
                               f"file, or restore the recipe and tuner.")
        pprint(f"{message} The checkpoint has no entries, so this is not an error.")

    experiment = tuner.get_probed_experiment(counts)
    if undeclared := [name for name in [*targets, *params] if name not in experiment.out_params]:
        raise RuntimeError(
            f"{type(tuner).__name__} probes or provides {undeclared} while probing (probe_targets / probe_params), but "
            f"its Experiment has no out-param of that name, so the trials would not provide what the probe counted "
            f"with. Add an out-param for each (a probe_params value with a mapping that returns it unchanged).")
    if record_probe and recorded != current:
        read_checkpoint(experiment, checkpoint_file)  # raises if the checkpoint does not fit the experiment
        with open(sidecar, 'w') as fh:
            fh.write(current.to_json())
    return experiment


def do_tuning[T: Tuner](args: RemoteFeedbackArgs, tuner: T, checkpoint_file: str, warmup_iterations: int = 10, iterations: int = 20, telegram_verbosity: int = 1) -> None:
    optimizer = CheckpointedOptimizer(
        f=functools.partial(do_tuning_run, args=args),
        experiment=build_experiment(tuner, args.recipe, checkpoint_file, record_probe=True),
        checkpoint_file=checkpoint_file,
        telegram_verbosity=telegram_verbosity,
    )

    if optimizer.n_checkpoint_loaded:
        pprint(f'Resumed from checkpoint: {optimizer.n_checkpoint_loaded} observations loaded from {checkpoint_file}')

    optimizer.maximize(warmup_iterations=warmup_iterations, iterations=iterations)
    assert optimizer.max is not None
    pprint(f'Tuning complete. Best target: {optimizer.max}')


def do_tuning_run(args: RemoteFeedbackArgs, **recipe_facing_args: dict[str, Any]) -> float:
    global _tuning_params
    _tuning_params = recipe_facing_args
    try:
        timing_value = do_remote_run(args, {})
    except RuntimeError as e:
        traceback.print_exception(e)
        # Probably created a bad split (empty loop, loop with no outputs) so discard this result
        return -float('inf')
    finally:
        _tuning_params = None

    return -timing_value  # Optimizer tries to maximize; lower time is better
