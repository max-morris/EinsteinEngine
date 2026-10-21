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

import dataclasses
import functools
import json
import os
import traceback
import typing
import uuid
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from EinsteinEngine.common.util import pprint
from EinsteinEngine.tuning.experiment import Experiment
from EinsteinEngine.intermediate.eqn_ordering import add_eqn_order
from EinsteinEngine.tuning.probe import EMPTY_PROBE, ProbeResult, RankByPostPopulation, active_params, \
    derive_add_eqn_order, describe_param, note_read, probe_recipe
from EinsteinEngine.tuning.remote_feedback import RemoteFeedbackArgs, TrialFailed, do_remote_run

_tuning_params: dict[str, Any] | None = None

def get_tuning_param[T](param_name: str, default: T) -> T:
    """
    The value of a tuner out-param, or `default` when not tuning. While tuning, a missing name is an error (a typo, or
    an out-param that returned None for this trial); use `get_optional_tuning_param` for knobs a tuner may leave out.
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
    """Like `get_tuning_param`, but returns `default` whenever the tuner does not provide `param_name`."""
    if (params := active_params()) is not None:
        note_read(param_name)
    else:
        params = _tuning_params
    return default if params is None else typing.cast(T, params.get(param_name, default))

class Tuner(ABC):
    def probe_targets(self) -> Collection[str]:
        """Out-param names of split predicates whose number of positions the Experiment depends on."""
        return ()

    def probe_params(self) -> dict[str, Any]:
        """
        Fixed recipe-facing out-params that change the probed counts (e.g. the split locus), provided while probing.
        Each of these and of `probe_targets()` must also be an out-param of the Experiment, with the same fixed value,
        so that the trials see what the probe counted.
        """
        return {}

    @abstractmethod
    def get_experiment(self) -> Experiment:
        ...

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        """Build the Experiment, given the counts for `probe_targets()`; override it if the search space depends on
        them."""
        return self.get_experiment()

    def describe_space(self) -> Mapping[str, str]:
        """
        What a checkpoint's coordinates mean, beyond their names and the probe: for example the choices an index
        param picks from, in order. Keyed by in-param name (or a name pattern), with stable descriptions
        (`describe_param`). It is recorded in the probe sidecar, and a checkpoint with entries refuses a tuner whose
        description differs. Default: nothing.
        """
        return {}


def probe_sidecar_path(checkpoint_file: str) -> str:
    """The file next to a checkpoint that records the probe its coordinates were recorded against."""
    return f'{checkpoint_file}.probe.json'


@dataclass(frozen=True)
class ProbeRecord:
    """
    The contents of a probe sidecar: what a checkpoint's coordinates were recorded against. `probe_params` describes
    each of the tuner's `probe_params()`; `derived_orders` records the add_eqn order each `rank_by_post_population(fn)`
    param was resolved to, since a split coordinate means "cut after the i-th group in this order".
    """
    counts: ProbeResult
    probe_params: Mapping[str, str]
    derived_orders: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    #  The tuner's describe_space(); None in a sidecar written before it was recorded (then it is not checked).
    space: Mapping[str, str] | None = None
    #  A random id given to the study when its sidecar is first written, and kept when it is rewritten, so that two
    #  studies with the same tuner and recipe (e.g. one per compiler) are told apart. None in an older sidecar.
    study: str | None = None
    #  The arguments the recipe was probed with (its sys.argv[1:]); None in an older sidecar (taken as none).
    recipe_args: tuple[str, ...] | None = None
    #  The checkpoint the study was created for (absolute path when the sidecar was first written), to warn when a
    #  sidecar is copied into another study; None in an older sidecar.
    origin: str | None = None

    def to_json(self) -> str:
        obj: dict[str, Any] = {'counts': dict(self.counts), 'probe_params': dict(self.probe_params)}
        if self.derived_orders:
            obj['derived_orders'] = {name: list(order) for name, order in sorted(self.derived_orders.items())}
        if self.space is not None:
            obj['space'] = dict(sorted(self.space.items()))
        if self.study is not None:
            obj['study'] = self.study
        if self.recipe_args is not None:
            obj['recipe_args'] = list(self.recipe_args)
        if self.origin is not None:
            obj['origin'] = self.origin
        return json.dumps(obj, indent=2) + '\n'

    def identity(self) -> dict[str, Any]:
        """What identifies the study this record belongs to, in a canonical form: the study id, the probe counts, the
        probe params, the derived orders, the search space and the recipe arguments (not the file's formatting, nor
        where the study was created)."""
        return {'study': self.study, 'counts': dict(sorted(self.counts.items())),
                'probe_params': dict(sorted(self.probe_params.items())),
                'derived_orders': {name: list(order) for name, order in sorted(self.derived_orders.items())},
                'space': None if self.space is None else dict(sorted(self.space.items())),
                'recipe_args': list(self.recipe_args or ())}

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
            raise ValueError(f'Expected "derived_orders" to map probe params to lists of add_eqn indices, got '
                             f'{derived!r}')
        space = obj.get('space')
        if space is not None and (not isinstance(space, dict) or not all(isinstance(v, str) for v in space.values())):
            raise ValueError(f'Expected "space" to map param names to descriptions, got {space!r}')
        study, origin = obj.get('study'), obj.get('origin')
        if any(x is not None and not isinstance(x, str) for x in (study, origin)):
            raise ValueError(f'Expected "study" and "origin" to be strings, got {study!r} and {origin!r}')
        recipe_args = obj.get('recipe_args')
        if recipe_args is not None and (not isinstance(recipe_args, list)
                                        or not all(isinstance(a, str) for a in recipe_args)):
            raise ValueError(f'Expected "recipe_args" to be a list of strings, got {recipe_args!r}')
        return ProbeRecord(ProbeResult.from_counts(obj.get('counts')), dict(obj['probe_params']),
                           {name: tuple(order) for name, order in derived.items()}, space, study,
                           None if recipe_args is None else tuple(recipe_args), origin)

    def differences(self, current: 'ProbeRecord') -> list[str]:
        """How `current` differs from this (recorded) record, one phrase per difference."""
        changes: list[str] = []
        derived_was = {name: list(order) for name, order in self.derived_orders.items()}
        derived_now = {name: list(order) for name, order in current.derived_orders.items()}
        spaces = [] if self.space is None or current.space is None else [
            ('the search space of', self.space, current.space)]
        for kind, was, now in (('the number of split positions for', self.counts, current.counts),
                               ('the probe param', self.probe_params, current.probe_params),
                               ('the add_eqn order derived for',
                                derived_was, derived_now), *spaces):
            for name in sorted(set(was) | set(now)):
                if was.get(name) != now.get(name):
                    changes.append(f"{kind} {name!r} was {_or_absent(was.get(name))} and is now "
                                   f"{_or_absent(now.get(name))}")
        if list(self.recipe_args or ()) != list(current.recipe_args or ()):
            changes.append(f"the recipe arguments were {list(self.recipe_args or ())} and are now "
                           f"{list(current.recipe_args or ())}")
        return changes


def _or_absent(value: object) -> str:
    return 'absent' if value is None else repr(value)


#  The out-param a rank_by_post_population probe param must be (after a split tuner's prefix).
EARLY_ORDERING_FN = 'early_ordering_fn'


def early_ordering_prefix(name: str) -> str | None:
    """The split tuner prefix of an early ordering function out-param (`early_ordering_fn` or
    `<prefix>_early_ordering_fn`), or None if `name` is not one."""
    if name == EARLY_ORDERING_FN:
        return ''
    if name.endswith('_' + EARLY_ORDERING_FN):
        return name.removesuffix(EARLY_ORDERING_FN)
    return None


def _count_checkpoint_entries(checkpoint_file: str) -> int:
    if not os.path.exists(checkpoint_file):
        return 0
    with open(checkpoint_file) as fh:
        return sum(1 for line in fh if line.strip())


def build_experiment(tuner: Tuner, recipe: str | None, checkpoint_file: str, *,
                     record_probe: bool = False, recipe_args: Sequence[str] = ()) -> Experiment:
    """
    Build `tuner`'s Experiment, probing `recipe` if the tuner has probe targets. A checkpoint's coordinates are only
    meaningful against the probe they were recorded with, and the search space the tuner describes
    (`Tuner.describe_space`), so both are recorded in a sidecar next to the checkpoint. A
    `rank_by_post_population(fn)` probe param is resolved to `add_eqn_order(<order>)`, derived from the recipe unless
    the sidecar records one for that param.

    - Without `recipe`, the sidecar's counts and derived orders are used, so it must exist if the tuner probes.
    - If the sidecar does not match, this raises unless the checkpoint has no entries yet.
    - `record_probe=True` writes the sidecar when it is missing or stale, after checking that every checkpoint entry
      loads against the new Experiment. It always writes one: a study's id (batches need it) is made then.
    - `recipe_args` are the recipe's own arguments (its sys.argv[1:]), which the probe runs it with and the sidecar
      records, since they can change the equations. Without `recipe`, they must be the recorded ones.

    The Experiment carries the tuner's search space (`Experiment.described_space`), for checks against the sidecar.
    """
    targets = list(tuner.probe_targets())
    params = tuner.probe_params()
    space = dict(tuner.describe_space())
    ranked = {name: value for name, value in params.items() if isinstance(value, RankByPostPopulation)}
    args = tuple(recipe_args)
    if not targets and not ranked and not space and not record_probe:
        experiment = tuner.get_probed_experiment(EMPTY_PROBE)
        experiment.described_space = space
        return experiment
    if bad := sorted(name for name in ranked if early_ordering_prefix(name) is None):
        raise RuntimeError(f"rank_by_post_population can only be the early_ordering_fn (with a split tuner's prefix), "
                           f"not {bad}.")

    sidecar = probe_sidecar_path(checkpoint_file)
    recorded: ProbeRecord | None = None
    if os.path.exists(sidecar):
        with open(sidecar) as fh:
            try:
                recorded = ProbeRecord.from_json(fh.read())
            except ValueError as e:
                raise RuntimeError(f"The probe file {sidecar} (next to checkpoint {checkpoint_file}) is not valid: "
                                   f"{e}") from e

    descriptions = {name: describe_param(value) for name, value in sorted(params.items())}
    derived_orders: Mapping[str, tuple[int, ...]]
    if not targets and not ranked:
        counts, derived_orders = EMPTY_PROBE, {}
    elif recipe is not None:
        orders: dict[str, tuple[int, ...]] = dict()
        for name, rule in ranked.items():
            if recorded is not None and name in recorded.derived_orders and \
                    recorded.probe_params.get(name) == descriptions[name]:
                orders[name] = recorded.derived_orders[name]
                pprint(f"Using the add_eqn order {list(orders[name])} recorded in {sidecar} for {rule!r}.")
            else:
                pprint(f"Deriving the add_eqn order for {rule!r} from {recipe}...")
                others = {k: v for k, v in params.items() if k not in ranked}
                prefix = early_ordering_prefix(name)
                assert prefix is not None
                function_name, orders[name] = derive_add_eqn_order(recipe, rule.ordering_fn, others, prefix=prefix,
                                                                   argv=args)
                pprint(f"{function_name}: {rule!r} derived the add_eqn order {list(orders[name])}.")
        derived_orders = orders
        resolved = {**params, **{name: add_eqn_order(order) for name, order in derived_orders.items()}}
        if targets:
            pprint(f"Probing {recipe} for {targets}...")
            counts = probe_recipe(recipe, targets, resolved, argv=args)
            pprint(f"Probe result: {dict(counts)}")
        else:
            counts = EMPTY_PROBE
    elif recorded is not None:
        counts = recorded.counts
        derived_orders = recorded.derived_orders
        if set(counts) != set(targets):
            raise RuntimeError(
                f"The probe file {sidecar} records {sorted(counts)}, but the tuner probes {sorted(targets)}.")
        if missing := sorted(set(ranked) - set(derived_orders)):
            raise RuntimeError(f"The probe file {sidecar} records no derived add_eqn order for {missing}.")
    else:
        raise RuntimeError(
            f"The tuner probes the recipe for {targets or sorted(ranked)}, but there is no probe file {sidecar} and "
            f"no recipe was given to probe.")
    if recipe is None and recorded is not None and list(recorded.recipe_args or ()) != list(args):
        raise RuntimeError(f"The probe file {sidecar} was recorded with the recipe arguments "
                           f"{list(recorded.recipe_args or ())}, not {list(args)}, and there is no recipe to probe "
                           f"again.")
    #  The study id and origin are kept from the recorded sidecar, or made when the sidecar is first written.
    current = ProbeRecord(counts, descriptions, dict(derived_orders), space,
                          recorded.study if recorded is not None and recorded.study is not None else uuid.uuid4().hex,
                          args, recorded.origin if recorded is not None and recorded.origin is not None
                          else os.path.abspath(checkpoint_file))

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
            f"{type(tuner).__name__} probes or provides {undeclared} while probing, but its Experiment has no "
            f"out-param of that name. Add an out-param for each (a probe_params value with a mapping that returns it "
            f"unchanged).")
    for name in ranked:
        experiment.add_out_param(name, lambda _params, fn=add_eqn_order(derived_orders[name]): fn)
    experiment.described_space = space
    #  A study moved or mirrored to another tree keeps its directory and file names; warn when those differ.
    if record_probe and recorded is not None and recorded.origin is not None and \
            _last_parts(recorded.origin) != _last_parts(os.path.abspath(checkpoint_file)):
        pprint(f"Warning: the probe file {sidecar} was created for the study at {recorded.origin}. If this is another "
               f"study (not the same one moved or mirrored), it must not share that study's id: use "
               f"tuning.fork_probe_file.")
    if record_probe and recorded != current:
        from EinsteinEngine.tuning.checkpointed_optimizer import checkpoint_lock, read_checkpoint  # see do_tuning
        read_checkpoint(experiment, checkpoint_file)  # raises if the checkpoint does not fit the experiment
        if recorded is not None and _count_checkpoint_entries(checkpoint_file) > 0:
            if recorded.space is None:
                pprint(f"Warning: recording the search space in {sidecar}, whose checkpoint already has entries: "
                       f"they are taken to have been recorded against this tuner's current space.")
            if recorded.study is None:
                pprint(f"Warning: giving the study of {sidecar} an id; batches asked before this cannot be told.")
        with checkpoint_lock(checkpoint_file):
            if recorded is None and os.path.exists(sidecar):  # written meanwhile by another process: keep its id
                with open(sidecar) as fh:
                    meanwhile = ProbeRecord.from_json(fh.read())
                current = dataclasses.replace(current, study=meanwhile.study or current.study,
                                              origin=meanwhile.origin or current.origin)
            with open(sidecar, 'w') as fh:
                fh.write(current.to_json())
    return experiment


def _last_parts(path: str) -> tuple[str, str]:
    return os.path.basename(os.path.dirname(path)), os.path.basename(path)


def fork_probe_file(from_checkpoint: str, to_checkpoint: str) -> None:
    """
    Start a new study at `to_checkpoint` from the probe file of the study at `from_checkpoint` (to reuse its probe and
    derived orders, which can be slow to derive), with a new study id. Refuses if the new study already has a probe
    file or entries.
    """
    src, dst = probe_sidecar_path(from_checkpoint), probe_sidecar_path(to_checkpoint)
    if os.path.exists(dst) or _count_checkpoint_entries(to_checkpoint) > 0:
        raise RuntimeError(f"The study at {to_checkpoint} already has a probe file or entries.")
    with open(src) as fh:
        record = ProbeRecord.from_json(fh.read())
    with open(dst, 'w') as fh:
        fh.write(dataclasses.replace(record, study=uuid.uuid4().hex, origin=os.path.abspath(to_checkpoint)).to_json())


def do_tuning[T: Tuner](args: RemoteFeedbackArgs, tuner: T, checkpoint_file: str, warmup_iterations: int = 10, iterations: int = 20, telegram_verbosity: int = 1) -> None:
    # Imported here: recipes import this module (for get_tuning_param) through `import EinsteinEngine`, which should not
    # load optuna.
    from EinsteinEngine.tuning.checkpointed_optimizer import CheckpointedOptimizer
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
    except TrialFailed as e:
        traceback.print_exception(e)
        # Probably created a bad split (empty loop, loop with no outputs) so discard this result
        return -float('inf')
    finally:
        _tuning_params = None

    return -timing_value  # Optimizer tries to maximize; lower time is better
