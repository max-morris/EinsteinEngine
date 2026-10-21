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

"""
Tuners built from other tuners.

`CompositeTuner` searches several tuners' spaces at once: for example one split tuner per function (each with its own
prefix), a CseOptimizationTuner and a few code-specific `DiscreteKnobTuner`s in one study. Its Experiment is the
union of theirs, so it works with `tuning.build_experiment`, the probe sidecar, `CheckpointedOptimizer`,
`generate_best` and `plot_tuning` like any other tuner. The sub-tuners' parameter names must be distinct.
"""

import os
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from EinsteinEngine.common.describe_param import describe_param
from EinsteinEngine.intermediate.split_locus import SplitLocus
from EinsteinEngine.tuning.experiment import Discrete, Experiment
from EinsteinEngine.tuning.probe import EMPTY_PROBE, ProbeResult
from EinsteinEngine.tuning.tuning import Tuner

#  Warnings are attributed to the first caller outside this package.
_TUNING_DIR = os.path.dirname(__file__) + os.sep


def _same_value(a: Any, b: Any) -> bool:
    """Whether two probe-param values are the same value: identical, or equal and of the same type (so 1, 1.0 and
    True differ). Equal descriptions are not enough, since `describe_param` does not describe a function's code."""
    if a is b:
        return True
    try:
        return type(a) is type(b) and bool(a == b)
    except Exception:  # noqa: BLE001 - a value whose == fails is not known to be the same
        return False


def _label(i: int, tuner: Tuner) -> str:
    prefix = getattr(tuner, 'prefix', '')
    return f"{i} ({type(tuner).__name__}{f', prefix {prefix!r}' if prefix else ''})"


class CompositeTuner(Tuner):
    """
    The product of several tuners' search spaces.

    - Probe targets: the union of the sub-tuners' `probe_targets()`. A target claimed by two sub-tuners is refused,
      since their Experiments would both declare it as an out-param; so is a target that another sub-tuner provides as
      a probe param.
    - Probe params: the union of the sub-tuners' `probe_params()`. Two sub-tuners may provide the same probe param
      only with the same value; different values are refused, since a probe sees one value per name. (The split
      tuners never share one: their names carry their prefixes. Sharing is for custom tuners.) Each sub-tuner also
      declares its probe params as out-params (see `Tuner.probe_params`), so the merged Experiment keeps one
      out-param for each shared probe param.
    - Experiment: each sub-tuner's Experiment, built from the same probe (each sees the counts of its own targets),
      merged in the order the sub-tuners were given. An in-param or out-param name declared by two sub-tuners is
      refused. A checkpoint stores the in-params by name, so the order of the sub-tuners does not change what it
      means; `describe_space` (recorded in the probe sidecar) is the union of the sub-tuners'.

    The probe counts split positions with every other out-param at the recipe's default. That is exact at the early
    locus, whose positions are the add_eqn calls; at a later locus, another sub-tuner's out-params (CSE options, for
    example) can change the number of positions, so a CompositeTuner warns when it combines one with other tuners.
    """

    def __init__(self, *tuners: Tuner) -> None:
        if not tuners:
            raise ValueError("CompositeTuner needs at least one tuner")
        self.tuners = list(tuners)
        targets: dict[str, int] = {}
        for i, tuner in enumerate(self.tuners):
            for target in tuner.probe_targets():
                if target in targets:
                    raise ValueError(f"CompositeTuner: probe target {target!r} is claimed by tuners "
                                     f"{_label(targets[target], self.tuners[targets[target]])} and "
                                     f"{_label(i, tuner)}; give each its own names (e.g. a split tuner's prefix)")
                targets[target] = i
        self._targets = list(targets)
        params: dict[str, Any] = {}
        providers: dict[str, list[int]] = {}
        for i, tuner in enumerate(self.tuners):
            for name, value in tuner.probe_params().items():
                if name in targets:
                    raise ValueError(f"CompositeTuner: {name!r} is a probe target of tuner "
                                     f"{_label(targets[name], self.tuners[targets[name]])} and a probe param of tuner "
                                     f"{_label(i, tuner)}")
                if name in params and not _same_value(params[name], value):
                    j = providers[name][0]
                    raise ValueError(f"CompositeTuner: probe param {name!r} is {describe_param(params[name])} for "
                                     f"tuner {_label(j, self.tuners[j])} but {describe_param(value)} for tuner "
                                     f"{_label(i, tuner)}; a probe sees one value per name")
                params.setdefault(name, value)
                providers.setdefault(name, []).append(i)
        self._params = params
        self._shared = {name: set(owners) for name, owners in providers.items() if len(owners) > 1}
        space: dict[str, str] = {}
        space_owner: dict[str, int] = {}
        for i, tuner in enumerate(self.tuners):
            for name, description in tuner.describe_space().items():
                if name in space:
                    j = space_owner[name]
                    raise ValueError(f"CompositeTuner: tuners {_label(j, self.tuners[j])} and {_label(i, tuner)} both "
                                     f"describe the search space of {name!r}; give each its own names")
                space[name], space_owner[name] = description, i
        self._space = space
        if len(self.tuners) > 1 and any(isinstance(v, SplitLocus) and v is not SplitLocus.Early
                                        for v in params.values()):
            warnings.warn("CompositeTuner: a split tuner at a locus after the early one is combined with other tuners. "
                          "Its positions are probed with the other tuners' out-params at the recipe's defaults; if "
                          "those change the equations (e.g. CSE options), trials can see a different number of "
                          "positions than was probed.", skip_file_prefixes=(_TUNING_DIR,))

    def probe_targets(self) -> Collection[str]:
        return list(self._targets)

    def probe_params(self) -> dict[str, Any]:
        return dict(self._params)

    def describe_space(self) -> Mapping[str, str]:
        return dict(self._space)

    def get_experiment(self) -> Experiment:
        if self._targets:
            raise RuntimeError(f"CompositeTuner has probe targets {self._targets}. Build its experiment with "
                               f"tuning.build_experiment, or call get_probed_experiment with a probe result.")
        return self._merge([tuner.get_probed_experiment(EMPTY_PROBE) for tuner in self.tuners])

    def get_probed_experiment(self, probe: ProbeResult) -> Experiment:
        if missing := [target for target in self._targets if target not in probe]:
            raise RuntimeError(f"CompositeTuner: the probe result {dict(probe)} has no count for {missing}")
        return self._merge([tuner.get_probed_experiment(ProbeResult({t: probe[t] for t in tuner.probe_targets()}))
                            for tuner in self.tuners])

    def _merge(self, experiments: Sequence[Experiment]) -> Experiment:
        merged = Experiment()
        in_owner: dict[str, int] = {}
        out_owner: dict[str, int] = {}
        for i, e in enumerate(experiments):
            for name, in_param in e.in_params.items():
                if name in in_owner:
                    j = in_owner[name]
                    raise ValueError(f"CompositeTuner: in-param {name!r} is declared by tuners "
                                     f"{_label(j, self.tuners[j])} and {_label(i, self.tuners[i])}; give each its own "
                                     f"names")
                in_owner[name] = i
                merged.add_in_param(in_param)
            for name, out_param in e.out_params.items():
                if name in out_owner and {out_owner[name], i} <= self._shared.get(name, set()):
                    continue  # a shared probe param, with the same value for both (checked in __init__)
                if name in out_owner:
                    j = out_owner[name]
                    raise ValueError(f"CompositeTuner: out-param {name!r} is provided by tuners "
                                     f"{_label(j, self.tuners[j])} and {_label(i, self.tuners[i])}; a recipe reads "
                                     f"one value per name")
                out_owner[name] = i
                merged.add_out_param(out_param)
        return merged


class DiscreteKnobTuner(Tuner):
    """
    One discrete knob: the in-param `name` picks an index into `choices`, and the out-param `out_param` (default
    `name`) gives the recipe the chosen value. For code-specific choices such as a loop layout or launch bounds, read
    in the recipe with `get_tuning_param(out_param)` (or `get_optional_tuning_param`, to run without the tuner).
    A checkpoint records the index; the choices are recorded in the probe sidecar (`describe_space`), so a study
    whose choices have changed (reordered, inserted, removed) is refused rather than misread. A choice cannot be None,
    which would drop the out-param; use an explicit value (e.g. 'default') and map it in the recipe.
    """

    def __init__(self, name: str, choices: Sequence[Any], *, out_param: str | None = None) -> None:
        if not choices:
            raise ValueError(f"DiscreteKnobTuner {name!r} needs at least one choice")
        if any(choice is None for choice in choices):
            raise ValueError(f"DiscreteKnobTuner {name!r}: a choice cannot be None (it would drop the out-param); use "
                             f"an explicit value, e.g. 'default'")
        self.name = name
        self.choices = list(choices)
        self.out_param = name if out_param is None else out_param

    def __repr__(self) -> str:
        return f"DiscreteKnobTuner({self.name!r}, {self.choices!r}, out_param={self.out_param!r})"

    def describe_space(self) -> Mapping[str, str]:
        return {self.name: f'{describe_param(self.choices)} -> {self.out_param}'}

    def get_experiment(self) -> Experiment:
        e = Experiment()
        e.add_in_param(self.name, Discrete(range(len(self.choices))))
        e.add_out_param(self.out_param, lambda params: self.choices[int(params[self.name])])
        return e
