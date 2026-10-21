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

`CompositeTuner` searches several tuners' spaces at once: for example a split tuner, a CseOptimizationTuner and a few
code-specific `DiscreteKnobTuner`s in one study. Its Experiment is the union of theirs, so it works with
`tuning.build_experiment`, the probe sidecar, `CheckpointedOptimizer`, `generate_best` and `plot_tuning` like any other
tuner. The sub-tuners' parameter names must be distinct; a name used by two of them is refused rather than shared.
"""

from collections.abc import Collection, Sequence
from typing import Any

from EinsteinEngine.tuning.experiment import Discrete, Experiment
from EinsteinEngine.common.describe_param import describe_param
from EinsteinEngine.tuning.probe import EMPTY_PROBE, ProbeResult
from EinsteinEngine.tuning.tuning import Tuner


def _same_value(a: Any, b: Any) -> bool:
    """Whether two probe-param values are the same value: identical or equal. Equal descriptions are not enough, since
    `describe_param` does not describe a function's code."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - a value whose == fails is not known to be the same
        return False


class CompositeTuner(Tuner):
    """
    The product of several tuners' search spaces.

    - Probe targets: the union of the sub-tuners' `probe_targets()`. A target claimed by two sub-tuners is refused,
      since their Experiments would both declare it as an out-param.
    - Probe params: the union of the sub-tuners' `probe_params()`. Two sub-tuners may provide the same probe param
      only with the same value (for example, one ordering function shared by several split tuners); different values
      are refused, since a probe sees one value per name. Each sub-tuner declares its probe params as out-params
      too (see `Tuner.probe_params`), so the merged Experiment keeps one out-param for each shared probe param.
    - Experiment: each sub-tuner's Experiment, built from the same probe (each sees the counts of its own targets),
      merged in the order the sub-tuners were given. An in-param or out-param name declared by two sub-tuners is
      refused. A checkpoint stores the in-params by name, so the order of the sub-tuners does not change what it
      means, but the sub-tuners must stay the same for a checkpoint to resume.
    """

    def __init__(self, *tuners: Tuner) -> None:
        if not tuners:
            raise ValueError("CompositeTuner needs at least one tuner")
        self.tuners = list(tuners)
        targets: dict[str, int] = {}
        for i, tuner in enumerate(self.tuners):
            for target in tuner.probe_targets():
                if target in targets:
                    raise ValueError(f"CompositeTuner: probe target {target!r} is claimed by tuners {targets[target]} "
                                     f"and {i} ({type(self.tuners[targets[target]]).__name__} and "
                                     f"{type(tuner).__name__}); give each its own names (e.g. a prefix)")
                targets[target] = i
        self._targets = list(targets)
        params: dict[str, Any] = {}
        providers: dict[str, list[int]] = {}
        for i, tuner in enumerate(self.tuners):
            for name, value in tuner.probe_params().items():
                if name in params and not _same_value(params[name], value):
                    raise ValueError(f"CompositeTuner: probe param {name!r} is {describe_param(params[name])} for "
                                     f"tuner {providers[name][0]} but {describe_param(value)} for tuner {i}; a probe "
                                     f"sees one value per name")
                params.setdefault(name, value)
                providers.setdefault(name, []).append(i)
        self._params = params
        self._shared = {name: set(owners) for name, owners in providers.items() if len(owners) > 1}

    def probe_targets(self) -> Collection[str]:
        return list(self._targets)

    def probe_params(self) -> dict[str, Any]:
        return dict(self._params)

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
                    raise ValueError(f"CompositeTuner: in-param {name!r} is declared by tuners {in_owner[name]} and "
                                     f"{i}; give each its own names")
                in_owner[name] = i
                merged.add_in_param(in_param)
            for name, out_param in e.out_params.items():
                if name in out_owner and {out_owner[name], i} <= self._shared.get(name, set()):
                    continue  # a shared probe param, with the same value for both (checked in __init__)
                if name in out_owner:
                    raise ValueError(f"CompositeTuner: out-param {name!r} is provided by tuners {out_owner[name]} and "
                                     f"{i}; a recipe reads one value per name")
                out_owner[name] = i
                merged.add_out_param(out_param)
        return merged


class DiscreteKnobTuner(Tuner):
    """
    One discrete knob: the in-param `name` picks an index into `choices`, and the out-param `out_param` (default
    `name`) gives the recipe the chosen value. For code-specific choices such as a loop layout or launch bounds,
    read in the recipe with `get_optional_tuning_param(out_param, default)`. A checkpoint records the index, so a
    study must keep `choices` in the same order to resume.
    """

    def __init__(self, name: str, choices: Sequence[Any], *, out_param: str | None = None) -> None:
        if not choices:
            raise ValueError(f"DiscreteKnobTuner {name!r} needs at least one choice")
        self.name = name
        self.choices = list(choices)
        self.out_param = name if out_param is None else out_param

    def get_experiment(self) -> Experiment:
        e = Experiment()
        e.add_in_param(self.name, Discrete(range(len(self.choices))))
        e.add_out_param(self.out_param, lambda params: self.choices[int(params[self.name])])
        return e
