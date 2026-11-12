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
Measure the cost model's weights on the tuning target.

A tuning run that opts in runs `microbenchmarks/` on the target once, through the same remote as its trials, fits a
weight profile, and records both in a sidecar next to the checkpoint (`<checkpoint>.microbench.json`). Every trial of
that checkpoint, on this run and every resumed run, uses the recorded weights, since a checkpoint's scores are only
meaningful against one cost model. The sidecar records the target (host, remote path, command, and a digest of the
benchmark sources); resuming against another target is refused once the checkpoint has entries.
"""

import hashlib
import json
import os
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from EinsteinEngine.common.util import pprint
from EinsteinEngine.generators.sympy_complexity import ComplexityWeights, weights_from_profile
from EinsteinEngine.tuning.fit_weights import build_profile, load_results
from EinsteinEngine.tuning.remote_feedback import Remote, find_job_id


def _find_microbench_dir() -> Path:
    """The benchmark sources: installed with the package, or at the top of a checkout (and an editable install)."""
    package = Path(__file__).resolve().parents[1]
    installed = package / 'microbenchmarks'
    return installed if installed.is_dir() else package.parent / 'microbenchmarks'


MICROBENCH_DIR = _find_microbench_dir()
_SOURCES = ('CMakeLists.txt', 'Makefile', 'run.sh', 'common', 'src')
RESULTS_FILE = 'results.json'


class MicrobenchArgs(Protocol):
    remote_host: str
    microbench_remote_path: str | None
    microbench_command: str | None


def microbench_sidecar_path(checkpoint_file: str) -> str:
    return f'{checkpoint_file}.microbench.json'


def sources_digest(microbench_dir: Path = MICROBENCH_DIR) -> str:
    """A digest of the benchmark sources, so that a changed benchmark is a different target."""
    h = hashlib.sha256()
    for name in _SOURCES:
        top = microbench_dir / name
        files = sorted(p for p in top.rglob('*') if p.is_file()) if top.is_dir() else [top]
        for path in files:
            h.update(str(path.relative_to(microbench_dir)).encode() + b'\0')
            h.update(path.read_bytes() + b'\0')
    return h.hexdigest()


@dataclass(frozen=True)
class MicrobenchTarget:
    host: str
    remote_path: str
    command: str
    sources_digest: str

    def differences(self, current: 'MicrobenchTarget') -> list[str]:
        return [f"{name} was {was!r} and is now {now!r}"
                for (name, was), now in zip(asdict(self).items(), asdict(current).values()) if was != now]


@dataclass(frozen=True)
class MicrobenchRecord:
    target: MicrobenchTarget
    results: Any
    profile: dict[str, Any]

    def weights(self, source: str) -> ComplexityWeights:
        return weights_from_profile(self.profile, source)

    def to_json(self) -> str:
        return json.dumps({'target': asdict(self.target), 'results': self.results, 'profile': self.profile},
                          indent=2) + '\n'

    @staticmethod
    def from_json(text: str) -> 'MicrobenchRecord':
        obj = json.loads(text)
        if not isinstance(obj, dict) or not isinstance(obj.get('target'), dict) or \
                not isinstance(obj.get('profile'), dict) or 'results' not in obj:
            raise ValueError('Expected a JSON object with "target", "results" and "profile".')
        return MicrobenchRecord(MicrobenchTarget(**obj['target']), obj['results'], obj['profile'])


def read_record(checkpoint_file: str) -> MicrobenchRecord | None:
    sidecar = microbench_sidecar_path(checkpoint_file)
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as fh:
        try:
            return MicrobenchRecord.from_json(fh.read())
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"The microbenchmark file {sidecar} (next to checkpoint {checkpoint_file}) is not "
                               f"valid: {e}") from e


def recorded_weights(checkpoint_file: str) -> ComplexityWeights | None:
    """The weights a checkpoint's trials were generated with, or None if it uses the default profile."""
    record = read_record(checkpoint_file)
    return None if record is None else record.weights(microbench_sidecar_path(checkpoint_file))


def run_microbenchmarks(remote: Remote, remote_path: str, command: str) -> Any:
    """Copy the benchmark sources to `remote_path`, run `command` there, wait for the Slurm job it submits (if it
    prints a job id), and return the parsed `results.json` it leaves behind."""
    results = shlex.quote(f'{remote_path.rstrip("/")}/{RESULTS_FILE}')
    pprint(f"Syncing the microbenchmarks to {remote.host}:{remote_path}...")
    if (mkdir := remote.run(f"mkdir -p {shlex.quote(remote_path)}")).returncode != 0:
        raise RuntimeError(f"Could not create {remote_path} on {remote.host}:\n{mkdir.stderr}")
    remote.sync(f'{MICROBENCH_DIR}/', remote_path, ['--filter=:- .gitignore'])
    # The build tree is excluded from the sync, so it survives --delete; a stale results file must not.
    remote.run(f"rm -f {results}")

    pprint(f"Running the microbenchmarks: {command}")
    submit = remote.run(f"cd {remote_path} && {command}")
    output = f"{submit.stdout}\n{submit.stderr}"
    if submit.returncode != 0:
        raise RuntimeError(f"The microbenchmark command failed:\n{output}")
    if (job_id := find_job_id(output)) is not None:
        remote.wait_for_job(job_id)

    fetched = remote.run(f"cat {results}")
    if fetched.returncode != 0:
        raise RuntimeError(f"The microbenchmarks left no {RESULTS_FILE} in {remote_path}:\n{fetched.stderr}\n"
                           f"The command printed:\n{output}")
    try:
        return json.loads(fetched.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{remote_path}/{RESULTS_FILE} is not JSON: {e}") from e


def tuning_weights(args: MicrobenchArgs, checkpoint_file: str, n_entries: int) -> ComplexityWeights | None:
    """
    The weights a tuning run on `checkpoint_file` (with `n_entries` entries) must use: the ones recorded next to it,
    newly measured ones if the run opts in (`args.microbench_command`) and none are recorded, or None for the default
    profile.
    """
    sidecar = microbench_sidecar_path(checkpoint_file)
    record = read_record(checkpoint_file)

    if args.microbench_command is None:
        if record is not None:
            pprint(f"Using the weights measured on {record.target.host} recorded in {sidecar}.")
            return record.weights(sidecar)
        return None

    if args.microbench_remote_path is None:
        raise RuntimeError("--microbench-command needs --microbench-remote-path.")
    if not MICROBENCH_DIR.is_dir():
        raise RuntimeError(f"Measuring the weights needs the microbenchmark sources, which are neither installed "
                           f"with the package nor in {MICROBENCH_DIR}.")
    target = MicrobenchTarget(args.remote_host, args.microbench_remote_path, args.microbench_command,
                              sources_digest())
    if record is not None:
        if not (changes := record.target.differences(target)):
            pprint(f"Using the weights measured on {target.host} recorded in {sidecar}.")
            return record.weights(sidecar)
        message = f"The microbenchmarks recorded in {sidecar} were run on another target: {'; '.join(changes)}."
        if n_entries > 0:
            raise RuntimeError(f"{message} The checkpoint's trials were scored with those weights. Use a new "
                               f"checkpoint file, or restore the target.")
        pprint(f"{message} The checkpoint has no entries, so they are measured again.")
    elif n_entries > 0:
        raise RuntimeError(f"Checkpoint {checkpoint_file} has entries scored with the default weights, so it cannot "
                           f"switch to measured ones. Use a new checkpoint file to tune with --microbench-command.")

    results = run_microbenchmarks(Remote(target.host), target.remote_path, target.command)
    profile, warnings = build_profile(load_results(results))
    for warning in warnings:
        pprint(f"Warning: {warning}")
    record = MicrobenchRecord(target, results, profile)
    weights = record.weights(sidecar)
    with open(sidecar, 'w') as fh:
        fh.write(record.to_json())
    pprint(f"Recorded the measured weights in {sidecar}.")
    return weights
