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

"""Tests for measuring the cost model's weights once per checkpoint (microbench.py), with localhost as the remote."""

import json
import os
import shlex
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from EinsteinEngine.tuning.microbench import microbench_sidecar_path, recorded_weights, tuning_weights
from EinsteinEngine.tuning.remote_feedback import find_job_id
from test_fit_weights import base_rows


class TestMicrobench(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.checkpoint = os.path.join(self.dir.name, 'checkpt.jsonl')
        self.runs = os.path.join(self.dir.name, 'runs')

    def tearDown(self) -> None:
        self.dir.cleanup()

    def args(self, sin_ns: float | None = 16.0, remote: str = 'remote') -> Any:
        """Stands in for run.sh: counts its runs and writes canned results (none if `sin_ns` is None)."""
        command = f"echo run >> {shlex.quote(self.runs)}"
        if sin_ns is not None:
            results = json.dumps({'results': base_rows(sin=sin_ns)})
            command += f" && printf '%s' {shlex.quote(results)} > results.json"
        return SimpleNamespace(remote_host='localhost', microbench_remote_path=os.path.join(self.dir.name, remote),
                               microbench_command=command)

    def n_runs(self) -> int:
        if not os.path.exists(self.runs):
            return 0
        with open(self.runs) as fh:
            return len(fh.readlines())

    def test_not_opted_in(self) -> None:
        self.assertIsNone(tuning_weights(SimpleNamespace(remote_host='localhost', microbench_remote_path=None,
                                                         microbench_command=None), self.checkpoint, 0))
        self.assertIsNone(recorded_weights(self.checkpoint))

    def test_measures_once_per_checkpoint(self) -> None:
        weights = tuning_weights(self.args(), self.checkpoint, 0)
        assert weights is not None
        self.assertEqual(weights.transcendental['sin'], 8)
        self.assertTrue(os.path.exists(microbench_sidecar_path(self.checkpoint)))
        # A resumed run reuses the recorded weights, with or without the option.
        self.assertEqual(tuning_weights(self.args(), self.checkpoint, 3), weights)
        no_option = SimpleNamespace(remote_host='elsewhere', microbench_remote_path=None, microbench_command=None)
        self.assertEqual(tuning_weights(no_option, self.checkpoint, 3), weights)
        self.assertEqual(recorded_weights(self.checkpoint), weights)
        self.assertEqual(self.n_runs(), 1)

    def test_another_target(self) -> None:
        tuning_weights(self.args(), self.checkpoint, 0)
        with self.assertRaisesRegex(RuntimeError, "remote_path was .* and is now"):
            tuning_weights(self.args(remote='other'), self.checkpoint, 1)
        # Without entries there is nothing to misinterpret, so it measures again.
        weights = tuning_weights(self.args(sin_ns=32.0, remote='other'), self.checkpoint, 0)
        assert weights is not None
        self.assertEqual(weights.transcendental['sin'], 16)
        self.assertEqual(self.n_runs(), 2)

    def test_checkpoint_scored_with_default_weights(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "default weights"):
            tuning_weights(self.args(), self.checkpoint, 2)
        self.assertEqual(self.n_runs(), 0)

    def test_stale_results_are_not_read(self) -> None:
        tuning_weights(self.args(), self.checkpoint, 0)
        os.remove(microbench_sidecar_path(self.checkpoint))
        with self.assertRaisesRegex(RuntimeError, "left no results.json"):
            tuning_weights(self.args(sin_ns=None), self.checkpoint, 0)

    def test_find_job_id(self) -> None:
        self.assertEqual(find_job_id("Submitted batch job 41\nSubmitted batch job 42\n"), '42')
        self.assertEqual(find_job_id("... Submit finished, job id is 127217\n"), '127217')
        self.assertIsNone(find_job_id("ran synchronously\n"))


if __name__ == "__main__":
    unittest.main()
