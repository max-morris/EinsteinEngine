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

"""Tests for polling a Slurm job with the --remote-poll-command template (remote_feedback.py), with localhost as the
remote and a stand-in for squeue."""

import argparse
import contextlib
import io
import os
import shlex
import tempfile
import unittest
from unittest import mock

from EinsteinEngine.tuning.remote_feedback import DEFAULT_POLL_COMMAND, POLL_INTERVAL_S, Remote, \
    add_poll_command_argument


class TestRemotePollCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.polls = os.path.join(self.dir.name, 'polls')

    def tearDown(self) -> None:
        self.dir.cleanup()

    def fake_squeue(self, n_running: int) -> str:
        """A poll command that logs the id it was given, after a `${VAR}` the shell must see, and prints a queue line
        for its first `n_running` polls."""
        polls = shlex.quote(self.polls)
        return (f"SETUP=sourced; echo \"${{SETUP}} {{job_id}}\" >> {polls}; "
                f"[ $(wc -l < {polls}) -gt {n_running} ] || echo '{{job_id}} gpu2 trial R'")

    def polled(self) -> list[str]:
        with open(self.polls) as fh:
            return fh.read().splitlines()

    def wait(self, poll_command: str, job_id: str = '4242') -> mock.MagicMock:
        with mock.patch('EinsteinEngine.tuning.remote_feedback.time.sleep') as sleep, \
                contextlib.redirect_stdout(io.StringIO()):
            Remote('localhost', poll_command).wait_for_job(job_id)
        return sleep

    def test_polls_until_the_job_leaves_the_queue(self) -> None:
        sleep = self.wait(self.fake_squeue(n_running=2))
        self.assertEqual(self.polled(), ['sourced 4242'] * 3)
        self.assertEqual(sleep.call_args_list, [mock.call(POLL_INTERVAL_S)] * 2)

    def test_a_job_already_gone(self) -> None:
        sleep = self.wait(self.fake_squeue(n_running=0))
        self.assertEqual(self.polled(), ['sourced 4242'])
        sleep.assert_not_called()

    def test_a_failed_poll_stops(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Failed while polling Slurm job 4242 with 'source /nonexistent"):
            self.wait("source /nonexistent/setup.sh && squeue -h -j {job_id}")

    def test_the_argument(self) -> None:
        parser = argparse.ArgumentParser()
        add_poll_command_argument(parser)
        self.assertEqual(parser.parse_args([]).remote_poll_command, 'squeue -h -j {job_id}')
        self.assertEqual(Remote('localhost').poll_command, DEFAULT_POLL_COMMAND)
        custom = 'source /sw/setup.sh && squeue -h -j {job_id}'
        self.assertEqual(parser.parse_args(['--remote-poll-command', custom]).remote_poll_command, custom)
        # Without the placeholder the command could not poll the job, so it is refused when parsed.
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            parser.parse_args(['--remote-poll-command', 'squeue -h'])
        self.assertIn('must contain {job_id}', err.getvalue())


if __name__ == '__main__':
    unittest.main()
