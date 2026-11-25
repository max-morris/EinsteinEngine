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

"""Tests for syncing the generated code to --remote-path (remote_feedback.py), with localhost as the remote."""

import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from EinsteinEngine.tuning.remote_feedback import add_sync_delete_argument, do_remote_run

# Stands in for a recipe: replaces the generated code in `local` with the files in `files`. The trials of a test run
# within a second of each other, so each trial's files differ in size: rsync skips files whose size and mtime match.
_RECIPE = """
import os
for name in os.listdir(local):
    os.remove(os.path.join(local, name))
for name in files:
    with open(os.path.join(local, name), 'w') as fh:
        fh.write(f'{name} ' + 'v' * trial)
"""


class TestRemoteSync(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.local = os.path.join(self.dir.name, 'generated')
        self.remote = os.path.join(self.dir.name, 'tree')
        os.makedirs(self.local)
        os.makedirs(self.remote)
        with open(os.path.join(self.remote, 'existing.cc'), 'w') as fh:
            fh.write('hand-written')
        self.recipe = os.path.join(self.dir.name, 'recipe.py')
        with open(self.recipe, 'w') as fh:
            fh.write(_RECIPE)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def args(self, sync_delete: bool) -> Any:
        return SimpleNamespace(recipe=self.recipe, local_path=f'{self.local}/', remote_host='localhost',
                               remote_path=f'{self.remote}/', remote_cactus_path=self.dir.name,
                               remote_command="echo 'Submitted batch job 7'", remote_timing_command='echo 2.5',
                               # Job 7 has already left the queue: the poll prints nothing.
                               remote_poll_command='true',
                               checkpoint_file=os.path.join(self.dir.name, 'checkpt.jsonl'), sync_delete=sync_delete)

    def trial(self, sync_delete: bool, trial: int, files: list[str]) -> None:
        self.assertEqual(do_remote_run(self.args(sync_delete), {'local': self.local, 'files': files, 'trial': trial}),
                         2.5)

    def tree(self) -> dict[str, str]:
        contents = {}
        for name in sorted(os.listdir(self.remote)):
            with open(os.path.join(self.remote, name)) as fh:
                contents[name] = fh.read()
        return contents

    def test_sync_delete(self) -> None:
        """By default the remote path ends up holding exactly this trial's generated code."""
        self.trial(True, 1, ['a.cc', 'b.cc'])
        self.assertEqual(self.tree(), {'a.cc': 'a.cc v', 'b.cc': 'b.cc v'})
        self.trial(True, 2, ['a.cc'])
        self.assertEqual(self.tree(), {'a.cc': 'a.cc vv'})

    def test_no_sync_delete(self) -> None:
        """--no-sync-delete adds and updates files, and leaves the rest of the remote path alone."""
        self.trial(False, 1, ['a.cc', 'b.cc'])
        self.assertEqual(self.tree(), {'a.cc': 'a.cc v', 'b.cc': 'b.cc v', 'existing.cc': 'hand-written'})
        self.trial(False, 2, ['a.cc'])
        self.assertEqual(self.tree(), {'a.cc': 'a.cc vv', 'b.cc': 'b.cc v', 'existing.cc': 'hand-written'})

    def test_argument(self) -> None:
        parser = argparse.ArgumentParser()
        add_sync_delete_argument(parser)
        self.assertTrue(parser.parse_args([]).sync_delete)
        self.assertTrue(parser.parse_args(['--sync-delete']).sync_delete)
        self.assertFalse(parser.parse_args(['--no-sync-delete']).sync_delete)


if __name__ == '__main__':
    unittest.main()
