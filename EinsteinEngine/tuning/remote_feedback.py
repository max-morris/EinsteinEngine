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

import argparse
import re
import runpy
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from typing import Protocol, Any

from EinsteinEngine.tuning import probe
from EinsteinEngine.tuning.clear_caches import clear_caches
from EinsteinEngine.common.util import pprint


class TrialFailed(RuntimeError):
    """The trial's configuration failed: the recipe raised or refused a requested cut, or the remote build, run or
    timing failed. Infrastructure errors (rsync, ssh, Slurm polling) propagate as plain RuntimeErrors."""


# ssh exits with 255 when the connection itself fails, whatever the remote command did.
_SSH_CONNECTION_FAILED = 255


class RemoteFeedbackArgs(Protocol):
    recipe: str
    local_path: str
    remote_host: str
    remote_path: str
    remote_cactus_path: str
    remote_command: str
    remote_timing_command: str
    checkpoint_file: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a recipe, execute on a remote machine, and collect timing information.")
    parser.add_argument("recipe", help="Path to the Einstein Engine recipe.")
    parser.add_argument("--local-path", type=str, required=True, help="Local path containing generated code.")
    parser.add_argument("--remote-host", type=str, required=True, help="Remote host to which generated code should be copied.")
    parser.add_argument("--remote-path", type=str, required=True, help="Remote path into which generated code should be copied.")
    parser.add_argument("--remote-cactus-path", type=str, required=True, help="Remote path containing the Cactus installation.")
    parser.add_argument("--remote-command", type=str, default="./build.sh && ./run-all.sh", help="Command to build and run on the remote machine.")
    parser.add_argument("--remote-timing-command", type=str, default="./timings.sh", help="Command to run timing on the remote machine. Must print a single number to optimize for.")

    args = parser.parse_args()

    do_remote_run(args, {})

class Remote:
    """The machine trials and microbenchmarks run on: commands go over ssh, or through the local shell when the host is
    localhost."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.is_local = host == "localhost"

    def run(self, cmd: str) -> subprocess.CompletedProcess[str]:
        invocation = ["bash", "-c", cmd] if self.is_local else ["ssh", self.host, cmd]
        result = subprocess.run(invocation, capture_output=True, text=True)
        if not self.is_local and result.returncode == _SSH_CONNECTION_FAILED:
            raise RuntimeError(f"ssh to {self.host} failed:\n{result.stderr}")
        return result

    def sync(self, local_path: str, remote_path: str, extra_args: Sequence[str] = ()) -> None:
        destination = remote_path if self.is_local else f"{self.host}:{remote_path}"
        result = subprocess.run(
            ["rsync", "-a", "--delete", "--itemize-changes", *extra_args, local_path, destination],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("rsync failed")

    def wait_for_job(self, job_id: str) -> None:
        pprint(f"Job {job_id} submitted on {self.host} with Slurm.")
        while True:
            squeue_result = self.run(f"squeue -h -j {job_id}")
            if squeue_result.returncode != 0:
                raise RuntimeError(
                    f"Failed while polling Slurm job {job_id}:\n{squeue_result.stdout}\n{squeue_result.stderr}")
            # `squeue -h -j <id>` prints nothing once the job has left the queue.
            if not squeue_result.stdout.strip():
                break
            pprint(f"Still waiting on job {job_id}...")
            time.sleep(60)
        pprint(f"Job {job_id} finished.")


def find_job_id(output: str) -> str | None:
    """The last Slurm job id that cactup ("Submit finished, job id is N") or sbatch ("Submitted batch job N") printed."""
    matches = re.findall(r"(?:Submit finished, job id is|Submitted batch job) (\d+)", output)
    return matches[-1] if matches else None


def do_remote_run(args: RemoteFeedbackArgs, globals_to_inject: dict[str, Any]) -> float:
    # When the "remote" host is localhost, skip scp/ssh entirely and run
    # everything locally. rsync still runs (to a local destination path) and
    # commands are executed through the local shell instead of over ssh.
    remote = Remote(args.remote_host)

    sys.argv = [args.recipe]
    with probe.collect_refused_cuts() as refused_cuts:
        try:
            runpy.run_path(args.recipe, run_name="__main__", init_globals=globals_to_inject)
        except Exception as e:
            traceback.print_exception(e)
            raise TrialFailed("Error when executing recipe") from e

    clear_caches()

    # Code generated without those cuts is not the configuration the optimizer chose, so it must not be scored.
    if refused_cuts:
        raise TrialFailed(f"The engine refused cuts this trial asked for: {'; '.join(refused_cuts)}.")

    pprint("Done generating. Syncing to remote..." if not remote.is_local else "Done generating. Syncing locally...")
    remote.sync(args.local_path, args.remote_path)
    pprint("Done syncing. Building..." if remote.is_local else "Done syncing to remote. Building on remote...")

    build_and_submit_result = remote.run(f"cd {args.remote_cactus_path} && {args.remote_command}")
    build_and_submit_output = f"{build_and_submit_result.stdout}\n{build_and_submit_result.stderr}"

    if build_and_submit_result.returncode != 0:
        raise TrialFailed(
            "Remote build/submit failed:\n"
            f"{build_and_submit_output}"
        )

    slurm_job_id = find_job_id(build_and_submit_output)
    if slurm_job_id is None:
        raise RuntimeError(
            "Could not parse Slurm job id from build/run output.\n"
            f"Output:\n{build_and_submit_output}"
        )
    remote.wait_for_job(slurm_job_id)

    timing_result = remote.run(f"cd {args.remote_cactus_path} && {args.remote_timing_command}")
    timing_output = f"{timing_result.stdout}\n{timing_result.stderr}"
    if timing_result.returncode != 0:
        raise TrialFailed(f"Remote timing run failed:\n{timing_output}")

    # The timing command is expected to print a single number (the value to
    # optimize for). Any parsing of a particular report format lives in the
    # timing script itself, keeping this workflow generic.
    try:
        timing_value = float(timing_result.stdout.strip())
    except ValueError:
        raise TrialFailed(
            f"Expected a single number from '{args.remote_timing_command}', "
            f"but could not parse one.\nOutput:\n{timing_output}"
        )

    print(f"Timing value: {timing_value:.3f}")

    return timing_value


if __name__ == "__main__":
    main()
