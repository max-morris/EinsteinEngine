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
from typing import Protocol, Any

from EinsteinEngine.tuning.clear_caches import clear_caches
from EinsteinEngine.common.util import pprint


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
    parser.add_argument("--local-path", type=str, default="/home/max/src/EmitCactus/EinsteinEngine/tuning/Cottonmouth/", help="Local path containing generated code.")
    parser.add_argument("--remote-host", type=str, default="qbd", help="Remote host to which generated code should be copied.")
    parser.add_argument("--remote-path", type=str, default="/home/mmorris/project/Cottonmouth/", help="Remote path into which generated code should be copied.")
    parser.add_argument("--remote-cactus-path", type=str, default="/home/mmorris/project/Cactus/", help="Remote path containing the Cactus installation.")
    parser.add_argument("--remote-command", type=str, default="./build.sh && ./run-all.sh", help="Command to build and run on the remote machine.")
    parser.add_argument("--remote-timing-command", type=str, default="./timings.sh", help="Command to run timing on the remote machine. Must print a single number to optimize for.")

    args = parser.parse_args()

    do_remote_run(args, {})

def do_remote_run(args: RemoteFeedbackArgs, globals_to_inject: dict[str, Any]) -> float:
    # When the "remote" host is localhost, skip scp/ssh entirely and run
    # everything locally. rsync still runs (to a local destination path) and
    # commands are executed through the local shell instead of over ssh.
    is_local = args.remote_host == "localhost"

    def run_command(cmd: str) -> subprocess.CompletedProcess[str]:
        if is_local:
            invocation = ["bash", "-c", cmd]
        else:
            invocation = ["ssh", args.remote_host, cmd]
        return subprocess.run(invocation, capture_output=True, text=True)

    sys.argv = [args.recipe]
    try:
        runpy.run_path(args.recipe, run_name="__main__", init_globals=globals_to_inject)
    except Exception as e:
        traceback.print_exception(e)
        raise RuntimeError(f"Error when executing recipe")

    clear_caches()

    pprint("Done generating. Syncing to remote..." if not is_local else "Done generating. Syncing locally...")

    rsync_destination = args.remote_path if is_local else f"{args.remote_host}:{args.remote_path}"
    rsync_result = subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--itemize-changes",
            args.local_path,
            rsync_destination
        ],
        capture_output=True,
        text=True
    )

    if rsync_result.returncode != 0:
        print(rsync_result.stdout)
        print(rsync_result.stderr)
        raise RuntimeError("rsync failed")

    pprint("Done syncing. Building..." if is_local else "Done syncing to remote. Building on remote...")

    build_and_submit_result = run_command(
        f"cd {args.remote_cactus_path} && {args.remote_command}"
    )

    build_and_submit_output = f"{build_and_submit_result.stdout}\n{build_and_submit_result.stderr}"

    if build_and_submit_result.returncode != 0:
        raise RuntimeError(
            "Remote build/submit failed:\n"
            f"{build_and_submit_output}"
        )

    job_matches = re.findall(r"Submit finished, job id is (\d+)", build_and_submit_output)
    if not job_matches:
        raise RuntimeError(
            "Could not parse Slurm job id from build/run output.\n"
            f"Output:\n{build_and_submit_output}"
        )
    slurm_job_id = job_matches[-1]

    pprint(f"Job {slurm_job_id} submitted on {args.remote_host} with Slurm.")

    while True:
        squeue_result = run_command(f"squeue -h -j {slurm_job_id}")

        if squeue_result.returncode != 0:
            raise RuntimeError(
                f"Failed while polling Slurm job {slurm_job_id}:\n"
                f"{squeue_result.stdout}\n{squeue_result.stderr}"
            )

        # `squeue -h -j <id>` prints nothing once the job has left the queue.
        if not squeue_result.stdout.strip():
            break

        pprint(f"Still waiting on job {slurm_job_id}...")
        time.sleep(60)

    pprint(f"Job {slurm_job_id} finished.")

    timing_result = run_command(
        f"cd {args.remote_cactus_path} && {args.remote_timing_command}"
    )
    timing_output = f"{timing_result.stdout}\n{timing_result.stderr}"
    if timing_result.returncode != 0:
        raise RuntimeError(f"Remote timing run failed:\n{timing_output}")

    # The timing command is expected to print a single number (the value to
    # optimize for). Any parsing of a particular report format lives in the
    # timing script itself, keeping this workflow generic.
    try:
        timing_value = float(timing_result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            f"Expected a single number from '{args.remote_timing_command}', "
            f"but could not parse one.\nOutput:\n{timing_output}"
        )

    print(f"Timing value: {timing_value:.3f}")

    return timing_value


if __name__ == "__main__":
    main()
