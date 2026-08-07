# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sub-batch checkpointing benchmark supervisor (TEMPORARY, commit 4).

Runs `qwen3_grpo_demo.py` under randomly injected hard preemptions and
restarts the process until it exits cleanly, once per benchmark arm:

  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_bench.py --mode sub_batch
  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_bench.py --mode baseline

The two arms are identical -- same geometry, same per-apply trainer
checkpointing, same chaos probability, same wall-clock anchor -- except
the sub-batch stream. Both emit the SAME wandb metric names, so plotting
`perf/sub_batch_progress_micro_steps` against
`perf/sub_batch_cumulative_time` overlays the recovery profiles directly;
the wandb run config (`sub_batch_checkpointing` / `bench_baseline`)
distinguishes the arms. `perf/sub_batch_time_saved` exists only on the
sub_batch arm (the baseline restores no mid-step work by definition).

Why a supervisor process: the chaos injector kills the trainee with
os._exit (skipping finally/atexit, like a real preemption), so restart
continuity cannot live inside the trainee. The supervisor also pins
TUNIX_BENCH_START_TIME once, which is what keeps cumulative-time curves
continuous across restarts on BOTH arms -- the baseline has no snapshot
to carry an anchor in.

Geometry defaults (num_generations=8 fixed in the demo): batch_size=8,
mini_batch_size=4, train_micro_batch_size=1, num_iterations=2 -> per
global step: 2 mini-batches x k=4 micro-steps x 2 epochs = 16 micro-steps
and 4 applies, so preemptions land mid-window, at apply boundaries, and
inside the replay sweep.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

_DEMO = pathlib.Path(__file__).parent / "qwen3_grpo_demo.py"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--mode",
      choices=("sub_batch", "baseline"),
      required=True,
      help="Which benchmark arm to run.",
  )
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--mini_batch_size", type=int, default=4)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--num_iterations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=20)
  parser.add_argument("--sb_chaos_prob", type=float, default=0.02)
  parser.add_argument(
      "--keep_eval",
      action="store_true",
      help=(
          "Keep the demo's held-out eval. Off by default: eval is noise"
          " for recovery benchmarking and re-fires on early-window"
          " restarts."
      ),
  )
  parser.add_argument(
      "--max_restarts",
      type=int,
      default=100,
      help=(
          "Abort after this many restarts (guards a genuine crash loop"
          " from masquerading as preemption recovery forever)."
      ),
  )
  parser.add_argument(
      "demo_args",
      nargs=argparse.REMAINDER,
      help="Extra args forwarded verbatim to qwen3_grpo_demo.py.",
  )
  args = parser.parse_args()

  env = dict(os.environ)
  # One anchor for the whole experiment, surviving every process death;
  # both trainee arms read it for perf/sub_batch_cumulative_time.
  env.setdefault("TUNIX_BENCH_START_TIME", repr(time.time()))
  # One checkpoint root per experiment arm, stable across restarts: the
  # demo's default root is per-launch timestamped, which would hand every
  # restarted process an empty directory and silently disable recovery in
  # both arms. Mode in the tag keeps the arms' checkpoints apart.
  env.setdefault(
      "TUNIX_BENCH_CKPT_TAG", f"bench-{args.mode}-{int(time.time())}"
  )
  # One wandb run per arm across restarts (wandb's own resume contract);
  # without this every restart opens a fresh run and fragments the curves.
  env.setdefault("WANDB_RUN_ID", env["TUNIX_BENCH_CKPT_TAG"])
  env.setdefault("WANDB_RESUME", "allow")

  # TRUE-completion marker: exit code 0 is NOT trusted (the vLLM/torch
  # shutdown path has been observed exiting 0 after an uncaught crash,
  # which would falsely stop the restart loop). The demo touches this file
  # only when train() actually returns; exit 0 without it is a crash.
  done_file = pathlib.Path(
      tempfile.gettempdir(), f"tunix_bench_done_{env['TUNIX_BENCH_CKPT_TAG']}"
  )
  done_file.unlink(missing_ok=True)
  env["TUNIX_BENCH_DONE_FILE"] = str(done_file)

  mode_flag = (
      "--sub_batch_checkpointing"
      if args.mode == "sub_batch"
      else "--bench_baseline"
  )
  cmd = [
      sys.executable,
      str(_DEMO),
      mode_flag,
      f"--batch_size={args.batch_size}",
      f"--mini_batch_size={args.mini_batch_size}",
      f"--train_micro_batch_size={args.train_micro_batch_size}",
      f"--num_iterations={args.num_iterations}",
      f"--max_steps={args.max_steps}",
      f"--sb_chaos_prob={args.sb_chaos_prob}",
      *([] if args.keep_eval else ["--skip_eval"]),
      *args.demo_args,
  ]

  restarts = 0
  while True:
    print(
        f"[sub-batch bench:{args.mode}] launch (restart #{restarts}):"
        f" {' '.join(cmd)}",
        flush=True,
    )
    code = subprocess.run(cmd, env=env, check=False).returncode
    if code == 0 and done_file.exists():
      print(
          f"[sub-batch bench:{args.mode}] clean exit after"
          f" {restarts} restart(s).",
          flush=True,
      )
      return 0
    if code == 0:
      print(
          f"[sub-batch bench:{args.mode}] exit code 0 WITHOUT the"
          " completion marker -- the run crashed through a shutdown path"
          " that swallowed the exit code. Treating as a crash.",
          flush=True,
      )
    restarts += 1
    if restarts > args.max_restarts:
      print(
          f"[sub-batch bench:{args.mode}] exceeded --max_restarts"
          f" ({args.max_restarts}); last exit code {code}. Aborting.",
          flush=True,
      )
      return code
    print(
        f"[sub-batch bench:{args.mode}] exit code {code}; restarting"
        " (simulated preemption recovery).",
        flush=True,
    )
    time.sleep(2)


if __name__ == "__main__":
  sys.exit(main())
