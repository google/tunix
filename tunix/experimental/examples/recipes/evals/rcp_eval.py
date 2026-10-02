# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for the MLPerf offline checkpoint evaluation loop.

Stdlib only: runs on the launcher host, which has no JAX/tunix install.

The training run writes `eval_checkpoints.jsonl` (one record per checkpoint,
with the weight-update `timestamp_ms` captured before the checkpoint is saved)
and the MLLOG `seed_<seed>.out`. The eval container appends eval_* events and a
run_stop backdated to the checkpoint `timestamp_ms` to that same MLLOG. This
script only reads both files; it never writes the MLLOG.

Subcommands:
  plan     Print one TSV row per checkpoint still to evaluate:
             step, samples_count, timestamp_ms, checkpoint_path, is_last,
             mllog_file
  status   Print the MLLOG state of one checkpoint: "done <acc> <run_stop>" or
           "missing".
  inspect  Human-readable summary of the manifest and MLLOG.
  check    Run mlperf_logging.compliance_checker on the MLLOG.
  rcp      Run mlperf_logging.rcp_checker on a results directory.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Optional, TypedDict

MLLOG_PREFIX = ":::MLLOG "


class Event(TypedDict):
  """One MLLOG line, as written by mlperf_logging.mllog."""

  namespace: str
  time_ms: int
  event_type: str
  key: str
  value: Any
  metadata: dict[str, Any]


class Record(TypedDict):
  """One eval_checkpoints.jsonl line, as written by run_deepswe_dist.py."""

  step: int
  checkpoint_path: str
  timestamp_ms: int
  samples_count: int
  global_batch_size: int
  batch_size: int
  num_generations: int
  val_start_at: int
  max_steps: int
  target_accuracy: float
  seed: int
  mllog_file: str


def _read(path: str) -> str:
  if path.startswith("gs://"):
    res = subprocess.run(
        ["gsutil", "cat", path], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
      sys.exit(f"Cannot read {path}: {res.stderr.strip()}")
    return res.stdout
  with open(path, encoding="utf-8") as f:
    return f.read()


def _mllog_events(text: str) -> Iterator[Event]:
  for line in text.splitlines():
    idx = line.find(MLLOG_PREFIX)
    if idx < 0:
      continue
    try:
      yield json.loads(line[idx + len(MLLOG_PREFIX):])
    except json.JSONDecodeError:
      continue


class MllogState:
  """What the offline-eval loop needs to know about the training MLLOG."""

  run_start_ms: Optional[int]
  has_block_stop: bool
  run_stop: Optional[Event]
  evals: dict[int, float]  # samples_count -> eval_accuracy
  rewards: dict[int, float]  # step -> train reward

  def _reset(self) -> None:
    self.run_start_ms = None
    self.has_block_stop = False
    self.run_stop = None
    self.evals = {}
    self.rewards = {}

  def __init__(self, text: str):
    self._reset()
    # mllog writes key, time_ms, value and metadata on every event.
    for ev in _mllog_events(text):
      key = ev["key"]
      meta = ev["metadata"]
      if key == "run_start":
        # A restarted run rewrites the log; only the last run counts.
        self._reset()
        self.run_start_ms = int(ev["time_ms"])
      elif key == "block_stop":
        self.has_block_stop = True
      elif key == "eval_accuracy" and "samples_count" in meta:
        self.evals[int(meta["samples_count"])] = float(ev["value"])
      elif key == "run_stop":
        self.run_stop = ev
      elif key == "tracked_stats":
        value = ev["value"]
        if isinstance(value, dict) and "reward" in value and "step" in meta:
          self.rewards[int(meta["step"])] = float(value["reward"])


def _parse_manifest(
    text: str, run_start_ms: Optional[int]
) -> tuple[list[Record], list[int]]:
  """Returns (records, dropped_stale_steps) for the current training run."""
  records: dict[int, Record] = {}
  dropped: list[int] = []
  for line in text.splitlines():
    if not line.strip():
      continue
    rec = json.loads(line)
    # Upserts are keyed by step, so an aborted earlier run can leave records
    # for steps the current run never reached. They predate its run_start.
    if run_start_ms is not None and int(rec["timestamp_ms"]) < int(
        run_start_ms
    ):
      dropped.append(int(rec["step"]))
      continue
    records[int(rec["step"])] = rec
  return [records[s] for s in sorted(records)], dropped


def _mllog_path(args: argparse.Namespace, records: list[Record]) -> str:
  if args.mllog:
    return args.mllog
  paths = {r["mllog_file"] for r in records if r["mllog_file"]}
  if len(paths) != 1:
    sys.exit(
        f"Expected one mllog_file in the manifest, got {sorted(paths)}; pass"
        " --mllog."
    )
  return paths.pop()


def _load(
    args: argparse.Namespace,
) -> tuple[list[Record], str, MllogState, list[int]]:
  """Loads the manifest and the MLLOG it points at.

  Returns (records to evaluate, mllog path, MLLOG state, stale steps dropped).
  """
  manifest_text = _read(args.manifest)
  raw, _ = _parse_manifest(manifest_text, None)
  if not raw:
    sys.exit(f"Checkpoint manifest is empty: {args.manifest}")
  mllog = _mllog_path(args, raw)
  state = MllogState(_read(mllog))
  records, dropped = _parse_manifest(manifest_text, state.run_start_ms)
  if not records:
    sys.exit(
        f"No manifest records after run_start={state.run_start_ms} in {mllog}"
    )
  steps = [int(r["step"]) for r in records]
  if steps != list(range(steps[0], steps[0] + len(steps))):
    sys.exit(f"Manifest steps must be contiguous: {steps}")
  return records, mllog, state, dropped


def _unrecorded_checkpoints(records: list[Record]) -> list[int]:
  """Checkpoint dirs past the last manifest step (saved but never recorded).

  The manifest record is appended after save_checkpoint returns, so a run
  that dies mid-save leaves a checkpoint with no timestamp. Best effort: only
  GCS paths of the form .../checkpoints/<step>/... are inspected.
  """
  last = str(records[-1]["checkpoint_path"])
  marker = "/checkpoints/"
  if not last.startswith("gs://") or marker not in last:
    return []
  parent = last[: last.index(marker) + len(marker)]
  res = subprocess.run(
      ["gsutil", "ls", parent], capture_output=True, text=True, check=False
  )
  if res.returncode != 0:
    return []
  last_step = int(records[-1]["step"])
  found = []
  for line in res.stdout.splitlines():
    name = line.strip().rstrip("/").rsplit("/", 1)[-1]
    if name.isdigit() and int(name) > last_step:
      found.append(int(name))
  return sorted(found)


def cmd_plan(args: argparse.Namespace) -> None:
  records, mllog, state, dropped = _load(args)
  if dropped:
    print(
        f"Ignoring stale manifest steps from an earlier run: {dropped}",
        file=sys.stderr,
    )
  if state.run_stop is not None:
    print(
        "run_stop already logged"
        f" ({state.run_stop['metadata']['status']});"
        " nothing to evaluate.",
        file=sys.stderr,
    )
    return
  # eval_accuracy samples_count must be contiguous, so evaluated checkpoints
  # must be a prefix of the manifest.
  done = [int(r["samples_count"]) in state.evals for r in records]
  if False in done and True in done[done.index(False):]:
    gap = int(records[done.index(False)]["step"])
    sys.exit(
        f"{mllog} has eval_accuracy for steps after unevaluated step {gap};"
        " the evaluated checkpoints must be contiguous."
    )
  for rec in records:
    samples = int(rec["samples_count"])
    if samples in state.evals:
      print(
          f"Skipping step {rec['step']}: eval_accuracy="
          f"{state.evals[samples]:.4f} already logged.",
          file=sys.stderr,
      )
      continue
    # Only the run's final checkpoint may emit run_stop(aborted); a run that
    # stopped early (e.g. preempted) must not.
    is_last = int(rec["step"]) == int(rec["max_steps"])
    print(
        "\t".join([
            str(int(rec["step"])),
            str(samples),
            str(int(rec["timestamp_ms"])),
            str(rec["checkpoint_path"]),
            "true" if is_last else "false",
            mllog,
        ])
    )


def cmd_status(args: argparse.Namespace) -> None:
  state = MllogState(_read(args.mllog))
  acc = state.evals.get(int(args.samples_count))
  if acc is None:
    print("missing")
    return
  status = "none"
  if state.run_stop:
    status = state.run_stop["metadata"]["status"]
  print(f"done {acc:.4f} {status}")


def cmd_inspect(args: argparse.Namespace) -> None:
  records, mllog, state, dropped = _load(args)
  gbs = int(records[0]["global_batch_size"])
  print(f"Manifest     : {args.manifest}")
  print(f"MLLOG        : {mllog}")
  print(
      f"GBS          : {gbs}  val_start_at: {records[0]['val_start_at']} "
      f" max_steps: {records[0]['max_steps']}"
  )
  print(f"Steps        : {int(records[0]['step'])}..{int(records[-1]['step'])}")
  if dropped:
    print(f"Stale steps  : {dropped} (older than run_start, ignored)")
  unrecorded = _unrecorded_checkpoints(records)
  if unrecorded:
    print(
        f"Unrecorded   : checkpoint dirs {unrecorded} exist past the last"
        f" manifest step {int(records[-1]['step'])} (no timestamp; not"
        " evaluable)"
    )
  print(
      "Train block  : "
      + ("closed" if state.has_block_stop else "open (no block_stop)")
  )
  if state.run_stop:
    meta = state.run_stop["metadata"]
    ttt = (int(state.run_stop["time_ms"]) - int(state.run_start_ms)) / 60000
    print(
        f"run_stop     : status={meta['status']}"
        f" samples={meta['samples_count']} time-to-train={ttt:.1f} min"
    )
  else:
    print("run_stop     : not logged yet")
  print(f"\n{'step':>5} {'samples':>8} {'reward':>7} {'pass@4':>7}  checkpoint")
  for rec in records:
    samples = int(rec["samples_count"])
    reward = state.rewards.get(int(rec["step"]))
    acc = state.evals.get(samples)
    reward_s = "-" if reward is None else f"{reward:.4f}"
    acc_s = "-" if acc is None else f"{acc:.4f}"
    print(
        f"{int(rec['step']):>5} {samples:>8} {reward_s:>7} {acc_s:>7}"
        f"  {rec['checkpoint_path']}"
    )


def _checker(
    module: str, argv: list[str]
) -> subprocess.CompletedProcess[bytes]:
  try:
    __import__("mlperf_logging")
  except ImportError:
    sys.exit(
        "mlperf_logging is not installed: pip install"
        " git+https://github.com/mlcommons/logging.git"
    )
  return subprocess.run([sys.executable, "-m", module, *argv], check=False)


def cmd_check(args: argparse.Namespace) -> None:
  with tempfile.TemporaryDirectory() as tmp:
    local = os.path.join(tmp, "result_0.txt")
    with open(local, "w", encoding="utf-8") as f:
      f.write(_read(args.mllog))
    res = _checker(
        "mlperf_logging.compliance_checker",
        [
            local,
            "--ruleset",
            args.ruleset,
            "--log_output",
            os.path.join(tmp, "compliance_checker.log"),
        ],
    )
  sys.exit(res.returncode)


def cmd_rcp(args: argparse.Namespace) -> None:
  if args.results_dir:
    with tempfile.TemporaryDirectory() as tmp:
      res = _checker(
          "mlperf_logging.rcp_checker",
          [
              args.results_dir,
              "--rcp_version",
              args.ruleset,
              "--log_output",
              os.path.join(tmp, "rcp_checker.log"),
          ],
      )
    sys.exit(res.returncode)
  # rcp_checker needs several runs, so a single log is copied three times.
  # This only shows that one seed converges no earlier than the RCP allows;
  # it is not a submission RCP result.
  print(
      "WARNING: single-seed smoke test (one log copied 3x); not a submission"
      " RCP result.",
      file=sys.stderr,
  )
  text = _read(args.mllog)
  with tempfile.TemporaryDirectory() as tmp:
    for i in range(3):
      path = os.path.join(tmp, f"result_{i}.txt")
      with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    res = _checker(
        "mlperf_logging.rcp_checker",
        [
            tmp,
            "--rcp_version",
            args.ruleset,
            "--log_output",
            os.path.join(tmp, "rcp_checker.log"),
        ],
    )
  sys.exit(res.returncode)


def main(argv: Optional[list[str]] = None) -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  sub = p.add_subparsers(dest="cmd", required=True)

  for name in ("plan", "inspect"):
    sp = sub.add_parser(name)
    sp.add_argument("--manifest", required=True)
    sp.add_argument(
        "--mllog", default="", help="Overrides manifest mllog_file."
    )

  sp = sub.add_parser("status")
  sp.add_argument("--mllog", required=True)
  sp.add_argument("--samples_count", required=True)

  for name in ("check", "rcp"):
    sp = sub.add_parser(name)
    sp.add_argument("--mllog", default="")
    sp.add_argument(
        "--manifest", default="", help="Reads mllog_file from the manifest."
    )
    sp.add_argument(
        "--ruleset", default=os.environ.get("MLPERF_RULESET", "6.1.0")
    )
    if name == "rcp":
      sp.add_argument("--results_dir", default="")

  args = p.parse_args(argv)
  if args.cmd in ("check", "rcp") and not args.mllog:
    if args.manifest:
      args.mllog = _mllog_path(
          args, _parse_manifest(_read(args.manifest), None)[0]
      )
    elif not getattr(args, "results_dir", ""):
      p.error("--mllog or --manifest is required")
  {
      "plan": cmd_plan,
      "status": cmd_status,
      "inspect": cmd_inspect,
      "check": cmd_check,
      "rcp": cmd_rcp,
  }[args.cmd](args)


if __name__ == "__main__":
  main()
