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

"""Sub-batch checkpointing correctness validation (TEMPORARY, commit 4).

Runs `qwen3_grpo_demo.py` twice with identical geometry where every
sub-batch dimension is exercised (num_generations > 1, num_iterations > 1,
gradient accumulation windows of k = mini/micro micro-steps):

  1. CONTROL arm: never preempted, runs to completion.
  2. PREEMPT arm: deterministically hard-killed (os._exit) at named
     mid-window micro-steps (TUNIX_SB_PREEMPT_AT -- the learner waits for
     that micro-step's snapshot to be DURABLE first, so the resume provably
     lands on that exact mid-accumulation key), restarted until completion.

Both arms append machine-readable evidence to a JSONL event log
(TUNIX_SB_EVENT_LOG; wandb-independent): one event per snapshot
(key/window/local/iter/denom), preemption, and resume. Afterwards a report
is generated that PROVES, from the events plus the per-attempt stdout logs:

  - each preemption landed mid-window (local > 0) at the requested iter,
    and the resume restored that exact key with the grad-accum buffer;
  - the standard-mode accounting invariant `denom == local` held on every
    snapshot (no double-counted and no lost micro-steps), including the
    first post-resume snapshot (local = preempt_local + 1);
  - every micro-step of every window of every global step was snapshotted
    exactly once (distinct-key coverage), and every expected apply fired;
  - per-apply losses and per-global-step training metrics are approximately
    equal across the arms (rollout sampling is stochastic, so expect the
    same scale/trend, not bit-equality -- the design contract is
    statistical equivalence).

Default geometry: batch=4 mini=4 micro=1 G=4 mu=2 max_steps=3
  -> k=4 micro-steps per window, 2 windows (applies) per global step,
     8 micro-steps per global step, 24 total; preempts at iters 6 and 11
     (window 2 local 2 of step 0, window 3 local 2 of step 1).

Usage (on the TPU VM, from the repo root, venv active):

  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_validate.py
  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_validate.py \
      --report_only validate_runs/<timestamp>
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

_DEMO = pathlib.Path(__file__).parent / "qwen3_grpo_demo.py"

_TRAIN_STEP_RE = re.compile(
    r"Train step (\d+) training loss: ([0-9.eE+-]+)"
)
# The demo's per-global-step summary, e.g.:
# [step 0] train_reward=0.000 ... loss=0.0001 grad_norm=0.0273 ... kl=0.0026
_STEP_SUMMARY_RE = re.compile(r"\[step (\d+)\] (.*)")
_KV_RE = re.compile(r"([a-z_]+)=([-+0-9.eE]+)")


def _run_arm(
    arm: str,
    args,
    workdir: pathlib.Path,
    preempt_at: set[int],
) -> None:
  """Runs one arm (restarting after preemptions) until true completion."""
  t0 = int(time.time())
  tag = f"validate-{arm}-{t0}"
  event_log = workdir / f"events_{arm}.jsonl"
  log_dir = workdir / f"logs_{arm}"
  log_dir.mkdir(parents=True, exist_ok=True)

  env = dict(os.environ)
  env.setdefault("ROLLOUT_ENGINE", "vanilla")
  env["TUNIX_BENCH_CKPT_TAG"] = tag
  env["TUNIX_BENCH_START_TIME"] = repr(time.time())
  env["TUNIX_SB_EVENT_LOG"] = str(event_log)
  env["PYTHONUNBUFFERED"] = "1"
  # The evidence channel for this harness is the event log + stdout; keep
  # wandb local so network/monotonic-step quirks cannot perturb the runs.
  env.setdefault("WANDB_MODE", "offline")
  env["WANDB_RUN_ID"] = tag
  done_file = workdir / f"done_{arm}"
  done_file.unlink(missing_ok=True)
  env["TUNIX_BENCH_DONE_FILE"] = str(done_file)

  cmd = [
      sys.executable,
      str(_DEMO),
      "--sub_batch_checkpointing",
      f"--batch_size={args.batch_size}",
      f"--mini_batch_size={args.mini_batch_size}",
      f"--train_micro_batch_size={args.train_micro_batch_size}",
      f"--num_generations={args.num_generations}",
      f"--num_iterations={args.num_iterations}",
      f"--max_steps={args.max_steps}",
      "--skip_eval",
      *args.demo_args,
  ]

  attempt = 0
  while True:
    fired = {
        e["iter_steps"]
        for e in _read_events(event_log)
        if e.get("event") == "preempt"
    }
    remaining = sorted(preempt_at - fired)
    if remaining:
      env["TUNIX_SB_PREEMPT_AT"] = ",".join(str(v) for v in remaining)
    else:
      env.pop("TUNIX_SB_PREEMPT_AT", None)
    print(
        f"[validate:{arm}] attempt {attempt}"
        f" (pending preempts: {remaining or 'none'}):"
        f" {' '.join(cmd)}",
        flush=True,
    )
    attempt_log = log_dir / f"attempt_{attempt:03d}.log"
    with open(attempt_log, "wb") as logf:
      proc = subprocess.Popen(
          cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
      )
      assert proc.stdout is not None
      for line in proc.stdout:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        logf.write(line)
      code = proc.wait()
    if code == 0 and done_file.exists():
      print(f"[validate:{arm}] complete after {attempt} restart(s).",
            flush=True)
      return
    attempt += 1
    if attempt > args.max_restarts:
      raise RuntimeError(
          f"[validate:{arm}] exceeded --max_restarts ({args.max_restarts});"
          f" last exit code {code}. See {log_dir}."
      )
    print(
        f"[validate:{arm}] exit code {code}; restarting in"
        f" {args.restart_backoff:.0f}s (TPU teardown after a hard kill"
        " takes ~60s -- relaunching early fails with vfio 'Device busy').",
        flush=True,
    )
    time.sleep(args.restart_backoff)


def _read_events(path: pathlib.Path) -> list[dict]:
  if not path.exists():
    return []
  events = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        try:
          events.append(json.loads(line))
        except json.JSONDecodeError:
          pass  # a torn write from a hard kill mid-line
  return events


def _parse_logs(log_dir: pathlib.Path):
  """Extracts per-apply losses and per-global-step summary metrics from all
  attempt logs. Later attempts overwrite earlier ones for the same step:
  a re-executed apply from a discarded (pre-crash) lineage is superseded by
  the surviving lineage's value, which is the one the run's final weights
  actually descend from."""
  apply_loss: dict[int, float] = {}
  step_metrics: dict[int, dict[str, float]] = {}
  for log in sorted(log_dir.glob("attempt_*.log")):
    text = log.read_text(errors="replace")
    for m in _TRAIN_STEP_RE.finditer(text):
      apply_loss[int(m.group(1))] = float(m.group(2))
    for m in _STEP_SUMMARY_RE.finditer(text):
      step = int(m.group(1))
      step_metrics[step] = {
          k: float(v) for k, v in _KV_RE.findall(m.group(2))
      }
  return apply_loss, step_metrics


def _dedup_last_by_key(snapshots: list[dict]) -> dict[int, dict]:
  out: dict[int, dict] = {}
  for e in snapshots:
    out[e["key"]] = e
  return out


def _write_report(args, workdir: pathlib.Path) -> bool:
  k = args.mini_batch_size // args.train_micro_batch_size
  applies_per_step = args.num_iterations * (
      args.batch_size // args.mini_batch_size
  )
  iters_per_step = applies_per_step * k
  total_iters = args.max_steps * iters_per_step
  total_applies = args.max_steps * applies_per_step

  lines: list[str] = []
  ok = True

  def check(name: str, passed: bool, detail: str) -> None:
    nonlocal ok
    ok = ok and passed
    lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

  lines.append("# Sub-batch checkpointing validation report")
  lines.append("")
  lines.append(
      f"Geometry: batch={args.batch_size} mini={args.mini_batch_size}"
      f" micro={args.train_micro_batch_size} G={args.num_generations}"
      f" mu={args.num_iterations} max_steps={args.max_steps}"
      f" -> k={k} micro-steps/window, {applies_per_step} applies/step,"
      f" {iters_per_step} micro-steps/step,"
      f" {total_iters} micro-steps and {total_applies} applies total."
  )
  lines.append("")

  arms = {}
  for arm in ("control", "preempt"):
    events = _read_events(workdir / f"events_{arm}.jsonl")
    snaps = [e for e in events if e.get("event") == "snapshot"]
    arms[arm] = {
        "events": events,
        "snapshots": snaps,
        "by_key": _dedup_last_by_key(snaps),
        "preempts": [e for e in events if e.get("event") == "preempt"],
        "resumes": [e for e in events if e.get("event") == "resume"],
    }
    arms[arm]["apply_loss"], arms[arm]["step_metrics"] = _parse_logs(
        workdir / f"logs_{arm}"
    )

  # --- 1. Preemption / resume pairing (preempt arm) -----------------------
  lines.append("## 1. Preemptions and resumes (preempt arm)")
  lines.append("")
  pre = arms["preempt"]["preempts"]
  res = arms["preempt"]["resumes"]
  check(
      "preemption count",
      len(pre) == len(args.preempt_set),
      f"requested at iters {sorted(args.preempt_set)}, fired {len(pre)}"
      f" at {[e['iter_steps'] for e in pre]}",
  )
  for e in pre:
    check(
        f"preempt@iter{e['iter_steps']} was mid-window",
        e["iter_steps"] % k != 0,
        f"key={e['key']} window={e['window']} local={e['local']}"
        f" iter%k={e['iter_steps'] % k} (nonzero means inside gradient"
        " accumulation, not at an apply boundary)",
    )
  # Pair each preempt with the chronologically next resume.
  paired = 0
  for e in pre:
    match = next(
        (
            r
            for r in res
            if r["ts"] > e["ts"] and r["iter_steps"] == e["iter_steps"]
        ),
        None,
    )
    if match is not None:
      paired += 1
      lines.append(
          f"  - preempted after durable key {e['key']} (window"
          f" {e['window']}, local {e['local']}, iter {e['iter_steps']})"
          f" -> resumed at iter {match['iter_steps']}"
          f" with grad-accum buffer restored ="
          f" {match.get('had_training_state')}"
      )
  check(
      "every preemption resumed from its exact snapshot",
      paired == len(pre),
      f"{paired}/{len(pre)} preempt->resume pairs matched on iter_steps"
      " (the durable-wait in the injector guarantees this)",
  )
  for r in res:
    check(
        f"resume@iter{r['iter_steps']} restored the buffer",
        bool(r.get("had_training_state")),
        "training_state (partial gradient + denom) present in the restored"
        " snapshot",
    )
  lines.append("")

  # --- 2. Accounting invariant: denom == iter_steps % k -------------------
  lines.append(
      "## 2. Micro-step accounting (denom == iter_steps % k on every"
      " snapshot)"
  )
  lines.append("")
  lines.append(
      "In standard mode the accumulator's denominator counts micro-steps"
      " since the last apply, so on every snapshot it must equal"
      " iter_steps mod k: exactly 0 at apply boundaries, counting up in"
      " between. A double-trained micro-step would overshoot, a lost one"
      " undershoot -- including across a preemption/resume."
  )
  lines.append("")
  for arm in ("control", "preempt"):
    snaps = arms[arm]["snapshots"]
    bad = [
        e
        for e in snaps
        if e.get("denom") is not None
        and int(e["denom"]) != e["iter_steps"] % k
    ]
    check(
        f"{arm}: denom == iter_steps % k on all snapshots",
        not bad and bool(snaps),
        f"{len(snaps)} snapshots checked;"
        + (
            " no violations"
            if not bad
            else " VIOLATIONS (key, denom, iter):"
            f" {[(e['key'], e['denom'], e['iter_steps']) for e in bad]}"
        ),
    )
  # Post-resume continuity: first snapshot after each resume continues the
  # window from local+1 with denom == local+1.
  snaps_p = arms["preempt"]["snapshots"]
  for r in arms["preempt"]["resumes"]:
    after = [e for e in snaps_p if e["ts"] > r["ts"]]
    if not after:
      check(f"continuity after resume@iter{r['iter_steps']}", False,
            "no snapshot events after the resume")
      continue
    first = min(after, key=lambda e: e["ts"])
    r_key = r.get("key")
    r_local = (r_key % 1_000_000) if r_key is not None else None
    passed = (
        first["iter_steps"] == r["iter_steps"] + 1
        and (r_local is None or first["local"] == r_local + 1)
        and (
            first.get("denom") is None
            or int(first["denom"]) == first["iter_steps"] % k
        )
    )
    check(
        f"continuity after resume@iter{r['iter_steps']}",
        passed,
        f"first post-resume snapshot: key={first['key']}"
        f" local={first['local']} denom={first.get('denom')}"
        f" iter={first['iter_steps']} (expected local"
        f" {r_local + 1 if r_local is not None else '?'} with matching"
        " denom: the restored buffer was extended, not restarted)",
    )
  lines.append("")

  # --- 3. Coverage: every micro-step, every apply -------------------------
  lines.append("## 3. Coverage")
  lines.append("")
  for arm in ("control", "preempt"):
    keys = arms[arm]["by_key"]
    check(
        f"{arm}: every micro-step snapshotted",
        len(keys) == total_iters,
        f"{len(keys)}/{total_iters} distinct snapshot keys"
        + (
            ""
            if len(keys) == total_iters
            else f" (windows seen: {sorted({k // 1_000_000 for k in keys})})"
        ),
    )
    losses = arms[arm]["apply_loss"]
    check(
        f"{arm}: every apply executed",
        len(losses) == total_applies
        and set(losses) == set(range(1, total_applies + 1)),
        f"applies seen: {sorted(losses)} (expected 1..{total_applies})",
    )
  lines.append("")

  # --- 4. Metric comparison across arms -----------------------------------
  lines.append("## 4. Training metrics: control vs preempted+resumed")
  lines.append("")
  lines.append("Per-apply loss (last surviving lineage per step):")
  lines.append("")
  lines.append("| apply | control | preempt | abs diff |")
  lines.append("|---|---|---|---|")
  c_loss, p_loss = arms["control"]["apply_loss"], arms["preempt"][
      "apply_loss"
  ]
  diffs = []
  for step in sorted(set(c_loss) | set(p_loss)):
    c, p = c_loss.get(step), p_loss.get(step)
    d = abs(c - p) if c is not None and p is not None else None
    if d is not None:
      diffs.append(d)
    lines.append(
        f"| {step} | {c if c is not None else '-'} |"
        f" {p if p is not None else '-'} |"
        f" {f'{d:.3e}' if d is not None else '-'} |"
    )
  lines.append("")
  if diffs:
    lines.append(
        f"Mean |loss diff| = {sum(diffs) / len(diffs):.3e}, max ="
        f" {max(diffs):.3e}. Rollouts are sampled, so exact equality is"
        " not expected (the design contract is statistical equivalence);"
        " the arms should agree in scale and trend."
    )
  lines.append("")
  lines.append("Per-global-step summary metrics:")
  lines.append("")
  metric_names = sorted(
      {
          m
          for arm in arms.values()
          for sm in arm["step_metrics"].values()
          for m in sm
      }
  )
  for name in metric_names:
    row_c = [
        arms["control"]["step_metrics"].get(s, {}).get(name)
        for s in range(args.max_steps)
    ]
    row_p = [
        arms["preempt"]["step_metrics"].get(s, {}).get(name)
        for s in range(args.max_steps)
    ]
    lines.append(f"- `{name}`: control={row_c} preempt={row_p}")
  lines.append("")

  lines.append("## Verdict")
  lines.append("")
  lines.append(
      "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED (see above)"
  )
  report = "\n".join(lines) + "\n"
  out = workdir / "report.md"
  out.write_text(report)
  print("\n" + report)
  print(f"[validate] report written to {out}", flush=True)
  return ok


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--mini_batch_size", type=int, default=4)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--num_generations", type=int, default=4)
  parser.add_argument("--num_iterations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=3)
  parser.add_argument(
      "--preempt_at", default="6,11",
      help="Comma-separated trainer iter_steps at which the preempt arm"
      " hard-kills itself, AFTER that micro-step's snapshot is durable."
      " Pick values that are NOT multiples of k (= mini/micro) so the kill"
      " lands mid-accumulation.",
  )
  parser.add_argument("--restart_backoff", type=float, default=75.0)
  parser.add_argument("--max_restarts", type=int, default=10)
  parser.add_argument(
      "--report_only", default=None,
      help="Skip running; regenerate the report from an existing workdir.",
  )
  parser.add_argument(
      "demo_args", nargs=argparse.REMAINDER,
      help="Extra args forwarded verbatim to qwen3_grpo_demo.py.",
  )
  args = parser.parse_args()
  if args.demo_args and args.demo_args[0] == "--":
    args.demo_args = args.demo_args[1:]

  k = args.mini_batch_size // args.train_micro_batch_size
  args.preempt_set = {
      int(v) for v in args.preempt_at.split(",") if v.strip()
  }
  if k < 2 or args.num_iterations < 2 or args.num_generations < 2:
    raise SystemExit(
        "validation requires num_generations, num_iterations and"
        f" k = mini/micro all > 1 (got G={args.num_generations},"
        f" mu={args.num_iterations}, k={k})."
    )
  boundary = [v for v in args.preempt_set if v % k == 0]
  if boundary:
    raise SystemExit(
        f"--preempt_at values {boundary} are apply boundaries (multiples"
        f" of k={k}); this harness exists to test MID-window preemption."
    )

  if args.report_only:
    workdir = pathlib.Path(args.report_only)
    return 0 if _write_report(args, workdir) else 1

  workdir = pathlib.Path("validate_runs") / str(int(time.time()))
  workdir.mkdir(parents=True, exist_ok=True)
  print(f"[validate] workdir: {workdir}", flush=True)

  free_gb = shutil.disk_usage(".").free / 1e9
  if free_gb < 40:
    print(
        f"[validate] WARNING: only {free_gb:.0f}GB free. Each k>1 window"
        " writes a param-sized grad buffer per mid-window snapshot on top"
        " of the per-apply trainer checkpoint; consider freeing old"
        " checkpoint tags first.",
        flush=True,
    )

  # Preempt arm first: it is the arm under test, so its failures surface
  # before the control arm's hour is spent.
  _run_arm("preempt", args, workdir, preempt_at=args.preempt_set)
  _run_arm("control", args, workdir, preempt_at=set())
  return 0 if _write_report(args, workdir) else 1


if __name__ == "__main__":
  sys.exit(main())
