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

"""Sub-batch vs stock checkpointing under identical preemption (TEMPORARY).

Runs `qwen3_grpo_demo.py` under `sub_batch_bench_hooks` in up to four arms
with identical geometry, differing only in the feature under test:

  1. SUBBATCH arm: --sub_batch_checkpointing on; hard-killed (os._exit) once
     training passes each named trainer iteration (TUNIX_SB_PREEMPT_AT),
     after its checkpoints and snapshots are durable, and restarted until it
     completes.
  2. STOCK arm: the feature off, killed at the SAME iterations.
  3. SUBBATCH_CLEAN arm: the feature on, never killed. Isolates the
     snapshot overhead from the recovery behavior.
  4. STOCK_CLEAN arm: the feature off, never killed. The undisturbed
     reference: wall-clock floor and ground-truth training trajectory.

  The clean pair (skippable via --skip_clean_arms) turns the wall-clock
  comparison into a proper 2x2: (subbatch_clean - stock_clean) is the
  feature's overhead with no preemption, and (preempted - clean) within
  each mode is what the SAME kills cost that mode.

Both arms checkpoint the trainer at every global-step boundary and nowhere
else, so a restart in either arm resumes at the start of the step it
interrupted and retrains that step's micro-steps so far. The stock arm also
regenerates the step's rollouts; the sub-batch arm restores the rollouts its
snapshots hold and generates only the rest. That is the cost the feature
removes.

  PREEMPTION MODES: --preempt_at names exact trainer iterations
  (deterministic, identical across both preempted arms). Alternatively
  --sb_chaos_prob kills RANDOMLY before an update_actor call with that
  probability (e.g. 0.004 = 0.4%); when it is set and --preempt_at is not
  explicitly given, it REPLACES the deterministic schedule rather than
  stacking on top. Chaos kill points differ per arm (each process rolls its
  own dice), so that comparison is statistical rather than paired. Chaos
  kills emit no `preempt` events and do not wait for checkpoints, so a
  restart after one can resume from an earlier step.

Each fired kill is subtracted from the pending schedule before relaunch,
so no arm is ever killed twice at the same iteration.

Each arm appends JSONL events (TUNIX_SB_EVENT_LOG; wandb-independent): a
`start` per attempt, a `train` per update_actor call, a `rollout` per
generated group and a `preempt` per deterministic kill; sub-batch arms add
`snapshot` (one per background save) and `resume`. The report checks, per
arm:

  - every requested kill fired once, and every kill was followed by a
    restart (that restored its snapshots, in sub-batch arms);
  - every optimizer update ran once, plus once more per restart that resumed
    before it (several updates per step);
  - every micro-step trained once, plus once more per restart that resumed
    before it (without sequence packing);
  - every rollout group was generated, and sub-batch arms never regenerated
    a restored one;

and reports regenerated rollouts, retrained micro-steps, snapshot save times
and wall-clock.

Training metrics go to REAL wandb (whatever mode the environment
configures): each restart attempt is its own wandb run, all grouped per
arm via WANDB_RUN_GROUP. The checks above never depend on wandb.

SEQUENCE PACKING: pass --max_seq_token_per_tpu. Packs per update vary, so
kill points are trainer iterations as each attempt numbers them, and the
micro-step accounting is skipped.

Because both arms restart a whole step, the comparison is equal-correctness
for any geometry (mini < batch, mu > 1); the stock arm no longer resumes
mid-step and skips data (b/483779605).

Default geometry: batch=4 mini=4 micro=1 G=4 mu=1 max_steps=6
  -> k=4 micro-steps/window, 1 apply/step, 24 micro-steps total; kills at
     iters 6,10,14,18,22 (local 2 of steps 1..5 -- one mid-window kill in
     each of five distinct global steps).

Usage (on the TPU VM, from the repo root, venv active):

  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_benchmark.py
  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_benchmark.py \\
      --report_only validate_runs/<timestamp>
"""

import argparse
import collections
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

_DEMO = pathlib.Path(__file__).parent / "qwen3_grpo_demo.py"

# Arm -> (sub-batch checkpointing on, killed), in run order: the arm under
# test first, so its failures surface before the other arms' hours are spent.
_ARMS = {
    "subbatch": (True, True),
    "stock": (False, True),
    "subbatch_clean": (True, False),
    "stock_clean": (False, False),
}


def _run_arm(
    arm: str,
    args,
    workdir: pathlib.Path,
    preempt_at: set[int],
    chaos: float = 0.0,
) -> None:
  """Runs one arm (restarting after preemptions) until true completion."""
  sub_batch, _ = _ARMS[arm]
  t0 = int(time.time())
  t_wall = time.time()
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
  # Each restart attempt gets its OWN wandb run, all grouped per arm under
  # WANDB_RUN_GROUP: fresh runs per attempt are sturdier than cross-process
  # run resumption, and the wandb UI compares arms by group.
  env["WANDB_RUN_GROUP"] = tag
  done_file = workdir / f"done_{arm}"
  done_file.unlink(missing_ok=True)
  env["TUNIX_BENCH_DONE_FILE"] = str(done_file)

  cmd = [
      sys.executable,
      str(_DEMO),
      "--sub_batch_checkpointing" if sub_batch else "--bench_baseline",
      f"--batch_size={args.batch_size}",
      f"--mini_batch_size={args.mini_batch_size}",
      f"--train_micro_batch_size={args.train_micro_batch_size}",
      f"--num_generations={args.num_generations}",
      f"--num_iterations={args.num_iterations}",
      f"--max_steps={args.max_steps}",
      "--skip_eval",
      *(
          [f"--max_seq_token_per_tpu={args.max_seq_token_per_tpu}"]
          if args.max_seq_token_per_tpu
          else []
      ),
      *([f"--sb_chaos_prob={chaos}"] if chaos > 0 else []),
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
    # Fresh wandb run per attempt, named so the group reads chronologically.
    env["WANDB_RUN_ID"] = f"{tag}-a{attempt:03d}"
    env["WANDB_NAME"] = f"{tag}-a{attempt:03d}"
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
      (workdir / f"timing_{arm}.json").write_text(
          json.dumps({
              "tag": tag,
              "attempts": attempt + 1,
              "restarts": attempt,
              "wall_s": time.time() - t_wall,
          })
      )
      print(
          f"[validate:{arm}] complete after {attempt} restart(s).", flush=True
      )
      return
    attempt += 1
    if args.max_restarts is not None and attempt > args.max_restarts:
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


def _arm_stats(events: list[dict], args) -> dict:
  """Derives an arm's accounting from its events."""
  by_type = collections.defaultdict(list)
  for e in sorted(events, key=lambda e: e["ts"]):
    by_type[e["event"]].append(e)
  starts, trains = by_type["start"], by_type["train"]
  restarts = starts[1:]

  updates = collections.Counter()
  update_loss = {}
  for e in trains:
    performed = range(e["train_before"] + 1, e["train_steps"] + 1)
    updates.update(performed)
    if performed and e.get("loss") is not None:
      # The trainer's loss buffer holds the call's last update; a later
      # attempt's rerun of that update supersedes it.
      update_loss[performed[-1]] = e["loss"]

  # Each restart resumes at the start b of the global step it interrupted,
  # after the attempt before it reached t, so it redoes (b, t]: micro-steps
  # and, with several updates per step, updates.
  trained = collections.Counter()
  for e in trains:
    trained.update(range(e["iter_before"] + 1, e["iter_steps"] + 1))
  expected_trained = collections.Counter(range(1, args.total_iters + 1))
  expected_updates = collections.Counter(range(1, args.total_updates + 1))
  retrained = []
  reran_updates = []
  for previous, s in zip(starts, restarts):
    attempt = [e for e in trains if previous["ts"] < e["ts"] < s["ts"]]
    b = s["iter_steps"]
    t = max((e["iter_steps"] for e in attempt), default=b)
    retrained.append(max(0, t - b))
    expected_trained.update(range(b + 1, t + 1))
    b = s["train_steps"]
    t = max((e["train_steps"] for e in attempt), default=b)
    reran_updates.append(max(0, t - b))
    expected_updates.update(range(b + 1, t + 1))

  # Restored groups must not be generated again by the attempt restoring
  # them, which ends at the next start.
  regenerated_restored = []
  for r in by_type["resume"]:
    end = min((s["ts"] for s in starts if s["ts"] > r["ts"]), default=None)
    restored = set(r["groups"])
    regenerated_restored.extend(
        e["group_id"]
        for e in by_type["rollout"]
        if e["ts"] > r["ts"]
        and (end is None or e["ts"] < end)
        and e["group_id"] in restored
    )

  return {
      "by_type": by_type,
      "restarts": restarts,
      "updates": updates,
      "expected_updates": expected_updates,
      "reran_updates": reran_updates,
      "update_loss": update_loss,
      "trained": trained,
      "expected_trained": expected_trained,
      "retrained": retrained,
      "rollouts": collections.Counter(e["group_id"] for e in by_type["rollout"]),
      "regenerated_restored": regenerated_restored,
  }


def _write_report(args, workdir: pathlib.Path) -> bool:
  lines: list[str] = []
  ok = True

  def check(name: str, passed: bool, detail: str) -> None:
    nonlocal ok
    ok = ok and passed
    lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

  lines.append("# Sub-batch checkpointing benchmark report")
  lines.append("")
  geometry = (
      f"Geometry: batch={args.batch_size} mini={args.mini_batch_size}"
      f" G={args.num_generations} mu={args.num_iterations}"
      f" max_steps={args.max_steps} -> {args.updates_per_step} updates/step,"
      f" {args.total_updates} updates total"
  )
  if args.packed:
    geometry += (
        f"; sequence packing, max_seq_token_per_tpu="
        f"{args.max_seq_token_per_tpu}."
    )
  else:
    geometry += (
        f"; micro={args.train_micro_batch_size} -> k={args.k}"
        f" micro-steps/update, {args.total_iters} micro-steps total."
    )
  lines.append(geometry)
  lines.append("")

  arms = {}
  for arm in _ARMS:
    events = _read_events(workdir / f"events_{arm}.jsonl")
    timing_file = workdir / f"timing_{arm}.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else None
    if events or timing:
      arms[arm] = {
          **_arm_stats(events, args),
          "timing": timing,
      }

  # --- 1. Preemptions and restarts --------------------------------------
  lines.append("## 1. Preemptions and restarts")
  lines.append("")
  if args.sb_chaos_prob > 0:
    lines.append(
        f"Chaos mode: random kills at p={args.sb_chaos_prob} per update_actor"
        " call (per-arm dice, so kill points differ across arms). Chaos kills"
        " emit no `preempt` events; their restarts still count below."
    )
    lines.append("")
  for arm, stats in arms.items():
    sub_batch, killed = _ARMS[arm]
    preempts = stats["by_type"]["preempt"]
    restarts = stats["restarts"]
    expected_kills = len(args.preempt_set) if killed else 0
    check(
        f"{arm}: deterministic preemptions",
        len(preempts) == expected_kills,
        f"requested {expected_kills}, fired {len(preempts)} at trainer"
        f" iterations {sorted(e['iter_steps'] for e in preempts)}",
    )
    check(
        f"{arm}: every kill restarted",
        all(any(s["ts"] > p["ts"] for s in restarts) for p in preempts),
        f"{len(restarts)} restart(s) after {len(preempts)} kill(s)",
    )
    if sub_batch:
      resumes = stats["by_type"]["resume"]
      check(
          f"{arm}: every restart restored its snapshots",
          len(resumes) == len(restarts),
          f"{len(resumes)} restore(s) for {len(restarts)} restart(s);"
          f" restored groups per restore: {[len(r['groups']) for r in resumes]}",
      )
    if not killed and restarts:
      lines.append(
          f"  - NOTE: clean arm restarted {len(restarts)} time(s) (unplanned"
          " crash); its wall-clock is no longer a pure no-preemption"
          " reference."
      )
  lines.append("")

  # --- 2. Optimizer updates ---------------------------------------------
  lines.append("## 2. Optimizer updates")
  lines.append("")
  lines.append(
      "Each update runs once, plus once more per restart that resumed before"
      " it: a restart resumes at the start of the global step it interrupted,"
      " so it reruns the step's updates that ran before the kill (only"
      " possible with several updates per step)."
  )
  lines.append("")
  for arm, stats in arms.items():
    updates, expected = stats["updates"], stats["expected_updates"]
    mismatched = {
        u: (updates[u], expected[u])
        for u in sorted(set(updates) | set(expected))
        if updates[u] != expected[u]
    }
    reran = stats["reran_updates"]
    check(
        f"{arm}: update multiplicity",
        not mismatched,
        f"{sum(reran)} update(s) rerun over {len(reran)} restart(s) ({reran})"
        + (
            ""
            if not mismatched
            else f"; MISMATCHES (update: ran, expected): {mismatched}"
        ),
    )
  lines.append("")

  # --- 3. Micro-steps -----------------------------------------------------
  lines.append("## 3. Micro-step accounting")
  lines.append("")
  if args.packed:
    lines.append(
        "Skipped with sequence packing: a restart renumbers trainer"
        " iterations from the step's start, and packs per update vary."
    )
  else:
    lines.append(
        "Every micro-step trains once, plus once more per restart that"
        " resumed before it: a restart that resumes at micro-step b, the"
        " start of the step it interrupted, after its attempt reached t"
        " retrains (b, t]. Any other multiplicity means lost or double-trained"
        " data."
    )
    lines.append("")
    for arm, stats in arms.items():
      trained, expected = stats["trained"], stats["expected_trained"]
      mismatched = {
          i: (trained[i], expected[i])
          for i in sorted(set(trained) | set(expected))
          if trained[i] != expected[i]
      }
      retrained = stats["retrained"]
      check(
          f"{arm}: micro-step multiplicity",
          not mismatched,
          f"{sum(retrained)} micro-step(s) retrained over"
          f" {len(retrained)} restart(s) ({retrained})"
          + (
              ""
              if not mismatched
              else f"; MISMATCHES (iter: trained, expected): {mismatched}"
          ),
      )
  lines.append("")

  # --- 4. Rollouts ----------------------------------------------------------
  lines.append("## 4. Rollouts")
  lines.append("")
  total_groups = args.max_steps * args.batch_size
  for arm, stats in arms.items():
    rollouts = stats["rollouts"]
    missing = total_groups - len(set(rollouts) & set(range(total_groups)))
    regenerated = sum(n - 1 for n in rollouts.values())
    check(
        f"{arm}: every rollout group generated",
        missing == 0,
        f"{len(rollouts)} distinct groups of {total_groups},"
        f" {regenerated} regenerated",
    )
    if _ARMS[arm][0]:
      bad = stats["regenerated_restored"]
      check(
          f"{arm}: no restored group regenerated",
          not bad,
          f"regenerated after their restore: {bad or 'none'}",
      )
  lines.append("")

  # --- 5. Training metrics ------------------------------------------------
  lines.append("## 5. Training metrics")
  lines.append("")
  lines.append(
      "Metrics are logged to wandb: each restart attempt is its own run,"
      " grouped per arm. Compare the arms by their run groups in the wandb"
      " UI (filter/group by `group`):"
  )
  lines.append("")
  arm_names = list(arms)
  for arm in arm_names:
    tag = (arms[arm]["timing"] or {}).get("tag", f"validate-{arm}-<t0>")
    lines.append(f"- `{arm}` arm group: `{tag}`")
  lines.append("")
  lines.append("Per-update loss, read from the trainer as each update ran:")
  lines.append("")
  lines.append("| update | " + " | ".join(arm_names) + " | spread |")
  lines.append("|" + "---|" * (len(arm_names) + 2))
  all_updates = sorted({u for arm in arm_names for u in arms[arm]["update_loss"]})
  for update in all_updates:
    vals = [arms[arm]["update_loss"].get(update) for arm in arm_names]
    known = [v for v in vals if v is not None]
    spread = (max(known) - min(known)) if len(known) > 1 else None
    lines.append(
        f"| {update} | "
        + " | ".join(f"{v:.6g}" if v is not None else "-" for v in vals)
        + f" | {f'{spread:.3e}' if spread is not None else '-'} |"
    )
  lines.append("")

  # --- 6. Efficiency: the 2x2 (mode x preemption) cost picture ------------
  lines.append("## 6. Efficiency (mode x preemption)")
  lines.append("")
  for arm in arm_names:
    stats = arms[arm]
    t = stats["timing"]
    rollouts = stats["rollouts"]
    detail = (
        f"{t['attempts']} attempt(s), wall-clock {t['wall_s']:.0f}s"
        if t
        else "no timing sidecar (incomplete run?)"
    )
    detail += (
        f"; {sum(n - 1 for n in rollouts.values())} regenerated rollout"
        " group(s)"
    )
    if not args.packed:
      detail += f", {sum(stats['retrained'])} retrained micro-step(s)"
    saves = [e["save_s"] for e in stats["by_type"]["snapshot"]]
    if saves:
      waited = [
          e["save_s"] for e in stats["by_type"]["snapshot"] if e["wait"]
      ]
      detail += (
          f"; {len(saves)} snapshots ({len(waited)} waited), save time"
          f" total {sum(saves):.1f}s, mean {sum(saves) / len(saves):.3f}s,"
          f" max {max(saves):.3f}s"
      )
    lines.append(f"- {arm}: {detail}")
  wall = {
      arm: arms[arm]["timing"]["wall_s"]
      for arm in arm_names
      if arms[arm]["timing"]
  }
  if {"subbatch_clean", "stock_clean"} <= set(wall):
    lines.append(
        "- feature overhead with NO preemption: subbatch_clean -"
        f" stock_clean = {wall['subbatch_clean'] - wall['stock_clean']:+.0f}s"
    )
  if {"subbatch", "subbatch_clean"} <= set(wall):
    lines.append(
        "- preemption cost under sub-batch: subbatch - subbatch_clean ="
        f" {wall['subbatch'] - wall['subbatch_clean']:+.0f}s"
    )
  if {"stock", "stock_clean"} <= set(wall):
    lines.append(
        "- preemption cost under stock: stock - stock_clean ="
        f" {wall['stock'] - wall['stock_clean']:+.0f}s"
    )
  lines.append("")
  tags = {
      arm: arms[arm]["timing"]["tag"]
      for arm in arm_names
      if arms[arm]["timing"]
  }
  ref = "stock_clean" if "stock_clean" in tags else "stock"
  if ref in tags and len(tags) > 1:
    lines.append(
        f"Final-weights equivalence vs the `{ref}` reference (run beside a"
        " live job, CPU-only):"
    )
    lines.append("")
    lines.append("```")
    for arm in arm_names:
      if arm == ref or arm not in tags:
        continue
      lines.append("python3 examples/math_gsm8k/sub_batch_diff.py \\")
      lines.append(
          f"    <CHECKPOINT_ROOT>/{tags[arm]}/actor/{args.total_updates} \\"
      )
      lines.append(f"    <CHECKPOINT_ROOT>/{tags[ref]}/actor/{args.total_updates}")
    lines.append("```")
    lines.append("")

  lines.append("## Verdict")
  lines.append("")
  lines.append("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED (see above)")
  report = "\n".join(lines) + "\n"
  out = workdir / "report.md"
  out.write_text(report)
  print("\n" + report)
  print(f"[validate] report written to {out}", flush=True)
  return ok


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--mini_batch_size", type=int, default=4)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--num_generations", type=int, default=4)
  parser.add_argument("--num_iterations", type=int, default=1)
  parser.add_argument("--max_steps", type=int, default=6)
  parser.add_argument(
      "--max_seq_token_per_tpu",
      type=int,
      default=None,
      help="Token budget per packed row; runs every arm with sequence packing.",
  )
  parser.add_argument(
      "--preempt_at", default=None,
      help="Comma-separated trainer iterations after which both PREEMPTED"
      " arms hard-kill themselves (each arm once its recovery points are"
      " durable). Without packing, pick values that are NOT multiples of k"
      " (= mini/micro) so the kills land mid-accumulation, spread across"
      " distinct global steps. Default: 6,10,14,18,22 -- unless"
      " --sb_chaos_prob is set, in which case the default becomes EMPTY"
      " (chaos replaces the schedule); pass both explicitly to stack them.",
  )
  parser.add_argument(
      "--sb_chaos_prob", type=float, default=0.0,
      help="Probability of a RANDOM hard kill before each update_actor call"
      " in the two PREEMPTED arms (e.g. 0.004 = 0.4%%); the clean arms never"
      " receive it. When set without an explicit --preempt_at it REPLACES"
      " the deterministic schedule. Random kill points differ per arm (each"
      " process rolls its own dice), so the comparison becomes statistical"
      " rather than paired.",
  )
  parser.add_argument(
      "--skip_clean_arms", action="store_true",
      help="Run only the two preempted arms (the original two-arm form),"
      " skipping the never-preempted subbatch_clean/stock_clean baselines"
      " and halving the wall-clock. The report adapts automatically.",
  )
  parser.add_argument("--restart_backoff", type=float, default=75.0)
  parser.add_argument(
      "--max_restarts",
      type=int,
      default=None,
      help="Maximum restarts per arm. Default: unlimited, until the run"
      " completes.",
  )
  parser.add_argument(
      "--report_only", default=None,
      help="Skip running; regenerate the report from an existing workdir.",
  )
  parser.add_argument(
      "demo_args", nargs=argparse.REMAINDER,
      help="Extra args forwarded verbatim to qwen3_grpo_demo.py.",
  )
  args, unknown_args = parser.parse_known_args()
  # Flags this parser does not know are the demo's too.
  args.demo_args = [
      arg for arg in unknown_args + args.demo_args if arg != "--"
  ]

  args.packed = args.max_seq_token_per_tpu is not None
  args.k = args.mini_batch_size // args.train_micro_batch_size
  args.updates_per_step = args.num_iterations * (
      args.batch_size // args.mini_batch_size
  )
  args.total_updates = args.max_steps * args.updates_per_step
  args.total_iters = args.total_updates * args.k
  if args.preempt_at is None:
    # Chaos replaces the deterministic schedule unless the caller stacks
    # them explicitly; with neither knob the default schedule applies.
    args.preempt_at = "" if args.sb_chaos_prob > 0 else "6,10,14,18,22"
  args.preempt_set = {
      int(v) for v in args.preempt_at.split(",") if v.strip()
  }
  if not args.preempt_set and args.sb_chaos_prob == 0:
    print(
        "[validate] WARNING: no --preempt_at and no --sb_chaos_prob -- the"
        " 'preempted' arms will never be killed and duplicate the clean"
        " ones.",
        flush=True,
    )
  if args.num_generations < 2:
    raise SystemExit("grouped generation requires --num_generations > 1.")
  if not args.packed and args.k < 2:
    raise SystemExit(
        "without sequence packing, the comparison requires gradient"
        f" accumulation: k = mini/micro > 1 (got k={args.k})."
    )
  if not args.packed:
    step_iters = args.k * args.updates_per_step
    boundary = [v for v in args.preempt_set if v % step_iters == 0]
    if boundary:
      print(
          f"[validate] NOTE: --preempt_at values {boundary} land on"
          f" global-step boundaries (multiples of {step_iters} micro-steps);"
          " those kills follow a durable step and retrain nothing.",
          flush=True,
      )
    beyond = [v for v in args.preempt_set if v > args.total_iters]
    if beyond:
      raise SystemExit(
          f"--preempt_at values {beyond} lie beyond the run"
          f" ({args.total_iters} micro-steps); they would never fire."
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
        f"[validate] WARNING: only {free_gb:.0f}GB free. Every arm writes a"
        " trainer checkpoint per global step; consider freeing old checkpoint"
        " tags first.",
        flush=True,
    )

  # Both preempted arms get the SAME kill schedule (and the same chaos
  # probability, each rolling its own dice) -- the delta between them is
  # the feature. The clean pair never receives either kill mechanism.
  for arm, (_, killed) in _ARMS.items():
    if not killed and args.skip_clean_arms:
      continue
    _run_arm(
        arm,
        args,
        workdir,
        preempt_at=args.preempt_set if killed else set(),
        chaos=args.sb_chaos_prob if killed else 0.0,
    )
  return 0 if _write_report(args, workdir) else 1


if __name__ == "__main__":
  sys.exit(main())
