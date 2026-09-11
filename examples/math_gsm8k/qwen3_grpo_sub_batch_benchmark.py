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

Runs `qwen3_grpo_demo.py` twice with identical geometry and an IDENTICAL
deterministic preemption schedule, differing only in the feature under
test:

  1. SUBBATCH arm: --sub_batch_checkpointing on; hard-killed (os._exit) at
     the named mid-window micro-steps (TUNIX_SB_PREEMPT_AT -- the learner
     waits for that micro-step's snapshot to be DURABLE first), restarted
     until completion. Resumes on the exact mid-accumulation key.
  2. STOCK arm: the feature off; hard-killed at the SAME micro-steps (the
     learner's stock-baseline hook waits for the actor checkpoint queue to
     be durable first), restarted until completion. Resumes at the last
     apply boundary and RE-EXECUTES the interrupted window's rollouts and
     micro-steps -- that re-executed work is the cost the feature removes.
  3. SUBBATCH_CLEAN arm: the feature on, never killed. Isolates the pure
     per-micro-step snapshot overhead from the recovery behavior.
  4. STOCK_CLEAN arm: the feature off, never killed. The undisturbed
     reference: wall-clock floor and ground-truth training trajectory.

  The clean pair (skippable via --skip_clean_arms) turns the wall-clock
  comparison into a proper 2x2: (subbatch_clean - stock_clean) is the
  feature's overhead with no preemption, and (preempted - clean) within
  each mode is what the SAME kills cost that mode.

  PREEMPTION MODES: --preempt_at names exact micro-steps (deterministic,
  identical across both preempted arms). Alternatively --sb_chaos_prob
  kills RANDOMLY with that per-micro-batch probability (e.g. 0.004 = 0.4%);
  when it is set and --preempt_at is not explicitly given, it REPLACES the
  deterministic schedule rather than stacking on top. Chaos kill points
  differ per arm (each process rolls its own dice), so that comparison is
  statistical rather than paired, and chaos kills emit no `preempt` events
  -- restarts surface as `resume` events, which every accounting check
  below is built to handle.

Each fired kill is subtracted from the pending schedule before relaunch,
so no arm is ever killed twice at the same micro-step -- essential for the
stock arm, which legitimately re-executes the killed iter after resuming.

Both arms append machine-readable evidence to a JSONL event log
(TUNIX_SB_EVENT_LOG; wandb-independent): the subbatch arm one event per
snapshot (key/window/local/iter/denom), the stock arm one `stock_iter`
event per training call, plus preempt/resume events in both. The report
then PROVES, from the events plus the per-attempt stdout logs:

  - both arms fired every requested kill exactly once, mid-window;
  - subbatch: exact-key resume with the grad-accum buffer, the accounting
    invariant `denom == (iter % k) x unit` on every snapshot, and
    every micro-step trained EXACTLY once (distinct-key coverage);
  - stock: resume at the expected apply boundary, every micro-step trained
    AT LEAST once, and the re-executed multiplicity exactly matching the
    kill->resume gaps (the measured cost);
  - every apply executed in both arms.

Training metrics go to REAL wandb (whatever mode the environment
configures): each restart attempt is its own wandb run, all grouped per
arm via WANDB_RUN_GROUP, so the arms are compared in the wandb UI rather
than by log parsing. The correctness guarantees above never depend on
wandb availability.

FAIR-GEOMETRY REQUIREMENT: the stock resume path is only CORRECT when
mini_batch_size == batch_size and num_iterations == 1 (one apply per
global step, so the per-apply checkpoint stamp is truthful). With richer
geometry the stock arm silently skips the interrupted step's remaining
data on resume (b/483779605) -- the harness then still runs, as a
demonstration of that data loss, but the comparison is no longer
equal-correctness.

Default geometry: batch=4 mini=4 micro=1 G=4 mu=1 max_steps=6
  -> k=4 micro-steps/window, 1 apply/step, 24 micro-steps total; kills at
     iters 6,10,14,18,22 (local 2 of steps 1..5 -- one mid-window kill in
     each of five distinct global steps).

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
    sub_batch: bool,
    chaos: float = 0.0,
) -> None:
  """Runs one arm (restarting after preemptions) until true completion."""
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
  # Metrics live in REAL wandb (whatever mode the environment configures --
  # online by default). Each restart attempt gets its OWN wandb run, all
  # grouped per arm under WANDB_RUN_GROUP: fresh runs per attempt are
  # sturdier than cross-process run resumption, and the wandb UI compares
  # arms by group. The correctness guarantees of this harness (event log,
  # done-file, coverage checks) never depend on wandb being reachable.
  env["WANDB_RUN_GROUP"] = tag
  # Progress metrics in stock mode are gated on this env (with sub-batch
  # enabled they are on regardless); set it here so the stock arms emit
  # perf/sub_batch_progress_micro_steps without any demo-side flag.
  env["TUNIX_SB_BENCH_METRICS"] = "1"
  done_file = workdir / f"done_{arm}"
  done_file.unlink(missing_ok=True)
  env["TUNIX_BENCH_DONE_FILE"] = str(done_file)

  cmd = [
      sys.executable,
      str(_DEMO),
      *(["--sub_batch_checkpointing"] if sub_batch else ["--bench_baseline"]),
      f"--batch_size={args.batch_size}",
      f"--mini_batch_size={args.mini_batch_size}",
      f"--train_micro_batch_size={args.train_micro_batch_size}",
      f"--num_generations={args.num_generations}",
      f"--num_iterations={args.num_iterations}",
      f"--max_steps={args.max_steps}",
      "--skip_eval",
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
      print(f"[validate:{arm}] complete after {attempt} restart(s).",
            flush=True)
      return
    attempt += 1
    if args.max_restarts is not None and args.max_restarts >= 0 and attempt > args.max_restarts:
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

  # Arm registry: (name, uses sub-batch, subject to the kill schedule).
  # The clean pair may be absent (--skip_clean_arms or an older workdir);
  # every per-arm check below runs only for arms that actually produced
  # evidence, so the report degrades to the two-arm form automatically.
  ARM_DEFS = (
      ("subbatch", True, True),
      ("stock", False, True),
      ("subbatch_clean", True, False),
      ("stock_clean", False, False),
  )
  arms = {}
  for arm, _, _ in ARM_DEFS:
    events = _read_events(workdir / f"events_{arm}.jsonl")
    snaps = [e for e in events if e.get("event") == "snapshot"]
    arms[arm] = {
        "events": events,
        "snapshots": snaps,
        "by_key": _dedup_last_by_key(snaps),
        "stock_iters": [e for e in events if e.get("event") == "stock_iter"],
        "preempts": [e for e in events if e.get("event") == "preempt"],
        "resumes": [e for e in events if e.get("event") == "resume"],
    }
    arms[arm]["apply_loss"], arms[arm]["step_metrics"] = _parse_logs(
        workdir / f"logs_{arm}"
    )
    timing_file = workdir / f"timing_{arm}.json"
    arms[arm]["timing"] = (
        json.loads(timing_file.read_text()) if timing_file.exists() else None
    )
  present = [
      (arm, sb, killed)
      for arm, sb, killed in ARM_DEFS
      if arms[arm]["events"] or arms[arm]["timing"]
  ]
  sb_arms = [arm for arm, sb, _ in present if sb]
  stock_arms = [arm for arm, sb, _ in present if not sb]

  # A subbatch arm's loss lines can be permanently lost to a kill: the
  # console/wandb flush lags one apply behind by design (peft_trainer
  # overlaps that I/O with the next window's compute), and unlike stock
  # (whose resume re-executes the interrupted window and naturally
  # reproduces the lost line) a committed subbatch apply is never
  # re-executed, so nothing regenerates it. The boundary-keyed snapshot
  # event durably carries the trainer's own loss for that apply (see
  # _bench_read_loss), captured before that lag can strike -- prefer it
  # over the fragile console line, which is now only a fallback for older
  # event logs that predate this field.
  durable_loss = {}
  for arm in sb_arms:
    durable_loss[arm] = {
        key // 1_000_000: e["loss"]
        for key, e in arms[arm]["by_key"].items()
        if key % 1_000_000 == 0
        and key // 1_000_000 >= 1
        and e.get("loss") is not None
    }
    arms[arm]["apply_loss"] = {
        **arms[arm]["apply_loss"],
        **durable_loss[arm],
    }

  # --- 1. Preemptions and resumes (identical schedule for killed arms) ----
  lines.append("## 1. Preemptions and resumes")
  lines.append("")
  if args.sb_chaos_prob > 0:
    lines.append(
        f"Chaos mode: random kills at p={args.sb_chaos_prob} per micro-batch"
        " (per-arm dice, so kill points differ across arms). Chaos kills"
        " emit no `preempt` events; their restarts appear as `resume`"
        " events below and in the attempt counts of section 6."
    )
    lines.append("")
  for arm, arm_sb, arm_killed in present:
    pre = arms[arm]["preempts"]
    res = arms[arm]["resumes"]
    expected_kills = len(args.preempt_set) if arm_killed else 0
    check(
        f"{arm}: deterministic preemption count",
        len(pre) == expected_kills,
        f"requested {expected_kills}"
        f" ({sorted(args.preempt_set) if arm_killed else 'clean arm'}),"
        f" fired {len(pre)} at {sorted(e['iter_steps'] for e in pre)}",
    )
    if not arm_killed and res:
      # Not a failure by itself (an unplanned crash -- e.g. an environment
      # bug -- restarts and resumes correctly), but the arm is then no
      # longer a CLEAN baseline; the wall-clock comparison in section 6
      # must be read with that in mind.
      lines.append(
          f"  - NOTE: clean arm restarted {len(res)} time(s) (unplanned"
          " crash); its wall-clock is no longer a pure no-preemption"
          " reference."
      )
    for e in pre:
      check(
          f"{arm}: preempt@iter{e['iter_steps']} was mid-window",
          e["iter_steps"] % k != 0,
          f"iter%k={e['iter_steps'] % k} (nonzero means inside gradient"
          " accumulation, not at an apply boundary)",
      )
    paired = 0
    if arm_sb:
      # Exact-key resume: the durable-wait in the injector guarantees the
      # resume lands on precisely the killed micro-step's snapshot.
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
          f"{arm}: every preemption resumed from its exact snapshot",
          paired == len(pre),
          f"{paired}/{len(pre)} preempt->resume pairs matched on iter_steps",
      )
      for r in res:
        r_key = r.get("key")
        mid_window = r_key is not None and r_key % 1_000_000 > 0
        if mid_window:
          # Only mid-window resumes carry a buffer; a resume from a
          # boundary key (possible under random --sb_chaos_prob kills whose
          # snapshot landed at an apply) legitimately restores an empty
          # accumulator.
          check(
              f"{arm}: resume@iter{r['iter_steps']} restored the buffer",
              bool(r.get("had_training_state")),
              "training_state (partial gradient + denom) present in the"
              " restored mid-window snapshot",
          )
        else:
          lines.append(
              f"  - resume@iter{r['iter_steps']} from boundary key {r_key}:"
              " empty accumulator by construction (no buffer check)"
          )
    else:
      # Stock resume: the trainer restores the last durable apply, so the
      # resume must land exactly at that boundary -- (t // k) * k for a
      # kill at iter t (fair geometry) -- and everything after it inside
      # the window is re-executed. That gap is the measured cost.
      for e in pre:
        t = e["iter_steps"]
        expected_b = (t // k) * k
        match = next((r for r in res if r["ts"] > e["ts"]), None)
        if match is None:
          check(f"{arm}: preempt@iter{t} resumed", False,
                "no resume event after the kill")
          continue
        paired += 1
        b = match["iter_steps"]
        check(
            f"{arm}: preempt@iter{t} resumed at the last apply boundary",
            b == expected_b,
            f"resumed at iter {b} (expected {expected_b}); re-executes"
            f" {t - b} micro-steps plus the window's rollout generation",
        )
    lines.append("")

  # --- 2. Sub-batch accounting: denom proportional to iter_steps % k ------
  lines.append(
      "## 2. Sub-batch micro-step accounting (denom == (iter_steps % k) x"
      " unit on every snapshot)"
  )
  lines.append("")
  lines.append(
      "The accumulator's denominator counts accumulation units since the"
      " last apply -- micro-steps on some trainer builds, sequences"
      " (micro-steps x per-micro batch size) on others. Either way, on"
      " every snapshot it must be exactly 0 at apply boundaries and equal"
      " (iter_steps mod k) x unit in between, for one CONSTANT unit across"
      " the whole run. A double-trained micro-step would overshoot, a lost"
      " one undershoot -- including across a preemption/resume."
  )
  lines.append("")
  for arm in sb_arms:
    snaps = [
        e for e in arms[arm]["snapshots"] if e.get("denom") is not None
    ]
    bad = []
    units = set()
    for e in snaps:
      pos = e["iter_steps"] % k
      d = float(e["denom"])
      if pos == 0:
        if d != 0:
          bad.append((e["key"], d, e["iter_steps"]))
      elif d <= 0 or d % pos != 0:
        bad.append((e["key"], d, e["iter_steps"]))
      else:
        units.add(d / pos)
    if len(units) > 1:
      bad.append(("inconsistent units", sorted(units), "-"))
    check(
        f"{arm}: denom == (iter_steps % k) x unit on all snapshots",
        not bad and bool(snaps),
        f"{len(snaps)} snapshots checked, unit ="
        f" {sorted(units) if units else '?'};"
        + (
            " no violations"
            if not bad
            else f" VIOLATIONS (key, denom, iter): {bad}"
        ),
    )
    # Post-resume continuity: first snapshot after each resume continues
    # the window from local+1 (clean arms only ever hit this after an
    # unplanned crash, where the same contract must hold).
    snaps_p = arms[arm]["snapshots"]
    for r in arms[arm]["resumes"]:
      after = [e for e in snaps_p if e["ts"] > r["ts"]]
      if not after:
        check(f"{arm}: continuity after resume@iter{r['iter_steps']}",
              False, "no snapshot events after the resume")
        continue
      first = min(after, key=lambda e: e["ts"])
      r_key = r.get("key")
      r_local = (r_key % 1_000_000) if r_key is not None else None
      pos = first["iter_steps"] % k
      d = first.get("denom")
      denom_ok = (
          d is None
          or (
              float(d) == 0
              if pos == 0
              else float(d) > 0 and float(d) % pos == 0
          )
      )
      # A resume at local k-1 restores a window whose NEXT micro-step
      # contains the apply: the following snapshot then correctly starts
      # the next window at local 0 (denom 0). Only mid-window resumes
      # below k-1 continue the same window at local+1.
      expected_local = 0 if pos == 0 else (
          r_local + 1 if r_local is not None else None
      )
      passed = (
          first["iter_steps"] == r["iter_steps"] + 1
          and (expected_local is None or first["local"] == expected_local)
          and denom_ok
      )
      check(
          f"{arm}: continuity after resume@iter{r['iter_steps']}",
          passed,
          f"first post-resume snapshot: key={first['key']}"
          f" local={first['local']} denom={first.get('denom')}"
          f" iter={first['iter_steps']} (expected local"
          f" {expected_local if expected_local is not None else '?'}:"
          " the restored buffer was extended through the window -- rolling"
          " to the next window's boundary when the resume sat one"
          " micro-step short of the apply)",
      )
  lines.append("")

  # --- 3. Stock accounting: exactly-once except the paid re-execution -----
  lines.append("## 3. Stock micro-step accounting")
  lines.append("")
  lines.append(
      "Stock arms have no snapshots; their `stock_iter` events record every"
      " training call. Every micro-step must train AT LEAST once, and the"
      " re-trained multiplicity must exactly match the kill->resume gaps"
      " (each resume at boundary b after progress reached t re-executes"
      " (b, t]). Any other multiplicity means lost or double-counted data."
      " A clean arm with no resumes must therefore be exactly-once"
      " throughout. (A crash can tear the log's final line, which would"
      " undercount one iter -- deterministic kills flush before dying, so"
      " only unplanned crashes can produce that artifact.)"
  )
  lines.append("")
  stock_observed: dict[str, dict[int, int]] = {}
  stock_reexecuted: dict[str, int] = {}
  for arm in stock_arms:
    stock_events = arms[arm]["stock_iters"]
    observed: dict[int, int] = {}
    for e in stock_events:
      it, n = e["iter_steps"], e.get("chunks", 1)
      for i in range(it - n + 1, it + 1):
        observed[i] = observed.get(i, 0) + 1
    expected = {i: 1 for i in range(1, total_iters + 1)}
    reexecuted = 0
    for r in arms[arm]["resumes"]:
      b = r["iter_steps"]
      prev = [
          e["iter_steps"] for e in stock_events if e["ts"] < r["ts"]
      ]
      t = max(prev, default=0)
      reexecuted += max(0, t - b)
      for i in range(b + 1, t + 1):
        expected[i] = expected.get(i, 0) + 1
    stock_observed[arm] = observed
    stock_reexecuted[arm] = reexecuted
    mismatched = {
        i: (observed.get(i, 0), expected.get(i, 0))
        for i in sorted(set(observed) | set(expected))
        if observed.get(i, 0) != expected.get(i, 0)
    }
    check(
        f"{arm}: multiplicity == 1 + re-executions on every micro-step",
        not mismatched and bool(stock_events),
        f"{len(stock_events)} training events over {total_iters}"
        f" micro-steps, {reexecuted} re-executed micro-step(s) across"
        f" {len(arms[arm]['resumes'])} resume(s);"
        + (
            " no mismatches"
            if not mismatched
            else f" MISMATCHES (iter: observed, expected): {mismatched}"
        ),
    )
  lines.append("")

  # --- 4. Coverage: every micro-step, every apply -------------------------
  lines.append("## 4. Coverage")
  lines.append("")
  for arm in sb_arms:
    keys = arms[arm]["by_key"]
    check(
        f"{arm}: every micro-step snapshotted exactly once",
        len(keys) == total_iters,
        f"{len(keys)}/{total_iters} distinct snapshot keys"
        + (
            ""
            if len(keys) == total_iters
            else f" (windows seen: {sorted({k // 1_000_000 for k in keys})})"
        ),
    )
    # Apply evidence comes from the BOUNDARY SNAPSHOT KEYS (T000000 is
    # written by the chunk whose apply advanced train_steps to T), not from
    # the trainer's console loss lines: those print late (behind the save
    # finalize queue), a hard kill destroys the pending ones, and a resumed
    # lineage never re-executes restored applies -- so a preempted run's
    # console loss lines alone would legitimately be incomplete for its
    # early applies while their boundary keys prove they ran. The loss
    # values above are now recovered from the durable per-apply snapshot
    # field instead (see the durable_loss merge), which the next check
    # confirms closed that gap.
    applies_evidenced = {
        key // 1_000_000
        for key in keys
        if key % 1_000_000 == 0 and key // 1_000_000 >= 1
    }
    check(
        f"{arm}: every apply executed",
        applies_evidenced == set(range(1, total_applies + 1)),
        f"boundary keys prove applies {sorted(applies_evidenced)} (expected"
        f" 1..{total_applies}); loss lines captured for"
        f" {sorted(arms[arm]['apply_loss'])}",
    )
    check(
        f"{arm}: every apply's loss recovered from the durable event log",
        applies_evidenced <= set(durable_loss[arm]),
        f"{len(durable_loss[arm])}/{len(applies_evidenced)} evidenced"
        " applies carry a durable loss (older event logs predating this"
        " field fall back to the fragile console line, which a hard kill"
        " before the one-apply-lagged flush can permanently miss)",
    )
  for arm in stock_arms:
    stock_covered = set(stock_observed[arm])
    check(
        f"{arm}: every micro-step trained",
        stock_covered >= set(range(1, total_iters + 1)),
        f"{len(stock_covered)}/{total_iters} micro-steps covered by"
        " stock_iter events",
    )
    # Stock arms re-execute killed applies, so -- unlike the subbatch
    # arms -- their loss lines ARE complete execution evidence: the
    # surviving lineage re-prints any apply whose line a kill destroyed.
    stock_applies = set(arms[arm]["apply_loss"])
    check(
        f"{arm}: every apply executed",
        stock_applies >= set(range(1, total_applies + 1)),
        f"loss lines captured for applies {sorted(stock_applies)}"
        f" (expected 1..{total_applies})",
    )
  lines.append("")

  # --- 5. Training metrics live in wandb ----------------------------------
  lines.append("## 5. Training metrics")
  lines.append("")
  lines.append(
      "Metrics are logged to wandb: each restart attempt is its own run,"
      " grouped per arm. Compare the arms by their run groups in the wandb"
      " UI (filter/group by `group`):"
  )
  lines.append("")
  arm_names = [arm for arm, _, _ in present]
  for arm in arm_names:
    tag = (arms[arm]["timing"] or {}).get("tag", f"validate-{arm}-<t0>")
    lines.append(f"- `{arm}` arm group: `{tag}`")
  lines.append("")
  lines.append(
      "Supplementary: per-apply losses (subbatch arms from the durable"
      " event log, stock arms from the surviving lineage's console lines):"
  )
  lines.append("")
  lines.append("| apply | " + " | ".join(arm_names) + " | spread |")
  lines.append("|" + "---|" * (len(arm_names) + 2))
  all_steps = sorted(
      {step for arm in arm_names for step in arms[arm]["apply_loss"]}
  )
  for step in all_steps:
    vals = [arms[arm]["apply_loss"].get(step) for arm in arm_names]
    known = [v for v in vals if v is not None]
    spread = (max(known) - min(known)) if len(known) > 1 else None
    lines.append(
        f"| {step} | "
        + " | ".join(str(v) if v is not None else "-" for v in vals)
        + f" | {f'{spread:.3e}' if spread is not None else '-'} |"
    )
  lines.append("")

  # --- 6. Efficiency: the 2x2 (mode x preemption) cost picture ------------
  lines.append("## 6. Efficiency (mode x preemption)")
  lines.append("")
  for arm in arm_names:
    t = arms[arm]["timing"]
    if t:
      lines.append(
          f"- {arm}: {t['attempts']} attempt(s) ({t['restarts']}"
          f" restart(s)), wall-clock {t['wall_s']:.0f}s"
      )
    else:
      lines.append(f"- {arm}: no timing sidecar (older workdir?)")
  for arm in stock_arms:
    if stock_reexecuted.get(arm):
      lines.append(
          f"- {arm} re-executed {stock_reexecuted[arm]} micro-step(s) plus"
          " each killed window's full rollout generation; subbatch arms"
          " re-execute 0 (count-skip resume on the exact snapshot)."
      )
  wall = {
      arm: arms[arm]["timing"]["wall_s"]
      for arm in arm_names
      if arms[arm]["timing"]
  }
  if {"subbatch_clean", "stock_clean"} <= set(wall):
    lines.append(
        f"- feature overhead with NO preemption: subbatch_clean -"
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
      lines.append(
          "python3 examples/math_gsm8k/qwen3_grpo_sub_batch_ckpt_diff.py \\"
      )
      lines.append(
          f"    <CHECKPOINT_ROOT>/{tags[arm]}/actor/{total_applies} \\"
      )
      lines.append(f"    <CHECKPOINT_ROOT>/{tags[ref]}/actor/{total_applies}")
    lines.append("```")
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
  parser.add_argument("--num_iterations", type=int, default=1)
  parser.add_argument("--max_steps", type=int, default=6)
  parser.add_argument(
      "--preempt_at", default=None,
      help="Comma-separated trainer iter_steps at which both PREEMPTED arms"
      " hard-kill themselves (each arm after its recovery point is"
      " durable). Pick values that are NOT multiples of k (= mini/micro)"
      " so the kills land mid-accumulation, spread across distinct global"
      " steps. Default: 6,10,14,18,22 -- unless --sb_chaos_prob is set, in"
      " which case the default becomes EMPTY (chaos replaces the schedule);"
      " pass both explicitly to stack them.",
  )
  parser.add_argument(
      "--sb_chaos_prob", type=float, default=0.0,
      help="Per-micro-batch probability of a RANDOM hard kill in the two"
      " PREEMPTED arms (e.g. 0.004 = 0.4%%); the clean arms never receive"
      " it. When set without an explicit --preempt_at it REPLACES the"
      " deterministic schedule. Random kill points differ per arm (each"
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
      help=(
          "Maximum restarts allowed per arm. Default is None (unlimited"
          " restarts until the run completes)."
      ),
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
  if unknown_args:
    if unknown_args[0] == "--":
      unknown_args = unknown_args[1:]
    args.demo_args.extend(unknown_args)
  if args.demo_args and args.demo_args[0] == "--":
    args.demo_args = args.demo_args[1:]

  k = args.mini_batch_size // args.train_micro_batch_size
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
  if k < 2 or args.num_generations < 2:
    raise SystemExit(
        "the comparison requires gradient accumulation and grouped"
        f" generation: k = mini/micro > 1 and num_generations > 1 (got"
        f" k={k}, G={args.num_generations})."
    )
  if args.num_iterations != 1 or args.mini_batch_size != args.batch_size:
    print(
        "[validate] WARNING: mini != batch or mu > 1 -- the STOCK arm's"
        " resume is known-incorrect under this geometry (it silently skips"
        " the interrupted step's remaining data, b/483779605). The run"
        " proceeds as a demonstration of that loss, but it is NOT an"
        " equal-correctness comparison; expect stock-arm accounting"
        " failures in the report.",
        flush=True,
    )
  boundary = [v for v in args.preempt_set if v % k == 0]
  if boundary:
    print(
        f"[validate] NOTE: --preempt_at values {boundary} land on apply"
        f" boundaries (multiples of k={k}). Kills fire after the apply and"
        " checkpoint are completed and durable.",
        flush=True,
    )
  iters_per_step = k * args.num_iterations * (
      args.batch_size // args.mini_batch_size
  )
  beyond = [
      v for v in args.preempt_set if v > args.max_steps * iters_per_step
  ]
  if beyond:
    raise SystemExit(
        f"--preempt_at values {beyond} lie beyond the run"
        f" ({args.max_steps} steps x {iters_per_step} micro-steps); they"
        " would never fire."
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

  # Subbatch arm first: it is the arm under test, so its failures surface
  # before the other arms' hours are spent. Both preempted arms get the
  # SAME kill schedule (and the same chaos probability, each rolling its
  # own dice) -- the delta between them is the feature. The clean pair
  # never receives either kill mechanism; it is the 2x2's control row.
  _run_arm("subbatch", args, workdir, preempt_at=args.preempt_set,
           sub_batch=True, chaos=args.sb_chaos_prob)
  _run_arm("stock", args, workdir, preempt_at=args.preempt_set,
           sub_batch=False, chaos=args.sb_chaos_prob)
  if not args.skip_clean_arms:
    _run_arm("subbatch_clean", args, workdir, preempt_at=set(),
             sub_batch=True, chaos=0.0)
    _run_arm("stock_clean", args, workdir, preempt_at=set(),
             sub_batch=False, chaos=0.0)
  return 0 if _write_report(args, workdir) else 1


if __name__ == "__main__":
  sys.exit(main())
