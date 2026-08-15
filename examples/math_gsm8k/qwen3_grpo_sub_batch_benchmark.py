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
  done_file = workdir / f"done_{arm}"
  done_file.unlink(missing_ok=True)
  env["TUNIX_BENCH_DONE_FILE"] = str(done_file)

  cmd = [
      sys.executable,
      str(_DEMO),
      *(["--sub_batch_checkpointing"] if sub_batch else []),
      f"--batch_size={args.batch_size}",
      f"--mini_batch_size={args.mini_batch_size}",
      f"--train_micro_batch_size={args.train_micro_batch_size}",
      f"--num_generations={args.num_generations}",
      f"--num_iterations={args.num_iterations}",
      f"--max_steps={args.max_steps}",
      "--skip_eval",
      *(
          [f"--sb_chaos_prob={args.sb_chaos_prob}"]
          if args.sb_chaos_prob > 0
          else []
      ),
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
  for arm in ("subbatch", "stock"):
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

  # --- 1. Preemptions and resumes (both arms, identical schedule) ---------
  lines.append("## 1. Preemptions and resumes")
  lines.append("")
  for arm in ("subbatch", "stock"):
    pre = arms[arm]["preempts"]
    res = arms[arm]["resumes"]
    check(
        f"{arm}: preemption count",
        len(pre) == len(args.preempt_set),
        f"requested at iters {sorted(args.preempt_set)}, fired {len(pre)}"
        f" at {sorted(e['iter_steps'] for e in pre)}",
    )
    for e in pre:
      check(
          f"{arm}: preempt@iter{e['iter_steps']} was mid-window",
          e["iter_steps"] % k != 0,
          f"iter%k={e['iter_steps'] % k} (nonzero means inside gradient"
          " accumulation, not at an apply boundary)",
      )
    paired = 0
    if arm == "subbatch":
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
          "subbatch: every preemption resumed from its exact snapshot",
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
              f"subbatch: resume@iter{r['iter_steps']} restored the buffer",
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
          check(f"stock: preempt@iter{t} resumed", False,
                "no resume event after the kill")
          continue
        paired += 1
        b = match["iter_steps"]
        check(
            f"stock: preempt@iter{t} resumed at the last apply boundary",
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
  snaps = [
      e for e in arms["subbatch"]["snapshots"] if e.get("denom") is not None
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
      "subbatch: denom == (iter_steps % k) x unit on all snapshots",
      not bad and bool(snaps),
      f"{len(snaps)} snapshots checked, unit ="
      f" {sorted(units) if units else '?'};"
      + (
          " no violations"
          if not bad
          else f" VIOLATIONS (key, denom, iter): {bad}"
      ),
  )
  # Post-resume continuity: first snapshot after each resume continues the
  # window from local+1 with denom == local+1.
  snaps_p = arms["subbatch"]["snapshots"]
  for r in arms["subbatch"]["resumes"]:
    after = [e for e in snaps_p if e["ts"] > r["ts"]]
    if not after:
      check(f"continuity after resume@iter{r['iter_steps']}", False,
            "no snapshot events after the resume")
      continue
    first = min(after, key=lambda e: e["ts"])
    r_key = r.get("key")
    r_local = (r_key % 1_000_000) if r_key is not None else None
    pos = first["iter_steps"] % k
    d = first.get("denom")
    denom_ok = (
        d is None
        or (float(d) == 0 if pos == 0 else float(d) > 0 and float(d) % pos == 0)
    )
    # A resume at local k-1 restores a window whose NEXT micro-step contains
    # the apply: the following snapshot then correctly starts the next
    # window at local 0 (denom 0). Only mid-window resumes below k-1
    # continue the same window at local+1.
    expected_local = 0 if pos == 0 else (
        r_local + 1 if r_local is not None else None
    )
    passed = (
        first["iter_steps"] == r["iter_steps"] + 1
        and (expected_local is None or first["local"] == expected_local)
        and denom_ok
    )
    check(
        f"continuity after resume@iter{r['iter_steps']}",
        passed,
        f"first post-resume snapshot: key={first['key']}"
        f" local={first['local']} denom={first.get('denom')}"
        f" iter={first['iter_steps']} (expected local"
        f" {expected_local if expected_local is not None else '?'}:"
        " the restored buffer was extended through the window -- rolling"
        " to the next window's boundary when the resume sat one micro-step"
        " short of the apply)",
    )
  lines.append("")

  # --- 3. Stock accounting: exactly-once except the paid re-execution -----
  lines.append("## 3. Stock micro-step accounting")
  lines.append("")
  lines.append(
      "The stock arm has no snapshots; its `stock_iter` events record every"
      " training call. Every micro-step must train AT LEAST once, and the"
      " re-trained multiplicity must exactly match the kill->resume gaps"
      " (each resume at boundary b after progress reached t re-executes"
      " (b, t]). Any other multiplicity means lost or double-counted data."
      " (A crash can tear the log's final line, which would undercount one"
      " iter -- deterministic kills flush before dying, so only unplanned"
      " crashes can produce that artifact.)"
  )
  lines.append("")
  stock_events = arms["stock"]["stock_iters"]
  observed: dict[int, int] = {}
  for e in stock_events:
    it, n = e["iter_steps"], e.get("chunks", 1)
    for i in range(it - n + 1, it + 1):
      observed[i] = observed.get(i, 0) + 1
  expected = {i: 1 for i in range(1, total_iters + 1)}
  reexecuted = 0
  for r in arms["stock"]["resumes"]:
    b = r["iter_steps"]
    prev = [
        e["iter_steps"] for e in stock_events if e["ts"] < r["ts"]
    ]
    t = max(prev, default=0)
    reexecuted += max(0, t - b)
    for i in range(b + 1, t + 1):
      expected[i] = expected.get(i, 0) + 1
  mismatched = {
      i: (observed.get(i, 0), expected.get(i, 0))
      for i in sorted(set(observed) | set(expected))
      if observed.get(i, 0) != expected.get(i, 0)
  }
  check(
      "stock: multiplicity == 1 + re-executions on every micro-step",
      not mismatched and bool(stock_events),
      f"{len(stock_events)} training events over {total_iters} micro-steps,"
      f" {reexecuted} re-executed micro-step(s) across"
      f" {len(arms['stock']['resumes'])} resume(s);"
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
  keys = arms["subbatch"]["by_key"]
  check(
      "subbatch: every micro-step snapshotted exactly once",
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
  # lineage never re-executes restored applies -- so early applies of a
  # preempted run legitimately have no loss line while their boundary
  # keys prove they ran.
  applies_evidenced = {
      key // 1_000_000
      for key in keys
      if key % 1_000_000 == 0 and key // 1_000_000 >= 1
  }
  check(
      "subbatch: every apply executed",
      applies_evidenced == set(range(1, total_applies + 1)),
      f"boundary keys prove applies {sorted(applies_evidenced)} (expected"
      f" 1..{total_applies}); loss lines captured for"
      f" {sorted(arms['subbatch']['apply_loss'])}",
  )
  stock_covered = set(observed)
  check(
      "stock: every micro-step trained",
      stock_covered >= set(range(1, total_iters + 1)),
      f"{len(stock_covered)}/{total_iters} micro-steps covered by"
      " stock_iter events",
  )
  # The stock arm re-executes killed applies, so -- unlike the subbatch
  # arm -- its loss lines ARE complete execution evidence: the surviving
  # lineage re-prints any apply whose line a kill destroyed.
  stock_applies = set(arms["stock"]["apply_loss"])
  check(
      "stock: every apply executed",
      stock_applies >= set(range(1, total_applies + 1)),
      f"loss lines captured for applies {sorted(stock_applies)} (expected"
      f" 1..{total_applies})",
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
  for arm in ("subbatch", "stock"):
    tag = (arms[arm]["timing"] or {}).get("tag", f"validate-{arm}-<t0>")
    lines.append(f"- `{arm}` arm group: `{tag}`")
  lines.append("")
  lines.append(
      "Supplementary: per-apply losses parsed from the attempt logs (the"
      " last surviving lineage per apply; early applies of a preempted"
      " subbatch run may be missing here -- boundary keys above are the"
      " execution evidence):"
  )
  lines.append("")
  lines.append("| apply | subbatch | stock | abs diff |")
  lines.append("|---|---|---|---|")
  c_loss, p_loss = arms["subbatch"]["apply_loss"], arms["stock"][
      "apply_loss"
  ]
  for step in sorted(set(c_loss) | set(p_loss)):
    c, p = c_loss.get(step), p_loss.get(step)
    d = abs(c - p) if c is not None and p is not None else None
    lines.append(
        f"| {step} | {c if c is not None else '-'} |"
        f" {p if p is not None else '-'} |"
        f" {f'{d:.3e}' if d is not None else '-'} |"
    )
  lines.append("")

  # --- 6. Efficiency: what the identical kills cost each arm --------------
  lines.append("## 6. Efficiency under identical preemption")
  lines.append("")
  for arm in ("subbatch", "stock"):
    t = arms[arm]["timing"]
    if t:
      lines.append(
          f"- {arm}: {t['attempts']} attempt(s) ({t['restarts']}"
          f" restart(s)), wall-clock {t['wall_s']:.0f}s"
      )
    else:
      lines.append(f"- {arm}: no timing sidecar (older workdir?)")
  lines.append(
      f"- stock re-executed {reexecuted} micro-step(s) plus each killed"
      " window's full rollout generation; subbatch re-executed 0"
      " (count-skip resume on the exact snapshot)."
  )
  lines.append("")
  sb_tag = (arms["subbatch"]["timing"] or {}).get("tag")
  st_tag = (arms["stock"]["timing"] or {}).get("tag")
  if sb_tag and st_tag:
    lines.append(
        "Final-weights equivalence (run beside a live job, CPU-only):"
    )
    lines.append("")
    lines.append("```")
    lines.append(
        "python3 examples/math_gsm8k/qwen3_grpo_sub_batch_ckpt_diff.py \\"
    )
    lines.append(
        f"    <CHECKPOINT_ROOT>/{sb_tag}/actor/{total_applies} \\"
    )
    lines.append(f"    <CHECKPOINT_ROOT>/{st_tag}/actor/{total_applies}")
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
      "--preempt_at", default="6,10,14,18,22",
      help="Comma-separated trainer iter_steps at which BOTH arms"
      " hard-kill themselves (each arm after its recovery point is"
      " durable). Pick values that are NOT multiples of k (= mini/micro)"
      " so the kills land mid-accumulation, spread across distinct global"
      " steps. May be empty when --sb_chaos_prob drives random preemption"
      " instead.",
  )
  parser.add_argument(
      "--sb_chaos_prob", type=float, default=0.0,
      help="Per-micro-batch probability of a RANDOM hard kill in BOTH arms"
      " (forwarded to the demo), on top of / instead of the deterministic"
      " --preempt_at kills. Random kill points differ per arm, so the"
      " comparison becomes statistical rather than paired.",
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
    raise SystemExit(
        f"--preempt_at values {boundary} are apply boundaries (multiples"
        f" of k={k}); this harness exists to test MID-window preemption."
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
  # before the stock arm's hour is spent. Both arms get the SAME kill
  # schedule -- the delta between them is the feature.
  _run_arm("subbatch", args, workdir, preempt_at=args.preempt_set,
           sub_batch=True)
  _run_arm("stock", args, workdir, preempt_at=args.preempt_set,
           sub_batch=False)
  return 0 if _write_report(args, workdir) else 1


if __name__ == "__main__":
  sys.exit(main())
