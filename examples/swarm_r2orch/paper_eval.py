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
"""Zero-skill deployment evaluator for R2Orch IID (6) and OOD benchmarks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
from typing import Any

from examples.swarm_r2orch import scoring as tunix_scoring

_SUITE_PRIORITY: dict[str, int] = {
    "aime": 0,
    "math_hard": 0,
    "alfworld": 1,
    "mbppplus": 2,
    "hotpotqa": 3,
    "musique": 3,
    "medqa": 4,
    "nq_open": 5,
    "triviaqa": 5,
}


def _reference_from_row(row: dict[str, Any]) -> Any:
  if "answers" in row and isinstance(row["answers"], list) and row["answers"]:
    return row["answers"]
  answers = row.get("target_answers")
  if isinstance(answers, list) and answers:
    return answers[0] if len(answers) == 1 else answers
  return row.get("reference", row.get("answer"))


def _task_spec_from_row(row: dict[str, Any]) -> Any:
  from r2orch.observability import TaskSpec  # pylint: disable=g-import-not-at-top

  posed_as = str(row.get("posed_as") or row.get("dataset") or "unknown")
  kind = str(row.get("kind") or posed_as)
  task_id = str(row.get("id") or row.get("task_id") or row.get("source_id"))
  type_by_dataset = {
      "aime": "math",
      "hotpotqa": "multi_hop_qa",
      "nq_open": "factual_qa",
      "medqa": "medical_mcq",
      "mbppplus": "code_function",
      "alfworld": "alfworld",
  }
  verifier_by_dataset = {
      "aime": "integer_answer",
      "hotpotqa": "qa_exact_match",
      "nq_open": "qa_exact_match",
      "medqa": "letter_choice",
      "mbppplus": "mbppplus_tests",
      "alfworld": "alfworld_environment",
  }
  task_type = str(row.get("task_type") or type_by_dataset.get(posed_as, "general"))
  verifier = str(row.get("verifier") or verifier_by_dataset.get(posed_as, "auto"))
  if kind == "math_hard":
    verifier = "math_hard_symbolic"
  meta = {
      **dict(row.get("metadata") or {}),
      "dataset": posed_as,
      "ood_kind": kind,
      "split": str(row.get("split", "test")),
      "task_type": task_type,
      "verifier": verifier,
  }
  if "options" in row and "options" not in meta:
    meta["options"] = list(row["options"])
  return TaskSpec(
      task_id=task_id,
      prompt=str(row["prompt"]),
      reference=_reference_from_row(row),
      task_type=task_type,
      metadata=meta,
  )


def load_eval_suite(path: Path, *, limit: int = 0) -> list[Any]:
  rows = [
      json.loads(line)
      for line in Path(path).read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if limit > 0:
    rows = rows[:limit]
  return [_task_spec_from_row(r) for r in rows]


def _build_eval_deadline(dataset: str, started: float) -> Any:
  from r2orch.deadline import RolloutDeadline  # pylint: disable=g-import-not-at-top

  ds = dataset.strip().lower()
  long_math = ds in {"aime", "alfworld", "math_hard"}
  total_s = 2700.0 if long_math else 1200.0
  req_s = 1800.0 if long_math else 900.0
  return RolloutDeadline(
      total_timeout_s=total_s,
      no_progress_timeout_s=req_s,
      request_timeout_s=req_s,
      started_monotonic=started,
      exclude_failed_request_time=True,
      absolute_wall_timeout_s=total_s + 180.0,
  )


class _MathHardSymbolicVerifier:
  """Symbolic/numeric equivalence verifier for OOD MATH-Hard (`\boxed{}`)."""

  name = "math_hard_symbolic"
  supports_intermediate_scoring = True

  @classmethod
  def install_if_needed(cls, app: Any, ver_name: str) -> None:
    if ver_name != cls.name:
      return
    inst = cls()
    adapters = getattr(app.solver.verifier, "adapters", None)
    if isinstance(adapters, dict):
      adapters[cls.name] = inst
    else:
      app.solver.verifier = inst

  def verify(self, task: Any, prediction: str) -> Any:
    from r2orch.observability import VerificationResult  # pylint: disable=g-import-not-at-top

    score = float(
        tunix_scoring.score_math_answer(
            str(prediction or ""), str(task.reference), allow_symbolic=True
        )
    )
    return VerificationResult(score, score >= 1.0, self.name, f"score={score}")


def _score_eval_prediction(
    ver_name: str, pred: str, payload: dict[str, Any], task: Any, cfg: Any
) -> float:
  if ver_name == "math_hard_symbolic":
    return float(
        tunix_scoring.score_math_answer(
            pred, str(task.reference), allow_symbolic=True
        )
    )
  if ver_name in {
      "qa_exact_match",
      "letter_choice",
      "integer_answer",
      "mbppplus_tests",
  }:
    from r2orch.benchmark_scoring import benchmark_score  # pylint: disable=g-import-not-at-top

    py_bin = cfg.r2orch.paths.evalplus_python or "python3"
    return float(benchmark_score(ver_name, pred, task, python=py_bin))
  ver = payload.get("verification") or {}
  return float(ver.get("score", 0.0))


def _solve_one_task(
    base_cfg: Any,
    tokenizer: Any,
    task: Any,
    seed: int,
    eval_dir: Path,
) -> dict[str, Any]:
  from r2orch.application import create_adaptive_application  # pylint: disable=g-import-not-at-top

  safe_id = task.task_id.replace("/", "_")
  ds_label = str(task.metadata.get("ood_kind") or task.metadata.get("dataset") or "")
  ver_name = str(task.metadata.get("verifier", ""))
  cfg = replace(
      base_cfg,
      seed=seed,
      skillbank_enabled=False,
      trace_path=eval_dir / "traces" / f"{safe_id}-s{seed}.jsonl",
      canvas=replace(
          base_cfg.canvas,
          submission_journal_dir=str(eval_dir / "journals" / f"{safe_id}-s{seed}"),
      ),
  )
  app = create_adaptive_application(cfg, mock=False, director_tokenizer=tokenizer)
  _MathHardSymbolicVerifier.install_if_needed(app, ver_name)
  t0 = time.monotonic()
  app.set_rollout_deadline(_build_eval_deadline(ds_label, t0))
  try:
    app_res = app.solve(
        task.prompt,
        task_id=task.task_id,
        task_type=task.task_type,
        reference=task.reference,
        run_id=f"eval-{safe_id}-s{seed}",
        metadata=task.metadata,
        private_verifier_payload=task.private_verifier_payload,
    )
    payload = app_res.to_dict()
    pred = str(payload.get("answer") or payload.get("raw_answer") or "")
    score = _score_eval_prediction(ver_name, pred, payload, task, cfg)
    nodes = (payload.get("final_graph") or {}).get("nodes") or {}
    n_agents = len(nodes)
    return {
        "task_id": task.task_id,
        "dataset": ds_label,
        "score": score,
        "finished": bool(payload.get("finished")),
        "agents": n_agents,
        "multi_agent": n_agents >= 2,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return {
        "task_id": task.task_id,
        "dataset": ds_label,
        "score": 0.0,
        "finished": False,
        "agents": 0,
        "multi_agent": False,
        "timeout": "WallClockLimit" in type(exc).__name__,
        "error": f"{type(exc).__name__}: {exc}",
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
  finally:
    app.close()


def _summarize_suite_rows(
    suite_name: str,
    n_tasks: int,
    rows_by_run: list[list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
  run_accs = [
      100.0 * statistics.fmean(r["score"] for r in rrows) if rrows else 0.0
      for rrows in rows_by_run
  ]
  all_rows = [r for rrows in rows_by_run for r in rrows]
  multi_frac = (
      100.0 * statistics.fmean(1.0 if r["multi_agent"] else 0.0 for r in all_rows)
      if all_rows
      else 0.0
  )
  (output_dir / f"{suite_name}_rows.jsonl").write_text(
      "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows),
      encoding="utf-8",
  )
  return {
      "tasks": n_tasks,
      "runs": len(run_accs),
      "acc_mean": round(statistics.fmean(run_accs), 2),
      "acc_sd": round(statistics.stdev(run_accs), 2) if len(run_accs) > 1 else 0.0,
      "run_accs": [round(a, 2) for a in run_accs],
      "multi_agent_pct": round(multi_frac, 2),
      "errors": sum(1 for r in all_rows if "error" in r),
  }


def evaluate_zero_skill_suites(
    config: Any,
    tokenizer: Any,
    suite_files: dict[str, Path],
    *,
    output_dir: Path,
    served_director: str,
    concurrency: int = 32,
    runs: int = 1,
    base_seed: int = 10_000,
    limit_per_suite: int = 0,
) -> dict[str, Any]:
  """Evaluates zero-skill deployment across IID and/or OOD suites concurrently."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  eval_cfg = replace(
      config,
      skillbank_enabled=False,
      solver_model=replace(config.solver_model, served_model=served_director),
  )
  loaded_suites: dict[str, list[Any]] = {
      name: load_eval_suite(Path(p), limit=limit_per_suite)
      for name, p in suite_files.items()
      if Path(p).is_file()
  }
  n_runs = max(1, int(runs))
  jobs: list[tuple[int, str, int, int, Any, int, Path]] = []
  for sname, tasks in loaded_suites.items():
    prio = _SUITE_PRIORITY.get(sname, 3)
    for r_idx in range(n_runs):
      seed_base = base_seed + r_idx * 100_000
      run_dir = output_dir / sname / f"run-{r_idx}"
      for t_idx, task in enumerate(tasks):
        jobs.append((prio, sname, r_idx, t_idx, task, seed_base + t_idx, run_dir))
  jobs.sort(key=lambda item: (item[0], item[2], item[3]))

  rows_map: dict[str, list[list[dict[str, Any]]]] = {
      sname: [[] for _ in range(n_runs)] for sname in loaded_suites
  }
  if jobs:
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
      fut_pairs = [
          (sname, r_idx, pool.submit(_solve_one_task, eval_cfg, tokenizer, t, sd, rdir))
          for _, sname, r_idx, _, t, sd, rdir in jobs
      ]
      for sname, r_idx, fut in fut_pairs:
        rows_map[sname][r_idx].append(fut.result())

  summary_by_suite: dict[str, Any] = {}
  for sname, tasks in loaded_suites.items():
    res_meta = _summarize_suite_rows(sname, len(tasks), rows_map[sname], output_dir)
    summary_by_suite[sname] = res_meta
    print(
        f"[eval] {sname}: {res_meta['acc_mean']:.2f}%"
        f" (multi={res_meta['multi_agent_pct']:.1f}%, n={len(tasks)})",
        flush=True,
    )
  mean_acc = (
      round(statistics.fmean(v["acc_mean"] for v in summary_by_suite.values()), 2)
      if summary_by_suite
      else 0.0
  )
  report = {
      "served_director": served_director,
      "zero_skill": True,
      "suites": summary_by_suite,
      "mean_acc": mean_acc,
  }
  (output_dir / "eval_summary.json").write_text(
      json.dumps(report, indent=2) + "\n", encoding="utf-8"
  )
  return report
