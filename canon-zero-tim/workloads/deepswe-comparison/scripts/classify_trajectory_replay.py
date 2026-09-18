#!/usr/bin/env python3
"""Fail-closed classifier for the P58.23 optimized B2xG2 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import classify_decode_prefill_probe as base


EXPECTED_JOURNAL_SHA256 = (
    "091a9273c2067876fbee1996ee853e3c8e861352e307cd5fb94fea2563aec456"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "482d7934a95207d0d77bb4857fbb200d7b367cbf437dda6585937b20909afa8f"
)
EXPECTED_TASK_IMAGES = (
    "namanjain12/scrapy_final:439a3e59b8e858441f8d97dbc32f398db392330d",
    "namanjain12/scrapy_final:439a3e59b8e858441f8d97dbc32f398db392330d",
)


def classify(
    root: Path, *, source_sha: str, expected_hostname: str
) -> dict:
  root = root.resolve()
  report = base.classify(
      root,
      source_sha=source_sha,
      expected_hostname=expected_hostname,
  )
  if report.get("verdict") != "PASS" or report.get("outcome") != (
      "ZERO_TIM_BACKWARD_NO_COMMIT_PASS"
  ):
    raise ValueError(
        "recorded-trajectory replay did not pass strict alignment/backward: "
        f"{report.get('verdict')}/{report.get('outcome')}"
    )
  carrier = report.get("carrier_provenance", {})
  expected_carrier = {
      "q4_tp4_trajectory_replay": True,
      "replay_journal_sha256": EXPECTED_JOURNAL_SHA256,
      "q4_tp4_short_backward": True,
      "task_images": list(EXPECTED_TASK_IMAGES),
      "max_prompt_length": 2048,
      "max_response_length": 512,
      "global_prompts": 2,
      "global_trajectories": 4,
      "system_optimization": {
          "carrier": "P28+P30+P71-fwd",
          "p59_rank_parallel_backward": False,
          "p59_reason": "DP1 one-host cannot execute rank-parallel backward",
          "p28_segmented_forward": True,
          "p28_segmented_train": True,
          "p30_sparse_grad_assembly": True,
          "p30_reuse_segmented_engine": True,
          "p71_scan": "fwd",
      },
      "stage": "backward-no-commit",
  }
  changed = {
      key: carrier.get(key)
      for key, expected in expected_carrier.items()
      if carrier.get(key) != expected
  }
  if changed:
    raise ValueError(f"recorded-trajectory replay manifest drifted: {changed}")

  provenance_path = root / "replay_provenance.json"
  provenance = base._load_json(provenance_path)  # pylint: disable=protected-access
  expected_provenance = {
      "schema": "canon.p58.recorded-trajectory-replay.v1",
      "evidence_kind": "recorded-trajectory-prefix-backward-diagnostic",
      "journal_sha256": EXPECTED_JOURNAL_SHA256,
      "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
      "source_model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "source_sampling_contract": {
          "temperature": 1.0,
          "top_p": 1.0,
          "top_k": 0,
      },
      "source_sampling_identity": (
          "p58s22lr3_20260829t2256z@"
          "16c224aa80eb6b3a544be19f693c0542ab4b0dcb:"
          "rows7,0x2:B2G2"
      ),
      "prompt_identity": "same-strict-exact-real-prompt-repeated-twice",
      "environment_calls": 0,
      "rollout_decode_calls": 0,
  }
  changed = {
      key: provenance.get(key)
      for key, expected in expected_provenance.items()
      if provenance.get(key) != expected
  }
  if changed:
    raise ValueError(f"recorded-trajectory replay provenance drifted: {changed}")
  rows = provenance.get("rows")
  if not isinstance(rows, list) or len(rows) != 4:
    raise ValueError("recorded-trajectory replay provenance must contain 4 rows")
  expected_rows = (
      (0, 0, 0, 432, 363, 1.0),
      (0, 1, 1, 333, 264, 0.0),
      (1, 2, 0, 432, 363, 1.0),
      (1, 3, 1, 333, 264, 0.0),
  )
  for row, expected in zip(rows, expected_rows):
    values = (
        row.get("source_group_id"),
        row.get("source_row"),
        row.get("source_pair_index"),
        row.get("prefix_length"),
        row.get("prefix_action_tokens"),
        row.get("terminal_reward"),
    )
    if values != expected:
      raise ValueError(
          f"recorded-trajectory replay row changed: {values} != {expected}"
      )
    for key in (
        "prompt_tokens_sha256",
        "prefix_tokens_sha256",
        "prefix_action_mask_sha256",
        "prefix_old_logprobs_sha256",
    ):
      value = row.get(key)
      if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"recorded-trajectory replay row lacks {key}")

  metrics = base._load_last_jsonl(  # pylint: disable=protected-access
      root / "batch_metrics.jsonl"
  )
  expected_metrics = {
      "trajectories": 4,
      "complete_trajectories": 4,
      "compact_filtered_trajectories": 0,
      "solved_trajectories": 2,
      "prompt_groups": 2,
      "mixed_prompt_groups": 2,
      "effective_prompt_groups": 2,
      "nonzero_advantages": 4,
  }
  changed = {
      key: metrics.get(key)
      for key, expected in expected_metrics.items()
      if metrics.get(key) != expected
  }
  if changed:
    raise ValueError(f"recorded-trajectory replay metrics drifted: {changed}")
  groups = metrics.get("groups")
  if (
      not isinstance(groups, list)
      or len(groups) != 2
      or any(group.get("raw_rewards") != [1.0, 0.0] for group in groups)
      or any(group.get("category") != "mixed" for group in groups)
  ):
    raise ValueError("recorded-trajectory replay mixed reward group changed")

  trajectory_paths = sorted(root.glob("batch-*.trajectories.jsonl.gz"))
  records = base._load_trajectories(trajectory_paths[0])  # pylint: disable=protected-access
  if len(records) != 4:
    raise ValueError("recorded-trajectory replay journal must contain 4 rows")
  for record, expected in zip(records, expected_rows):
    trajectory = record.get("trajectory", {})
    replay = trajectory.get("replay_provenance", {})
    if (
        record.get("status") != "SUCCEEDED"
        or record.get("task_identity", {}).get("docker_image")
        != EXPECTED_TASK_IMAGES[expected[0]]
        or len(trajectory.get("conversation_tokens", [])) != expected[3]
        or sum(trajectory.get("conversation_masks", [])) != expected[4]
        or replay.get("source_row") != expected[1]
    ):
      raise ValueError("recorded-trajectory replay durable journal changed")

  raw = (root / "raw.log").read_text(encoding="utf-8", errors="replace")
  required_markers = (
      "[P58.23.SYSTEM_OPT] PASS carrier=P28+P30+P71-fwd",
      "[P58.23.REPLAY] LOAD_PASS groups=2 generations=2 trajectories=4",
      "[P58.23.REPLAY] SAMPLING_PROVENANCE_PASS temperature=1.0 "
      "top_p=1.0 top_k=0",
      "[P58.23.REPLAY] PRODUCER_BYPASS verdict=PASS environment=0 "
      "rollout_decode=0",
      "[P58.23.REPLAY] ADVANTAGE_PASS groups=2 generations=2",
      "[P58.23.REPLAY] POST_BACKWARD_BATCH_PASS trajectories=4 "
      "microsteps=2 N_action=1254",
      "injected=0",
  )
  missing = [marker for marker in required_markers if marker not in raw]
  if missing:
    raise ValueError(f"recorded-trajectory replay lacks markers: {missing}")
  for forbidden in ("creating RepoEnv", "[SWEEnv group="):
    if forbidden in raw:
      raise ValueError(
          f"recorded-trajectory replay unexpectedly invoked environment: {forbidden}"
      )
  process = report.get("process_status")
  if not isinstance(process, dict) or process.get("training_process_status") != 0:
    raise ValueError("recorded-trajectory replay training process did not exit 0")

  report.update({
      "schema": "canon.p58.trajectory-replay.classification.v1",
      "outcome": "ZERO_TIM_RECORDED_TRAJECTORY_BACKWARD_NO_COMMIT_PASS",
      "replay_provenance": provenance,
      "claim": (
          "PASS proves strict A=B=C and finite nonzero repeat-exact "
          "backward-no-commit only for immutable action-boundary prefixes of "
          "two real B2xG2 prompt groups on direct Qwen3-4B DP1xTP4 using "
          "P28/P30/P71-fwd. It is "
          "not a fresh rollout, does not re-run rewards, and does not certify "
          "TP8 or production training."
      ),
  })
  report["artifacts"]["replay_provenance"] = provenance_path.name
  report["artifacts"]["replay_provenance_sha256"] = base._sha256(  # pylint: disable=protected-access
      provenance_path
  )
  return report


def _package(root: Path, output: Path) -> None:
  names = [
      "raw.log",
      "run_manifest.json",
      "probe_process_status.json",
      "replay_provenance.json",
      "batch_metrics.jsonl",
      "pre_alignment.jsonl",
      "alignment.jsonl",
      "backward_no_commit.json",
      next(root.glob("batch-*.trajectories.jsonl.gz")).name,
      output.name,
  ]
  (root / "RETURN_FILES").write_text(
      "".join(f"{name}\n" for name in names), encoding="utf-8"
  )
  (root / "SHA256SUMS").write_text(
      "".join(
          f"{base._sha256(root / name)}  {name}\n"  # pylint: disable=protected-access
          for name in names
      ),
      encoding="utf-8",
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifact-dir", type=Path, required=True)
  parser.add_argument("--source-sha", required=True)
  parser.add_argument("--expected-hostname", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--package", action="store_true")
  args = parser.parse_args()
  try:
    result = classify(
        args.artifact_dir,
        source_sha=args.source_sha,
        expected_hostname=args.expected_hostname,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"trajectory replay classifier error: {exc}", file=sys.stderr)
    return 1
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  if args.package:
    _package(args.artifact_dir, args.output)
  print(json.dumps(result, sort_keys=True, separators=(",", ":")))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
