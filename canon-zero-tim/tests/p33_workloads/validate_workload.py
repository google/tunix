#!/usr/bin/env python3
"""Validate and serialize one default-off DP16 workload contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tunix.rl import dp_workloads


def build_record(name: str, *, require_reduction_admission: bool) -> dict:
  workload = dp_workloads.get_workload(name)
  dp_workloads.validate_environment(
      workload,
      require_reduction_admission=require_reduction_admission,
  )
  run_stage = (
      os.environ.get("CANON_P33_RUN_STAGE", "full")
      if require_reduction_admission
      else "full"
  )
  command = workload.command(run_stage=run_stage)
  response_length = next(
      int(value.split("=", 1)[1])
      for value in command
      if value.startswith("--max_response_length=")
  )
  return {
      "verdict": "PASS",
      "scope": (
          "launch" if require_reduction_admission else "contract-only"
      ),
      "workload": workload.name,
      "model_id": workload.model_id,
      "topology": {
          "dp": workload.dp_size,
          "tp": workload.tp_size,
          "devices": workload.total_devices,
      },
      "batch": {
          "global_prompts": workload.global_prompts,
          "local_prompts": workload.local_prompts,
          "generations": workload.num_generations,
          "global_trajectories": workload.global_trajectories,
          "local_trajectories": workload.local_trajectories,
          "gradient_groups": workload.gradient_groups,
      },
      "tokens": {
          "prompt": workload.max_prompt_length,
          "response": response_length,
          "local_m": workload.local_m,
          "global_m": workload.global_m,
      },
      "max_steps": dp_workloads.requested_max_steps(
          workload,
          {
              "CANON_P33_RUN_STAGE": run_stage,
              "CANON_P33_NO_COMMIT": os.environ.get(
                  "CANON_P33_NO_COMMIT", "0"
              ),
          },
      ),
      "run_stage": run_stage,
      "periodic_evaluation": workload.periodic_evaluation,
      "wandb_project": workload.wandb_project,
      "command": list(command),
      "dp_reduction_admitted": (
          os.environ.get("CANON_P32_DP_REDUCTION_ADMITTED") == "1"
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", required=True, choices=("gsm8k", "frozenlake"))
  parser.add_argument("--output", required=True)
  parser.add_argument("--launch", action="store_true")
  args = parser.parse_args()

  output = Path(args.output)
  if output.exists():
    raise FileExistsError(f"refusing to overwrite workload record: {output}")
  record = build_record(
      args.name, require_reduction_admission=bool(args.launch)
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  print(
      "[P33.WORKLOAD] VERDICT PASS "
      f"scope={record['scope']} workload={record['workload']} "
      f"dp={record['topology']['dp']} tp={record['topology']['tp']} "
      f"global_trajectories={record['batch']['global_trajectories']} "
      f"local_trajectories={record['batch']['local_trajectories']} "
      f"reduction_admitted={int(record['dp_reduction_admitted'])}",
      flush=True,
  )
  print(f"[P33.WORKLOAD] classification={output}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
