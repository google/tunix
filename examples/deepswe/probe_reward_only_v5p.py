#!/usr/bin/env python3
"""One-host v5p L1/L2 probe for DeepSWE reward-only evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time

import deepswe_eval_artifacts as artifacts
import deepswe_reward_only_parity as parity
import eval_deepswe


def _payload_bytes(output) -> int:
  payload = {
      "text": output.text,
      "tokens": [item.tolist() for item in output.tokens],
      "logprobs": artifacts.serializable(output.logprobs),
  }
  return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _capture_engine_rng(runtime):
  import jax

  runner = runtime.sampler._model_runner  # pylint: disable=protected-access
  current = runner.rng_params_for_sampling
  return jax.device_get(current), current.sharding


def _restore_engine_rng(runtime, snapshot) -> None:
  import jax

  value, sharding = snapshot
  runner = runtime.sampler._model_runner  # pylint: disable=protected-access
  runner.rng_params_for_sampling = jax.device_put(value, sharding)


def _sample_once(
    runtime, prompt: str, *, return_logprobs: bool, rng_snapshot
):
  # TPU/JAX rejects SamplingParams.seed. Restoring the idle engine's exact
  # engine-level key gives both diagnostic arms the same prompt-start RNG
  # state without pretending production requests have independent seeds.
  _restore_engine_rng(runtime, rng_snapshot)
  runtime.sampler.config.return_logprobs = return_logprobs
  started = time.monotonic()
  output = runtime.sampler(
      prompt,
      max_generation_steps=128,
      temperature=1.0,
      top_p=1.0,
      top_k=0,
      echo=False,
      request_timeout_s=300,
  )
  return output, time.monotonic() - started


def _container_ids() -> set[str]:
  import docker

  client = docker.from_env()
  if client.ping() is not True:
    raise RuntimeError("Docker daemon did not answer one-host cleanup probe")
  return {container.id for container in client.containers.list(all=True)}


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repeats", type=int, default=3)
  args = parser.parse_args()
  if args.repeats < 2:
    raise ValueError("one-host timing requires at least two repeats per arm")
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite one-host report: {args.output}")

  config, physical_shard, output_dir = eval_deepswe._build_config()  # pylint: disable=protected-access
  if not config.onehost_probe or config.topology != "4":
    raise ValueError("probe_reward_only_v5p requires the isolated one-host config")
  runtime = eval_deepswe._Runtime(config)  # pylint: disable=protected-access
  prompt = runtime.tokenizer.apply_chat_template(
      [{"role": "user", "content": "Return the word READY and nothing else."}],
      tokenize=False,
      add_generation_prompt=True,
  )
  rng_snapshot = _capture_engine_rng(runtime)

  # Compile/warm both shapes before collecting timing. L2 never treats token
  # identity as a hard gate.
  _sample_once(
      runtime, prompt, return_logprobs=True, rng_snapshot=rng_snapshot
  )
  _sample_once(
      runtime, prompt, return_logprobs=False, rng_snapshot=rng_snapshot
  )
  arm_outputs = {True: [], False: []}
  arm_times = {True: [], False: []}
  # Alternate AB/BA so the timing delta is not just monotonic host drift.
  for repeat_index in range(args.repeats):
    order = (True, False) if repeat_index % 2 == 0 else (False, True)
    for return_logprobs in order:
      output, elapsed = _sample_once(
          runtime,
          prompt,
          return_logprobs=return_logprobs,
          rng_snapshot=rng_snapshot,
      )
      arm_outputs[return_logprobs].append(output)
      arm_times[return_logprobs].append(elapsed)
  with_logprobs = arm_outputs[True][-1]
  reward_only = arm_outputs[False][-1]
  on_times = arm_times[True]
  off_times = arm_times[False]
  if with_logprobs.logprobs is None:
    raise RuntimeError("logprob observation arm returned no logprobs")
  if reward_only.logprobs is not None:
    raise RuntimeError("reward-only arm returned a logprob payload")
  l2 = parity.classify_l2_tokens(
      with_logprobs.tokens[0].tolist(), reward_only.tokens[0].tolist()
  )

  all_entries = eval_deepswe._load_clean_entries(config)  # pylint: disable=protected-access
  _, physical_entries = eval_deepswe._select_shards(  # pylint: disable=protected-access
      all_entries, config, physical_shard
  )
  trajectory_path = output_dir / "onehost.trajectories.jsonl"
  before_containers = _container_ids()
  completed, timed_out = asyncio.run(
      eval_deepswe._run_evaluation(  # pylint: disable=protected-access
          config,
          physical_entries,
          [(physical_entries[0], 0, 0)],
          trajectory_path,
          runtime=runtime,
      )
  )
  after_containers = _container_ids()
  records = artifacts.load_records(
      [trajectory_path],
      config=config,
      allowed_task_keys={artifacts.task_key(physical_entries[0])},
  )
  cleanup_new_containers = sorted(after_containers - before_containers)
  l1 = {
      "completed": completed,
      "timed_out": timed_out,
      "records": len(records),
      "record_valid": len(records) == 1 and records[0].get("valid") is True,
      "trajectory_mode": records[0].get("trajectory_mode") if records else None,
      "sampled_by": records[0].get("sampled_by") if records else None,
      "cleanup_new_containers": cleanup_new_containers,
  }
  l1["verdict"] = "PASS" if (
      completed == 1
      and not timed_out
      and l1["record_valid"]
      and l1["trajectory_mode"] == "reward_only_no_logprobs"
      and l1["sampled_by"] == config.sampled_by
      and not cleanup_new_containers
  ) else "FAIL"
  report = {
      "schema": "canon.p46.deepswe-eval.reward-only-onehost.v1",
      "source_commit": config.source_commit,
      "trajectory_mode": config.trajectory_mode,
      "sampled_by": config.sampled_by,
      "sampling_rng_mode": config.sampling_rng_mode,
      "engine_seed": config.seed_base,
      "parity_rng_reset": "same_engine_key_before_each_idle_request",
      "request_contract": {
          "logprob_arm": {"sampled_logprobs": 1, "prompt_logprobs": 0},
          "reward_only_arm": {
              "sampled_logprobs": None,
              "prompt_logprobs": None,
              "host_extraction": "skipped",
          },
      },
      "timing": {
          "repeats": args.repeats,
          "logprob_arm_secs": on_times,
          "reward_only_arm_secs": off_times,
          "logprob_arm_median_secs": statistics.median(on_times),
          "reward_only_arm_median_secs": statistics.median(off_times),
      },
      "artifact_bytes": {
          "logprob_arm": _payload_bytes(with_logprobs),
          "reward_only_arm": _payload_bytes(reward_only),
          "trajectory_jsonl": trajectory_path.stat().st_size,
      },
      "l1": l1,
      "l2": l2,
      "l3": "NOT_RUN_REQUIRES_64_CHIP_PAIRED_N16",
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  if l1["verdict"] != "PASS":
    print(json.dumps(report, sort_keys=True), flush=True)
    return 1
  print(
      "P46_REWARD_ONLY_ONEHOST_PASS "
      f"l1=PASS l2={l2['classification']} report={args.output}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
