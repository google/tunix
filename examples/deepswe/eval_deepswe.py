#!/usr/bin/env python3
"""Resumable Qwen3-4B-Instruct DeepSWE clean-data evaluation.

One logical report covers 32 clean tasks x 16 stochastic samples. Execution is
deliberately split into eight 4-task x 16-sample physical shards so a shard is
one 64-trajectory concurrency wave with a one-hour hard boundary. Every full
trajectory is fsynced before another result is accepted. Reusing the same
logical-shard index resumes only an exact configuration fingerprint.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from deepswe_eval_artifacts import EvalConfig
from deepswe_eval_artifacts import aggregate_tasks
from deepswe_eval_artifacts import append_record
from deepswe_eval_artifacts import load_records
from deepswe_eval_artifacts import remaining_samples
from deepswe_eval_artifacts import sha256_file
from deepswe_eval_artifacts import task_key
from deepswe_eval_artifacts import trajectory_record
from deepswe_eval_artifacts import write_reports


logger = logging.getLogger("deepswe_eval")


def _required(name: str, fallback: str | None = None) -> str:
  value = os.environ.get(name, "")
  if not value and fallback:
    value = os.environ.get(fallback, "")
  if not value:
    raise ValueError(f"missing required environment variable {name}")
  return value


def _int(name: str, default: int) -> int:
  value = int(os.environ.get(name, str(default)))
  if value < 0:
    raise ValueError(f"{name} must be nonnegative")
  return value


def _build_config() -> tuple[EvalConfig, int, Path]:
  model_id = os.environ.get(
      "MODEL_VERSION", "Qwen/Qwen3-4B-Instruct-2507"
  )
  model_base = os.environ.get(
      "MODEL_BASE_DIR",
      os.environ.get(
          "CANON_P46_MODEL_BASE_DIR", "/mnt/disks/linchai_data/models"
      ),
  )
  model_path = Path(model_base) / model_id.removeprefix("Qwen/")
  whitelist_path = Path(
      _required("GOLD_JSONL", "CANON_P46_GOLD_JSONL")
  ).resolve()
  output_dir = Path(
      _required("OUTPUT_DIR", "CANON_P46_OUTPUT_DIR")
  ).resolve()
  config = EvalConfig(
      model_id=model_id,
      model_path=str(model_path.resolve()),
      dataset_name=os.environ.get(
          "DATASET_NAME", "R2E-Gym/R2E-Gym-Subset"
      ),
      dataset_revision=os.environ.get(
          "DATASET_REVISION", "2e8108ff942f24fcb5686badfaf7f9a8808566d5"
      ),
      dataset_split=os.environ.get("DATASET_SPLIT", "train"),
      dataset_rows=_int("EXPECTED_DATASET_ROWS", 4578),
      whitelist_path=str(whitelist_path),
      whitelist_sha256=_required(
          "GOLD_JSONL_SHA256", "CANON_P46_GOLD_JSONL_SHA256"
      ),
      whitelist_rows=_int("EXPECTED_CLEAN_ROWS", 1851),
      source_commit=_required("CANON_EXPECT_COMMIT"),
      client_image=_required("CANON_CLIENT_IMAGE"),
      topology=_required("CANON_P46_TOPOLOGY"),
      max_model_len=_int("MAX_MODEL_LEN", 20_480),
      max_response_length=_int("MAX_RESPONSE_LENGTH", 16_384),
      max_steps=_int("MAX_STEPS", 50),
      temperature=float(os.environ.get("TEMPERATURE", "1.0")),
      top_p=float(os.environ.get("TOP_P", "1.0")),
      top_k=int(os.environ.get("TOP_K", "0")),
      n_sample=_int("N_SAMPLE", 16),
      logical_tasks=_int("EVAL_LOGICAL_TASKS", 32),
      shard_tasks=_int("EVAL_SHARD_TASKS", 4),
      shard_index=_int("CANON_P46_LOGICAL_SHARD_INDEX", 0),
      max_concurrency=_int("MAX_CONCURRENT", 64),
      trajectory_timeout_secs=_int("TRAJECTORY_TIMEOUT_SECS", 3000),
      per_turn_timeout_secs=_int("PER_TURN_TIMEOUT_SECS", 300),
      step_timeout_secs=_int("STEP_TIMEOUT_SECS", 600),
      reward_timeout_secs=_int("REWARD_TIMEOUT_SECS", 600),
      cleanup_timeout_secs=_int("CLEANUP_TIMEOUT_SECS", 300),
      shard_timeout_secs=_int("SHARD_TIMEOUT_SECS", 3600),
      seed_base=_int("SEED_BASE", 42),
      prefix_cache=os.environ.get("ENABLE_PREFIX_CACHE", "0") == "1",
  )
  config.validate()
  physical_shard = _int("CANON_P46_PHYSICAL_SHARD_INDEX", 0)
  physical_shards = config.logical_tasks // config.shard_tasks
  if physical_shard >= physical_shards:
    raise ValueError(
        f"EVAL_PHYSICAL_SHARD_INDEX must be below {physical_shards}"
    )
  return config, physical_shard, output_dir


def _load_clean_entries(config: EvalConfig) -> list[dict[str, Any]]:
  from datasets import load_dataset

  whitelist_path = Path(config.whitelist_path)
  actual_sha = sha256_file(whitelist_path)
  if actual_sha != config.whitelist_sha256:
    raise ValueError(
        "clean whitelist SHA-256 mismatch: "
        f"expected={config.whitelist_sha256} actual={actual_sha}"
    )
  wanted: set[str] = set()
  with whitelist_path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      if not line.strip():
        continue
      record = json.loads(line)
      image = record.get("docker_image")
      if not isinstance(image, str) or not image:
        raise ValueError(
            f"clean whitelist line {line_number} lacks docker_image"
        )
      if image in wanted:
        raise ValueError(f"duplicate docker_image in clean whitelist: {image}")
      wanted.add(image)
  if len(wanted) != config.whitelist_rows:
    raise ValueError(
        "clean whitelist row contract changed: "
        f"expected={config.whitelist_rows} actual={len(wanted)}"
    )

  dataset = load_dataset(
      config.dataset_name,
      revision=config.dataset_revision,
      split=config.dataset_split,
      cache_dir=os.environ.get(
          "DATASET_CACHE", "/mnt/disks/linchai_data/huggingface/datasets"
      ),
      num_proc=32,
  )
  if len(dataset) != config.dataset_rows:
    raise ValueError(
        "source dataset row contract changed: "
        f"expected={config.dataset_rows} actual={len(dataset)}"
    )
  entries = [dict(entry) for entry in dataset if entry.get("docker_image") in wanted]
  actual = {task_key(entry) for entry in entries}
  if actual != wanted or len(entries) != config.whitelist_rows:
    missing = sorted(wanted - actual)[:5]
    raise ValueError(
        "clean whitelist join is not exact: "
        f"kept={len(entries)} unique={len(actual)} missing={missing}"
    )
  entries.sort(key=task_key)
  logger.info(
      "P46 clean-data gate PASS dataset=%d whitelist=%d sha256=%s",
      len(dataset),
      len(entries),
      actual_sha,
  )
  return entries


def _select_shards(
    all_entries: Sequence[Mapping[str, Any]],
    config: EvalConfig,
    physical_shard: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
  logical_start = config.shard_index * config.logical_tasks
  logical = list(all_entries[logical_start : logical_start + config.logical_tasks])
  if not logical:
    raise ValueError(
        f"logical shard {config.shard_index} starts beyond the clean dataset"
    )
  physical_start = physical_shard * config.shard_tasks
  physical = logical[physical_start : physical_start + config.shard_tasks]
  if not physical:
    raise ValueError(
        "physical shard is empty for this final partial logical group"
    )
  return logical, physical


class _Runtime:
  """Late-imported TPU/Kubernetes runtime so artifact tests stay CPU-only."""

  def __init__(self, config: EvalConfig):
    self.config = config
    model_path = Path(config.model_path)
    if not model_path.is_dir() or not any(model_path.iterdir()):
      raise FileNotFoundError(
          "P46 evaluation requires an existing local checkpoint; refusing to "
          f"download {config.model_path}"
      )

    if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
      import pathwaysutils

      pathwaysutils.initialize()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from kubernetes import client
    from kubernetes import config as k8s_config
    from transformers import AutoTokenizer
    from tunix.generate import mappings
    from tunix.generate import tokenizer_adapter as tok_adapter
    from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
    from tunix.models.qwen3 import model as model_lib
    from tunix.models.qwen3 import params as params_lib
    from tunix.rl.agentic.parser.chat_template_parser import parser
    import numpy as np

    try:
      k8s_config.load_incluster_config()
    except k8s_config.config_exception.ConfigException:
      k8s_config.load_kube_config()
    client.CoreV1Api().list_namespace(timeout_seconds=5)

    devices = jax.devices()
    expected_devices = int(config.topology)
    if len(devices) != expected_devices or len(devices) % 8:
      raise ValueError(
          "P46 evaluation device inventory mismatch: "
          f"expected={expected_devices} actual={len(devices)}"
      )
    dp_size = len(devices) // 8
    mesh = Mesh(np.array(devices).reshape(dp_size, 8), ("fsdp", "tp"))
    self.tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=True
    )
    self.tokenizer_for_agentic = tok_adapter.TokenizerAdapter(self.tokenizer)
    self.chat_parser = parser.QwenChatTemplateParser(self.tokenizer)
    model_config = model_lib.ModelConfig.qwen3_4b_instruct_2507()
    logger.info(
        "Loading %s on evaluation mesh DP%dxTP8 from %s",
        config.model_id,
        dp_size,
        config.model_path,
    )
    model = params_lib.create_model_from_safe_tensors(
        config.model_path, model_config, mesh, dtype=jnp.bfloat16
    )
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=None, model=model, backend="vllm_jax"
    )
    vllm_max_num_seqs = max(1, config.max_concurrency // dp_size)
    sampler_config = VllmConfig(
        mesh=mesh,
        hbm_utilization=float(os.environ.get("VLLM_HBM_UTILIZATION", "0.6")),
        init_with_random_weights=True,
        tpu_backend_type="jax",
        server_mode=True,
        tensor_parallel_size=8,
        data_parallel_size=dp_size,
        mapping_config=mapping_config,
        engine_kwargs={
            "model": config.model_path,
            "max_model_len": config.max_model_len,
            "max_num_seqs": vllm_max_num_seqs,
            "max_num_batched_tokens": config.max_response_length,
            "enable_prefix_caching": False,
            "kv_cache_metrics": True,
            "disable_log_stats": False,
        },
    )
    self.sampler = VllmSampler(
        tokenizer=self.tokenizer, config=sampler_config
    )
    from flax import nnx

    self.sampler.load_checkpoint(nnx.state(model))
    logger.info(
        "P46 evaluation runtime PASS devices=%d dp=%d tp=8 prefix_cache=off",
        len(devices),
        dp_size,
    )

  def model_call(
      self,
      chat_completions,
      env,
      *,
      max_generation_steps=None,
      request_timeout_s=None,
      **unused_kwargs,
  ):
    config = self.config
    prompt = self.chat_parser.parse(
        chat_completions, add_generation_prompt=True, is_first_msg=True
    )
    prompt_tokens = len(self.tokenizer.encode(prompt))
    remaining_context = config.max_model_len - prompt_tokens
    if remaining_context <= 0:
      raise ValueError(
          f"prompt exceeds max_model_len: {prompt_tokens}/{config.max_model_len}"
      )
    generation_steps = min(
        max_generation_steps or config.max_response_length,
        remaining_context,
    )
    return self.sampler(
        prompt,
        max_generation_steps=generation_steps,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=env.extra_kwargs["sample_seed"],
        echo=False,
        request_timeout_s=request_timeout_s,
    )

async def _run_evaluation(
    config: EvalConfig,
    entries: Sequence[Mapping[str, Any]],
    pending: Sequence[tuple[Mapping[str, Any], int]],
    output_path: Path,
) -> tuple[int, bool]:
  from guarded_swe_env import GuardedSWEEnv
  from swe_agent import SWEAgent
  from swe_env import SWEEnv
  from tunix.rl.agentic import utils as agentic_utils
  from tunix.rl.agentic.agents import agent_types
  from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
  from tunix.rl.agentic.trajectory import trajectory_collect_engine

  runtime = _Runtime(config)
  enable_guard = os.environ.get("ENABLE_GUARD", "false").lower() == "true"
  plan = list(pending)
  started: dict[int, float] = {}

  class EvalTrajectoryCollectEngine(
      trajectory_collect_engine.TrajectoryCollectEngine
  ):
    async def collect(self, mode: str = "Conversation"):
      try:
        return await super().collect(mode)
      except TimeoutError as exc:
        if "cleanup" in str(exc).lower():
          raise
        logger.exception("isolating timed-out evaluation trajectory")
      except Exception:
        logger.exception("isolating failed evaluation trajectory")
      self.agent.trajectory.status = agent_types.TrajectoryStatus.FAILED
      self.agent.trajectory.reward = 0.0
      return self.agent.trajectory

  def pairs_generator():
    env_cls = GuardedSWEEnv if enable_guard else SWEEnv
    for pair_index, (entry, sample_index) in enumerate(plan):
      agent = SWEAgent()
      env = env_cls(
          entry=dict(entry),
          max_steps=config.max_steps,
          pair_index=pair_index,
          group_id=task_key(entry),
          step_timeout=config.step_timeout_secs,
          reward_timeout=config.reward_timeout_secs,
      )
      env.extra_kwargs["sample_index"] = sample_index
      env.extra_kwargs["sample_seed"] = config.sample_seed(
          task_key(entry), sample_index
      )
      started[pair_index] = time.monotonic()
      yield agent, env

  orchestrator = RolloutOrchestrator(
      engine_cls=EvalTrajectoryCollectEngine,
      engine_kwargs={
          "model_call": runtime.model_call,
          "timeout": config.trajectory_timeout_secs,
          "per_turn_timeout": config.per_turn_timeout_secs,
          "cleanup_timeout": config.cleanup_timeout_secs,
          "max_response_length": config.max_response_length,
          "tokenizer": runtime.tokenizer_for_agentic,
          "chat_parser": runtime.chat_parser,
      },
      max_concurrency=config.max_concurrency,
      rollout_sync_lock=agentic_utils.RolloutSyncLock(),
  )
  producer = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs_generator(),
          group_size=1,
          group_key_fn=lambda i, env, traj: (
              env.extra_kwargs["group_id"],
              env.extra_kwargs["sample_index"],
          ),
          collect_mode="Trajectory",
      )
  )
  completed = 0

  async def consume() -> None:
    nonlocal completed
    await asyncio.sleep(0)
    async for batch in orchestrator.yield_batches(batch_size=1):
      for item in batch:
        entry, sample_index = plan[item.pair_index]
        record = trajectory_record(
            config,
            entry=entry,
            sample_index=sample_index,
            trajectory=item.traj,
            elapsed_secs=time.monotonic() - started[item.pair_index],
        )
        append_record(output_path, record)
        completed += 1
        logger.info(
            "P46_EVAL_TRAJECTORY task=%s sample=%d status=%s reward=%s "
            "completed=%d/%d",
            record["task_key"],
            sample_index,
            record["status"],
            record["reward"],
            completed,
            len(plan),
        )
    await producer

  timed_out = False
  try:
    await asyncio.wait_for(consume(), timeout=config.shard_timeout_secs)
  except asyncio.TimeoutError:
    timed_out = True
    logger.error(
        "P46_EVAL_SHARD_TIMEOUT completed=%d/%d deadline=%ds",
        completed,
        len(plan),
        config.shard_timeout_secs,
    )
  finally:
    if not producer.done():
      producer.cancel()
    await asyncio.gather(producer, return_exceptions=True)
  return completed, timed_out


def main() -> int:
  logging.basicConfig(
      stream=sys.stdout,
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s",
  )
  config, physical_shard, output_dir = _build_config()
  all_entries = _load_clean_entries(config)
  logical_entries, physical_entries = _select_shards(
      all_entries, config, physical_shard
  )
  trajectory_dir = output_dir / "trajectories"
  pattern = trajectory_dir / f"{config.run_tag}.*.jsonl"
  existing = load_records(
      glob.glob(str(pattern)),
      config=config,
      allowed_task_keys=(task_key(entry) for entry in logical_entries),
  )
  pending = remaining_samples(physical_entries, existing, config=config)
  logger.info(
      "P46_EVAL_START tag=%s logical_shard=%d physical_shard=%d "
      "logical_tasks=%d physical_tasks=%d existing=%d pending=%d",
      config.run_tag,
      config.shard_index,
      physical_shard,
      len(logical_entries),
      len(physical_entries),
      len(existing),
      len(pending),
  )
  timed_out = False
  if pending:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    output_path = trajectory_dir / (
        f"{config.run_tag}.p{physical_shard}."
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    )
    _, timed_out = asyncio.run(
        _run_evaluation(config, physical_entries, pending, output_path)
    )

  accumulated = load_records(
      glob.glob(str(pattern)),
      config=config,
      allowed_task_keys=(task_key(entry) for entry in logical_entries),
  )
  reports = aggregate_tasks(logical_entries, accumulated, config=config)
  incomplete = sum(item["category"] == "incomplete" for item in reports)
  logger.info(
      "P46_EVAL_PROGRESS tag=%s trajectories=%d/%d incomplete_tasks=%d",
      config.run_tag,
      len(accumulated),
      len(logical_entries) * config.n_sample,
      incomplete,
  )
  if timed_out:
    return 2
  if incomplete:
    print(
        f"P46_EVAL_SUBSHARD_PASS tag={config.run_tag} "
        f"physical_shard={physical_shard} pending_logical_tasks={incomplete}",
        flush=True,
    )
    return 0
  report_dir = output_dir / "reports"
  summary = write_reports(report_dir, reports, config=config)
  print(
      "P46_EVAL_LOGICAL_REPORT_PASS "
      f"tag={config.run_tag} tasks={summary['tasks']} "
      f"solve_ratio={summary['solve_ratio']} "
      f"summary_sha256={summary['summary_sha256']}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
