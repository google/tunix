# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SW3 R2Orch Inner-Loop trainer: trainable Graph Director + frozen Workers.

Run from the repository root with `python -m examples.swarm_r2orch.train`.
Supports a deterministic CPU categorical fixture (`--backend cpu`) and multi-slice
or single-slice TPU training with Qwen3-4B (`--backend qwen`), partitioning the
accelerator mesh into three isolated sub-meshes (`director_mesh`, `worker_mesh`,
`train_mesh`) and asserting zero weight drift on the frozen Worker policy.
"""

import argparse
import asyncio
import collections
from collections.abc import Callable, Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path

from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from examples.swarm_math import train as math_train
from examples.swarm_r2orch import cpu_fixture
from examples.swarm_r2orch import scoring
from examples.swarm_r2orch import workflow
from tunix.common import configs
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.swarm import runtime as runtime_lib
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.utils import sharding_utils

ROOT = Path(__file__).parent
WORKER_SYSTEM_PROMPT = "You are a helpful, accurate, and concise specialist Worker."


@dataclasses.dataclass(frozen=True)
class PolicyBundle:
  """Initialized Director trainer actor, dual samplers, encoders, and hooks."""

  actor: nnx.Module
  director_train_sampler: object
  director_eval_sampler: object
  worker_sampler: object
  encode_director: Callable[[str], Sequence[int]]
  encode_worker: Callable[[str], Sequence[int]]
  pad_id: int
  eos_id: int
  sync_director: Callable[[], None]
  worker_state_fn: Callable[[], object]
  close_fn: Callable[[], None]


def load_tasks(path: str | Path) -> list[scoring.R2OrchTask]:
  """Loads and validates uniform `R2OrchTask` rows from a JSONL file."""
  rows = [
      scoring.R2OrchTask(**json.loads(line))
      for line in Path(path).read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if not rows or len({r.task_id for r in rows}) != len(rows):
    raise ValueError("Task file must be non-empty with unique task_id values.")
  return rows


def validate_r2orch_config(config: dict[str, object]) -> None:
  """Validates SW3 R2Orch training and numerical invariants."""
  if config["enable_thinking"]:
    raise ValueError("R2Orch inner-loop recipe requires enable_thinking=false.")
  if config["rollout_engine"] not in ("vanilla", "vllm"):
    raise ValueError("rollout_engine must be 'vanilla' or 'vllm'.")
  if (
      config["director_train_temperature"] != 1.0
      or config["director_train_top_p"] != 1.0
  ):
    raise ValueError(
        "Director training rollouts require temperature=1.0 and top_p=1.0."
    )
  if config["use_flash_attention"] or config["dtype"] != "float32":
    raise ValueError(
        "TPU gradient stability requires dtype='float32' and"
        " use_flash_attention=false."
    )
  if config["director_policy_id"] == config["worker_policy_id"]:
    raise ValueError(
        "Director and frozen Worker must have distinct policy_id identifiers."
    )
  d_frac = float(config["director_rollout_fraction"])
  w_frac = float(config["worker_rollout_fraction"])
  if min(d_frac, w_frac) < 0.0 or d_frac + w_frac >= 1.0:
    raise ValueError("Invalid director/worker rollout mesh fractions.")


def pytree_leaf_norms(state: object) -> np.ndarray:
  """Computes per-leaf float32 L2 norms on-device for fast drift attestation."""
  leaves = jax.tree_util.tree_leaves(state)
  return np.asarray(
      [
          float(jnp.linalg.norm(jnp.asarray(leaf, dtype=jnp.float32)))
          for leaf in leaves
      ],
      dtype=np.float64,
  )


def build_tri_meshes(
    config: dict[str, object], *, backend: str
) -> tuple[jax.sharding.Mesh, jax.sharding.Mesh, jax.sharding.Mesh]:
  """Builds `(train_mesh, director_mesh, worker_mesh)` across visible devices."""
  devices = np.asarray(
      jax.devices()[:1] if backend == "cpu" else jax.devices()
  )
  tp = int(config["tp_size"])
  d_frac = float(config["director_rollout_fraction"])
  w_frac = float(config["worker_rollout_fraction"])
  if backend == "cpu" or config["rollout_engine"] != "vllm" or (d_frac + w_frac) == 0.0:
    mesh = math_train.reshape_mesh(
        devices, tp, int(config["micro_batch_size"]), "micro_batch_size"
    )
    return mesh, mesh, mesh
  d_count = int(len(devices) * d_frac)
  w_count = int(len(devices) * w_frac)
  if min(d_count, w_count) < tp or d_count + w_count >= len(devices):
    raise ValueError(
        f"Cannot split {len(devices)} devices with fractions ({d_frac}, {w_frac})"
        f" and tp_size={tp}."
    )
  director_mesh = math_train.reshape_mesh(
      devices[:d_count],
      tp,
      int(config["rollout_batch_size"]),
      "rollout_batch_size",
  )
  worker_mesh = math_train.reshape_mesh(
      devices[d_count : d_count + w_count],
      tp,
      int(config["rollout_batch_size"]),
      "rollout_batch_size",
  )
  train_mesh = math_train.reshape_mesh(
      devices[d_count + w_count :],
      tp,
      int(config["micro_batch_size"]),
      "micro_batch_size",
  )
  return train_mesh, director_mesh, worker_mesh


def _build_single_vllm_engine(
    *,
    server_id: str,
    actor: nnx.Module,
    tokenizer: object,
    config: dict[str, object],
    mesh: jax.sharding.Mesh,
    model_path_str: str,
    eos_tokens: list[int],
    max_prompt_length: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    return_logprobs: bool,
):
  """Instantiates and weight-initializes one `InprocessVllmSampler` on `mesh`."""
  from tunix.dist_rl.rollout import inprocess_vllm_sampler  # pylint: disable=g-import-not-at-top
  from tunix.generate import mappings  # pylint: disable=g-import-not-at-top

  for env_key, env_val in (
      ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
      ("VLLM_TPU_RPA_VERSION", "2"),
      ("DISABLE_MOSAIC_ATTN", "1"),
      ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
      ("NEW_MODEL_DESIGN", "1"),
  ):
    os.environ.setdefault(env_key, env_val)
  tp_size = int(mesh.shape["tp"])
  dp_size = int(mesh.shape["fsdp"])
  rcfg = configs.RolloutConfig(
      max_prompt_length=max_prompt_length,
      max_tokens_to_generate=max_response_length,
      kv_cache_size=max_prompt_length + max_response_length,
      temperature=temperature,
      top_p=top_p,
      top_k=0,
      eos_tokens=eos_tokens,
      return_logprobs=return_logprobs,
      tensor_parallel_size=tp_size,
      data_parallel_size=dp_size,
      rollout_mapping_config=mappings.MappingConfig.from_model(
          actor, backend="vllm_jax"
      ),
      rollout_vllm_model_version=model_path_str,
      rollout_vllm_server_mode=True,
      rollout_vllm_init_with_random_weights=True,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_hbm_utilization=float(config["vllm_hbm_utilization"]),
      rollout_vllm_reshard_chunk_size=int(config["vllm_reshard_chunk_size"]),
      rollout_vllm_async_scheduling=bool(config["vllm_async_scheduling"]),
      rollout_vllm_max_num_seqs=int(config["vllm_max_num_seqs"]),
      rollout_vllm_max_num_batched_tokens=int(
          config["vllm_max_num_batched_tokens"]
      ),
      rollout_vllm_kwargs={
          "enable_prefix_caching": bool(config["vllm_enable_prefix_caching"]),
          "dtype": "bfloat16",
          "tokenizer": model_path_str,
          "seed": int(config["seed"]),
          "kv_cache_metrics": True,
          "disable_log_stats": False,
      },
  )
  with mesh:
    raw_engine = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id=server_id,
        tokenizer=tokenizer,
        config=rcfg,
        mesh=mesh,
        max_concurrency=int(config["vllm_max_num_seqs"]) * dp_size,
    )
    raw_engine.initialize()
    raw_engine.update_params(nnx.state(actor, nnx.Param))
  return raw_engine, math_train.VllmMathSampler(raw_engine, tokenizer, eos_tokens)


def create_policy_bundle(
    backend: str,
    config: dict[str, object],
    train_mesh: jax.sharding.Mesh,
    director_mesh: jax.sharding.Mesh,
    worker_mesh: jax.sharding.Mesh,
    model_path: Path | None,
) -> PolicyBundle:
  """Builds trainable Director and frozen Worker samplers for CPU or Qwen TPU."""
  if backend == "cpu":
    actor = cpu_fixture.make_director_policy()
    director_rollout = nnx.clone(actor)
    frozen_worker = cpu_fixture.make_frozen_worker_policy()

    def sync_cpu() -> None:
      nnx.update(
          director_rollout,
          jax.tree.map(lambda x: x.copy(), nnx.state(actor)),
      )

    sync_cpu()
    d_sampler = cpu_fixture.CategoricalSampler(director_rollout)
    w_sampler = cpu_fixture.CategoricalSampler(frozen_worker)
    return PolicyBundle(
        actor=actor,
        director_train_sampler=d_sampler,
        director_eval_sampler=d_sampler,
        worker_sampler=w_sampler,
        encode_director=cpu_fixture.encode_director,
        encode_worker=cpu_fixture.encode_worker,
        pad_id=0,
        eos_id=0,
        sync_director=sync_cpu,
        worker_state_fn=lambda: nnx.state(frozen_worker, nnx.Param),
        close_fn=lambda: None,
    )

  if (
      config["model_id"] != "Qwen/Qwen3-4B"
      or model_path is None
      or not Path(model_path).is_dir()
      or Path(model_path).resolve().name != config["model_revision"]
  ):
    raise ValueError(
        "Qwen backend requires --model-path pointing to the pinned Qwen3-4B"
        " HF snapshot directory."
    )
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  model_path_str = str(model_path)
  tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
  actor, _ = automodel.AutoModel.from_pretrained(
      config["model_id"],
      train_mesh,
      model_path=model_path_str,
      remat_config=config["remat_config"],
      dtype=config["dtype"],
      load_dtype=config["load_dtype"],
      use_flash_attention=config["use_flash_attention"],
      flash_attention_block_size=config["flash_attention_block_size"],
  )
  eos_tokens = math_train._resolve_eos_tokens(  # pylint: disable=protected-access
      Path(model_path), tokenizer.eos_token_id
  )
  encode_dir = workflow.role_chat_encoder(
      tokenizer, workflow.DIRECTOR_SYSTEM_PROMPT, int(config["max_prompt_length"])
  )
  encode_wrk = workflow.role_chat_encoder(
      tokenizer, WORKER_SYSTEM_PROMPT, int(config["worker_max_prompt_length"])
  )
  if config["rollout_engine"] == "vllm":
    # Initialize the frozen Worker engine FIRST from untouched base weights;
    # `sync_director` never touches `worker_raw` again.
    worker_raw, worker_wrapped = _build_single_vllm_engine(
        server_id="r2orch_frozen_worker",
        actor=actor,
        tokenizer=tokenizer,
        config=config,
        mesh=worker_mesh,
        model_path_str=model_path_str,
        eos_tokens=eos_tokens,
        max_prompt_length=int(config["worker_max_prompt_length"]),
        max_response_length=int(config["worker_max_response_length"]),
        temperature=float(config["worker_temperature"]),
        top_p=float(config["worker_top_p"]),
        return_logprobs=False,
    )
    director_raw, director_wrapped = _build_single_vllm_engine(
        server_id="r2orch_director",
        actor=actor,
        tokenizer=tokenizer,
        config=config,
        mesh=director_mesh,
        model_path_str=model_path_str,
        eos_tokens=eos_tokens,
        max_prompt_length=int(config["max_prompt_length"]),
        max_response_length=int(config["max_response_length"]),
        temperature=float(config["director_train_temperature"]),
        top_p=float(config["director_train_top_p"]),
        return_logprobs=True,
    )

    def sync_vllm() -> None:
      weights = nnx.state(actor, nnx.Param)
      with director_mesh:
        director_raw.update_params(weights)

    def close_vllm() -> None:
      asyncio.run(director_wrapped.close())
      asyncio.run(worker_wrapped.close())

    return PolicyBundle(
        actor=actor,
        director_train_sampler=director_wrapped,
        director_eval_sampler=director_wrapped,
        worker_sampler=worker_wrapped,
        encode_director=encode_dir,
        encode_worker=encode_wrk,
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
        sync_director=sync_vllm,
        worker_state_fn=lambda: worker_raw.transformer_state,
        close_fn=close_vllm,
    )

  frozen_model = nnx.clone(actor)
  nnx.update(frozen_model, jax.tree.map(lambda x: x.copy(), nnx.state(actor)))
  worker_native = math_train.NativeMathSampler(
      actor=frozen_model,
      tokenizer=tokenizer,
      stage="worker",
      rollout_model=frozen_model,
      batch_size=int(config["rollout_batch_size"]),
      rollout_config=configs.RolloutConfig(
          max_prompt_length=int(config["worker_max_prompt_length"]),
          max_tokens_to_generate=int(config["worker_max_response_length"]),
          kv_cache_size=int(config["worker_max_prompt_length"])
          + int(config["worker_max_response_length"]),
          temperature=float(config["worker_temperature"]),
          top_p=float(config["worker_top_p"]),
          top_k=None,
          eos_tokens=eos_tokens,
          seed=int(config["seed"]),
          return_logprobs=True,
      ),
  )
  dir_model = nnx.clone(actor)
  nnx.update(dir_model, jax.tree.map(lambda x: x.copy(), nnx.state(actor)))
  dir_samplers = [
      math_train.NativeMathSampler(
          actor=actor,
          tokenizer=tokenizer,
          stage=stage,
          rollout_model=dir_model,
          batch_size=int(config["rollout_batch_size"]),
          rollout_config=configs.RolloutConfig(
              max_prompt_length=int(config["max_prompt_length"]),
              max_tokens_to_generate=int(config["max_response_length"]),
              kv_cache_size=int(config["max_prompt_length"])
              + int(config["max_response_length"]),
              temperature=float(config[f"director_{stage}_temperature"]),
              top_p=float(config[f"director_{stage}_top_p"]),
              top_k=None,
              eos_tokens=eos_tokens,
              seed=int(config["seed"]),
              return_logprobs=True,
          ),
      )
      for stage in ("train", "eval")
  ]

  def sync_native() -> None:
    dir_samplers[0].sync()
    dir_samplers[1].sampler.transformer_state = dir_samplers[
        0
    ].sampler.transformer_state

  return PolicyBundle(
      actor=actor,
      director_train_sampler=dir_samplers[0],
      director_eval_sampler=dir_samplers[1],
      worker_sampler=worker_native,
      encode_director=encode_dir,
      encode_worker=encode_wrk,
      pad_id=tokenizer.pad_token_id,
      eos_id=tokenizer.eos_token_id,
      sync_director=sync_native,
      worker_state_fn=lambda: worker_native.sampler.transformer_state,
      close_fn=lambda: None,
  )


def select_stratified_tasks(
    tasks: Sequence[scoring.R2OrchTask],
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> list[scoring.R2OrchTask]:
  """Selects `batch_size` tasks balanced round-robin across dataset domains."""
  by_domain: dict[str, list[scoring.R2OrchTask]] = collections.defaultdict(list)
  for task in tasks:
    by_domain[task.dataset].append(task)
  domains = sorted(by_domain)
  per_domain_orders: dict[str, list[scoring.R2OrchTask]] = {}
  for idx, dom in enumerate(domains):
    bucket = by_domain[dom]
    rng = np.random.default_rng(seed + idx * 1009 + step // max(1, len(bucket)))
    perm = rng.permutation(len(bucket))
    per_domain_orders[dom] = [bucket[int(i)] for i in perm]
  picked: list[scoring.R2OrchTask] = []
  seen_ids: set[str] = set()
  cursor = step * max(1, batch_size // len(domains))
  while len(picked) < batch_size:
    for dom in domains:
      if len(picked) >= batch_size:
        break
      ordered = per_domain_orders[dom]
      cand = ordered[cursor % len(ordered)]
      if cand.task_id not in seen_ids:
        seen_ids.add(cand.task_id)
        picked.append(cand)
    cursor += 1
  return picked


def summarize_outcomes(
    outcomes: Sequence[workflow.R2OrchOutcome], *, k: int
) -> dict[str, object]:
  """Aggregates accuracy, pass@k, Worker cost, and DAG topology statistics."""
  per_task_hits: dict[str, list[float]] = collections.defaultdict(list)
  per_domain_outcomes: dict[str, list[workflow.R2OrchOutcome]] = (
      collections.defaultdict(list)
  )
  for out in outcomes:
    per_task_hits[f"{out.dataset}::{out.episode.task_id}"].append(out.correctness)
    per_domain_outcomes[out.dataset].append(out)
  domain_tasks: dict[str, list[list[float]]] = collections.defaultdict(list)
  for key, hits in per_task_hits.items():
    dom = key.split("::", 1)[0]
    domain_tasks[dom].append(hits)
  by_dataset = {}
  for dom in sorted(domain_tasks):
    rows = domain_tasks[dom]
    dom_outs = per_domain_outcomes[dom]
    by_dataset[dom] = {
        "tasks": len(rows),
        "avg_at_k": float(np.mean([np.mean(h) for h in rows])),
        "pass_at_k": float(np.mean([max(h) for h in rows])),
        "finish_rate": float(np.mean([o.finished for o in dom_outs])),
        "mean_director_turns": float(
            np.mean([o.director_turns for o in dom_outs])
        ),
        "mean_worker_calls": float(np.mean([o.worker_calls for o in dom_outs])),
        "mean_worker_cost": float(np.mean([o.worker_cost for o in dom_outs])),
        "multi_agent_frac": float(
            np.mean([o.num_nodes >= 2 for o in dom_outs])
        ),
        "multi_layer_frac": float(
            np.mean([o.max_layer >= 1 for o in dom_outs])
        ),
        "parallel_wave_frac": float(
            np.mean([o.parallel_wave for o in dom_outs])
        ),
    }
  return {
      "accuracy": float(np.mean([o.correctness for o in outcomes])),
      "attempts": len(outcomes),
      "k": k,
      "finish_rate": float(np.mean([o.finished for o in outcomes])),
      "mean_director_turns": float(
          np.mean([o.director_turns for o in outcomes])
      ),
      "mean_worker_calls": float(np.mean([o.worker_calls for o in outcomes])),
      "mean_worker_cost": float(np.mean([o.worker_cost for o in outcomes])),
      "single_agent_frac": float(
          np.mean([o.num_nodes == 1 for o in outcomes])
      ),
      "multi_agent_frac": float(np.mean([o.num_nodes >= 2 for o in outcomes])),
      "multi_layer_frac": float(np.mean([o.max_layer >= 1 for o in outcomes])),
      "parallel_wave_frac": float(np.mean([o.parallel_wave for o in outcomes])),
      "macro_avg_at_k": float(
          np.mean([b["avg_at_k"] for b in by_dataset.values()])
      ),
      "macro_pass_at_k": float(
          np.mean([b["pass_at_k"] for b in by_dataset.values()])
      ),
      "benchmarks": by_dataset,
  }


class R2OrchCollector:
  """Collects R2Orch episodes binding trainable Director and frozen Workers."""

  def __init__(
      self,
      bundle: PolicyBundle,
      tasks: Sequence[scoring.R2OrchTask],
      config: dict[str, object],
      *,
      trace_path: Path | None = None,
  ):
    self.bundle = bundle
    self.tasks = list(tasks)
    self.config = config
    self.trace_path = trace_path
    self.last_round_summary: dict[str, object] = {}
    self.flow = workflow.GraphDirectorWorkflow(
        max_rounds=int(config["max_rounds"]),
        max_agents=int(config["max_agents"]),
        relay_max_chars=int(config["relay_max_chars"]),
        cost_lambda=float(config["cost_lambda"]),
    )

  async def collect_outcomes(
      self,
      tasks: Sequence[scoring.R2OrchTask],
      version: int,
      step: int,
      *,
      evaluate: bool,
  ) -> list[workflow.R2OrchOutcome]:
    cfg = self.config
    attempts = int(cfg["eval_attempts"] if evaluate else cfg["group_size"])
    stage = "eval" if evaluate else "train"
    d_sampler = (
        self.bundle.director_eval_sampler
        if evaluate
        else self.bundle.director_train_sampler
    )
    coros = []
    for index, task in enumerate(tasks):
      for attempt in range(attempts):
        seed = int(
            np.random.SeedSequence(
                [int(cfg["seed"]), step, index, attempt, int(evaluate)]
            ).generate_state(1)[0]
        )
        director_binding = runtime_lib.PolicyBinding(
            policy_id=str(cfg["director_policy_id"]),
            policy_version=version,
            sampler=d_sampler,
            encode=self.bundle.encode_director,
            sampling_params=sampler_lib.SamplingParams(
                max_tokens=int(cfg["max_response_length"]),
                seed=seed,
                temperature=float(cfg[f"director_{stage}_temperature"]),
                top_p=float(cfg[f"director_{stage}_top_p"]),
                return_logprobs=True,
            ),
            max_prompt_length=int(cfg["max_prompt_length"]),
            trainable=True,
        )
        worker_binding = runtime_lib.PolicyBinding(
            policy_id=str(cfg["worker_policy_id"]),
            policy_version=0,
            sampler=self.bundle.worker_sampler,
            encode=self.bundle.encode_worker,
            sampling_params=sampler_lib.SamplingParams(
                max_tokens=int(cfg["worker_max_response_length"]),
                seed=seed + 100_000,
                temperature=float(cfg["worker_temperature"]),
                top_p=float(cfg["worker_top_p"]),
                return_logprobs=False,
            ),
            max_prompt_length=int(cfg["worker_max_prompt_length"]),
            trainable=False,
        )
        rt = runtime_lib.RecordingRuntime.from_bindings(
            {
                workflow.DIRECTOR_ROLE: director_binding,
                workflow.WORKER_ROLE: worker_binding,
            },
            episode_id=f"{stage}/{step}/{task.task_id}/{attempt}",
        )
        coros.append(self.flow.run_detailed(task, rt, evaluate=evaluate))
    return list(await asyncio.gather(*coros))

  async def __call__(self, version: int, step: int):
    chosen = select_stratified_tasks(
        self.tasks,
        step=step,
        batch_size=int(self.config["tasks_per_step"]),
        seed=int(self.config["seed"]),
    )
    outcomes = await self.collect_outcomes(chosen, version, step, evaluate=False)
    self.last_round_summary = summarize_outcomes(
        outcomes, k=int(self.config["group_size"])
    )
    episodes = [o.episode for o in outcomes]
    if (
        self.trace_path is not None
        and step % int(self.config["trace_every_steps"]) == 0
    ):
      with Path(self.trace_path).open("a", encoding="utf-8") as stream:
        for out in outcomes:
          stream.write(
              json.dumps({
                  "step": step,
                  "dataset": out.dataset,
                  "correctness": out.correctness,
                  "worker_cost": out.worker_cost,
                  "finished": out.finished,
                  "director_turns": out.director_turns,
                  "worker_calls": out.worker_calls,
                  "num_nodes": out.num_nodes,
                  "max_layer": out.max_layer,
                  "episode": dataclasses.asdict(out.episode),
              })
              + "\n"
          )
    return episodes


async def evaluate_suite(
    collector: R2OrchCollector, version: int
) -> dict[str, object]:
  """Evaluates `collector.tasks` and returns domain + topology metrics."""
  outcomes = await collector.collect_outcomes(
      collector.tasks, version, 0, evaluate=True
  )
  return summarize_outcomes(outcomes, k=int(collector.config["eval_attempts"]))


class R2OrchRoundHooks:
  """Logs round telemetry, validates heldout tasks, and verifies zero Worker drift."""

  def __init__(
      self,
      collector: R2OrchCollector,
      eval_collector: R2OrchCollector,
      worker: trainer_worker.TrainerWorker,
      bundle: PolicyBundle,
      initial_worker_norms: np.ndarray,
      config: dict[str, object],
      output: Path,
      *,
      checkpointing: bool,
  ):
    self.collector = collector
    self.eval_collector = eval_collector
    self.worker = worker
    self.bundle = bundle
    self.initial_worker_norms = initial_worker_norms
    self.config = config
    self.metrics_path = output / "metrics.jsonl"
    self.checkpointing = checkpointing
    self.program: swarm_program.SwarmRLProgram | None = None
    self.start_step = 0
    self.state: dict[str, object] = {"initial_eval": None, "validations": []}

  def check_frozen_worker_drift(self) -> float:
    """Asserts that frozen Worker weights have zero drift from step 0."""
    current = pytree_leaf_norms(self.bundle.worker_state_fn())
    drift = float(np.max(np.abs(current - self.initial_worker_norms)))
    if drift != 0.0:
      raise RuntimeError(
          f"Frozen Worker parameter drift detected: max_norm_diff={drift}"
      )
    return drift

  def log(self, record: dict[str, object]) -> None:
    with self.metrics_path.open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(record) + "\n")

  def checkpoint_metadata(self, version: int) -> dict[str, object]:
    assert self.program is not None
    return {
        "swarm_step": self.program.step,
        "swarm_policy_version": version,
        "history": self.program.history,
        "recipe": self.config,
        **self.state,
    }

  def resume_from(self, engine, *, step: int | None, latest: bool) -> int:
    if step is None and not latest:
      return 0
    assert self.program is not None
    meta = self.worker.restore_checkpoint(step=step)
    if "swarm_step" in meta:
      self.program.restore_state(
          step=meta["swarm_step"], history=meta["history"]
      )
      engine.version = meta["swarm_policy_version"]
      self.start_step = meta["swarm_step"]
      self.state = {k: meta[k] for k in self.state if k in meta}
    if meta["step"]:
      engine.sync()
    return int(meta["step"])

  async def __call__(self, version: int, step: int):
    assert self.program is not None
    if step > self.start_step:
      drift = self.check_frozen_worker_drift()
      self.log({
          "event": "train",
          **self.program.history[-1],
          "rollout_summary": self.collector.last_round_summary,
          "frozen_worker_max_norm_drift": drift,
      })
      if step % int(self.config["val_every_steps"]) == 0:
        val_res = {
            "step": step,
            "frozen_worker_max_norm_drift": drift,
            **await evaluate_suite(self.eval_collector, version),
        }
        self.state["validations"].append(val_res)  # pytype: disable=attribute-error
        self.log({"event": "validation", **val_res})
      if (
          self.checkpointing
          and step % int(self.config["checkpoint_every_steps"]) == 0
      ):
        self.worker.save_checkpoint(
            self.checkpoint_metadata(version), step=step, force=True
        )
    return await self.collector(version, step)


def _run_extra_suites(
    bundle: PolicyBundle,
    config: dict[str, object],
    suites: dict[str, Sequence[scoring.R2OrchTask]],
    version: int,
) -> dict[str, object]:
  results = {}
  for name, suite_tasks in suites.items():
    sc = R2OrchCollector(bundle, suite_tasks, config)
    results[name] = asyncio.run(evaluate_suite(sc, version))
  return results


def _execute_r2orch_loop(
    *,
    backend: str,
    mode: str,
    restored_step: int,
    config: dict[str, object],
    bundle: PolicyBundle,
    trainer: peft_trainer_v2.PeftTrainer,
    t_worker: trainer_worker.TrainerWorker,
    engine: local_engine.LocalSwarmEngine,
    collector: R2OrchCollector,
    eval_collector: R2OrchCollector,
    hooks: R2OrchRoundHooks,
    program: swarm_program.SwarmRLProgram,
    initial_director_norms: np.ndarray,
    extra_suites: dict[str, Sequence[scoring.R2OrchTask]],
    output_dir: Path,
) -> dict[str, object]:
  """Executes evaluation or training rounds and writes `report.json`."""
  try:
    if mode == "eval":
      eval_report: dict[str, object] = {
          "backend": backend,
          "mode": mode,
          "restored_step": restored_step,
          "eval": asyncio.run(evaluate_suite(eval_collector, engine.version)),
          "extra_evals": _run_extra_suites(
              bundle, config, extra_suites, engine.version
          ),
          "frozen_worker_max_norm_drift": hooks.check_frozen_worker_drift(),
      }
      (output_dir / "report.json").write_text(
          json.dumps(eval_report, indent=2) + "\n", encoding="utf-8"
      )
      return eval_report

    if hooks.state["initial_eval"] is None:
      hooks.state["initial_eval"] = asyncio.run(
          evaluate_suite(eval_collector, 0)
      )
      hooks.log({"event": "initial_eval", **hooks.state["initial_eval"]})
    remaining = int(config["num_steps"]) - program.step
    if remaining > 0:
      program.run(engine, num_steps=remaining)
      drift = hooks.check_frozen_worker_drift()
      hooks.log({
          "event": "train",
          **program.history[-1],
          "rollout_summary": collector.last_round_summary,
          "frozen_worker_max_norm_drift": drift,
      })
    final_director_norms = pytree_leaf_norms(
        nnx.state(bundle.actor, nnx.Param)
    )
    final_drift = hooks.check_frozen_worker_drift()
    director_delta_l2 = float(
        np.linalg.norm(final_director_norms - initial_director_norms)
    )
    report: dict[str, object] = {
        "backend": backend,
        "mode": mode,
        "restored_step": restored_step,
        "upstream_reference": (
            "beita6969/R2Orch@12ceb7563aa50439687f21ba5da60ad74a75eb89"
        ),
        "director_norm_delta_l2": director_delta_l2,
        "frozen_worker_max_norm_drift": final_drift,
        **hooks.state,
        "history": program.history,
        "final_eval": asyncio.run(
            evaluate_suite(eval_collector, program.policy_version)
        ),
        "extra_evals": _run_extra_suites(
            bundle, config, extra_suites, program.policy_version
        ),
    }
    if hooks.checkpointing:
      t_worker.save_checkpoint(
          hooks.checkpoint_metadata(program.policy_version),
          step=program.step,
          force=True,
      )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
  finally:
    bundle.close_fn()
    trainer.close()


def _sync_barrier_mesh(backend: str, train_mesh: jax.sharding.Mesh) -> None:
  barrier_mesh = (
      train_mesh
      if backend == "cpu"
      else jax.sharding.Mesh(np.asarray(jax.devices()), ("all",))
  )
  jax.block_until_ready(
      sharding_utils.place_array_on_target(
          np.zeros((), dtype=np.float32),
          jax.sharding.NamedSharding(barrier_mesh, jax.sharding.PartitionSpec()),
      )
  )


def run(
    backend: str,
    config: dict[str, object],
    train_tasks: Sequence[scoring.R2OrchTask],
    eval_tasks: Sequence[scoring.R2OrchTask],
    output: str | Path,
    *,
    extra_eval_suites: dict[str, Sequence[scoring.R2OrchTask]] | None = None,
    model_path: Path | None = None,
    checkpoint_dir: str | None = None,
    mode: str = "train",
    resume: bool = False,
    restore_step: int | None = None,
) -> dict[str, object]:
  """Runs SW3 R2Orch Director training (or `mode='eval'`) and writes `report.json`."""
  validate_r2orch_config(config)
  if int(config["tasks_per_step"]) > len(train_tasks):
    raise ValueError("tasks_per_step exceeds number of training tasks.")
  if {t.task_id for t in train_tasks} & {t.task_id for t in eval_tasks}:
    raise ValueError("Training and evaluation task IDs must be disjoint.")
  output_dir = Path(output).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  if any(output_dir.iterdir()):
    raise ValueError("Use an empty output directory.")
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["swarm_r2orch"])
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    pathwaysutils.initialize()
  train_mesh, director_mesh, worker_mesh = build_tri_meshes(
      config, backend=backend
  )
  _sync_barrier_mesh(backend, train_mesh)
  resolved_ckpt = (
      None
      if checkpoint_dir == "none"
      else (checkpoint_dir or str(output_dir / "checkpoints"))
  )
  with train_mesh:
    bundle = create_policy_bundle(
        backend, config, train_mesh, director_mesh, worker_mesh, model_path
    )
    initial_director_norms = pytree_leaf_norms(
        nnx.state(bundle.actor, nnx.Param)
    )
    initial_worker_norms = pytree_leaf_norms(bundle.worker_state_fn())
    optimizer = (
        optax.sgd(float(config["learning_rate"]))
        if backend == "cpu"
        else optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                float(config["learning_rate"]), b1=0.9, b2=0.999, weight_decay=0.01
            ),
        )
    )
    trainer = peft_trainer_v2.PeftTrainer(
        bundle.actor,
        optimizer,
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=int(config["num_steps"]) + 1,
            max_steps=int(config["num_steps"]),
            gradient_accumulation_steps=2,
            checkpoint_root_directory=resolved_ckpt,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type=(
            "inprocess_vllm" if config["rollout_engine"] == "vllm" else "vanilla"
        ),
    )
    t_worker = trainer_worker.TrainerWorker(
        lambda: trainer,
        logps_chunk_size=int(config["logps_chunk_size"]),
        execution_context=train_mesh,
    )
    t_worker.initialize()
    engine = local_engine.LocalSwarmEngine(t_worker, bundle.sync_director)
    collector = R2OrchCollector(
        bundle, train_tasks, config, trace_path=output_dir / "train_traces.jsonl"
    )
    eval_collector = R2OrchCollector(bundle, eval_tasks, config)
    hooks = R2OrchRoundHooks(
        collector,
        eval_collector,
        t_worker,
        bundle,
        initial_worker_norms,
        config,
        output_dir,
        checkpointing=resolved_ckpt is not None,
    )
    program = swarm_program.SwarmRLProgram(
        hooks,
        swarm_program.SwarmConfig(
            group_size=int(config["group_size"]),
            micro_batch_size=int(config["micro_batch_size"]),
            max_prompt_length=int(config["max_prompt_length"]),
            max_response_length=int(config["max_response_length"]),
            pad_id=bundle.pad_id,
            eos_id=bundle.eos_id,
            normalization=str(config["normalization"]),
            invalid_action_penalty=float(config["invalid_action_penalty"]),
            max_logprob_error=float(config["max_logprob_error"]),
            loss_reduction=str(config["loss_reduction"]),
            loss_microbatch_rows=int(config["loss_microbatch_rows"]),
            sampler_is=(
                None if config["sampler_is"] is None else str(config["sampler_is"])
            ),
            epsilon_high=float(config["epsilon_high"]),
        ),
    )
    hooks.program = program
    restored_step = hooks.resume_from(
        engine, step=restore_step, latest=resume
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return _execute_r2orch_loop(
        backend=backend,
        mode=mode,
        restored_step=restored_step,
        config=config,
        bundle=bundle,
        trainer=trainer,
        t_worker=t_worker,
        engine=engine,
        collector=collector,
        eval_collector=eval_collector,
        hooks=hooks,
        program=program,
        initial_director_norms=initial_director_norms,
        extra_suites=extra_eval_suites or {},
        output_dir=output_dir,
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("cpu", "qwen"), default="cpu")
  parser.add_argument("--mode", choices=("train", "eval"), default="train")
  parser.add_argument("--config", type=Path)
  parser.add_argument("--train-data", type=Path)
  parser.add_argument("--eval-data", type=Path)
  parser.add_argument("--test-iid-data", type=Path)
  parser.add_argument("--test-ood-data", type=Path)
  parser.add_argument("--model-path", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoint-dir", type=str)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--restore-step", type=int)
  parser.add_argument("--rollout-engine", choices=("vanilla", "vllm"))
  parser.add_argument("--tp-size", type=int)
  parser.add_argument("--micro-batch-size", type=int)
  parser.add_argument("--rollout-batch-size", type=int)
  parser.add_argument("--tasks-per-step", type=int)
  parser.add_argument("--num-steps", type=int)
  parser.add_argument("--eval-attempts", type=int)
  parser.add_argument("--val-every-steps", type=int)
  parser.add_argument("--checkpoint-every-steps", type=int)
  args = parser.parse_args()
  if args.backend == "qwen" and (args.train_data is None or args.eval_data is None):
    parser.error("Qwen requires --train-data and --eval-data JSONL files.")
  cfg_path = args.config or ROOT / "configs" / (
      "cpu_smoke.json" if args.backend == "cpu" else "qwen3_4b.json"
  )
  config = json.loads(cfg_path.read_text(encoding="utf-8"))
  for key, val in (
      ("rollout_engine", args.rollout_engine),
      ("tp_size", args.tp_size),
      ("micro_batch_size", args.micro_batch_size),
      ("rollout_batch_size", args.rollout_batch_size),
      ("tasks_per_step", args.tasks_per_step),
      ("num_steps", args.num_steps),
      ("eval_attempts", args.eval_attempts),
      ("val_every_steps", args.val_every_steps),
      ("checkpoint_every_steps", args.checkpoint_every_steps),
  ):
    if val is not None:
      config[key] = val
  extra_suites = {}
  if args.test_iid_data is not None:
    extra_suites["test_iid"] = load_tasks(args.test_iid_data)
  if args.test_ood_data is not None:
    extra_suites["test_ood"] = load_tasks(args.test_ood_data)
  logging.set_verbosity(logging.INFO)
  report = run(
      args.backend,
      config,
      load_tasks(args.train_data or ROOT / "data/cpu_train.jsonl"),
      load_tasks(args.eval_data or ROOT / "data/cpu_eval.jsonl"),
      args.output,
      extra_eval_suites=extra_suites,
      model_path=args.model_path,
      checkpoint_dir=args.checkpoint_dir,
      mode=args.mode,
      resume=args.resume,
      restore_step=args.restore_step,
  )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
