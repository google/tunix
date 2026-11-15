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

"""Local transport entry point for shared Qwen3-4B or a tiny CPU fixture.

Run from the repository root with `python -m examples.swarm_math.train`.
The CPU backend exercises real native training without downloading weights.
The Qwen backend requires a local HF snapshot and accelerator capacity.
"""

import argparse
import asyncio
import collections
import dataclasses
import hashlib
import json
import os
from pathlib import Path

from absl import flags
from absl import logging
from flax import nnx
import jax
import numpy as np
import optax
from examples.swarm_math import cpu_fixture
from examples.swarm_math import workflow
from tunix.common import configs
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.swarm import runtime
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.utils import sharding_utils

ROOT = Path(__file__).parent


def load_tasks(path):
  tasks = [
      workflow.MathTask(**json.loads(line))
      for line in Path(path).read_text().splitlines()
      if line.strip()
  ]
  if not tasks or len({t.task_id for t in tasks}) != len(tasks):
    raise ValueError("Task data must be nonempty with unique task_id values.")
  if any(
      not all(
          isinstance(x, str) and x.strip()
          for x in (t.task_id, t.problem, t.answer)
      )
      for t in tasks
  ):
    raise ValueError(
        "Each task needs nonempty string task_id, problem and answer."
    )
  return tasks


# (requirement, error message) pairs checked by `validate_config`.
_RECIPE_REQUIREMENTS = (
    (
        lambda c: c["model_sharing"] and not c["enable_thinking"],
        "This recipe requires shared weights and enable_thinking=false.",
    ),
    (
        lambda c: c["rollout_engine"] in ("vanilla", "vllm"),
        "rollout_engine must be 'vanilla' or 'vllm'.",
    ),
    (
        lambda c: c["train_temperature"] == 1.0 and c["train_top_p"] == 1.0,
        "V0 training requires temperature=1 and full-distribution sampling.",
    ),
    # The TPU splash-attention backward gave wrong gradients on this recipe's
    # left-padded batches (see docs/swarm_training_validation.md).
    (
        lambda c: not c["use_flash_attention"],
        "use_flash_attention must be false for swarm_math.",
    ),
    # Early-layer Qwen3 norm and RoPE attention gradients lose cosine fidelity
    # in bfloat16 on long multi-turn contexts; keep trainer compute in float32.
    (
        lambda c: c["dtype"] == "float32",
        "dtype must be 'float32' for swarm_math.",
    ),
    (
        lambda c: c["sampler_is"] in (None, "token"),
        "sampler_is must be null or 'token'.",
    ),
)


def validate_config(config):
  for satisfied, message in _RECIPE_REQUIREMENTS:
    if not satisfied(config):
      raise ValueError(message)
  positive_keys = (
      "tasks_per_step",
      "num_steps",
      "max_solver_turns",
      "eval_attempts",
      "micro_batch_size",
      "rollout_batch_size",
      "tp_size",
      "flash_attention_block_size",
      "vllm_max_num_seqs",
      "vllm_max_num_batched_tokens",
      "vllm_reshard_chunk_size",
      "checkpoint_every_steps",
      "val_every_steps",
      "trace_every_steps",
  )
  if any(config[k] < 1 for k in positive_keys) or config["logps_chunk_size"] < 0:
    raise ValueError(
        "Task, step, turn, evaluation, batch, vLLM and TP counts must be"
        " positive."
    )
  if (
      not 0 <= config["rollout_split_fraction"] < 1
      or not 0 < config["vllm_hbm_utilization"] <= 1
      or not 0 < config["eval_top_p"] <= 1
  ):
    raise ValueError("Invalid fraction or top_p bound in config.")
  for key, allow_zero in (
      ("max_logprob_error", True),
      ("learning_rate", False),
      ("eval_temperature", False),
  ):
    val = config[key]
    if not np.isfinite(val) or (val < 0 if allow_zero else val <= 0):
      raise ValueError(f"{key} out of range: {val!r}.")


class MathCollector:
  """Serial local transport; independent episodes may later run on workers."""

  def __init__(self, sampler, encode, tasks, config, *, trace_path=None):
    self.sampler, self.encode, self.tasks, self.config = (
        sampler,
        encode,
        tasks,
        config,
    )
    self.trace_path = trace_path
    self.workflow = workflow.SolverVerifierWorkflow(
        max_solver_turns=config["max_solver_turns"]
    )

  async def episodes(
      self, tasks, version, step, *, evaluate=False, max_solver_turns=None
  ):
    config = self.config
    attempts = config["eval_attempts"] if evaluate else config["group_size"]
    stage = "eval" if evaluate else "train"
    flow = (
        self.workflow
        if max_solver_turns is None
        else workflow.SolverVerifierWorkflow(max_solver_turns=max_solver_turns)
    )
    coros = []
    for index, task in enumerate(tasks):
      for attempt in range(attempts):
        seed = int(
            np.random.SeedSequence(
                [config["seed"], step, index, attempt, int(evaluate)]
            ).generate_state(1)[0]
        )
        rt = runtime.RecordingRuntime(
            self.sampler,
            self.encode,
            sampler_lib.SamplingParams(
                max_tokens=config["max_response_length"],
                seed=seed,
                temperature=config[f"{stage}_temperature"],
                top_p=config[f"{stage}_top_p"],
            ),
            policy_id=config["policy_id"],
            policy_version=version,
            episode_id=f"{stage}/{step}/{task.task_id}/{attempt}",
            max_prompt_length=config["max_prompt_length"],
        )
        coros.append(flow.run(task, rt))
    return list(await asyncio.gather(*coros))

  async def __call__(self, version, step):
    size = self.config["tasks_per_step"]
    # Deterministic epoch permutation; do not duplicate a task within a group.
    order = np.random.default_rng(
        self.config["seed"] + step // max(1, len(self.tasks) // size)
    ).permutation(len(self.tasks))
    start = (step * size) % len(self.tasks)
    indices = [
        order[(start + offset) % len(self.tasks)] for offset in range(size)
    ]
    episodes = await self.episodes(
        [self.tasks[i] for i in indices], version, step
    )
    if (
        self.trace_path is not None
        and step % self.config["trace_every_steps"] == 0
    ):
      with Path(self.trace_path).open("a") as stream:
        for episode in episodes:
          stream.write(json.dumps(dataclasses.asdict(episode)) + "\n")
    return episodes


class NativeMathSampler:
  """Keep native static decode settings explicit and evaluation seeds stable.

  VanillaSampler uses its configured random stream, not SamplingParams.seed.
  Resetting only that stream for paired evaluation avoids reallocating model
  buffers or recompiling XLA executables. Concurrent episode calls are batched
  to static `batch_size` so multi-device FSDP activation sharding stays valid.
  """

  def __init__(
      self,
      actor,
      tokenizer,
      rollout_config,
      stage,
      *,
      rollout_model=None,
      batch_size: int = 1,
  ):
    if batch_size < 1:
      raise ValueError("batch_size must be positive.")
    from tunix.dist_rl.rollout import vanilla_sampler  # pylint: disable=g-import-not-at-top

    self.actor = actor
    self.tokenizer = tokenizer
    self.config = rollout_config
    self.stage = stage
    self.batch_size = batch_size
    self._queue: list[
        tuple[
            sampler_lib.SamplingRequest,
            asyncio.Future[sampler_lib.SamplingResponse],
        ]
    ] = []
    self._flusher: asyncio.Task[None] | None = None
    if rollout_model is None:
      rollout_model = nnx.clone(actor)
      # nnx.clone clones graph objects but shares JAX buffers. Native optimizer
      # JITs donate actor buffers; a serving snapshot needs independent storage.
      nnx.update(
          rollout_model, jax.tree.map(lambda x: x.copy(), nnx.state(actor))
      )
    self.sampler = vanilla_sampler.VanillaSampler(
        server_id=f"swarm-{stage}",
        transformer=rollout_model,
        tokenizer=tokenizer,
        config=rollout_config,
    )

  def reset(self):
    seed = 0 if self.config.seed is None else int(self.config.seed)
    self.sampler._rng = jax.random.PRNGKey(seed)  # pylint: disable=protected-access

  async def sample(self, request):
    params = request.sampling_params
    if (
        params.temperature != self.config.temperature
        or params.top_p != self.config.top_p
    ):
      raise ValueError(
          "Requested sampling distribution differs from the native static"
          " config."
      )
    if self.batch_size == 1:
      return await self.sampler.sample(
          request,
          max_prompt_length=self.config.max_prompt_length,
          include_eos=True,
      )
    loop = asyncio.get_running_loop()
    future: asyncio.Future[sampler_lib.SamplingResponse] = loop.create_future()
    self._queue.append((request, future))
    if self._flusher is None or self._flusher.done():
      self._flusher = loop.create_task(self._flush_queued())
    return await future

  async def _flush_queued(self) -> None:
    await asyncio.sleep(0)
    while self._queue:
      chunk = self._queue[: self.batch_size]
      del self._queue[: self.batch_size]
      # A collective deadline can cancel an awaiting agent while its request
      # is queued or in flight. Never deliver to an already completed Future.
      chunk = [(req, fut) for req, fut in chunk if not fut.done()]
      if not chunk:
        continue
      requests = [req for req, _ in chunk]
      if len(requests) < self.batch_size:
        pad_req = dataclasses.replace(requests[0], request_id="__pad__")
        requests.extend([pad_req] * (self.batch_size - len(requests)))
      try:
        responses = await self.sampler.sample(
            requests,
            max_prompt_length=self.config.max_prompt_length,
            include_eos=True,
        )
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        pending = [fut for _, fut in chunk + self._queue]
        self._queue.clear()
        for fut in pending:
          if not fut.done():
            if isinstance(exc, asyncio.CancelledError):
              fut.cancel()
            else:
              fut.set_exception(exc)
        return
      for (_, fut), resp in zip(chunk, responses[: len(chunk)], strict=True):
        if not fut.done():
          fut.set_result(resp)

  def sync(self):
    # Some CPU leaves start SingleDeviceSharding and become NamedSharding in
    # the optimizer JIT. Preserve each serving leaf's placement explicitly.
    copied = jax.tree.map(
        lambda source, target: jax.device_put(
            source, target.sharding, may_alias=False
        ),
        nnx.state(self.actor, nnx.Param),
        self.sampler.transformer_state,
    )
    self.sampler.transformer_state = copied


class VllmMathSampler:
  """Adapt InprocessVllmSampler so EOS text matches NativeMathSampler."""

  def __init__(self, sampler, tokenizer, eos_tokens):
    self.sampler = sampler
    self._tokenizer = tokenizer
    self._eos_set = frozenset(int(tok) for tok in eos_tokens)

  async def sample(self, request):
    params = request.sampling_params
    if params is not None and (
        params.top_k is None or params.seed is not None
    ):
      request = dataclasses.replace(
          request,
          sampling_params=dataclasses.replace(
              params,
              top_k=0 if params.top_k is None else params.top_k,
              seed=None,
          ),
      )
    response = await self.sampler.sample(request)
    tok_ids = response.token_ids
    if len(tok_ids) > 0 and int(tok_ids[-1]) in self._eos_set:
      return dataclasses.replace(
          response,
          text=self._tokenizer.decode(
              tok_ids[:-1].tolist(), skip_special_tokens=False
          ),
      )
    return response

  async def close(self):
    await self.sampler.stop()


def _resolve_eos_tokens(model_dir: Path, default_eos_id: int) -> list[int]:
  gen_cfg_path = model_dir / "generation_config.json"
  if not gen_cfg_path.is_file():
    return [default_eos_id]
  gen_cfg = json.loads(gen_cfg_path.read_text())
  if "eos_token_id" not in gen_cfg:
    return [default_eos_id]
  raw = gen_cfg["eos_token_id"]
  return [int(x) for x in raw] if isinstance(raw, list) else [int(raw)]


def reshape_mesh(
    devices: np.ndarray, tp_size: int, batch_size: int, label: str
) -> jax.sharding.Mesh:
  if len(devices) % tp_size != 0:
    raise ValueError(
        f"Device count {len(devices)} is not divisible by tp_size={tp_size}."
    )
  fsdp_size = len(devices) // tp_size
  if batch_size % fsdp_size != 0:
    raise ValueError(
        f"{label} ({batch_size}) must be divisible by fsdp_size={fsdp_size}."
    )
  return jax.sharding.Mesh(
      devices.reshape(fsdp_size, tp_size), ("fsdp", "tp")
  )


def build_meshes(
    config, *, backend: str | None = None
) -> tuple[jax.sharding.Mesh, jax.sharding.Mesh]:
  devices = np.asarray(
      jax.devices()[:1] if backend == "cpu" else jax.devices()
  )
  split = config["rollout_split_fraction"]
  if backend == "cpu" or config["rollout_engine"] != "vllm" or split == 0.0:
    mesh = reshape_mesh(
        devices,
        config["tp_size"],
        config["micro_batch_size"],
        "micro_batch_size",
    )
    if config["rollout_engine"] == "vanilla":
      reshape_mesh(
          devices,
          config["tp_size"],
          config["rollout_batch_size"],
          "rollout_batch_size",
      )
    return mesh, mesh
  rollout_count = int(len(devices) * split)
  if not 0 < rollout_count < len(devices):
    raise ValueError(
        f"Invalid rollout_split_fraction={split} across"
        f" {len(devices)} devices."
    )
  return (
      reshape_mesh(
          devices[rollout_count:],
          config["tp_size"],
          config["micro_batch_size"],
          "micro_batch_size",
      ),
      reshape_mesh(
          devices[:rollout_count],
          config["tp_size"],
          config["rollout_batch_size"],
          "rollout_batch_size",
      ),
  )


def _create_vllm_samplers(
    actor, tokenizer, config, rollout_mesh, model_path_str, eos_tokens
):
  from tunix.dist_rl.rollout import inprocess_vllm_sampler  # pylint: disable=g-import-not-at-top
  from tunix.generate import mappings  # pylint: disable=g-import-not-at-top

  if not all(
      isinstance(config[k], bool)
      for k in ("vllm_async_scheduling", "vllm_enable_prefix_caching")
  ):
    raise ValueError("vLLM scheduling switches must be booleans.")
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  tp_size = int(rollout_mesh.shape["tp"])
  dp_size = int(rollout_mesh.shape["fsdp"])
  rollout_config = configs.RolloutConfig(
      max_prompt_length=config["max_prompt_length"],
      max_tokens_to_generate=config["max_response_length"],
      kv_cache_size=config["max_prompt_length"]
      + config["max_response_length"],
      temperature=config["train_temperature"],
      top_p=config["train_top_p"],
      top_k=0,
      eos_tokens=eos_tokens,
      return_logprobs=True,
      tensor_parallel_size=tp_size,
      data_parallel_size=dp_size,
      rollout_mapping_config=mappings.MappingConfig.from_model(
          actor, backend="vllm_jax"
      ),
      rollout_vllm_model_version=model_path_str,
      rollout_vllm_server_mode=True,
      rollout_vllm_init_with_random_weights=True,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_hbm_utilization=config["vllm_hbm_utilization"],
      rollout_vllm_reshard_chunk_size=config["vllm_reshard_chunk_size"],
      # With both switches on, a DP=8 RPA-v2 diagnostic saw ~43-nat first-token
      # logprob spikes on repeated identical Qwen prompts; turning both off
      # removed them (as in the qwen3_grpo_demo/frozenlake Qwen3 recipes).
      rollout_vllm_async_scheduling=config["vllm_async_scheduling"],
      rollout_vllm_max_num_seqs=config["vllm_max_num_seqs"],
      rollout_vllm_max_num_batched_tokens=config["vllm_max_num_batched_tokens"],
      rollout_vllm_kwargs={
          "enable_prefix_caching": config["vllm_enable_prefix_caching"],
          "dtype": "bfloat16",
          "tokenizer": model_path_str,
          "seed": config["seed"],
          "kv_cache_metrics": True,
          "disable_log_stats": False,
      },
  )
  # One submitting thread per engine sequence slot, so evaluation can fill
  # every data-parallel replica instead of a fixed training batch.
  max_concurrency = config["vllm_max_num_seqs"] * dp_size
  with rollout_mesh:
    engine = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="swarm_vllm",
        tokenizer=tokenizer,
        config=rollout_config,
        mesh=rollout_mesh,
        max_concurrency=max_concurrency,
    )
    engine.initialize()

  def sync():
    weights = nnx.state(actor, nnx.Param)
    with rollout_mesh:
      engine.update_params(weights)

  sync()
  wrapped = VllmMathSampler(engine, tokenizer, eos_tokens)
  return wrapped, wrapped, sync


def create_backends(
    backend, config, mesh, model_path=None, *, rollout_mesh=None
):
  if backend == "cpu":
    if config["model_id"] != "tiny-categorical-fixture":
      raise ValueError("Use cpu_smoke.json with the CPU fixture.")
    actor = cpu_fixture.TinyPolicy()
    rollout = nnx.clone(actor)

    def sync_cpu():
      nnx.update(rollout, jax.tree.map(lambda x: x.copy(), nnx.state(actor)))

    sync_cpu()
    return (
        actor,
        cpu_fixture.TinySampler(rollout),
        cpu_fixture.TinySampler(rollout),
        cpu_fixture.encode,
        0,
        0,
        sync_cpu,
    )
  if (
      config["model_id"] != "Qwen/Qwen3-4B"
      or not model_path
      or not Path(model_path).is_dir()
  ):
    raise ValueError(
        "Qwen requires model_id=Qwen/Qwen3-4B and a local --model-path HF"
        " snapshot."
    )
  if Path(model_path).resolve().name != config["model_revision"]:
    raise ValueError(
        "Use the HF cache snapshot directory for the pinned model_revision (see"
        " README)."
    )
  # Imports are lazy: the CPU fixture does not need model downloads or KV cache.
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  model_path_str = str(model_path)
  tokenizer = AutoTokenizer.from_pretrained(
      model_path_str, local_files_only=True
  )
  actor, _ = automodel.AutoModel.from_pretrained(
      config["model_id"],
      mesh,
      model_path=model_path_str,
      remat_config=config["remat_config"],
      dtype=config["dtype"],
      load_dtype=config["load_dtype"],
      use_flash_attention=config["use_flash_attention"],
      flash_attention_block_size=config["flash_attention_block_size"],
  )
  eos_tokens = _resolve_eos_tokens(Path(model_path), tokenizer.eos_token_id)
  if config["rollout_engine"] == "vllm":
    train_sampler, eval_sampler, sync = _create_vllm_samplers(
        actor,
        tokenizer,
        config,
        rollout_mesh or mesh,
        model_path_str,
        eos_tokens,
    )
  else:
    rollout_model = nnx.clone(actor)
    nnx.update(
        rollout_model, jax.tree.map(lambda x: x.copy(), nnx.state(actor))
    )
    samplers = [
        NativeMathSampler(
            actor=actor,
            tokenizer=tokenizer,
            stage=stage,
            rollout_model=rollout_model,
            batch_size=config["rollout_batch_size"],
            rollout_config=configs.RolloutConfig(
                max_prompt_length=config["max_prompt_length"],
                max_tokens_to_generate=config["max_response_length"],
                kv_cache_size=config["max_prompt_length"]
                + config["max_response_length"],
                temperature=config[f"{stage}_temperature"],
                top_p=config[f"{stage}_top_p"],
                top_k=None,
                eos_tokens=eos_tokens,
                seed=config["seed"],
                return_logprobs=True,
            ),
        )
        for stage in ("train", "eval")
    ]

    def sync():
      samplers[0].sync()
      samplers[1].sampler.transformer_state = samplers[
          0
      ].sampler.transformer_state

    train_sampler, eval_sampler = samplers
  return (
      actor,
      train_sampler,
      eval_sampler,
      workflow.chat_encoder(tokenizer, config["max_prompt_length"]),
      tokenizer.pad_token_id,
      tokenizer.eos_token_id,
      sync,
  )


async def evaluate(collector, version, *, single_solver=False):
  """Team accuracy plus per-benchmark avg@k and pass@k over `eval_attempts`.

  A benchmark is the task-ID prefix (`aime24/3` -> `aime24`). As in the
  upstream report, macro averages weight every benchmark equally. Per-task
  outcomes are kept so subsets (e.g. decontaminated tasks) can be rescored.
  """
  if isinstance(collector.sampler, NativeMathSampler):
    collector.sampler.reset()
  episodes = await collector.episodes(
      collector.tasks,
      version,
      0,
      evaluate=True,
      max_solver_turns=1 if single_solver else None,
  )
  per_task = collections.defaultdict(list)
  for episode in episodes:
    per_task[episode.task_id].append(int(episode.reward == 1.0))
  groups = collections.defaultdict(list)
  for task_id, hits in per_task.items():
    groups[task_id.split("/")[0]].append(hits)
  benchmarks = {
      name: {
          "tasks": len(rows),
          "avg_at_k": float(np.mean([np.mean(h) for h in rows])),
          "pass_at_k": float(np.mean([max(h) for h in rows])),
      }
      for name, rows in sorted(groups.items())
  }
  return {
      "accuracy": float(np.mean([e.reward for e in episodes])),
      "attempts": len(episodes),
      "k": collector.config["eval_attempts"],
      "mean_calls": float(np.mean([len(e.calls) for e in episodes])),
      "mean_generated_tokens": float(
          np.mean(
              [sum(len(c.completion_ids) for c in e.calls) for e in episodes]
          )
      ),
      "macro_avg_at_k": float(
          np.mean([b["avg_at_k"] for b in benchmarks.values()])
      ),
      "macro_pass_at_k": float(
          np.mean([b["pass_at_k"] for b in benchmarks.values()])
      ),
      "benchmarks": benchmarks,
      "per_task": dict(per_task),
  }


class RoundHooks:
  """Checkpoint, validate and log at committed round boundaries.

  `SwarmRLProgram` calls its collector at the start of each round, after the
  previous round was trained and published. A checkpoint taken there commits a
  complete boundary: resuming at `step` replays the same deterministic task
  window from the restored weights and optimizer state.
  """

  def __init__(
      self, collector, eval_collector, worker, config, output, *, checkpointing
  ):
    self.collector = collector
    self.eval_collector = eval_collector
    self.worker = worker
    self.config = config
    self.metrics_path = Path(output) / "metrics.jsonl"
    self.checkpointing = checkpointing
    self.program = None
    self.start_step = 0
    self.state = {"initial_team_eval": None, "validations": []}

  def metadata(self, version):
    """Program state saved alongside the trainer checkpoint."""
    return {
        "swarm_step": self.program.step,
        "swarm_policy_version": version,
        "history": self.program.history,
        "recipe": self.config,
        **self.state,
    }

  def resume_from(self, engine, *, step, latest):
    """Restores checkpoint `step`, or with `latest` the newest one if any.

    Program state is restored only from boundaries written by `metadata()`;
    other checkpoints (e.g. for evaluation) restore just the weights. Restored
    weights are pushed to the serving engine before any sampling.

    Returns:
      The restored trainer step, or 0 when nothing was restored.
    """
    if step is None and not latest:
      return 0
    metadata = self.worker.restore_checkpoint(step=step)
    if "swarm_step" in metadata:
      self.program.restore_state(
          step=metadata["swarm_step"], history=metadata["history"]
      )
      engine.version = metadata["swarm_policy_version"]
      self.start_step = metadata["swarm_step"]
      self.state = {key: metadata[key] for key in self.state}
    if metadata["step"]:
      engine.sync()
    return metadata["step"]

  def log(self, record):
    with self.metrics_path.open("a") as stream:
      stream.write(json.dumps(record) + "\n")

  async def __call__(self, version, step):
    if step > self.start_step:
      self.log({"event": "train", **self.program.history[-1]})
      if step % self.config["val_every_steps"] == 0:
        result = {"step": step, **await evaluate(self.eval_collector, version)}
        self.state["validations"].append(result)
        self.log({"event": "validation", **result})
      if (
          self.checkpointing
          and step % self.config["checkpoint_every_steps"] == 0
      ):
        self.worker.save_checkpoint(
            self.metadata(version), step=step, force=True
        )
    return await self.collector(version, step)


def _execute_run(
    *,
    actor,
    before,
    trainer,
    worker,
    engine,
    sampler,
    eval_collector,
    program,
    hooks,
    config,
    manifest,
    output,
    mode,
):
  try:
    if mode == "eval":
      report = {
          **manifest,
          "team_eval": asyncio.run(evaluate(eval_collector, engine.version)),
      }
      (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
      return report
    if hooks.state["initial_team_eval"] is None:
      logging.info(
          "Running initial team evaluation (%d tasks).",
          len(eval_collector.tasks),
      )
      hooks.state["initial_team_eval"] = asyncio.run(
          evaluate(eval_collector, 0)
      )
      hooks.log(
          {"event": "initial_team_eval", **hooks.state["initial_team_eval"]}
      )
    remaining = config["num_steps"] - program.step
    if remaining > 0:
      program.run(engine, num_steps=remaining)
      hooks.log({"event": "train", **program.history[-1]})
    report = {
        **manifest,
        **hooks.state,
        "history": program.history,
        "final_team_eval": asyncio.run(
            evaluate(eval_collector, program.policy_version)
        ),
        "final_single_solver_eval": asyncio.run(
            evaluate(
                eval_collector, program.policy_version, single_solver=True
            )
        ),
    }
    if before is not None:
      after = np.asarray(actor.logits[...]).copy()
      report["parameter_delta_l2"] = float(np.linalg.norm(after - before))
      np.savez(output / "parameters.npz", initial=before, final=after)
    if hooks.checkpointing:
      worker.save_checkpoint(
          hooks.metadata(program.policy_version), step=program.step, force=True
      )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
  finally:
    if hasattr(sampler, "close"):
      asyncio.run(sampler.close())
    trainer.close()


def run(
    backend,
    config,
    train_tasks,
    eval_tasks,
    output,
    *,
    model_path=None,
    checkpoint_dir: str | None = None,
    mode: str = "train",
    resume: bool = False,
    restore_step: int | None = None,
):
  """Train (or, with `mode="eval"`, only evaluate) the shared-policy team.

  `resume` restores the latest committed round boundary if one exists;
  `restore_step` restores a specific checkpoint (e.g. for evaluation).
  """
  validate_config(config)
  if config["tasks_per_step"] > len(train_tasks):
    raise ValueError("tasks_per_step exceeds the number of training tasks.")
  if {t.task_id for t in train_tasks} & {t.task_id for t in eval_tasks}:
    raise ValueError("Training and evaluation task IDs must be disjoint.")
  # Upstream benchmarks share a few exact questions with the training corpus;
  # report them instead of silently changing either reference dataset.
  eval_ids = {t.problem.strip(): t.task_id for t in eval_tasks}
  train_problems = {t.problem.strip() for t in train_tasks}
  overlap = sorted(eval_ids[p] for p in eval_ids.keys() & train_problems)
  output = Path(output).resolve()
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()):
    raise ValueError(
        "Use an empty output directory; existing run artifacts are preserved."
    )
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["swarm_math"])
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    pathwaysutils.initialize()
  resolved_ckpt_dir = (
      None
      if checkpoint_dir == "none"
      else (checkpoint_dir or str(output / "checkpoints"))
  )
  mesh, rollout_mesh = build_meshes(config, backend=backend)
  logging.info(
      "Built train_mesh=%s (%d devices) rollout_mesh=%s (%d devices).",
      mesh.shape,
      mesh.size,
      rollout_mesh.shape,
      rollout_mesh.size,
  )
  barrier_mesh = (
      mesh
      if backend == "cpu" or mesh == rollout_mesh
      else jax.sharding.Mesh(np.asarray(jax.devices()), ("all",))
  )
  # Wait for late-bound TPU workers with a single 4-byte op before loading
  # large weight shards onto the mesh.
  jax.block_until_ready(
      sharding_utils.place_array_on_target(
          np.zeros((), dtype=np.float32),
          jax.sharding.NamedSharding(barrier_mesh, jax.sharding.PartitionSpec()),
      )
  )
  with mesh:
    actor, sampler, eval_sampler, encode, pad_id, eos_id, sync = (
        create_backends(
            backend, config, mesh, model_path, rollout_mesh=rollout_mesh
        )
    )
    before = np.asarray(actor.logits[...]).copy() if backend == "cpu" else None
    optimizer = (
        optax.sgd(config["learning_rate"])
        if backend == "cpu"
        else optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                config["learning_rate"], b1=0.9, b2=0.999, weight_decay=0.01
            ),
        )
    )
    trainer = peft_trainer_v2.PeftTrainer(
        actor,
        optimizer,
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=config["num_steps"] + 1,
            max_steps=config["num_steps"],
            gradient_accumulation_steps=2,
            checkpoint_root_directory=resolved_ckpt_dir,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type=(
            "inprocess_vllm" if config["rollout_engine"] == "vllm" else "vanilla"
        ),
    )
    worker = trainer_worker.TrainerWorker(
        lambda: trainer,
        logps_chunk_size=config["logps_chunk_size"],
        execution_context=mesh,
    )
    worker.initialize()
    engine = local_engine.LocalSwarmEngine(worker, sync)
    collector = MathCollector(
        sampler,
        encode,
        train_tasks,
        config,
        trace_path=output / "train_traces.jsonl",
    )
    eval_collector = MathCollector(eval_sampler, encode, eval_tasks, config)
    hooks = RoundHooks(
        collector,
        eval_collector,
        worker,
        config,
        output,
        checkpointing=resolved_ckpt_dir is not None,
    )
    program = swarm_program.SwarmRLProgram(
        hooks,
        swarm_program.SwarmConfig(
            **{
                key: config[key]
                for key in (
                    "group_size",
                    "micro_batch_size",
                    "max_prompt_length",
                    "max_response_length",
                    "normalization",
                    "invalid_action_penalty",
                    "max_logprob_error",
                    "loss_reduction",
                    "loss_microbatch_rows",
                    "sampler_is",
                )
            },
            pad_id=pad_id,
            eos_id=eos_id,
        ),
    )
    hooks.program = program
    restored_step = hooks.resume_from(
        engine, step=restore_step, latest=resume
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "backend": backend,
        "mode": mode,
        "restored_step": restored_step,
        "eval_training_overlap": overlap,
        "model_path": str(model_path) if model_path else None,
        "reference_commit": "c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9",
        "train_data_sha256": (
            hashlib.sha256(
                json.dumps(
                    [dataclasses.asdict(t) for t in train_tasks], sort_keys=True
                ).encode()
            ).hexdigest()
        ),
        "eval_data_sha256": (
            hashlib.sha256(
                json.dumps(
                    [dataclasses.asdict(t) for t in eval_tasks], sort_keys=True
                ).encode()
            ).hexdigest()
        ),
    }
    return _execute_run(
        actor=actor,
        before=before,
        trainer=trainer,
        worker=worker,
        engine=engine,
        sampler=sampler,
        eval_collector=eval_collector,
        program=program,
        hooks=hooks,
        config=config,
        manifest=manifest,
        output=output,
        mode=mode,
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("cpu", "qwen"), default="cpu")
  parser.add_argument("--mode", choices=("train", "eval"), default="train")
  parser.add_argument("--config", type=Path)
  parser.add_argument("--train-data", type=Path)
  parser.add_argument("--eval-data", type=Path)
  parser.add_argument("--model-path", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoint-dir", type=str)
  parser.add_argument(
      "--resume",
      action="store_true",
      help="Continue from the latest checkpoint in --checkpoint-dir, if any.",
  )
  parser.add_argument("--restore-step", type=int)
  parser.add_argument("--rollout-engine", choices=("vanilla", "vllm"))
  parser.add_argument("--normalization", choices=("role", "episode"))
  parser.add_argument("--tp-size", type=int)
  parser.add_argument("--micro-batch-size", type=int)
  parser.add_argument("--rollout-batch-size", type=int)
  parser.add_argument("--tasks-per-step", type=int)
  parser.add_argument("--num-steps", type=int)
  parser.add_argument("--eval-attempts", type=int)
  args = parser.parse_args()
  if args.backend == "qwen" and (
      args.train_data is None or args.eval_data is None
  ):
    parser.error(
        "Qwen requires explicit disjoint --train-data and --eval-data JSONL"
        " files."
    )
  config_path = args.config or ROOT / "configs" / (
      "cpu_smoke.json" if args.backend == "cpu" else "qwen3_4b.json"
  )
  config = json.loads(config_path.read_text())
  for key, value in (
      ("rollout_engine", args.rollout_engine),
      ("normalization", args.normalization),
      ("tp_size", args.tp_size),
      ("micro_batch_size", args.micro_batch_size),
      ("rollout_batch_size", args.rollout_batch_size),
      ("tasks_per_step", args.tasks_per_step),
      ("num_steps", args.num_steps),
      ("eval_attempts", args.eval_attempts),
  ):
    if value is not None:
      config[key] = value
  logging.set_verbosity(logging.INFO)
  report = run(
      args.backend,
      config,
      load_tasks(args.train_data or ROOT / "data/cpu_train.jsonl"),
      load_tasks(args.eval_data or ROOT / "data/cpu_eval.jsonl"),
      args.output,
      model_path=args.model_path,
      checkpoint_dir=args.checkpoint_dir,
      mode=args.mode,
      resume=args.resume,
      restore_step=args.restore_step,
  )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
