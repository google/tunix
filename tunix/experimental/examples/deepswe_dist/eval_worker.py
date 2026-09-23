# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 evaluation worker on the distributed rollout stack."""

import asyncio
import logging
import os
import signal

from tunix.experimental.examples.deepswe_dist import eval_deepswe


def create_worker(a):
  """Load real inference weights once, then expose the standard RolloutWorker."""
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
  os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
  os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  # Preserve the vLLM-before-rollout-adapters import order used by rollout nodes.
  from tunix.generate import vllm_sampler
  import jax
  from jax.experimental import mesh_utils
  from jax.sharding import Mesh
  from transformers import AutoTokenizer
  from tunix.experimental.common import datatypes
  from tunix.experimental.examples.deepswe_dist import deepswe
  from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
  from tunix.experimental.worker import rollout_worker
  from tunix.generate import tokenizer_adapter
  from tunix.rl.agentic.parser.chat_template_parser import parser
  from examples.deepswe import sandbox_utils

  from maxtext.integration.vllm import maxtext_vllm_adapter

  maxtext_vllm_adapter.register()
  from etils import epath

  path = epath.Path(a.model_absolute_path)
  if not path.exists() and a.model_absolute_path.rstrip("/").endswith("/item"):
    alt_path = epath.Path(a.model_absolute_path.rstrip("/") + "s")
    if alt_path.exists():
      logging.info(
          "Resolved checkpoint path %s -> %s", a.model_absolute_path, alt_path
      )
      path = alt_path
  if not path.exists():
    raise FileNotFoundError(f"MaxText checkpoint not found: {path}")
  metadata_file = path / "_METADATA"
  if metadata_file.exists():
    try:
      import json  # pylint: disable=import-outside-toplevel

      meta = json.loads(metadata_file.read_text())
      if "use_zarr3" in meta:
        a.checkpoint_storage_use_zarr3 = bool(meta["use_zarr3"])
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Could not read Orbax _METADATA from %s: %s", path, exc)
  if a.use_ocdbt_with_pathways:
    from orbax.checkpoint._src.serialization import jax_array_handlers
    from orbax.checkpoint._src.serialization import type_handler_registry

    type_handler_registry.register_type_handler(
        jax.Array, jax_array_handlers.ArrayHandler(), override=True
    )
  mt_cfg = eval_deepswe.maxtext_config(a)
  mt_cfg["load_parameters_path"] = str(path)
  additional_config = {
      "enable_continue_decode": False,
      "maxtext_config": mt_cfg,
  }

  if jax.device_count() != a.mesh_fsdp * a.mesh_tp:
    raise ValueError(
        f"Expected {a.mesh_fsdp * a.mesh_tp} rollout chips; got"
        f" {jax.device_count()}"
    )
  mesh = Mesh(
      mesh_utils.create_device_mesh(
          (a.mesh_fsdp, a.mesh_tp),
          jax.devices(),
          allow_split_physical_axes=True,
      ),
      ("fsdp", "tp"),
  )
  tokenizer = AutoTokenizer.from_pretrained(a.tokenizer_path)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  eos_ids = []
  for token in ("<|im_end|>", "<|endoftext|>"):
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
      raise ValueError(f"{token} must be a single Qwen token; got {ids}")
    eos_ids.extend(ids)
  engine_kwargs = {
      "model": a.model_id,
      "tokenizer": a.tokenizer_path,
      "max_model_len": a.max_model_len,
      "max_num_seqs": a.vllm_max_num_seqs,
      "max_num_batched_tokens": a.vllm_max_num_batched_tokens,
      "enable_prefix_caching": a.enable_prefix_caching,
      "async_scheduling": os.environ.get(
          "VLLM_ASYNC_SCHEDULING", "0"
      ).lower() in ("1", "true"),
      "dtype": "bfloat16",
      "enable_expert_parallel": os.environ.get(
          "VLLM_ENABLE_EXPERT_PARALLEL", "0"
      ).lower() in ("1", "true"),
      "disable_log_stats": False,
      # Use the explicit sampling settings, not repository generation_config.
      "generation_config": "vllm",
      "seed": a.seed,
  }
  if os.environ.get("VLLM_LANGUAGE_MODEL_ONLY", "0").lower() in ("1", "true"):
    engine_kwargs["language_model_only"] = True
  if os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "0").lower() in (
      "1",
      "true",
  ):
    engine_kwargs["enable_chunked_prefill"] = True
  if os.environ.get("VLLM_KV_CACHE_DTYPE"):
    engine_kwargs["kv_cache_dtype"] = os.environ["VLLM_KV_CACHE_DTYPE"]
  if os.environ.get("VLLM_BLOCK_SIZE"):
    engine_kwargs["block_size"] = int(os.environ["VLLM_BLOCK_SIZE"])
  if os.environ.get("VLLM_MAMBA_CACHE_MODE"):
    engine_kwargs["mamba_cache_mode"] = os.environ["VLLM_MAMBA_CACHE_MODE"]
  if os.environ.get("VLLM_LIMIT_MM_PER_PROMPT"):
    mm_limits = {}
    for item in os.environ["VLLM_LIMIT_MM_PER_PROMPT"].split(","):
      if "=" in item:
        k, v = item.split("=", 1)
        mm_limits[k.strip()] = int(v.strip())
    if mm_limits:
      engine_kwargs["limit_mm_per_prompt"] = mm_limits
  engine_kwargs["hf_overrides"] = {
      "architectures": ["MaxTextForCausalLM"]
  }
  config = vllm_sampler.VllmConfig(
      server_mode=True,
      mesh=mesh,
      tensor_parallel_size=a.mesh_tp,
      data_parallel_size=a.mesh_fsdp,
      init_with_random_weights=False,
      hbm_utilization=a.vllm_utilization,
      additional_config=additional_config,
      engine_kwargs=engine_kwargs,
      eos_tokens=eos_ids,
      sampling_kwargs={
          "stop": ["</function>"],
          "include_stop_str_in_output": True,
          "skip_special_tokens": False,
      },
  )
  sampler = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id="deepswe-eval",
      tokenizer=tokenizer,
      config=config,
      weight_sync_mode="none",
      max_concurrency=a.max_concurrent,
  )

  class EvaluationWorker(rollout_worker.RolloutWorker):
    """Compact eval RPC over the same manager and collector as training."""

    _stop_event = None

    def evaluation_info(self):
      return eval_deepswe.model_profile(a)

    def shutdown(self):
      if self._stop_event is not None:
        self._stop_event.set()
      return True

    async def evaluate(self, fields):
      request = datatypes.RolloutRequest(**fields)
      response = await self.generate(request)
      # generate also enqueues each full trajectory for streaming consumers.
      # This controller consumes the direct return, so drain that extra copy.
      await self.pop_next_completed()
      return eval_deepswe.compact_result(response)

  if a.use_agent_sandbox:
    if a.docker_image_prefix:
      os.environ["IMAGE_REWRITE_PREFIX"] = a.docker_image_prefix
    entries = eval_deepswe.load_entries(a)
    # Populate the fleet plan with all dataset tasks so fleet.acquire claims
    # from the planned warmpools created by the controller's PrewarmDatasetIterator.
    sandbox_utils.init_global_fleet(
        tasks=entries,
        max_concurrency=a.max_concurrent,
        num_generations=a.num_rollouts_per_instance,
        batch_size=a.batch_size,
        max_warmpool_replicas=a.max_warmpool_size,
        scaffold=a.scaffold,
    )
  worker = EvaluationWorker(
      worker_id="deepswe-eval",
      config=rollout_worker.RolloutConfig(
          sampler_type="inprocess_vllm",
          weight_sync_mode="none",
          env_name=deepswe.DEEPSWE_ENV_NAME,
          agent_name=deepswe.DEEPSWE_AGENT_NAME,
          eos_tokens=eos_ids,
      ),
      sampler=sampler,
      tokenizer=tokenizer_adapter.TokenizerAdapter(tokenizer),
      chat_parser=parser.QwenChatTemplateParser(
          tokenizer, enable_thinking=a.enable_thinking
      ),
      max_concurrency=a.max_concurrent,
  )
  return worker


async def serve(a):
  # Initialization may compile for minutes. Publish RPC readiness only after
  # real checkpoint restore and sampler startup have succeeded.
  worker = create_worker(a)
  from tunix.experimental.worker import remote_execution
  from examples.deepswe import sandbox_utils

  server = remote_execution.GrpcRemoteExecutionServer(worker)
  try:
    await worker.sampler.start()
    worker.initialize()
    stop = asyncio.Event()
    worker._stop_event = stop
    await server.start_serving_async(a.port)
    logging.info(
        "DeepSWE eval worker ready on port %d: %s",
        a.port,
        eval_deepswe.model_profile(a),
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
      loop.add_signal_handler(sig, stop.set)
    await stop.wait()
  finally:
    worker.stop()
    await server.stop_serving()
    try:
      await worker.sampler.stop()
    finally:
      await asyncio.to_thread(sandbox_utils.teardown_global_fleet)
