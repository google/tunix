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
from tunix.utils import maxtext_utils



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
  if not (path / "_METADATA").exists():
    for subdir in ("model_params", "items", "default"):
      candidate = path / subdir
      if (candidate / "_METADATA").exists():
        logging.info(
            "Resolved composite checkpoint path %s -> %s", path, candidate
        )
        path = candidate
        break
  ckpt_prefuse_moe = False
  metadata_file = path / "_METADATA"
  if metadata_file.exists():
    try:
      import json  # pylint: disable=import-outside-toplevel

      meta_text = metadata_file.read_text()
      meta = json.loads(meta_text)
      if "use_ocdbt" in meta:
        a.checkpoint_storage_use_ocdbt = bool(meta["use_ocdbt"])
      if "use_zarr3" in meta:
        a.checkpoint_storage_use_zarr3 = bool(meta["use_zarr3"])
      if '"wi"' in meta_text:
        ckpt_prefuse_moe = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Could not read Orbax _METADATA from %s: %s", path, exc)
  is_ocdbt = bool(
      getattr(a, "checkpoint_storage_use_ocdbt", False)
      or (path / "manifest.ocdbt").exists()
  )
  if a.use_ocdbt_with_pathways and is_ocdbt:
    from orbax.checkpoint._src.serialization import jax_array_handlers
    from orbax.checkpoint._src.serialization import type_handler_registry

    logging.info(
        "Checkpoint at %s uses OCDBT; registering standard Orbax ArrayHandler.",
        path,
    )
    type_handler_registry.register_type_handler(
        jax.Array, jax_array_handlers.ArrayHandler(), override=True
    )
  else:
    logging.info(
        "Checkpoint at %s is not OCDBT (or use_ocdbt_with_pathways=False);"
        " keeping registered jax.Array handler.",
        path,
    )
  convert_in_memory = bool(a.scan_layers)
  mt_cfg = eval_deepswe.maxtext_config(a)
  mt_cfg["scan_layers"] = False
  if convert_in_memory:
    mt_cfg.pop("load_parameters_path", None)
  else:
    mt_cfg["load_parameters_path"] = str(path)
  additional_config = {
      "enable_continue_decode": False,
      "maxtext_config": mt_cfg,
  }
  raw_add_cfg = os.environ.get("VLLM_ADDITIONAL_CONFIG", "").strip()
  if raw_add_cfg:
    import json  # pylint: disable=import-outside-toplevel

    extra_add_cfg = json.loads(raw_add_cfg)
    if isinstance(extra_add_cfg, dict):
      extra_mt_cfg = extra_add_cfg.get("maxtext_config")
      if isinstance(extra_mt_cfg, dict):
        merged_mt_cfg = dict(extra_mt_cfg)
        merged_mt_cfg.update(mt_cfg)
        if convert_in_memory:
          merged_mt_cfg.pop("load_parameters_path", None)
        mt_cfg = merged_mt_cfg
      for k, v in extra_add_cfg.items():
        if k != "maxtext_config":
          additional_config[k] = v
      additional_config["maxtext_config"] = mt_cfg

  mesh_fsdp = a.mesh_fsdp
  mesh_dp = (
      getattr(a, "mesh_dp", None)
      or int(os.environ.get("VLLM_DATA_PARALLEL_SIZE", 0))
      or 1
  )
  mesh_tp = a.mesh_tp
  mesh_expert = a.mesh_expert
  multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "")
  if multihost_backend:
    if convert_in_memory:
      raise ValueError(
          "Converting scanned checkpoints in memory is not supported with"
          f" TPU_MULTIHOST_BACKEND={multihost_backend!r}. Use unscanned"
          " checkpoints or single-host/Pathways."
      )
    mesh = None
  else:
    expected_devices = mesh_fsdp * mesh_dp * mesh_tp * mesh_expert
    if jax.device_count() != expected_devices:
      raise ValueError(
          f"Expected {expected_devices} rollout chips (fsdp={mesh_fsdp},"
          f" dp={mesh_dp}, tp={mesh_tp}, expert={mesh_expert}); got"
          f" {jax.device_count()}"
      )
    if mesh_dp > 1 and mesh_fsdp > 1:
      mesh_shape = (mesh_dp, mesh_fsdp * mesh_expert, mesh_tp)
      axis_names = ("dp", "fsdp", "tp")
    elif mesh_dp > 1:
      mesh_shape = (mesh_dp * mesh_expert, mesh_tp)
      axis_names = ("dp", "tp")
    else:
      mesh_shape = (mesh_fsdp * mesh_expert, mesh_tp)
      axis_names = ("fsdp", "tp")

    mesh = Mesh(
        mesh_utils.create_device_mesh(
            mesh_shape,
            jax.devices(),
            allow_split_physical_axes=True,
        ),
        axis_names,
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
  if multihost_backend:
    engine_kwargs["distributed_executor_backend"] = multihost_backend
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
  if os.environ.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL"):
    engine_kwargs["prefix_cache_retention_interval"] = int(
        os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"]
    )
  if os.environ.get("VLLM_REASONING_PARSER"):
    engine_kwargs["reasoning_parser"] = os.environ["VLLM_REASONING_PARSER"]
  if os.environ.get("VLLM_LIMIT_MM_PER_PROMPT"):
    raw_mm = os.environ["VLLM_LIMIT_MM_PER_PROMPT"].strip()
    mm_limits = {}
    if raw_mm.startswith("{"):
      import json  # pylint: disable=import-outside-toplevel
      mm_limits = {k: int(v) for k, v in json.loads(raw_mm).items()}
    else:
      for item in raw_mm.split(","):
        if "=" in item:
          k, v = item.split("=", 1)
          mm_limits[k.strip()] = int(v.strip())
    if mm_limits:
      engine_kwargs["limit_mm_per_prompt"] = mm_limits
  engine_kwargs["hf_overrides"] = {
      "architectures": ["MaxTextForCausalLM"]
  }
  enable_dp_attention = mesh_dp > 1
  if raw_add_cfg:
    try:
      parsed_add = json.loads(raw_add_cfg)
      if "enable_dp_attention" in parsed_add.get("sharding", {}).get(
          "sharding_strategy", {}
      ):
        enable_dp_attention = bool(
            parsed_add["sharding"]["sharding_strategy"]["enable_dp_attention"]
        )
    except Exception:
      pass

  from examples.deepswe import template as deepswe_template

  sampling_kwargs = {"skip_special_tokens": False}
  if a.scaffold not in deepswe_template.OPENHANDS_SCAFFOLDS:
    sampling_kwargs["stop"] = ["</function>"]
    sampling_kwargs["include_stop_str_in_output"] = True

  config = vllm_sampler.VllmConfig(
      server_mode=True,
      mesh=mesh,
      tensor_parallel_size=a.mesh_tp,
      data_parallel_size=mesh_dp,
      expert_parallel_size=mesh_expert,
      enable_dp_attention=enable_dp_attention,
      init_with_random_weights=convert_in_memory,
      hbm_utilization=a.vllm_utilization,
      additional_config=additional_config,
      engine_kwargs=engine_kwargs,
      eos_tokens=eos_ids,
      sampling_kwargs=sampling_kwargs,
  )
  sampler = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id="deepswe-eval",
      tokenizer=tokenizer,
      config=config,
      weight_sync_mode="none",
      max_concurrency=a.max_concurrent,
  )
  if convert_in_memory:
    sampler.initialize()
    maxtext_utils.load_and_convert_scanned_checkpoint(
        path=path,
        sampler=sampler,
        mesh_tp=a.mesh_tp,
        ckpt_prefuse_moe=ckpt_prefuse_moe,
        maxtext_config_overrides=eval_deepswe.maxtext_config(a),
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
          agent_name=deepswe.get_agent_name(a.scaffold),
          agent_config={"scaffold": a.scaffold},
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
  try:
    worker = create_worker(a)
  except BaseException as e:
    logging.exception("Fatal error creating eval worker: %s", e)
    raise
  from tunix.experimental.worker import remote_execution
  from examples.deepswe import sandbox_utils

  server = remote_execution.GrpcRemoteExecutionServer(worker)
  try:
    await worker.sampler.start()
    worker.initialize()
    stop = asyncio.Event()
    worker._stop_event = stop
    await server.start_serving_async(a.port)
    marker_path = os.environ.get("FT_REGISTERED_MARKER")
    if marker_path:
      try:
        Path(marker_path).touch()
        logging.info("Created fail-fast registered marker %s", marker_path)
      except Exception:
        logging.warning("Failed to create marker %s", marker_path, exc_info=True)
    logging.info(
        "DeepSWE eval worker ready on port %d: %s",
        a.port,
        eval_deepswe.model_profile(a),
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
      loop.add_signal_handler(sig, stop.set)
    await stop.wait()
  except BaseException as e:
    logging.exception("Fatal error in eval worker serving loop: %s", e)
    raise
  finally:
    worker.stop()
    await server.stop_serving()
    try:
      await worker.sampler.stop()
    except Exception:
      logging.warning("Error stopping worker sampler", exc_info=True)
    finally:
      await asyncio.to_thread(sandbox_utils.teardown_global_fleet)
