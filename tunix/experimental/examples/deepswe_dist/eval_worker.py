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


def _iter_leaves(tree, jax_mod):
  tree_util = getattr(jax_mod, "tree_util", None)
  if tree_util is not None and hasattr(tree_util, "tree_leaves"):
    yield from tree_util.tree_leaves(tree)
    return
  if isinstance(tree, dict):
    for v in tree.values():
      yield from _iter_leaves(v, jax_mod)
  elif isinstance(tree, (list, tuple)):
    for v in tree:
      yield from _iter_leaves(v, jax_mod)
  else:
    yield tree


def _delete_pytree_buffers(tree, jax_mod, keep_tree=None):
  array_cls = getattr(jax_mod, "Array", ())
  if not isinstance(array_cls, type):
    return
  keep_ids = set()
  if keep_tree is not None:
    for leaf in _iter_leaves(keep_tree, jax_mod):
      val = getattr(leaf, "value", leaf)
      keep_ids.add(id(val))
  for leaf in _iter_leaves(tree, jax_mod):
    val = getattr(leaf, "value", leaf)
    if id(val) in keep_ids:
      continue
    if isinstance(val, array_cls) and not getattr(
        val, "is_deleted", lambda: False
    )():
      try:
        val.delete()
      except Exception:  # pylint: disable=broad-exception-caught
        pass


def load_and_convert_scanned_checkpoint(
    a, path, sampler, ckpt_prefuse_moe=False
):
  """Restores a scanned MaxText checkpoint and converts it into vLLM's unscanned state."""
  import gc  # pylint: disable=import-outside-toplevel
  from flax import nnx  # pylint: disable=import-outside-toplevel
  import jax  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.integration.vllm.weight_converter import MaxTextToMaxTextConverter  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR  # pylint: disable=import-outside-toplevel

  logging.info(
      "Restoring scanned MaxText checkpoint from %s and converting to unscanned"
      " vLLM weights in-memory...",
      path,
  )
  reinit_needed = False
  if hasattr(sampler.vllm_sampler, "delete_cache"):
    sampler.vllm_sampler.delete_cache()
    reinit_needed = True
  _delete_pytree_buffers(
      getattr(sampler.vllm_sampler, "transformer_state", None), jax
  )
  try:
    base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    scanned_overrides = dict(eval_deepswe.maxtext_config(a))
    scanned_overrides.update({
        "load_parameters_path": str(path),
        "scan_layers": True,
        "prefuse_moe_weights": ckpt_prefuse_moe,
        "attention": "dot_product",
        "model_call_mode": "",
    })
    scanned_cfg = pyconfig.initialize(
        ["", str(base_config_path)], **scanned_overrides
    )
    scanned_model, scanned_mesh = model_creation_utils.from_pretrained(
        scanned_cfg,
        devices=jax.devices(),
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )
    scanned_state = nnx.state(scanned_model, nnx.Param)
    del scanned_model, scanned_mesh
    gc.collect()

    converter = MaxTextToMaxTextConverter(
        config=scanned_cfg,
        tp=a.mesh_tp,
        prefuse_moe_weights=True,
        target_dtype=None,
    )
    orig_exec_group = getattr(converter, "_execute_group", None)
    prefused_moe_sources = {}
    orig_build_plan = getattr(converter, "_build_plan", None)
    if ckpt_prefuse_moe and callable(orig_build_plan):

      def _split_prefused_moe_and_build_plan(
          src_flat, tgt_flat, skip_paths=frozenset()
      ):
        for k in list(src_flat.keys()):
          if "layers" in k and k[-1] == "wi":
            wi = src_flat.pop(k)
            prefix = k[:-1]
            if callable(orig_exec_group):
              prefused_moe_sources[prefix] = wi
              src_flat[prefix + ("wi_0",)] = wi
              src_flat[prefix + ("wi_1",)] = wi
            else:
              half = wi.shape[-1] // 2
              src_flat[prefix + ("wi_0",)] = wi[..., :half]
              src_flat[prefix + ("wi_1",)] = wi[..., half:]
        return orig_build_plan(src_flat, tgt_flat, skip_paths)

      converter._build_plan = _split_prefused_moe_and_build_plan

    if callable(orig_exec_group):

      def _exec_and_free_group(group, src_flat, tgt_flat):
        prefused_wi = None
        if group.op == "fuse_moe" and group.source_keys:
          prefused_wi = prefused_moe_sources.pop(
              group.source_keys[0][:-1], None
          )
          if prefused_wi is not None:
            half = prefused_wi.shape[-1] // 2
            src_flat[group.source_keys[0]] = prefused_wi[..., :half]
            src_flat[group.source_keys[1]] = prefused_wi[..., half:]
        outs = orig_exec_group(group, src_flat, tgt_flat)
        out_arrays = [out for _, out in outs]
        if hasattr(jax, "block_until_ready"):
          jax.block_until_ready(out_arrays)
        _delete_pytree_buffers(
            [prefused_wi] + [src_flat.get(k) for k in group.source_keys],
            jax,
            keep_tree=out_arrays,
        )
        return outs

      converter._execute_group = _exec_and_free_group

    converted_state = converter.convert(
        scanned_state,
        target_state=sampler.vllm_sampler.transformer_state,
    )
    _delete_pytree_buffers(scanned_state, jax, keep_tree=converted_state)
    del scanned_state
    gc.collect()

    while isinstance(converted_state, dict) and "model" in converted_state:
      converted_state = converted_state["model"]
    sampler_cfg = getattr(sampler.vllm_sampler, "config", None)
    orig_free_kv = getattr(
        sampler_cfg, "free_kv_cache_during_weight_sync", None
    )
    if sampler_cfg is not None and orig_free_kv is not None:
      sampler_cfg.free_kv_cache_during_weight_sync = False
    try:
      sampler.vllm_sampler.update_params(converted_state)
    finally:
      if sampler_cfg is not None and orig_free_kv is not None:
        sampler_cfg.free_kv_cache_during_weight_sync = orig_free_kv
    _delete_pytree_buffers(
        converted_state,
        jax,
        keep_tree=getattr(sampler.vllm_sampler, "transformer_state", None),
    )
    del converted_state
    gc.collect()
    jax.clear_caches()
    if hasattr(jax, "effects_barrier"):
      jax.effects_barrier()
  finally:
    if reinit_needed and hasattr(sampler.vllm_sampler, "reinitialize_cache"):
      sampler.vllm_sampler.reinitialize_cache()
  logging.info("Completed in-memory scanned-to-unscanned weight conversion.")


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
      if "use_zarr3" in meta:
        a.checkpoint_storage_use_zarr3 = bool(meta["use_zarr3"])
      if '"wi"' in meta_text:
        ckpt_prefuse_moe = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Could not read Orbax _METADATA from %s: %s", path, exc)
  if a.use_ocdbt_with_pathways:
    from orbax.checkpoint._src.serialization import jax_array_handlers
    from orbax.checkpoint._src.serialization import type_handler_registry

    type_handler_registry.register_type_handler(
        jax.Array, jax_array_handlers.ArrayHandler(), override=True
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

  mesh_expert = getattr(a, "mesh_expert", 1)
  expected_devices = a.mesh_fsdp * a.mesh_tp * mesh_expert
  if jax.device_count() != expected_devices:
    raise ValueError(
        f"Expected {expected_devices} rollout chips; got"
        f" {jax.device_count()}"
    )
  mesh = Mesh(
      mesh_utils.create_device_mesh(
          (a.mesh_fsdp * mesh_expert, a.mesh_tp),
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
  config = vllm_sampler.VllmConfig(
      server_mode=True,
      mesh=mesh,
      tensor_parallel_size=a.mesh_tp,
      data_parallel_size=a.mesh_fsdp,
      expert_parallel_size=mesh_expert,
      init_with_random_weights=convert_in_memory,
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
  if convert_in_memory:
    load_and_convert_scanned_checkpoint(
        a, path, sampler, ckpt_prefuse_moe=ckpt_prefuse_moe
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
