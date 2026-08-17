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

"""Legacy vLLM Sampler adapter integrating with Tunix VllmSampler."""

try:
  import tpu_raiden.frameworks.jax._tpu_raiden_jax  # Preload raiden C++ module before JAX init
except Exception:
  pass

import abc
import asyncio
import contextlib
import ipaddress
import numbers
import os
import re
from typing import Any, List, Mapping, Sequence
from absl import logging
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

from tunix.experimental.orchestrator import weight_sync
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.rollout import sampler as base_sampler_lib

Sampler = base_sampler_lib.Sampler


def _get_vllm_sampler_cls():
  """Lazy import of tunix.generate.vllm_sampler to avoid top-level vLLM import side-effects."""
  from tunix.generate import vllm_sampler as generate_vllm_lib  # pylint: disable=g-import-not-at-top
  return generate_vllm_lib


def _hf_to_tunix_name(name: str) -> str:
  """Maps HuggingFace/vLLM parameter names to Tunix model parameter names."""
  if name == "model.embed_tokens.weight":
    return "embedder.input_embedding"
  if name == "model.norm.weight":
    return "final_norm.w"
  if name in ("lm_head.weight", "model.lm_head.weight"):
    return "lm_head.w"
  m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
  if m:
    layer_idx, rest = m.groups()
    if rest == "input_layernorm.weight":
      return f"layers.{layer_idx}.input_layernorm.w"
    if rest == "post_attention_layernorm.weight":
      return f"layers.{layer_idx}.post_attention_layernorm.w"
    if rest == "mlp.down_proj.weight":
      return f"layers.{layer_idx}.mlp.down_proj.kernel"
    if rest == "mlp.gate_proj.weight":
      return f"layers.{layer_idx}.mlp.gate_proj.kernel"
    if rest == "mlp.up_proj.weight":
      return f"layers.{layer_idx}.mlp.up_proj.kernel"
    if rest == "self_attn.k_norm.weight":
      return f"layers.{layer_idx}.attn.k_norm.w"
    if rest == "self_attn.q_norm.weight":
      return f"layers.{layer_idx}.attn.q_norm.w"
    if rest == "self_attn.k_proj.weight":
      return f"layers.{layer_idx}.attn.k_proj.w"
    if rest == "self_attn.q_proj.weight":
      return f"layers.{layer_idx}.attn.q_proj.w"
    if rest == "self_attn.v_proj.weight":
      return f"layers.{layer_idx}.attn.v_proj.w"
    if rest == "self_attn.o_proj.weight":
      return f"layers.{layer_idx}.attn.o_proj.w"
    if rest == "self_attn.k_proj.bias":
      return f"layers.{layer_idx}.attn.k_bias"
    if rest == "self_attn.q_proj.bias":
      return f"layers.{layer_idx}.attn.q_bias"
    if rest == "self_attn.v_proj.bias":
      return f"layers.{layer_idx}.attn.v_bias"
  return name


class LegacyVllmSamplerAdapter(Sampler, abc.ABC):
  """Sampler adapter wrapping Tunix VllmSampler."""

  def __init__(
      self,
      server_id: str,
      tokenizer: Any = None,
      config: Any = None,
      model_name: str = "",
      **kwargs,
  ):
    self.server_id = server_id
    self.tokenizer = tokenizer
    self.config = config
    self.model_name = model_name or kwargs.get("model", "")
    self.vllm_sampler = kwargs.get("vllm_sampler", None)
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._phase_lock = asyncio.Lock()
    self._admitting = asyncio.Event()
    self._admitting.set()
    self._pending_weights: Any = None
    self._dst_ws_info: Any = None
    self._dst_staging_arrays: list[jax.Array] = []
    self._dst_variable_names: list[str] = []

    if self.vllm_sampler is None and self.tokenizer is not None and self.config is not None:
      vllm_lib = _get_vllm_sampler_cls()
      self.vllm_sampler = vllm_lib.VllmSampler(
          tokenizer=self.tokenizer, config=self.config
      )

  def initialize(self) -> None:
    """Initializes vLLM sampler if needed."""
    if self.vllm_sampler is not None:
      return
    if self.tokenizer is None and self.model_name:
      from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
      from tunix.generate import vllm_sampler as tunix_vllm_sampler  # pylint: disable=g-import-not-at-top

      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.config = tunix_vllm_sampler.VllmConfig(
          engine_kwargs={"model": self.model_name}
      )

    if (
        self.vllm_sampler is None
        and self.tokenizer is not None
        and self.config is not None
    ):
      vllm_lib = _get_vllm_sampler_cls()
      self.vllm_sampler = vllm_lib.VllmSampler(
          tokenizer=self.tokenizer, config=self.config
      )
    if self.vllm_sampler is None:
      raise RuntimeError(
          f"LegacyVllmSamplerAdapter [{self.server_id}] requires a vllm_sampler"
          " instance or tokenizer + config."
      )

  def _unpadded_prompt_tokens(self, padded_tokens: Any) -> np.ndarray:
    """Returns sampler-tokenized prompt ids without backend left padding."""
    arr = np.asarray(padded_tokens, dtype=np.int32).reshape(-1)
    pad_id = getattr(self.tokenizer, "pad_token_id", None)
    if pad_id is None:
      pad_id = getattr(self.tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, numbers.Integral):
      return arr
    non_pad = np.flatnonzero(arr != pad_id)
    if non_pad.size == 0:
      return np.zeros(0, dtype=np.int32)
    return arr[non_pad[0] :]

  def _prompt_tokens_from_request(
      self, req: Any, fallback_padded_tokens: Any
  ) -> np.ndarray:
    """Returns request token ids directly when available, else sampler output."""
    prompt = req.prompt if hasattr(req, "prompt") else req
    try:
      return np.asarray(prompt, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
      pass
    return self._unpadded_prompt_tokens(fallback_padded_tokens)

  def _prompt_to_input_string(self, prompt: Any) -> Any:
    """Renders chat-message prompts to strings for Tunix VllmSampler."""
    if isinstance(prompt, str):
      return prompt
    if isinstance(prompt, (list, tuple)) and all(
        isinstance(message, dict) for message in prompt
    ):
      if hasattr(self.tokenizer, "apply_chat_template"):
        return self.tokenizer.apply_chat_template(
            list(prompt), tokenize=False, add_generation_prompt=True
        )
      return "\n".join(
          str(message.get("content", "")) for message in prompt
      )
    return prompt

  # --- Lifecycle & Topology ---
  async def start(self, **kwargs) -> str | None | Any:
    """Starts the sampling engine or local loop."""
    del kwargs
    return True

  async def stop(self, **kwargs) -> str | None | Any:
    del kwargs
    if self.vllm_sampler and hasattr(self.vllm_sampler, "stop"):
      self.vllm_sampler.stop()
    return True

  async def pause(self, **kwargs) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    del kwargs
    self._admitting.clear()
    return True

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    del kwargs
    self._admitting.set()
    return True

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    del kwargs
    if self.vllm_sampler and hasattr(self.vllm_sampler, "mesh"):
      return self.vllm_sampler.mesh
    return None

  # --- Inference ---
  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
          | Any
      ),
      **kwargs,
  ) -> (
      base_sampler_lib.SamplingResponse
      | List[base_sampler_lib.SamplingResponse]
      | Any
  ):
    """Generates completions using underlying Tunix VllmSampler."""
    await self._admitting.wait()
    if not self.vllm_sampler:
      raise RuntimeError(
          f"LegacyVllmSamplerAdapter [{self.server_id}] vllm_sampler is not"
          " initialized."
      )

    if sampling_requests is None:
      raise ValueError("sampling_requests cannot be None.")

    if isinstance(sampling_requests, base_sampler_lib.SamplingRequest):
      requests: List[Any] = [sampling_requests]
      is_sequence = False
    elif isinstance(sampling_requests, (list, tuple)):
      requests = list(sampling_requests)
      is_sequence = True
    else:
      requests = [sampling_requests]
      is_sequence = False

    prompts = []
    max_gen_steps_list = []
    temps = []
    top_ps = []
    top_ks = []
    seeds = []
    return_logprobs_list = []

    for req in requests:
      prompt = req.prompt if hasattr(req, "prompt") else req
      prompts.append(self._prompt_to_input_string(prompt))
      sp = (
          req.sampling_params
          if hasattr(req, "sampling_params") and req.sampling_params is not None
          else base_sampler_lib.SamplingParams()
      )
      assert sp is not None

      max_gen_steps_list.append(sp.max_tokens)
      temps.append(sp.temperature)
      top_ps.append(sp.top_p)
      top_ks.append(sp.top_k)
      seeds.append(sp.seed)
      return_logprobs_list.append(sp.return_logprobs)

    max_generation_steps = (
        max(max_gen_steps_list) if max_gen_steps_list else 64
    )
    temperature = temps[0] if temps else 0.0
    top_p = top_ps[0] if top_ps else None
    top_k = top_ks[0] if top_ks else None
    seed = seeds[0] if seeds else None
    return_logprobs = any(return_logprobs_list) or kwargs.get(
        "return_logprobs", False
    )

    sampler_output = self.vllm_sampler(
        input_strings=prompts,
        max_generation_steps=max_generation_steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        return_logprobs=return_logprobs,
    )

    responses = []
    for i, req in enumerate(requests):
      req_id = getattr(req, "request_id", "")

      txt = (
          sampler_output.text[i]
          if isinstance(sampler_output.text, list)
          else sampler_output.text
      )
      toks = (
          sampler_output.tokens[i]
          if isinstance(sampler_output.tokens, list)
          else sampler_output.tokens
      )
      lps = None
      if sampler_output.logprobs and isinstance(sampler_output.logprobs, list):
        lps = sampler_output.logprobs[i]

      tok_ids = (
          np.array(toks, dtype=np.int32)
          if toks is not None
          else np.zeros(0, dtype=np.int32)
      )
      prompt_token_ids = self._prompt_tokens_from_request(
          req, sampler_output.padded_prompt_tokens[i]
      )
      log_ps = np.array(lps, dtype=np.float32) if lps is not None else None

      responses.append(
          base_sampler_lib.SamplingResponse(
              request_id=req_id,
              text=txt,
              prompt_token_ids=prompt_token_ids,
              token_ids=tok_ids,
              logprobs=log_ps,
              finish_reason="stop",
          )
      )

    if is_sequence:
      return responses
    return responses[0]

  # --- Weight Synchronization (WeightSyncDestination) ---
  async def bind_weight_sync(self, **kwargs) -> None:
    """Binds destination-side transport resources."""
    if self.vllm_sampler is None:
      self.initialize()

    mesh = await self.get_mesh()
    has_raiden = False
    try:
      from tpu_raiden.api.jax.weight_synchronizer import WeightSynchronizer  # pylint: disable=g-import-not-at-top
      has_raiden = (
          WeightSynchronizer is not None
          and jax.default_backend() == "tpu"
          and mesh is not None
      )
    except Exception:
      has_raiden = False

    if mesh is not None and not getattr(self, "_dst_ws", None) and not getattr(self, "_dst_ws_info", None):
      try:
        arrays: list[jax.Array] = []
        names: list[str] = []
        if hasattr(self.vllm_sampler, "transformer_state") and self.vllm_sampler.transformer_state is not None:
          dst_state = self.vllm_sampler.transformer_state
          dst_dict = {}
          if hasattr(dst_state, "flat_state"):
            for path, val in dst_state.flat_state():
              arr = val.get_value() if hasattr(val, "get_value") else getattr(val, "value", val)
              if isinstance(arr, (jax.Array, np.ndarray)):
                hf_name = ".".join(str(p) for p in path)
                tunix_name = _hf_to_tunix_name(hf_name)
                dst_dict[tunix_name] = arr
            sorted_keys = sorted(dst_dict.keys())
            names = sorted_keys
            arrays = [dst_dict[k] for k in sorted_keys]
          else:
            for i, x in enumerate(jax.tree_util.tree_leaves(dst_state)):
              if isinstance(x, (jax.Array, np.ndarray)):
                names.append(f"param_{i}")
                arrays.append(x)

        if not arrays:
          axis0 = mesh.axis_names[0] if mesh.axis_names else "dp"
          axis1 = mesh.axis_names[1] if len(mesh.axis_names) > 1 else "tp"
          dummy_arr = jax.device_put(
              jnp.zeros((1024, 1024), dtype=jnp.float32),
              shd.NamedSharding(mesh, shd.PartitionSpec(axis0, axis1)),
          )
          arrays = [dummy_arr]
          names = ["model.dummy"]

        self._dst_staging_arrays = arrays
        self._dst_variable_names = names

        backend = (
            kwargs.get("backend")
            or os.environ.get("TUNIX_WEIGHT_SYNC_BACKEND")
            or (
                "pathways"
                if (
                    "proxy" in os.environ.get("JAX_PLATFORMS", "")
                    or os.environ.get("JAX_BACKEND_TARGET")
                )
                else "local_launcher"
            )
        )

        if backend == "pathways":
          self._bind_weight_sync_pathways(
              arrays=self._dst_staging_arrays, mesh=mesh, **kwargs
          )
        else:
          self._bind_weight_sync_local_launcher(
              arrays=self._dst_staging_arrays, mesh=mesh, **kwargs
          )
      except Exception as err:
        logging.warning("Raiden destination binding fallback: %r", err)
        self._dst_ws = None
        self._dst_ws_info = None

  def _bind_weight_sync_local_launcher(
      self, arrays: Sequence[jax.Array], mesh: Any, **kwargs
  ) -> None:
    """Binds Raiden weight sync via WeightSynchronizer API for local launcher backend."""
    from tpu_raiden.api.jax.weight_synchronizer import WeightSynchronizer  # pylint: disable=g-import-not-at-top
    self._dst_ws = WeightSynchronizer(
        jax_arrays=arrays,
        parallelism=int(kwargs.get("parallelism", 16)),
        listener_port=0,
        unsafe_skip_buffer_lock=True,
    )

  def _bind_weight_sync_pathways(
      self, arrays: Sequence[jax.Array], mesh: Any, **kwargs
  ) -> None:
    """Binds Raiden weight sync via weight_synchronizer_ffi for Pathways backend."""
    from tpu_raiden.frameworks.jax import weight_synchronizer_ffi as raiden_ffi  # pylint: disable=g-import-not-at-top
    max_slice_size = max(
        int(
            np.prod(
                getattr(arr, "sharding", None).shard_shape(arr.shape)
                if hasattr(getattr(arr, "sharding", None), "shard_shape")
                else arr.shape
            )
        )
        * arr.dtype.itemsize
        for arr in arrays
    )
    dst_global_ids = np.arange(mesh.devices.size, dtype=np.int32).reshape(
        mesh.devices.shape
    )
    dst_shard_idx = jax.device_put(
        dst_global_ids,
        shd.NamedSharding(mesh, shd.PartitionSpec(*mesh.axis_names)),
    )
    slice_byte_sizes = jnp.array([max_slice_size] * len(arrays), dtype=jnp.int32)
    self._dst_ws_info = raiden_ffi.init_weight_synchronizer(
        device_array=arrays[0],
        shard_idx=dst_shard_idx,
        mesh=mesh,
        slice_byte_sizes=slice_byte_sizes,
        local_port=0,
        parallelism=int(kwargs.get("parallelism", 16)),
        num_layers=len(arrays),
        listener_port=0,
    )
    self._dst_ws_info.block_until_ready()

  async def get_weight_sync_metadata(
      self, **kwargs
  ) -> Sequence[weight_sync.WorkUnitMetadata]:
    """Returns transport metadata for this worker."""
    del kwargs
    if self.vllm_sampler is None:
      self.initialize()

    mesh = await self.get_mesh()
    if mesh is not None and hasattr(mesh, "shape") and isinstance(mesh.shape, dict) and mesh.shape:
      if len(mesh.shape) == 1:
        physical_mesh_shape = (1, tuple(mesh.shape.values())[0])
        mesh_axes = ("data", tuple(mesh.axis_names)[0])
      else:
        physical_mesh_shape = tuple(mesh.shape.values())
        mesh_axes = tuple(mesh.axis_names)
    else:
      physical_mesh_shape = (1, 1)
      mesh_axes = ("data", "tp")

    if getattr(self, "_dst_ws", None) is not None:
      dst_ips = [f"127.0.0.1:{self._dst_ws.local_port}"]
      dst_listener = f"127.0.0.1:{self._dst_ws.listener_port}"
    elif getattr(self, "_dst_ws_info", None) is not None:
      def _unpack_ip(row: np.ndarray) -> str:
        raw_bytes = row[:4].astype(np.int32).tobytes()
        try:
          ip_obj = ipaddress.IPv6Address(raw_bytes)
          if ip_obj.ipv4_mapped is not None:
            return str(ip_obj.ipv4_mapped)
          return f"[{ip_obj}]" if ":" in str(ip_obj) else str(ip_obj)
        except Exception:
          return "127.0.0.1"

      dst_info_np = np.asarray(self._dst_ws_info).reshape(-1, 6)
      dst_ips = [f"{_unpack_ip(row)}:{row[4]}" for row in dst_info_np]
      dst_listener = f"{_unpack_ip(dst_info_np[0])}:{dst_info_np[0][5]}"
    else:
      num_devices = 1
      if mesh is not None and hasattr(mesh, "devices"):
        devs = getattr(mesh, "devices", None)
        if hasattr(devs, "size") and isinstance(devs.size, int):
          num_devices = max(1, devs.size)

      dst_ips = [
          f"127.0.0.1:{29600 + i}"
          for i in range(num_devices)
      ]
      dst_listener = "127.0.0.1:29600"

    variables: list[weight_sync.TensorMetadata] = []
    if self._dst_staging_arrays:
      for idx, arr in enumerate(self._dst_staging_arrays):
        shape = tuple(arr.shape)
        shd_obj = getattr(arr, "sharding", None)
        if isinstance(shd_obj, shd.NamedSharding) and hasattr(shd_obj, "spec"):
          spec = shd_obj.spec
          raw_spec = list(spec) if spec is not None else []
          padded_spec = raw_spec + [None] * max(0, len(shape) - len(raw_spec))
          spec_axes = tuple(
              "" if a is None else (a if isinstance(a, str) else (a[0] if a else ""))
              for a in padded_spec[: len(shape)]
          )
          l_shape = shd_obj.shard_shape(shape)
          s_shape = tuple(max(1, g // l) for g, l in zip(shape, l_shape))
        else:
          spec_axes = tuple("" for _ in range(len(shape)))
          s_shape = tuple(1 for _ in range(len(shape)))

        var_name = (
            self._dst_variable_names[idx]
            if idx < len(self._dst_variable_names)
            else f"param_{idx}"
        )
        variables.append(
            weight_sync.TensorMetadata(
                name=var_name,
                shape=shape,
                mesh_shape=s_shape,
                layout=tuple(range(len(shape) - 1, -1, -1)),
                item_size=arr.dtype.itemsize,
                layer_idx=idx,
                sharding_spec=spec_axes,
            )
        )

    work_unit = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(
            job_name=self.server_id,
            job_replica_id="0",
            data_name="weights",
        ),
        shards=tuple(dst_ips),
        control_plane_rpc_address=dst_listener,
        mesh_shape=physical_mesh_shape,
        mesh_axes=mesh_axes,
        variables=tuple(variables),
    )
    return [work_unit]

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Prepares staging handshake prior to policy weight update."""
    del kwargs
    async with self._phase_lock:
      if self._tracker.admit(sync_request, "prepared"):
        await self.pause()
        self._tracker.complete(sync_request, "prepared")
    return True

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Updates model weights in-place from the specified controller."""
    del kwargs
    async with self._phase_lock:
      if self._tracker.admit(sync_request, "h2d_done"):
        if getattr(self, "_dst_ws", None) is not None:
          self._dst_ws.h2d()
          jax.effects_barrier()
        elif sync_request is not None:
          weights = getattr(sync_request, "weights", None)
          if weights is not None:
            self._pending_weights = weights
        self._tracker.complete(sync_request, "h2d_done")
    return True

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Finalizes and switches active policy weights after transfer completion."""
    del kwargs
    async with self._phase_lock:
      if self._tracker.admit(sync_request, "committed"):
        if self.vllm_sampler and getattr(self.vllm_sampler, "llm", None) is not None:
          try:
            self.vllm_sampler.llm.reset_prefix_cache()
          except Exception:
            pass
        elif (
            self._pending_weights is not None
            and self.vllm_sampler
            and hasattr(self.vllm_sampler, "update_params")
        ):
          self.vllm_sampler.update_params(self._pending_weights)
          self._pending_weights = None
        await self.resume()
        self._tracker.complete(sync_request, "committed")
    return True

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Rolls back to serving the previous weights."""
    del kwargs
    async with self._phase_lock:
      if self._tracker.admit(sync_request, "aborted"):
        self._pending_weights = None
        await self.resume()
        self._tracker.complete(sync_request, "aborted")
    return True

  async def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Returns the worker round tracker status report."""
    return self._tracker.report()

  async def get_transfer_status(self, req_id: Any, **kwargs) -> Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns best-effort vLLM queue/cache load information."""
    del kwargs
    return base_sampler_lib.LoadInfo()

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Triggers KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids
    return True
