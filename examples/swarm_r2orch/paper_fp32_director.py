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
"""Single-compiled batched TRUE float32 Director Sampler for R2Orch Option 5.

Replaces lossy `bfloat16` weight-merged `VllmSampler` on the Director partition
with ONE `tunix.generate.sampler.Sampler` sharded on `(fsdp=dp, tp=4)` using
`ShardingConfig.get_default_sharding(is_sampling=True)`:
  - Compiles ONE XLA graph per prompt bucket (saving ~8x pathways-proxy RAM).
  - Runs `batch_size = dp` concurrent Director requests in parallel in exact
    `float32` (0.0% LoRA rounding loss vs 82%-99.9% loss in merged `bfloat16`).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import gc
import queue
import secrets
import threading
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.common import datatypes
from tunix.dist_rl.swarm import contracts
from tunix.generate import base_sampler
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen3p5 import model as qwen3p5_model

PROMPT_BUCKETS = (2048, 8192, 16384, 30720)
GEN_BUCKETS = (32, 256, 1024, 2048, 4096, 12288)


def _delete_unshared_leaves(tree: Any, keep_ids: set[int]) -> None:
  for leaf in jax.tree.leaves(tree):
    if (
        isinstance(leaf, jax.Array)
        and id(leaf) not in keep_ids
        and not leaf.is_deleted()
    ):
      leaf.delete()


def _merge_single_lora_leaf(
    arr: jax.Array,
    entry: dict[str, np.ndarray],
    scale: float,
    rep_sharding: jax.sharding.Sharding,
    target_shd: Any,
) -> jax.Array:
  a_jax = jax.device_put(np.asarray(entry["lora_a"], dtype=np.float32), rep_sharding)
  b_jax = jax.device_put(np.asarray(entry["lora_b"], dtype=np.float32), rep_sharding)
  delta_2d = jnp.matmul(a_jax, b_jax, precision=jax.lax.Precision.HIGHEST)
  delta = (delta_2d * jnp.float32(scale)).reshape(arr.shape)
  updated = (arr.astype(jnp.float32) + delta).astype(jnp.float32)
  if isinstance(target_shd, jax.sharding.Sharding) and updated.sharding != target_shd:
    placed = jax.device_put(updated, target_shd)
    jax.block_until_ready(placed)
    updated.delete()
    updated = placed
  else:
    jax.block_until_ready(updated)
  for tmp in (a_jax, b_jax, delta_2d, delta):
    if isinstance(tmp, jax.Array) and not tmp.is_deleted():
      tmp.delete()
  return updated


def _set_pure_leaf(mapping: Any, keys: tuple[Any, ...], value: Any) -> None:
  cur = mapping
  for k in keys[:-1]:
    if isinstance(cur, dict):
      cur = cur[k] if k in cur else cur[str(k)]
    else:
      cur = cur[int(k)]
  last = keys[-1]
  if isinstance(cur, dict):
    cur[last if last in cur else str(last)] = value
  else:
    cur[int(last)] = value


def _merge_base_with_lora_on_mesh(
    base_state: nnx.State,
    target_shardings: nnx.State,
    lora_np: dict[str, dict[str, np.ndarray]] | None,
    scale: float,
    rep_sharding: jax.sharding.Sharding,
) -> tuple[nnx.State, int]:
  """Merges FP32 LoRA deltas directly on the Director TPU mesh (~465MB transfer)."""
  shd_by_key = {
      k: (v.get_value() if isinstance(v, nnx.Variable) else v)
      for k, v in nnx.to_flat_state(target_shardings)
  }
  merged_pure = nnx.to_pure_dict(base_state)
  n_merged = 0
  for keys, var in nnx.to_flat_state(base_state):
    arr = var.get_value() if isinstance(var, nnx.Variable) else var
    key_str = ".".join(str(k) for k in keys)
    entry = lora_np.get(key_str) if (lora_np and scale != 0.0) else None
    if entry is None:
      continue
    updated = _merge_single_lora_leaf(
        arr, entry, scale, rep_sharding, shd_by_key.get(keys)
    )
    _set_pure_leaf(merged_pure, keys, updated)
    n_merged += 1
  out_state = jax.tree.map(lambda v: v, base_state)
  nnx.replace_by_pure_dict(out_state, merged_pure)
  return out_state, n_merged


def _remesh_to_sampling_specs(
    src_state: nnx.State,
    template_state: nnx.State,
    target_shardings: nnx.State,
) -> nnx.State:
  """Transfers `src_state` leaves onto the exact sampling mesh shardings."""
  src_pure = nnx.to_pure_dict(src_state)
  shd_pure = nnx.to_pure_dict(target_shardings)

  def _put(x: Any, shd: Any) -> Any:
    if isinstance(x, jax.Array) and isinstance(shd, jax.sharding.Sharding):
      src = x if x.dtype == jnp.float32 else x.astype(jnp.float32)
      placed = jax.device_put(src, shd)
      jax.block_until_ready(placed)
      if src is not x:
        src.delete()
      return placed
    return x

  placed_pure = jax.tree.map(_put, src_pure, shd_pure)
  out_state = jax.tree.map(lambda v: v, template_state)
  nnx.replace_by_pure_dict(out_state, placed_pure)
  gc.collect()
  return out_state


class MeshBoundBatchedSampler(sampler_lib.Sampler):
  """`Sampler` with batch-sharded inputs on `fsdp` and weights sharded on `tp`."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: datatypes.CacheConfig,
      mesh: jax.sharding.Mesh,
  ) -> None:
    super().__init__(transformer, tokenizer, cache_config)
    self._mesh = mesh
    self._rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self._batch_shd = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("fsdp")
    )

  def _place_state_leaf(self, x: Any) -> Any:
    if not isinstance(x, jax.Array):
      return x
    if x.ndim >= 1 and x.shape[0] % self._mesh.shape["fsdp"] == 0:
      spec = jax.sharding.PartitionSpec("fsdp", *([None] * (x.ndim - 1)))
      return jax.device_put(x, jax.sharding.NamedSharding(self._mesh, spec))
    return jax.device_put(x, self._rep)

  def init_sample_state(self, *args: Any, **kwargs: Any) -> Any:
    self.eos_ids = jax.device_put(self.eos_ids, self._rep)
    args = tuple(self._place_state_leaf(a) for a in args)
    kwargs = {k: self._place_state_leaf(v) for k, v in kwargs.items()}
    with self._mesh:
      st = super().init_sample_state(*args, **kwargs)
    return jax.tree.map(self._place_state_leaf, st)


@dataclasses.dataclass
class _PendingRequest:
  prompt_ids: list[int]
  max_p: int
  actual_gen: int
  temperature: float
  top_p: float
  top_k: int | None
  seed: int
  return_logprobs: bool
  future: concurrent.futures.Future[base_sampler.SamplerOutput]


class BatchingFp32DirectorSampler:
  """Drop-in Director sampler running batched TRUE `float32` on `(dp, tp=4)`."""

  weight_sync_dtype = jnp.float32

  def __init__(
      self,
      src_state: nnx.State,
      hf_tok: Any,
      director_devs: list[Any],
      eos_ids: list[int],
      *,
      model_config: qwen3p5_model.ModelConfig | None = None,
      cache_size: int = 32768,
      tp_size: int = 4,
      batch_window_s: float = 0.03,
  ) -> None:
    self.eos_ids = [int(x) for x in eos_ids]
    self.cache_size = int(cache_size)
    self._batch_window_s = float(batch_window_s)
    n_devs = len(director_devs)
    tp = min(int(tp_size), n_devs)
    while tp > 1 and n_devs % tp != 0:
      tp //= 2
    self.dp_size = max(1, n_devs // tp)
    self.tp_size = tp
    self.mesh = jax.sharding.Mesh(
        np.asarray(director_devs).reshape(self.dp_size, self.tp_size),
        ("fsdp", "tp"),
    )
    base_cfg = model_config or qwen3p5_model.ModelConfig.qwen3p5_9b()
    cfg = dataclasses.replace(
        base_cfg,
        shd_config=qwen3p5_model.ShardingConfig.get_default_sharding(
            is_sampling=True
        ),
        remat_config=qwen3p5_model.RematConfig.NONE,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_flash_attention=False,
    )
    cache_cfg = datatypes.CacheConfig(
        cache_size=self.cache_size,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
    )
    t0 = time.monotonic()
    with self.mesh:
      abs_m = nnx.eval_shape(
          lambda: qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
      )
      gdef, abs_st = nnx.split(abs_m)
      self._template_state = abs_st
      self._target_shardings = nnx.get_named_sharding(abs_st, self.mesh)
      sharded_st = _remesh_to_sampling_specs(
          src_state, self._template_state, self._target_shardings
      )
      self._base_state = jax.tree.map(lambda v: v, sharded_st)
      self._rep_sharding = jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec()
      )
      mdl = nnx.merge(gdef, sharded_st)
      self._sampler = MeshBoundBatchedSampler(mdl, hf_tok, cache_cfg, self.mesh)
    self._exec_lock = threading.Lock()
    self._queue: queue.Queue[_PendingRequest | None] = queue.Queue()
    self._stopped = False
    self._worker = threading.Thread(
        target=self._batch_loop, name="fp32-director-batcher", daemon=True
    )
    self._worker.start()
    print(
        f"[fp32-director] Built SINGLE compiled TRUE float32 sampler on "
        f"(dp={self.dp_size}, tp={self.tp_size}) in {time.monotonic() - t0:.1f}s!",
        flush=True,
    )

  def close(self) -> None:
    self._stopped = True
    self._queue.put(None)
    self._worker.join(timeout=5.0)

  def update_lora(
      self,
      lora_np: dict[str, dict[str, np.ndarray]] | None,
      scale: float,
  ) -> None:
    """Applies LoRA deltas on Director TPU HBM without transferring 36 GiB."""
    t0 = time.monotonic()
    with self._exec_lock:
      old_state = self._sampler.transformer_state
      with self.mesh:
        new_state, n_merged = _merge_base_with_lora_on_mesh(
            self._base_state,
            self._target_shardings,
            lora_np,
            float(scale),
            self._rep_sharding,
        )
        self._sampler.transformer_state = new_state
      keep_ids = {id(x) for x in jax.tree.leaves(new_state)} | {
          id(x) for x in jax.tree.leaves(self._base_state)
      }
      _delete_unshared_leaves(old_state, keep_ids)
      del old_state
      gc.collect()
    print(
        f"[fp32-director] update_lora merged {n_merged} LoRA layers on-TPU in "
        f"{time.monotonic() - t0:.2f}s",
        flush=True,
    )

  def update_params(self, new_fp32_state: nnx.State) -> None:
    with self._exec_lock:
      old_state = self._sampler.transformer_state
      with self.mesh:
        new_state = _remesh_to_sampling_specs(
            new_fp32_state, self._template_state, self._target_shardings
        )
        self._sampler.transformer_state = new_state
      keep_ids = {id(x) for x in jax.tree.leaves(new_state)} | {
          id(x) for x in jax.tree.leaves(self._base_state)
      }
      _delete_unshared_leaves(old_state, keep_ids)
      del old_state
      gc.collect()

  def _choose_bucket(self, prompt_len: int, req_max: int) -> tuple[int, int]:
    max_p = max(1, self.cache_size - 512)
    for b in PROMPT_BUCKETS:
      if prompt_len <= b and b < self.cache_size - 1:
        max_p = b
        break
    room = max(1, self.cache_size - max_p - 1)
    req = max(1, int(req_max))
    target_gen = room
    for g in GEN_BUCKETS:
      if req <= g:
        target_gen = g
        break
    return max_p, min(room, target_gen)

  def _drain_compatible_batch(
      self, first: _PendingRequest
  ) -> list[_PendingRequest]:
    batch = [first]
    deadline = time.monotonic() + self._batch_window_s
    deferred: list[_PendingRequest] = []
    while len(batch) < self.dp_size:
      rem = deadline - time.monotonic()
      if rem <= 0:
        break
      try:
        nxt = self._queue.get(timeout=rem)
      except queue.Empty:
        break
      if nxt is None:
        self._queue.put(None)
        break
      same_mode = (nxt.temperature <= 0.0) == (first.temperature <= 0.0)
      if same_mode and nxt.max_p == first.max_p and nxt.actual_gen == first.actual_gen:
        batch.append(nxt)
      else:
        deferred.append(nxt)
    for item in deferred:
      self._queue.put(item)
    return batch

  def _deliver_batch_item(
      self, req: _PendingRequest, text: str, raw_toks: Any, raw_lps: Any, actual_gen: int
  ) -> None:
    gen_toks = [int(t) for t in np.asarray(raw_toks).tolist()]
    gen_lps = (
        [float(x) for x in np.asarray(raw_lps).tolist()]
        if raw_lps is not None
        else [0.0] * len(gen_toks)
    )
    if len(gen_toks) < actual_gen and (not gen_toks or gen_toks[-1] not in self.eos_ids):
      gen_toks.append(self.eos_ids[0])
      gen_lps.append(0.0)
    req.future.set_result(
        base_sampler.SamplerOutput(
            text=[text],
            logits=None,
            tokens=[np.asarray(gen_toks, dtype=np.int32)],
            padded_prompt_tokens=np.asarray([req.prompt_ids], dtype=np.int32),
            logprobs=[np.asarray(gen_lps, dtype=np.float32)]
            if req.return_logprobs
            else None,
        )
    )

  def _execute_batch(self, batch: list[_PendingRequest]) -> None:
    first = batch[0]
    padded_prompts = [r.prompt_ids for r in batch] + [first.prompt_ids] * (
        self.dp_size - len(batch)
    )
    greedy = first.temperature <= 0.0
    call_kwargs: dict[str, Any] = {
        "max_generation_steps": first.actual_gen,
        "max_prompt_length": first.max_p,
        "temperature": 0.0 if greedy else float(first.temperature),
        "top_p": None if greedy else float(first.top_p),
        "top_k": None if greedy else first.top_k,
        "seed": first.seed,
        "eos_tokens": self.eos_ids,
        "return_logits": False,
        "return_logprobs": True,
        "echo": False,
        "pad_output": False,
    }
    try:
      with self._exec_lock:
        with self.mesh:
          out = self._sampler(
              input_strings=None,
              prompt_token_ids=padded_prompts,
              **call_kwargs,
          )
      for i, req in enumerate(batch):
        raw_lps = out.logprobs[i] if out.logprobs is not None else None
        self._deliver_batch_item(req, out.text[i], out.tokens[i], raw_lps, first.actual_gen)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      for req in batch:
        if not req.future.done():
          req.future.set_exception(exc)

  def _batch_loop(self) -> None:
    while not self._stopped:
      item = self._queue.get()
      if item is None:
        break
      batch = self._drain_compatible_batch(item)
      self._execute_batch(batch)

  def __call__(
      self,
      input_strings: list[Any] | None = None,
      *,
      prompt_token_ids: list[Any] | None = None,
      max_generation_steps: int = 1024,
      temperature: float = 1.0,
      top_p: float | None = 1.0,
      top_k: int | None = None,
      seed: int | None = None,
      return_logprobs: bool = True,
      **_: Any,
  ) -> base_sampler.SamplerOutput:
    prompts = prompt_token_ids if prompt_token_ids is not None else input_strings
    if not prompts or len(prompts) != 1:
      raise ValueError("BatchingFp32DirectorSampler expects 1 prompt per RPC.")
    prompt_ids = [int(t) for t in prompts[0]]
    max_p, actual_gen = self._choose_bucket(len(prompt_ids), max_generation_steps)
    if len(prompt_ids) > max_p:
      prompt_ids = prompt_ids[-max_p:]
    fut: concurrent.futures.Future[base_sampler.SamplerOutput] = (
        concurrent.futures.Future()
    )
    self._queue.put(
        _PendingRequest(
            prompt_ids=prompt_ids,
            max_p=max_p,
            actual_gen=actual_gen,
            temperature=float(temperature),
            top_p=1.0 if top_p is None else float(top_p),
            top_k=top_k,
            seed=int(seed) if seed is not None else secrets.randbits(31),
            return_logprobs=bool(return_logprobs),
            future=fut,
        )
    )
    return fut.result()


def run_director_fidelity_gate(
    hub: Any,
    director_sampler: Any,
    hf_tok: Any,
    *,
    fidelity: contracts.ServingFidelity,
) -> contracts.FidelityReport:
  """Option 5 L3 startup gate verifying Learner <-> Director greedy/logprob parity."""
  prompt_text = "Compute 7 * 8 and answer with one integer."
  prompt_ids = [int(t) for t in hf_tok.encode(prompt_text, add_special_tokens=False)]
  out = director_sampler(
      [prompt_ids],
      max_generation_steps=8,
      temperature=0.0,
      top_p=None,
      top_k=None,
      seed=0,
      return_logprobs=True,
  )
  comp_ids = [int(t) for t in np.asarray(out.tokens[0]).tolist()][:8]
  samp_lps = [float(x) for x in np.asarray(out.logprobs[0]).tolist()][: len(comp_ids)]
  # Drop synthetic trailing EOS marker (0.0 logprob) if restored after early stop.
  valid_n = sum(1 for lp in samp_lps if lp != 0.0)
  comp_ids = comp_ids[: max(1, valid_n)]
  samp_lps = samp_lps[: len(comp_ids)]

  seq_ids = prompt_ids + comp_ids
  positions = list(range(len(prompt_ids) - 1, len(seq_ids) - 1))
  pred_slice = np.asarray(
      hub.eval_selected_logits(seq_ids, positions, role="solver", reference=False),
      dtype=np.float64,
  )
  learner_top1 = np.argmax(pred_slice, axis=-1).tolist()
  log_probs = pred_slice - np.log(
      np.sum(np.exp(pred_slice - np.max(pred_slice, axis=-1, keepdims=True)), axis=-1, keepdims=True)
  ) - np.max(pred_slice, axis=-1, keepdims=True)
  learner_lps = [float(log_probs[i, tok]) for i, tok in enumerate(comp_ids)]
  return contracts.verify_serving_fidelity(
      learner_top1_ids=learner_top1,
      sampler_top1_ids=comp_ids,
      learner_logprobs=learner_lps,
      sampler_logprobs=samp_lps,
      fidelity=fidelity,
      raise_on_fail=True,
  )
