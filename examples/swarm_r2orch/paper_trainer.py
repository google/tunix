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
"""Tunix JAX/TPU PolicyTrainer hub for R2Orch Qwen3.5-9B self-play."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from datetime import UTC
import json
import math
from pathlib import Path
import statistics
import threading
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.models.qwen3p5 import model as qwen3p5_model
from tunix.models.qwen3p5 import params as qwen3p5_params

from examples.swarm_r2orch import paper_kernels
from examples.swarm_r2orch import paper_lora

MAX_PROPOSER_LABELS = 8


@dataclasses.dataclass
class RoleLearnerState:
  """Live LoRA parameters and optimizer state for one R2Orch role."""

  role: str
  lora_params: paper_lora.LoraTree
  opt_state: optax.OptState
  tx: optax.GradientTransformation
  lr_schedule: optax.Schedule
  step_fn: Any
  lora_scale: float
  optimizer_step_count: int = 0


class _NullContext:

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    del exc
    return False


def _build_b1_sharding_config() -> qwen3p5_model.ShardingConfig:
  base = qwen3p5_model.ShardingConfig.get_default_sharding()
  return dataclasses.replace(
      base,
      act_btd=(None, None, None),
      act_btf=(None, None, "tp"),
      act_btnh=(None, None, "tp", None),
  )


def _even_mini_batches(total: int, requested: int) -> list[list[int]]:
  if total <= 0:
    return []
  k = max(1, min(int(requested), total))
  base, extra = divmod(total, k)
  out: list[list[int]] = []
  cursor = 0
  for idx in range(k):
    width = base + (1 if idx < extra else 0)
    out.append(list(range(cursor, cursor + width)))
    cursor += width
  return out


def _opt_state_to_dict(opt_state: optax.OptState) -> dict[str, np.ndarray]:
  leaves = jax.tree_util.tree_leaves(opt_state)
  return {
      f"leaf_{idx:04d}": np.asarray(jax.device_get(leaf))
      for idx, leaf in enumerate(leaves)
  }


def _restore_opt_state(
    template: optax.OptState,
    arrays: dict[str, np.ndarray],
    mesh: jax.sharding.Mesh | None,
) -> optax.OptState:
  leaves, treedef = jax.tree_util.tree_flatten(template)
  if len(arrays) != len(leaves):
    return template
  replicated = (
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      if mesh is not None
      else None
  )
  restored = []
  for idx, tmpl in enumerate(leaves):
    arr = arrays[f"leaf_{idx:04d}"].astype(tmpl.dtype)
    use_mesh = replicated is not None and arr.ndim > 0
    val = jax.device_put(arr, replicated) if use_mesh else jnp.asarray(arr, dtype=tmpl.dtype)
    restored.append(val)
  return jax.tree_util.tree_unflatten(treedef, restored)


def _merge_ref_lookup(ref_maps: list[dict[int, Any]] | None) -> dict[int, Any] | None:
  if ref_maps is None:
    return None
  merged: dict[int, Any] = {}
  for m in ref_maps:
    merged.update(m)
  return merged


def _merge_old_lookup(
    specs: list[dict[str, Any]],
    old_maps: list[dict[int, Any]] | None,
) -> dict[int, float] | None:
  if old_maps is None:
    return None
  merged: dict[int, float] = {}
  for sp, m in zip(specs, old_maps):
    if sp["kind"] == "relation":
      p = sp["positions"][0]
      val = m[p]
      merged[p] = float(val[sp["chosen"]] if isinstance(val, np.ndarray) else val)
    elif sp["kind"] == "ordinary":
      for p, v in m.items():
        merged[p] = float(v)
  return merged


class SharedTunixLearnerHub:
  """Persistent JAX Qwen3.5-9B model + Proposer/Director LoRA learner states."""

  def __init__(
      self,
      model_path: str | Path | None,
      *,
      mesh: jax.sharding.Mesh | None = None,
      model_config: qwen3p5_model.ModelConfig | None = None,
      preloaded_model: qwen3p5_model.Qwen3p5 | None = None,
  ) -> None:
    self.mesh = mesh
    self._lock = threading.RLock()
    model = preloaded_model or self._load_base_model(model_path, model_config)
    self.config = model.config
    self.graphdef, self.base_state = nnx.split(model)
    self._eval_fn = paper_kernels.make_eval_logits_fn(self.graphdef)
    self._proposer_grad_fn = paper_kernels.make_proposer_grad_fn(
        self.graphdef, self.mesh
    )
    self._director_grad_fn = paper_kernels.make_director_grad_fn(
        self.graphdef, self.mesh
    )
    self.roles: dict[str, RoleLearnerState] = {}
    self.served_snapshots: dict[str, tuple[paper_lora.LoraTree, float]] = {}

  def _load_base_model(
      self,
      model_path: str | Path | None,
      model_config: qwen3p5_model.ModelConfig | None,
  ) -> qwen3p5_model.Qwen3p5:
    cfg = model_config or qwen3p5_model.ModelConfig.qwen3p5_9b()
    cfg = dataclasses.replace(
        cfg,
        shd_config=_build_b1_sharding_config(),
        remat_config=qwen3p5_model.RematConfig.DECODER,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_flash_attention=False,
    )
    ctx = self.mesh if self.mesh is not None else _NullContext()
    with ctx:
      return qwen3p5_params.create_model_from_safe_tensors(
          file_dir=str(model_path),
          config=cfg,
          mesh=self.mesh,
          dtype=jnp.float32,
      )

  def ensure_role(self, role: str, policy_cfg: Any, seed: int) -> RoleLearnerState:
    with self._lock:
      if role in self.roles:
        return self.roles[role]
      lr_sched = paper_kernels.build_lr_schedule(
          float(policy_cfg.learning_rate),
          warmup_steps=int(policy_cfg.warmup_steps),
          total_steps=int(policy_cfg.total_optimizer_steps),
          schedule_kind=str(policy_cfg.learning_rate_schedule),
      )
      prop_gn = float(policy_cfg.proposer_max_grad_norm or 0.0)
      max_gn = prop_gn if (role == "proposer" and prop_gn > 0) else float(policy_cfg.max_grad_norm)
      tx = optax.chain(
          optax.clip_by_global_norm(max_gn),
          optax.adamw(learning_rate=lr_sched, weight_decay=float(policy_cfg.weight_decay)),
      )
      lora_params = paper_lora.init_lora_params(
          self.base_state,
          rank=int(policy_cfg.lora_rank),
          seed=seed,
          mesh=self.mesh,
      )
      opt_state = tx.init(lora_params)
      if self.mesh is not None:
        rep = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )
        opt_state = jax.device_put(opt_state, rep)
      lora_params, opt_state, opt_steps = self._maybe_restore_role(
          Path(policy_cfg.checkpoint_root), lora_params, opt_state
      )
      state = RoleLearnerState(
          role=role,
          lora_params=lora_params,
          opt_state=opt_state,
          tx=tx,
          lr_schedule=lr_sched,
          step_fn=paper_kernels.make_optimizer_step_fn(tx, self.mesh),
          lora_scale=float(policy_cfg.lora_alpha) / float(policy_cfg.lora_rank),
          optimizer_step_count=opt_steps,
      )
      self.roles[role] = state
      return state

  def _maybe_restore_role(
      self,
      ckpt_root: Path,
      lora_params: paper_lora.LoraTree,
      opt_state: optax.OptState,
  ) -> tuple[paper_lora.LoraTree, optax.OptState, int]:
    latest_json = ckpt_root / "latest.json"
    if not latest_json.is_file():
      return lora_params, opt_state, 0
    latest = json.loads(latest_json.read_text(encoding="utf-8"))
    step_dir = Path(latest["path"])
    if not (step_dir / "lora_weights.npz").is_file():
      step_dir = ckpt_root / f"step-{int(latest.get('step', 0)):08d}"
    if not (step_dir / "lora_weights.npz").is_file():
      return lora_params, opt_state, 0
    loaded_p, opt_arrs, meta = paper_lora.load_lora_weights(step_dir, mesh=self.mesh)
    restored_opt = _restore_opt_state(opt_state, opt_arrs, self.mesh)
    return loaded_p, restored_opt, int(meta.get("optimizer_step_count", 0))

  def snapshot_adapter(
      self, served_name: str, role: str, *, lora_path: str | None = None
  ) -> None:
    with self._lock:
      if role in self.roles:
        st = self.roles[role]
        copied = jax.tree.map(lambda x: x.copy(), st.lora_params)
        self.served_snapshots[served_name] = (copied, st.lora_scale)
        return
      if lora_path and (Path(lora_path) / "lora_weights.npz").is_file():
        loaded_p, _, meta = paper_lora.load_lora_weights(
            Path(lora_path), mesh=self.mesh
        )
        scale = float(meta.get("lora_scale", 1.0))
        self.served_snapshots[served_name] = (loaded_p, scale)

  def lora_numpy_for_serving(
      self,
      role: str = "solver",
      *,
      served_name: str | None = None,
  ) -> tuple[dict[str, dict[str, np.ndarray]] | None, float]:
    """Exports only the compact FP32 LoRA adapter matrices (~465MB) to NumPy."""
    with self._lock:
      lora_p, scale = self._resolve_lora_for_eval(role, served_name, reference=False)
      if not lora_p:
        return None, float(scale)
      out: dict[str, dict[str, np.ndarray]] = {}
      for key_str, entry in lora_p.items():
        out[key_str] = {
            "lora_a": np.asarray(jax.device_get(entry["lora_a"]), dtype=np.float32),
            "lora_b": np.asarray(jax.device_get(entry["lora_b"]), dtype=np.float32),
        }
      return out, float(scale)

  def merged_state_for_vllm(
      self,
      role: str = "solver",
      *,
      served_name: str | None = None,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> nnx.State:
    with self._lock:
      lora_p, scale = self._resolve_lora_for_eval(role, served_name, reference=False)
      ctx = self.mesh if self.mesh is not None else _NullContext()
      with ctx:
        return paper_lora.merge_lora_state(
            self.base_state, lora_p or {}, scale=scale, target_dtype=dtype
        )

  def _resolve_lora_for_eval(
      self,
      role: str | None,
      served_name: str | None,
      *,
      reference: bool,
  ) -> tuple[paper_lora.LoraTree | None, float]:
    if reference:
      if role and role in self.roles:
        return self.roles[role].lora_params, 0.0
      return None, 1.0
    if served_name and served_name in self.served_snapshots:
      return self.served_snapshots[served_name]
    if role and role in self.roles:
      return self.roles[role].lora_params, self.roles[role].lora_scale
    return None, 1.0

  def eval_selected_logits(
      self,
      token_ids: tuple[int, ...] | list[int],
      positions: list[int],
      *,
      role: str | None = None,
      served_name: str | None = None,
      reference: bool = False,
  ) -> np.ndarray:
    seq_b = paper_kernels.select_bucket(len(token_ids), paper_kernels.SEQ_BUCKETS)
    pos_b = paper_kernels.select_bucket(len(positions), paper_kernels.POS_BUCKETS)
    toks, pos_ids, seg_ids = paper_kernels.pad_sequence_inputs(token_ids, seq_b)
    sel = np.zeros((pos_b,), dtype=np.int32)
    sel[: len(positions)] = np.asarray(positions, dtype=np.int32)
    with self._lock:
      lora_p, scale = self._resolve_lora_for_eval(role, served_name, reference=reference)
      ctx = self.mesh if self.mesh is not None else _NullContext()
      with ctx:
        logits = self._eval_fn(
            self.base_state,
            lora_p,
            jnp.float32(scale),
            jnp.asarray(toks),
            jnp.asarray(pos_ids),
            jnp.asarray(seg_ids),
            jnp.asarray(sel),
        )
      return np.asarray(jax.device_get(logits[: len(positions)]), dtype=np.float32)

  def predict_last_logits(
      self,
      prompt_token_ids: tuple[int, ...] | list[int],
      *,
      role: str | None = None,
      served_name: str | None = None,
  ) -> np.ndarray:
    return self.eval_selected_logits(
        prompt_token_ids,
        [len(prompt_token_ids) - 1],
        role=role,
        served_name=served_name,
    )[0]

  def run_director_grad(
      self,
      st: RoleLearnerState,
      token_ids: tuple[int, ...] | list[int],
      specs: list[dict[str, Any]],
      ref_maps: list[dict[int, Any]] | None,
      old_maps: list[dict[int, Any]] | None,
      hypers: dict[str, jax.Array],
  ) -> tuple[dict[str, jax.Array], paper_lora.LoraTree]:
    seq_tuple, packed = paper_kernels.pack_director_unit_inputs(
        token_ids,
        specs,
        _merge_ref_lookup(ref_maps),
        _merge_old_lookup(specs, old_maps),
    )
    with self._lock:
      ctx = self.mesh if self.mesh is not None else _NullContext()
      with ctx:
        return self._director_grad_fn(
            st.lora_params, self.base_state, seq_tuple, packed, hypers
        )

  def run_director_calls_grad(
      self,
      st: RoleLearnerState,
      calls: Any,
      prep: dict[str, Any],
      hypers: dict[str, jax.Array],
  ) -> tuple[dict[str, Any], paper_lora.LoraTree]:
    acc_g = paper_kernels.tree_zeros_like(st.lora_params)
    totals = {"pol": 0.0, "kl": 0.0, "clip": 0, "kl_cap": 0, "lr_sum": 0.0, "lr_max": 0.0}
    fused_maps: list[dict[int, Any]] = []
    need_old = prep["old_data"] is None
    for idx, (call, sp) in enumerate(zip(calls, prep["specs"])):
      if sp["kind"] == "empty":
        fused_maps.append({})
        continue
      ref_sub = [prep["ref_data"][idx]] if prep["ref_data"] is not None else None
      old_sub = [prep["old_data"][idx]] if not need_old else None
      aux, g_call = self.run_director_grad(st, call.token_ids, [sp], ref_sub, old_sub, hypers)
      acc_g = paper_kernels.tree_add(acc_g, g_call)
      part = paper_kernels.unpack_aux_scalars(aux)
      totals["pol"] += part["pol"]
      totals["kl"] += part["kl"]
      totals["clip"] += part["clip"]
      totals["kl_cap"] += part["kl_cap"]
      totals["lr_sum"] += part["lr_sum"]
      totals["lr_max"] = max(totals["lr_max"], part["lr_max"])
      if need_old:
        fused_maps.append(paper_kernels.record_fused_call_old(prep, sp, aux))
    if need_old:
      prep["old_data"] = fused_maps
    return totals, acc_g

  def apply_opt_step(
      self, st: RoleLearnerState, grads: paper_lora.LoraTree
  ) -> tuple[float, float]:
    with self._lock:
      lr_before = float(st.lr_schedule(st.optimizer_step_count))
      ctx = self.mesh if self.mesh is not None else _NullContext()
      with ctx:
        new_p, new_opt, gnorm = st.step_fn(st.lora_params, st.opt_state, grads)
      st.lora_params = new_p
      st.opt_state = new_opt
      st.optimizer_step_count += 1
      return float(gnorm), lr_before

  def save_role_ckpt(self, st: RoleLearnerState, config: Any, step: int) -> Path:
    with self._lock:
      opt_arrs = _opt_state_to_dict(st.opt_state)
      meta = {
          "role": st.role,
          "step": int(step),
          "optimizer_step_count": int(st.optimizer_step_count),
          "lora_scale": float(st.lora_scale),
      }
      return paper_lora.save_lora_checkpoint(
          Path(config.checkpoint_root),
          step=step,
          lora_params=st.lora_params,
          opt_arrays=opt_arrs,
          metadata=meta,
          publish_latest=bool(getattr(config, "publish_latest_checkpoint", True)),
      )

  def factory(
      self, role: str, policy_cfg: Any, seed: int = 0
  ) -> TunixRoleTrainerView:
    self.ensure_role(role, policy_cfg, seed)
    return TunixRoleTrainerView(self, role, policy_cfg, seed)


class TunixRoleTrainerView:
  """Lightweight per-cycle `PolicyTrainer` view backed by `SharedTunixLearnerHub`."""

  def __init__(
      self,
      hub: SharedTunixLearnerHub,
      role: str,
      config: Any,
      seed: int,
  ) -> None:
    self.hub = hub
    self.role = role
    self.config = config
    self.seed = seed

  def close(self) -> None:
    return None

  def update(self, batch: Any, *, step: int) -> Any:
    if batch.role != self.role:
      raise ValueError(f"Expected {self.role} batch, got {batch.role}")
    if batch.objective == "proposer_pair":
      return self.update_proposer_pair(batch, step=step)
    if batch.objective == "r2orch_director":
      return self.update_director(batch, step=step)
    raise ValueError(f"Unsupported objective: {batch.objective}")

  def _label_logprobs(self, sample: Any, *, reference: bool) -> list[float]:
    support = [int(v) for v in sample.metadata["r2orch"]["label_token_ids"]]
    logits = self.hub.eval_selected_logits(
        sample.token_ids,
        [len(sample.token_ids) - 1],
        role=self.role,
        reference=reference,
    )[0]
    sub = logits[support].astype(np.float64)
    sub -= np.max(sub)
    lse = math.log(float(np.sum(np.exp(sub))))
    return [float(v - lse) for v in sub]

  def update_proposer_pair(self, batch: Any, *, step: int) -> Any:
    from r2orch import proposer_pair_training as ppt  # pylint: disable=g-import-not-at-top
    from r2orch.training import PolicyUpdateResult  # pylint: disable=g-import-not-at-top

    t0 = time.perf_counter()
    samples = list(batch.samples)
    st = self.hub.roles[self.role]
    req_steps = int(self.config.proposer_optimizer_steps or self.config.ppo_mini_batches)
    max_gn = float(self.config.proposer_max_grad_norm or self.config.max_grad_norm)
    chunks = ppt.step_index_chunks(samples, req_steps)
    half_targets = [ppt.pair_terms(s)[2] / 2.0 for s in samples]
    rms = math.sqrt(statistics.fmean(v * v for v in half_targets)) if half_targets else 0.0
    refs = [self._label_logprobs(s, reference=True) for s in samples]
    currs = [self._label_logprobs(s, reference=False) for s in samples]
    diag = ppt.policy_diagnostics(ppt._diagnostic_records(samples, currs, refs))
    losses, gaps, norms, step_metrics = [], [], [], []
    for chunk in chunks:
      w0 = time.perf_counter()
      acc_g = paper_kernels.tree_zeros_like(st.lora_params)
      w_losses: list[float] = []
      for idx in chunk:
        lv, gv, gt = self._proposer_sample_grad(st, samples[idx], refs[idx], rms, len(chunk))
        acc_g = paper_kernels.tree_add(acc_g, gt)
        w_losses.append(lv)
        gaps.append(gv)
      norm_val, lr_now = self.hub.apply_opt_step(st, acc_g)
      losses.extend(w_losses)
      norms.append(norm_val)
      step_metrics.append(
          self._proposer_window_metric(
              step, chunk, samples, w_losses, norm_val, max_gn, lr_now, time.perf_counter() - w0
          )
      )
    extra = {
        **ppt.pair_signal_stats(samples, chunks),
        **diag,
        "r2orch_max_grad_norm": max_gn,
        "r2orch_grad_norm_pre_clip_mean": statistics.fmean(norms),
        "r2orch_grad_norm_pre_clip_max": max(norms),
        "r2orch_clip_fraction": sum(1 for v in norms if v > max_gn) / len(norms),
    }
    step_metrics[-1].update(ppt._metrics(batch, losses, gaps, extra))
    ckpt = self.hub.save_role_ckpt(st, self.config, step)
    rewards = [float(s.reward) for s in samples]
    return PolicyUpdateResult(
        self.role,
        step,
        statistics.fmean(losses),
        statistics.fmean(losses),
        0.0,
        len(samples),
        str(ckpt),
        samples=len(samples),
        optimizer_steps=len(step_metrics),
        learning_rate=float(st.lr_schedule(st.optimizer_step_count)),
        grad_norm=statistics.fmean(norms),
        reward_mean=statistics.fmean(rewards),
        reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        duration_seconds=time.perf_counter() - t0,
        step_metrics=tuple(step_metrics),
        loss_variant=ppt.LOSS_VARIANT,
        reference_provenance="lora_disabled_base",
    )

  def _proposer_sample_grad(
      self,
      st: RoleLearnerState,
      sample: Any,
      ref_row: list[float],
      rms: float,
      chunk_len: int,
  ) -> tuple[float, float, paper_lora.LoraTree]:
    from r2orch import proposer_pair_training as ppt  # pylint: disable=g-import-not-at-top

    a_idx, b_idx, target = ppt.pair_terms(sample)
    adv = (target / 2.0) / rms if rms > 1e-12 else 0.0
    raw_support = [int(v) for v in sample.metadata["r2orch"]["label_token_ids"]]
    sup_ids = np.zeros((MAX_PROPOSER_LABELS,), dtype=np.int32)
    sup_mask = np.zeros((MAX_PROPOSER_LABELS,), dtype=bool)
    ref_arr = np.zeros((MAX_PROPOSER_LABELS,), dtype=np.float32)
    k = len(raw_support)
    sup_ids[:k] = raw_support
    sup_mask[:k] = True
    ref_arr[:k] = np.asarray(ref_row, dtype=np.float32)
    seq_b = paper_kernels.select_bucket(len(sample.token_ids), paper_kernels.SEQ_BUCKETS)
    toks, pos_ids, seg_ids = paper_kernels.pad_sequence_inputs(sample.token_ids, seq_b)
    last_pos = np.asarray([len(sample.token_ids) - 1], dtype=np.int32)
    pair_inputs = {
        "lora_scale": jnp.float32(st.lora_scale),
        "support_ids": jnp.asarray(sup_ids),
        "support_mask": jnp.asarray(sup_mask),
        "ref_lp": jnp.asarray(ref_arr),
        "idx_a": jnp.int32(a_idx),
        "idx_b": jnp.int32(b_idx),
        "advantage": jnp.float32(adv),
        "kl_beta": jnp.float32(self.config.proposer_kl_beta),
        "gap_clip": jnp.float32(self.config.proposer_gap_clip),
        "inv_chunk_size": jnp.float32(1.0 / max(1, chunk_len)),
    }
    seq_tuple = (
        jnp.asarray(toks),
        jnp.asarray(pos_ids),
        jnp.asarray(seg_ids),
        jnp.asarray(last_pos),
    )
    with self.hub._lock:
      ctx = self.hub.mesh if self.hub.mesh is not None else _NullContext()
      with ctx:
        (loss_j, gap_j), grads = self.hub._proposer_grad_fn(
            st.lora_params, self.hub.base_state, seq_tuple, pair_inputs
        )
    return float(loss_j), float(gap_j), grads

  def _proposer_window_metric(
      self,
      step: int,
      chunk: list[int],
      samples: list[Any],
      w_losses: list[float],
      norm_val: float,
      max_gn: float,
      lr_now: float,
      dur: float,
  ) -> dict[str, Any]:
    from r2orch import proposer_pair_training as ppt  # pylint: disable=g-import-not-at-top

    rewards = [float(samples[i].reward) for i in chunk]
    targets = [ppt.pair_terms(samples[i])[2] for i in chunk]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "role": self.role,
        "role_optimizer_step": self.hub.roles[self.role].optimizer_step_count,
        "alternating_update_step": step,
        "epoch": 0,
        "microbatch_count": len(chunk),
        "loss": statistics.fmean(w_losses),
        "policy_loss": statistics.fmean(w_losses),
        "kl": 0.0,
        "entropy": 0.0,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "masked_tokens": len(chunk),
        "clip_fraction": 0.0,
        "grad_norm": norm_val,
        "learning_rate": lr_now,
        "duration_seconds": dur,
        "r2orch_step_pairs": len(chunk),
        "r2orch_step_nonzero": sum(1 for v in targets if abs(v) > 1e-12),
        "r2orch_step_eff_participation": ppt.effective_participation(targets),
        "r2orch_step_grad_norm_pre_clip": norm_val,
        "r2orch_step_clipped": bool(norm_val > max_gn),
    }

  def _prepare_director_sample(
      self,
      sample: Any,
      *,
      kl_rank: int,
      defer_old: bool,
  ) -> dict[str, Any]:
    from r2orch.policy_timeline import timeline_positions  # pylint: disable=g-import-not-at-top

    calls = sample.policy_calls
    r2_meta = sample.metadata.get("r2orch", {})
    condition = str(r2_meta.get("scaffold_condition", "on"))
    kl_coeff = float(r2_meta.get("kl_coefficient", self.config.kl_coefficient))
    stride = int(self.config.kl_reference_stride)
    use_ref = (
        (kl_coeff > 0.0)
        if stride == 1
        else (stride > 1 and kl_rank % stride == 0 and kl_coeff > 0.0)
    )
    tl_pos = (
        timeline_positions(calls, int(self.config.max_sequence_length))
        if self.config.raw_policy_backward_mode == "timeline"
        else None
    )
    specs = [
        paper_kernels.extract_call_targets(
            c, tl_pos[idx] if tl_pos is not None else None
        )
        for idx, c in enumerate(calls)
    ]
    ref_data = self._eval_sample_maps(calls, specs, tl_pos, reference=True) if use_ref else None
    old_data = None if defer_old else self._eval_sample_maps(calls, specs, tl_pos, reference=False)
    return {
        "condition": condition,
        "kl_coeff": kl_coeff,
        "kl_scale": float(stride) if (stride > 1 and use_ref) else 1.0,
        "kl_reference": use_ref,
        "pg_tokens": sum(len(sp["positions"]) for sp in specs),
        "timeline": tl_pos is not None,
        "specs": specs,
        "ref_data": ref_data,
        "old_data": old_data,
        "tis_weights": self._initial_tis(specs, old_data) if old_data is not None else [],
    }

  def _eval_sample_maps(
      self,
      calls: Any,
      specs: list[dict[str, Any]],
      tl_pos: list[list[int]] | None,
      *,
      reference: bool,
  ) -> list[dict[int, Any]]:
    if tl_pos is not None:
      flat = sorted({p for row in tl_pos for p in row})
      logits = self.hub.eval_selected_logits(
          calls[-1].token_ids, flat, role=self.role, reference=reference
      )
      by_pos = {p: logits[idx] for idx, p in enumerate(flat)}
      return [paper_kernels.spec_logprob_map(sp, by_pos) for sp in specs]
    maps: list[dict[int, Any]] = []
    for call, sp in zip(calls, specs):
      if sp["kind"] == "empty":
        maps.append({})
      else:
        logits = self.hub.eval_selected_logits(
            call.token_ids, sp["positions"], role=self.role, reference=reference
        )
        by_pos = {p: logits[idx] for idx, p in enumerate(sp["positions"])}
        maps.append(paper_kernels.spec_logprob_map(sp, by_pos))
    return maps

  def _initial_tis(
      self, specs: list[dict[str, Any]], old_maps: list[dict[int, Any]]
  ) -> list[float]:
    cap = float(self.config.tis_cap)
    weights: list[float] = []
    for sp, omap in zip(specs, old_maps):
      if sp["kind"] == "relation":
        p = sp["positions"][0]
        diff = max(-20.0, min(20.0, float(omap[p][sp["chosen"]]) - sp["behav"][0]))
        weights.append(min(cap, math.exp(diff)))
      elif sp["kind"] == "ordinary":
        for p, blp in zip(sp["positions"], sp["behav"]):
          diff = max(-20.0, min(20.0, float(omap[p]) - blp))
          weights.append(min(cap, math.exp(diff)))
    return weights

  def _director_sample_backward(
      self,
      st: RoleLearnerState,
      sample: Any,
      prep: dict[str, Any],
      *,
      token_count: int,
      stats: dict[str, Any],
  ) -> tuple[dict[str, Any], paper_lora.LoraTree]:
    kl_weight = float(prep["kl_coeff"] * prep["kl_scale"])
    hypers = {
        "lora_scale": jnp.float32(st.lora_scale),
        "advantage": jnp.float32(sample.advantage),
        "kl_weight": jnp.float32(kl_weight),
        "inv_token_count": jnp.float32(1.0 / max(1, token_count)),
        "clip_low": jnp.float32(self.config.clip_low),
        "clip_high": jnp.float32(self.config.clip_high),
        "tis_cap": jnp.float32(self.config.tis_cap),
        "kl_term_cap": jnp.float32(self.config.kl_term_cap),
        "use_fused_old": jnp.bool_(prep["old_data"] is None),
    }
    if prep["timeline"]:
      aux, grads = self.hub.run_director_grad(
          st,
          sample.policy_calls[-1].token_ids,
          prep["specs"],
          prep["ref_data"],
          prep["old_data"],
          hypers,
      )
      if prep["old_data"] is None:
        paper_kernels.record_fused_timeline_old(prep, aux)
      metrics = paper_kernels.unpack_aux_scalars(aux)
    else:
      metrics, grads = self.hub.run_director_calls_grad(
          st, sample.policy_calls, prep, hypers
      )
    terms = paper_kernels.accumulate_sample_stats(
        prep, kl_weight, metrics, stats, int(self.config.kl_reference_stride)
    )
    return terms, grads

  def _run_director_window(
      self,
      st: RoleLearnerState,
      samples: list[Any],
      prepared: list[dict[str, Any]],
      indices: list[int],
      stats: dict[str, Any],
      *,
      step: int,
      ep: int,
      fused: bool,
  ) -> tuple[dict[str, Any], float, float, float, int]:
    w0 = time.perf_counter()
    tok_count = max(1, sum(prepared[i]["pg_tokens"] for i in indices))
    acc_g = paper_kernels.tree_zeros_like(st.lora_params)
    w_loss, w_pol, w_kl, w_rew = [], [], [], []
    w_clip = w_mask = 0
    for idx in indices:
      terms, g_s = self._director_sample_backward(
          st, samples[idx], prepared[idx], token_count=tok_count, stats=stats
      )
      acc_g = paper_kernels.tree_add(acc_g, g_s)
      w_loss.append(terms["loss"])
      w_pol.append(terms["policy"])
      if terms["kl_valid"]:
        w_kl.append(terms["kl"])
      w_rew.append(float(samples[idx].reward))
      w_clip += terms["clipped"]
      w_mask += prepared[idx]["pg_tokens"]
    norm_val, lr_now = self.hub.apply_opt_step(st, acc_g)
    metric = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "role": self.role,
        "role_optimizer_step": st.optimizer_step_count,
        "alternating_update_step": step,
        "epoch": ep,
        "microbatch_count": len(indices),
        "loss": statistics.fmean(w_loss),
        "policy_loss": statistics.fmean(w_pol),
        "kl": statistics.fmean(w_kl or [0.0]),
        "entropy": 0.0,
        "reward_mean": statistics.fmean(w_rew),
        "reward_std": statistics.pstdev(w_rew) if len(w_rew) > 1 else 0.0,
        "masked_tokens": w_mask,
        "clip_fraction": w_clip / max(1, w_mask),
        "grad_norm": norm_val,
        "learning_rate": lr_now,
        "duration_seconds": time.perf_counter() - w0,
        "fused_old_logprobs": fused,
    }
    return metric, sum(w_loss), sum(w_pol), sum(w_kl), len(w_kl)

  def update_director(self, batch: Any, *, step: int) -> Any:
    from r2orch.training import PolicyUpdateResult  # pylint: disable=g-import-not-at-top

    t0 = time.perf_counter()
    samples = list(batch.samples)
    if not samples:
      raise ValueError("Cannot update solver from empty batch")
    st = self.hub.roles[self.role]
    mini_batches = _even_mini_batches(len(samples), int(self.config.ppo_mini_batches))
    fuse = bool(self.config.fuse_old_logprobs)
    deferred = set(mini_batches[0]) if fuse and mini_batches else set()
    seen_cond: dict[str, int] = {}
    prepared: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
      cond = str((sample.metadata.get("r2orch") or {}).get("scaffold_condition", "on"))
      kl_rank = seen_cond.get(cond, 0) + int(step)
      seen_cond[cond] = seen_cond.get(cond, 0) + 1
      prepared.append(
          self._prepare_director_sample(sample, kl_rank=kl_rank, defer_old=idx in deferred)
      )
    stats = paper_kernels.empty_director_stats()
    total_loss = total_policy = total_kl = 0.0
    kl_samples = 0
    step_metrics: list[dict[str, Any]] = []
    for ep in range(int(self.config.epochs)):
      for win, indices in enumerate(mini_batches):
        m_row, dl, dp_val, dkl, n_kl = self._run_director_window(
            st,
            samples,
            prepared,
            indices,
            stats,
            step=step,
            ep=ep,
            fused=bool(fuse and win == 0 and ep == 0),
        )
        step_metrics.append(m_row)
        total_loss += dl
        total_policy += dp_val
        total_kl += dkl
        kl_samples += n_kl
    step_metrics[-1].update(
        paper_kernels.director_summary_metrics(
            samples, prepared, stats, total_loss, self.config
        )
    )
    ckpt = self.hub.save_role_ckpt(st, self.config, step)
    divisor = max(1, len(samples) * int(self.config.epochs))
    rewards = [float(s.reward) for s in samples]
    grad_norms = [float(m["grad_norm"]) for m in step_metrics]
    return PolicyUpdateResult(
        self.role,
        step,
        total_loss / divisor,
        total_policy / divisor,
        total_kl / max(1, kl_samples),
        sum(p["pg_tokens"] for p in prepared),
        str(ckpt),
        0.0,
        samples=len(samples),
        optimizer_steps=len(step_metrics),
        learning_rate=float(st.lr_schedule(st.optimizer_step_count)),
        grad_norm=statistics.fmean(grad_norms) if grad_norms else 0.0,
        clip_fraction=stats["clipped"] / max(1, stats["pg_tokens"]),
        reward_mean=statistics.fmean(rewards),
        reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        duration_seconds=time.perf_counter() - t0,
        step_metrics=tuple(step_metrics),
        loss_variant="r2orch_director",
        reference_provenance="lora_disabled_base",
    )
