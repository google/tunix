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
"""Compiled JAX forward and value_and_grad kernels for R2Orch Qwen3.5-9B."""

from __future__ import annotations

import functools
import math
import statistics
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.swarm_r2orch import paper_lora

SEQ_BUCKETS: tuple[int, ...] = (
    512,
    8192,
    16384,
    32768,
    36864,
)
POS_BUCKETS: tuple[int, ...] = (16, 4096, 8192)


def select_bucket(length: int, buckets: tuple[int, ...]) -> int:
  for b in buckets:
    if b >= length:
      return b
  return ((length + 1023) // 1024) * 1024


def pad_sequence_inputs(
    token_ids: tuple[int, ...] | list[int],
    seq_bucket: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  n = len(token_ids)
  tokens = np.zeros((1, seq_bucket), dtype=np.int32)
  positions = np.zeros((1, seq_bucket), dtype=np.int32)
  seg_ids = np.zeros((1, seq_bucket), dtype=np.int32)
  tokens[0, :n] = np.asarray(token_ids, dtype=np.int32)
  positions[0, :n] = np.arange(n, dtype=np.int32)
  seg_ids[0, :n] = 1
  return tokens, positions, seg_ids


def forward_selected_logits(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    input_tokens: jax.Array,
    pos_ids: jax.Array,
    seg_ids: jax.Array,
    sel_positions: jax.Array,
) -> jax.Array:
  model = nnx.merge(graphdef, state)
  hidden, _ = model(
      input_tokens,
      pos_ids,
      cache=None,
      attention_mask=None,
      segment_ids=seg_ids,
      skip_lm_head=True,
  )
  chosen_hidden = jnp.take(hidden[0], sel_positions, axis=0)
  return model.compute_final_logits(chosen_hidden[None, :, :])[0]


def make_eval_logits_fn(graphdef: nnx.GraphDef):
  @jax.jit
  def _eval_fn(
      base_state: nnx.State,
      lora_params: paper_lora.LoraTree | None,
      lora_scale: jax.Array,
      input_tokens: jax.Array,
      pos_ids: jax.Array,
      seg_ids: jax.Array,
      sel_positions: jax.Array,
  ) -> jax.Array:
    state = (
        base_state
        if lora_params is None
        else paper_lora.merge_lora_state(
            base_state, lora_params, scale=lora_scale
        )
    )
    return forward_selected_logits(
        graphdef, state, input_tokens, pos_ids, seg_ids, sel_positions
    )

  return _eval_fn


def _proposer_sample_loss(
    lora_params: paper_lora.LoraTree,
    *,
    graphdef: nnx.GraphDef,
    base_state: nnx.State,
    lora_scale: jax.Array,
    input_tokens: jax.Array,
    pos_ids: jax.Array,
    seg_ids: jax.Array,
    last_pos: jax.Array,
    support_ids: jax.Array,
    support_mask: jax.Array,
    ref_lp: jax.Array,
    idx_a: jax.Array,
    idx_b: jax.Array,
    advantage: jax.Array,
    kl_beta: jax.Array,
    gap_clip: jax.Array,
    inv_chunk_size: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  merged = paper_lora.merge_lora_state(base_state, lora_params, scale=lora_scale)
  logits_last = forward_selected_logits(
      graphdef, merged, input_tokens, pos_ids, seg_ids, last_pos
  )[0]
  raw_support = jnp.take(logits_last, support_ids, axis=0)
  masked_logits = jnp.where(support_mask, raw_support, -1e9)
  current = jax.nn.log_softmax(masked_logits, axis=-1)
  probs = jnp.where(support_mask, jnp.exp(current), 0.0)
  kl = jnp.sum(probs * jnp.where(support_mask, current - ref_lp, 0.0))
  gap = (current[idx_a] - current[idx_b]) - (ref_lp[idx_a] - ref_lp[idx_b])
  g_det = jax.lax.stop_gradient(gap)
  is_clipped = (
      (gap_clip > 0.0) & ((advantage * g_det) > 0.0) & (jnp.abs(g_det) >= gap_clip)
  )
  pg = jnp.where(is_clipped, 0.0, -advantage * (current[idx_a] - current[idx_b]))
  raw_loss = pg + kl_beta * kl
  return raw_loss * inv_chunk_size, (raw_loss, gap)


def _replicated_sharding(
    mesh: jax.sharding.Mesh | None,
) -> jax.sharding.NamedSharding | None:
  if mesh is None:
    return None
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def make_proposer_grad_fn(
    graphdef: nnx.GraphDef, mesh: jax.sharding.Mesh | None = None
):
  rep = _replicated_sharding(mesh)

  def _grad_fn(
      lora_params: paper_lora.LoraTree,
      base_state: nnx.State,
      seq_inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      pair_inputs: dict[str, jax.Array],
  ) -> tuple[tuple[jax.Array, jax.Array], paper_lora.LoraTree]:
    input_tokens, pos_ids, seg_ids, last_pos = seq_inputs
    fn = functools.partial(
        _proposer_sample_loss,
        graphdef=graphdef,
        base_state=base_state,
        lora_scale=pair_inputs["lora_scale"],
        input_tokens=input_tokens,
        pos_ids=pos_ids,
        seg_ids=seg_ids,
        last_pos=last_pos,
        support_ids=pair_inputs["support_ids"],
        support_mask=pair_inputs["support_mask"],
        ref_lp=pair_inputs["ref_lp"],
        idx_a=pair_inputs["idx_a"],
        idx_b=pair_inputs["idx_b"],
        advantage=pair_inputs["advantage"],
        kl_beta=pair_inputs["kl_beta"],
        gap_clip=pair_inputs["gap_clip"],
        inv_chunk_size=pair_inputs["inv_chunk_size"],
    )
    (_, aux), grads = jax.value_and_grad(fn, has_aux=True)(lora_params)
    return aux, grads

  return jax.jit(_grad_fn, out_shardings=rep) if rep is not None else jax.jit(_grad_fn)


def _director_targets_loss(
    lora_params: paper_lora.LoraTree,
    *,
    graphdef: nnx.GraphDef,
    base_state: nnx.State,
    seq_inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    targets: dict[str, jax.Array],
    hypers: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
  input_tokens, pos_ids, seg_ids, sel_positions = seq_inputs
  merged = paper_lora.merge_lora_state(
      base_state, lora_params, scale=hypers["lora_scale"]
  )
  logits_mv = forward_selected_logits(
      graphdef, merged, input_tokens, pos_ids, seg_ids, sel_positions
  )
  all_lp = jax.nn.log_softmax(logits_mv, axis=-1)
  ord_lp = jnp.take_along_axis(
      all_lp, targets["target_ids"][:, None], axis=-1
  )[:, 0]
  off_logit = jnp.take_along_axis(
      logits_mv, targets["off_ids"][:, None], axis=-1
  )[:, 0]
  on_logit = jnp.take_along_axis(
      logits_mv, targets["on_ids"][:, None], axis=-1
  )[:, 0]
  pair_lp = jax.nn.log_softmax(
      jnp.stack([off_logit, on_logit], axis=-1), axis=-1
  )
  rel_lp = jnp.take_along_axis(
      pair_lp, targets["chosen_idx"][:, None], axis=-1
  )[:, 0]
  is_rel = targets["is_rel"]
  valid = targets["valid"]
  curr_lp = jnp.where(is_rel, rel_lp, ord_lp)
  eff_old = jnp.where(
      hypers["use_fused_old"], jax.lax.stop_gradient(curr_lp), targets["old_lp"]
  )
  tis_w = jnp.minimum(
      jnp.exp(jnp.clip(eff_old - targets["behav_lp"], -20.0, 20.0)),
      hypers["tis_cap"],
  )
  log_ratio = jnp.clip(curr_lp - eff_old, -20.0, 20.0)
  ratio = jnp.exp(log_ratio)
  adv = hypers["advantage"]
  low = 1.0 - hypers["clip_low"]
  high = 1.0 + hypers["clip_high"]
  surr = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, low, high))

  ord_kl_log = jnp.clip(targets["ref_lp"] - curr_lp, -20.0, 20.0)
  ord_kl_raw = jnp.exp(ord_kl_log) - ord_kl_log - 1.0
  cap = hypers["kl_term_cap"]
  ord_kl = jnp.where(cap > 0.0, jnp.minimum(ord_kl_raw, cap), ord_kl_raw)
  rel_kl = jnp.sum(
      jnp.exp(pair_lp) * (pair_lp - targets["ref_pair_lp"]), axis=-1
  )
  kl_term = jnp.where(
      targets["has_ref"], jnp.where(is_rel, rel_kl, ord_kl), 0.0
  )

  pol_sum = jnp.sum(valid * tis_w * surr)
  kl_sum = jnp.sum(valid * kl_term)
  contrib = (pol_sum + hypers["kl_weight"] * kl_sum) * hypers["inv_token_count"]
  is_clipped = ((ratio < low) | (ratio > high)) & (valid > 0.5)
  kl_capped = (
      (cap > 0.0)
      & (~is_rel)
      & targets["has_ref"]
      & (ord_kl >= cap)
      & (valid > 0.5)
  )
  abs_lr = jnp.where(valid > 0.5, jnp.abs(jax.lax.stop_gradient(log_ratio)), 0.0)
  aux = {
      "pol_sum": jax.lax.stop_gradient(pol_sum),
      "kl_sum": jax.lax.stop_gradient(kl_sum),
      "clipped": jnp.sum(is_clipped.astype(jnp.int32)),
      "kl_capped": jnp.sum(kl_capped.astype(jnp.int32)),
      "abs_lr_sum": jnp.sum(abs_lr),
      "abs_lr_max": jnp.max(abs_lr),
      "tis_weights": jax.lax.stop_gradient(tis_w),
      "eff_old": jax.lax.stop_gradient(eff_old),
  }
  return contrib, aux


def make_director_grad_fn(
    graphdef: nnx.GraphDef, mesh: jax.sharding.Mesh | None = None
):
  rep = _replicated_sharding(mesh)

  def _grad_fn(
      lora_params: paper_lora.LoraTree,
      base_state: nnx.State,
      seq_inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      targets: dict[str, jax.Array],
      hypers: dict[str, jax.Array],
  ) -> tuple[dict[str, jax.Array], paper_lora.LoraTree]:
    fn = functools.partial(
        _director_targets_loss,
        graphdef=graphdef,
        base_state=base_state,
        seq_inputs=seq_inputs,
        targets=targets,
        hypers=hypers,
    )
    (_, aux), grads = jax.value_and_grad(fn, has_aux=True)(lora_params)
    return aux, grads

  return jax.jit(_grad_fn, out_shardings=rep) if rep is not None else jax.jit(_grad_fn)


def make_optimizer_step_fn(
    tx: optax.GradientTransformation, mesh: jax.sharding.Mesh | None = None
):
  rep = _replicated_sharding(mesh)

  def _step_fn(
      lora_params: paper_lora.LoraTree,
      opt_state: optax.OptState,
      grads: paper_lora.LoraTree,
  ) -> tuple[paper_lora.LoraTree, optax.OptState, jax.Array]:
    grad_norm = optax.global_norm(grads)
    updates, new_opt_state = tx.update(grads, opt_state, lora_params)
    new_params = optax.apply_updates(lora_params, updates)
    return new_params, new_opt_state, grad_norm

  return jax.jit(_step_fn, out_shardings=rep) if rep is not None else jax.jit(_step_fn)


def tree_zeros_like(params: paper_lora.LoraTree) -> paper_lora.LoraTree:
  return jax.tree.map(jnp.zeros_like, params)


def tree_add(
    lhs: paper_lora.LoraTree, rhs: paper_lora.LoraTree
) -> paper_lora.LoraTree:
  return jax.tree.map(jnp.add, lhs, rhs)


def _flatten_target_rows(
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for spec in specs:
    if spec["kind"] == "relation":
      rows.append({
          "pos": spec["positions"][0],
          "is_rel": True,
          "tid": 0,
          "off_id": spec["off_id"],
          "on_id": spec["on_id"],
          "chosen": spec["chosen"],
          "behav": spec["behav"][0],
      })
    elif spec["kind"] == "ordinary":
      for p, tid, blp in zip(
          spec["positions"], spec["targets"], spec["behav"]
      ):
        rows.append({
            "pos": p,
            "is_rel": False,
            "tid": tid,
            "off_id": 0,
            "on_id": 0,
            "chosen": 0,
            "behav": blp,
        })
  return rows


def pack_director_unit_inputs(
    token_ids: tuple[int, ...] | list[int],
    specs: list[dict[str, Any]],
    ref_lookup: dict[int, Any] | None,
    old_lookup: dict[int, float] | None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], dict[str, jax.Array]]:
  rows = _flatten_target_rows(specs)
  seq_b = select_bucket(len(token_ids), SEQ_BUCKETS)
  pos_b = select_bucket(max(1, len(rows)), POS_BUCKETS)
  toks, pos_ids, seg_ids = pad_sequence_inputs(token_ids, seq_b)
  sel_pos = np.zeros((pos_b,), dtype=np.int32)
  valid = np.zeros((pos_b,), dtype=np.float32)
  is_rel = np.zeros((pos_b,), dtype=bool)
  target_ids = np.zeros((pos_b,), dtype=np.int32)
  off_ids = np.zeros((pos_b,), dtype=np.int32)
  on_ids = np.zeros((pos_b,), dtype=np.int32)
  chosen_idx = np.zeros((pos_b,), dtype=np.int32)
  behav_lp = np.zeros((pos_b,), dtype=np.float32)
  old_lp = np.zeros((pos_b,), dtype=np.float32)
  ref_lp = np.zeros((pos_b,), dtype=np.float32)
  ref_pair_lp = np.zeros((pos_bucket := pos_b, 2), dtype=np.float32)
  del pos_bucket
  has_ref = np.zeros((pos_b,), dtype=bool)
  for idx, r in enumerate(rows):
    p = r["pos"]
    sel_pos[idx] = p
    valid[idx] = 1.0
    is_rel[idx] = r["is_rel"]
    target_ids[idx] = r["tid"]
    off_ids[idx] = r["off_id"]
    on_ids[idx] = r["on_id"]
    chosen_idx[idx] = r["chosen"]
    behav_lp[idx] = r["behav"]
    if old_lookup is not None and p in old_lookup:
      old_lp[idx] = old_lookup[p]
    if ref_lookup is not None and p in ref_lookup:
      has_ref[idx] = True
      if r["is_rel"]:
        ref_pair_lp[idx] = ref_lookup[p]
      else:
        ref_lp[idx] = ref_lookup[p]
  seq_tuple = (
      jnp.asarray(toks),
      jnp.asarray(pos_ids),
      jnp.asarray(seg_ids),
      jnp.asarray(sel_pos),
  )
  packed = {
      "valid": jnp.asarray(valid),
      "is_rel": jnp.asarray(is_rel),
      "target_ids": jnp.asarray(target_ids),
      "off_ids": jnp.asarray(off_ids),
      "on_ids": jnp.asarray(on_ids),
      "chosen_idx": jnp.asarray(chosen_idx),
      "behav_lp": jnp.asarray(behav_lp),
      "old_lp": jnp.asarray(old_lp),
      "ref_lp": jnp.asarray(ref_lp),
      "ref_pair_lp": jnp.asarray(ref_pair_lp),
      "has_ref": jnp.asarray(has_ref),
  }
  return seq_tuple, packed


def build_lr_schedule(
    peak_lr: float,
    *,
    warmup_steps: int,
    total_steps: int,
    schedule_kind: str = "cosine",
) -> optax.Schedule:
  total = max(1, int(total_steps))
  warmup = max(0, min(int(warmup_steps), total - 1))
  if schedule_kind == "constant":
    if warmup <= 0:
      return optax.constant_schedule(peak_lr)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak_lr, warmup),
            optax.constant_schedule(peak_lr),
        ],
        [warmup],
    )
  decay_steps = max(1, total - warmup)
  if warmup <= 0:
    return optax.cosine_decay_schedule(init_value=peak_lr, decay_steps=decay_steps)
  return optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=peak_lr,
      warmup_steps=warmup,
      decay_steps=total,
      end_value=0.0,
  )


def extract_call_targets(
    call: Any,
    offset_check: list[int] | None = None,
) -> dict[str, Any]:
  rel_meta = call.metadata.get("relation_policy") or {}
  is_rel = call.relation_token_span is not None and bool(rel_meta)
  behav = [
      float(v)
      for v, keep in zip(call.behavior_log_probs, call.action_mask[1:])
      if keep
  ]
  if not behav:
    return {"kind": "empty", "positions": []}
  if is_rel:
    start, _ = call.relation_token_span
    off_id = int(rel_meta["token_ids"]["off"])
    on_id = int(rel_meta["token_ids"]["on"])
    chosen = 0 if int(call.token_ids[start]) == off_id else 1
    return {
        "kind": "relation",
        "positions": [int(start - 1)],
        "off_id": off_id,
        "on_id": on_id,
        "chosen": chosen,
        "behav": [behav[0]],
    }
  positions = (
      list(offset_check)
      if offset_check is not None
      else [i for i, keep in enumerate(call.action_mask[1:]) if keep]
  )
  if not positions:
    return {"kind": "empty", "positions": []}
  return {
      "kind": "ordinary",
      "positions": positions,
      "targets": [int(call.token_ids[p + 1]) for p in positions],
      "behav": behav,
  }


def spec_logprob_map(
    sp: dict[str, Any], by_pos: dict[int, np.ndarray]
) -> dict[int, Any]:
  if sp["kind"] == "empty":
    return {}
  if sp["kind"] == "relation":
    p = sp["positions"][0]
    row = by_pos[p]
    pair = np.asarray([row[sp["off_id"]], row[sp["on_id"]]], dtype=np.float64)
    pair -= np.max(pair)
    lse = math.log(float(np.sum(np.exp(pair))))
    return {p: np.asarray([pair[0] - lse, pair[1] - lse], dtype=np.float32)}
  out: dict[int, Any] = {}
  for p, tid in zip(sp["positions"], sp["targets"]):
    row = by_pos[p].astype(np.float64)
    mx = float(np.max(row))
    lse = mx + math.log(float(np.sum(np.exp(row - mx))))
    out[p] = float(row[tid] - lse)
  return out


def unpack_aux_scalars(aux: dict[str, jax.Array]) -> dict[str, Any]:
  return {
      "pol": float(aux["pol_sum"]),
      "kl": float(aux["kl_sum"]),
      "clip": int(aux["clipped"]),
      "kl_cap": int(aux["kl_capped"]),
      "lr_sum": float(aux["abs_lr_sum"]),
      "lr_max": float(aux["abs_lr_max"]),
  }


def record_fused_timeline_old(
    prep: dict[str, Any], aux: dict[str, jax.Array]
) -> None:
  total = prep["pg_tokens"]
  eff = np.asarray(jax.device_get(aux["eff_old"]))[:total]
  tis = np.asarray(jax.device_get(aux["tis_weights"]))[:total]
  prep["tis_weights"].extend(float(x) for x in tis)
  cursor = 0
  old_maps: list[dict[int, Any]] = []
  for sp in prep["specs"]:
    m = {p: float(eff[cursor + i]) for i, p in enumerate(sp["positions"])}
    cursor += len(sp["positions"])
    old_maps.append(m)
  prep["old_data"] = old_maps


def record_fused_call_old(
    prep: dict[str, Any], sp: dict[str, Any], aux: dict[str, jax.Array]
) -> dict[int, Any]:
  n_valid = len(sp["positions"])
  eff = np.asarray(jax.device_get(aux["eff_old"]))[:n_valid]
  tis = np.asarray(jax.device_get(aux["tis_weights"]))[:n_valid]
  prep["tis_weights"].extend(float(x) for x in tis)
  return {p: float(v) for p, v in zip(sp["positions"], eff)}


def empty_director_stats() -> dict[str, Any]:
  return {
      "kl_on": 0.0,
      "kl_on_tokens": 0,
      "kl_off": 0.0,
      "kl_off_tokens": 0,
      "policy_sum": 0.0,
      "clipped": 0,
      "pg_tokens": 0,
      "logratio_abs_sum": 0.0,
      "logratio_tokens": 0,
      "logratio_abs_max": 0.0,
      "kl_capped": 0,
  }


def accumulate_sample_stats(
    prep: dict[str, Any],
    kl_weight: float,
    m: dict[str, Any],
    stats: dict[str, Any],
    stride: int,
) -> dict[str, Any]:
  tokens = max(1, prep["pg_tokens"])
  cond_key = "kl_on" if prep["condition"] == "on" else "kl_off"
  kl_valid = bool(prep["kl_reference"]) or int(stride) == 1
  stats[cond_key] += m["kl"]
  stats[cond_key + "_tokens"] += prep["pg_tokens"] if kl_valid else 0
  stats["policy_sum"] += m["pol"]
  stats["clipped"] += m["clip"]
  stats["pg_tokens"] += prep["pg_tokens"]
  stats["kl_capped"] += m["kl_cap"]
  stats["logratio_abs_sum"] += m["lr_sum"]
  stats["logratio_tokens"] += prep["pg_tokens"]
  stats["logratio_abs_max"] = max(stats["logratio_abs_max"], m["lr_max"])
  return {
      "loss": (m["pol"] + kl_weight * m["kl"]) / tokens,
      "policy": m["pol"] / tokens,
      "kl": m["kl"] / tokens,
      "kl_valid": kl_valid,
      "clipped": m["clip"],
  }


def director_summary_metrics(
    samples: list[Any],
    prepared: list[dict[str, Any]],
    stats: dict[str, Any],
    total_loss: float,
    config: Any,
) -> dict[str, Any]:
  weights = [w for p in prepared for w in p["tis_weights"]]
  conds = [p["condition"] for p in prepared]
  cap = float(config.tis_cap)
  return {
      "r2orch_outcome_adv_abs_mean": statistics.fmean(
          abs(float(s.advantage)) for s in samples
      ),
      "r2orch_policy_loss_token_mean": (
          stats["policy_sum"] / max(1, stats["pg_tokens"])
      ),
      "r2orch_kl_on": stats["kl_on"] / max(1, stats["kl_on_tokens"]),
      "r2orch_kl_off": stats["kl_off"] / max(1, stats["kl_off_tokens"]),
      "r2orch_kl_capped_tokens": stats["kl_capped"],
      "r2orch_tis_weight_mean": statistics.fmean(weights) if weights else 0.0,
      "r2orch_tis_capped_fraction": (
          sum(1 for w in weights if w >= cap) / len(weights) if weights else 0.0
      ),
      "r2orch_samples_on": conds.count("on"),
      "r2orch_samples_off": conds.count("off"),
      "r2orch_loss": total_loss / max(1, len(samples) * int(config.epochs)),
      "r2orch_logratio_abs_mean": (
          stats["logratio_abs_sum"] / max(1, stats["logratio_tokens"])
      ),
      "r2orch_logratio_abs_max": stats["logratio_abs_max"],
  }
