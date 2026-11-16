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

"""R2Orch ADS+TSDS curriculum, Recursive Learnability reward, and JAX Proposer."""

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import heapq
import json
import math
import random

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from examples.swarm_r2orch import scoring
from examples.swarm_r2orch import skill_ledger
from examples.swarm_r2orch import workflow
from tunix.dist_rl.train import peft_trainer_v2
from tunix.sft import utils as sft_utils

MIN_EVIDENCE_ROLLOUTS = 2
HISTORY_KEEP = 16
THIN_CANDIDATES = 4

PROPOSER_SYSTEM_PROMPT = (
    "You choose the next training task for a graph-orchestration Director."
    " Each lettered candidate is a fixed-pool task prepared by ADS boundary"
    " scheduling or TSDS retrieval near the anchor. Prefer a task whose"
    " outcome is still uncertain for the Director: neither reliably solved nor"
    " hopeless. Each candidate shows scope_cards, scope_card_tokens,"
    " previous_success, and selection_count. Answer with one candidate letter."
)


@dataclasses.dataclass
class ADSClusterState:
  """Mutable difficulty-band and learnability state of one ADS cluster."""

  cluster_id: str
  cluster_size: int
  prob: float
  mini_positions: set[int]
  success_rate: float = 0.5
  learnability: float = 1.0
  observations: int = 0


@dataclasses.dataclass(frozen=True)
class CandidateList:
  """One Proposer candidate list (`A..H`) anchored in a single domain."""

  list_id: str
  cycle: int
  dataset: str
  scope: str
  anchor_id: str
  candidates: tuple[dict[str, object], ...]
  prompt_text: str


@dataclasses.dataclass(frozen=True)
class PairSelection:
  """Two Proposer picks (`a, b`) sampled from one `CandidateList`."""

  cand_list: CandidateList
  choices: tuple[dict[str, object], ...]
  same_choice: bool
  behavior_logprobs: tuple[float, ...]


def task_embedding(task: scoring.R2OrchTask, *, dim: int = 128) -> np.ndarray:
  """Extracts precomputed embedding or deterministic unit hash vector."""
  raw = task.curriculum.get("embedding")
  if isinstance(raw, Sequence) and raw:
    return np.asarray([float(x) for x in raw], dtype=np.float32)
  seed = int.from_bytes(
      hashlib.sha256(task.prompt.encode("utf-8")).digest()[:8], "big"
  )
  vec = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
  norm = float(np.linalg.norm(vec))
  return vec / max(norm, 1e-6)


def spread_positions(size: int, count: int) -> set[int]:
  """Spreads `count` initial boundary indices evenly across `range(size)`."""
  cnt = min(int(count), int(size))
  if cnt >= size:
    return set(range(size))
  if cnt <= 1:
    return {size // 2}
  return {round(i * (size - 1) / (cnt - 1)) for i in range(cnt)}


def nearest_free_position(
    desired: int, direction: int, occupied: set[int], size: int, fallback: int
) -> int:
  """Finds the closest unoccupied index in `[0, size)` biased by `direction`."""
  clamped = min(max(desired, 0), size - 1)
  if clamped not in occupied:
    return clamped
  for radius in range(1, size):
    for cand in (clamped + direction * radius, clamped - direction * radius):
      if 0 <= cand < size and cand not in occupied:
        return cand
  return fallback


def compute_top_kdes(
    candidates: np.ndarray,
    top_indices: np.ndarray,
    *,
    kde_k: int,
    sigma: float,
) -> np.ndarray:
  """Computes truncated parabolic KDE densities on unique top-k neighbors."""
  if sigma == 0.0:
    return np.ones(top_indices.shape, dtype=np.float64)
  unique_idx = np.unique(top_indices)
  feats = candidates[unique_idx]
  sq = ((feats[:, None, :] - feats[None, :, :]) ** 2).sum(axis=-1)
  nearest = np.sort(sq, axis=-1)[:, :kde_k]
  kde = np.maximum(np.maximum(1.0 - nearest / (sigma**2), 0.0).sum(axis=-1), 1e-12)
  kde_map = {int(idx): float(kde[pos]) for pos, idx in enumerate(unique_idx)}
  return np.asarray(
      [[kde_map[int(idx)] for idx in row] for row in top_indices],
      dtype=np.float64,
  )


def tsds_probabilities(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    max_k: int = 32,
    kde_k: int = 16,
    sigma: float = 0.1,
    alpha: float = 0.5,
    c: float = 1.0,
) -> np.ndarray:
  """TSDS KNN-KDE optimal transport probability assignment over `candidates`."""
  eff_k = max(1, min(int(max_k), len(candidates)))
  eff_kde = max(1, min(int(kde_k), len(candidates)))
  sq = ((query[:, None, :] - candidates[None, :, :]) ** 2).sum(axis=-1)
  top_idx = np.argsort(sq, axis=-1)[:, :eff_k]
  top_dists = np.sqrt(np.take_along_axis(sq, top_idx, axis=1))
  q_cnt, c_cnt = top_idx.shape[0], candidates.shape[0]
  if eff_k == 1:
    probs = np.zeros(c_cnt, dtype=np.float64)
    for row in top_idx:
      probs[int(row[0])] += 1.0 / q_cnt
    return probs / probs.sum()
  top_kdes = compute_top_kdes(candidates, top_idx, kde_k=eff_kde, sigma=sigma)
  last_k = [0] * q_cnt
  heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(q_cnt)]
  heapq.heapify(heap)
  w_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(q_cnt)]
  mass, total_cost = 1.0, 0.0
  costs = np.zeros(q_cnt, dtype=np.float64)
  while heap:
    cnt, cur_k, cur_q = heapq.heappop(heap)
    mass = cnt
    total_cost -= costs[cur_q]
    costs[cur_q] = top_dists[cur_q][cur_k + 1] * cnt - w_sum[cur_q]
    total_cost += costs[cur_q]
    if (alpha / c) * total_cost >= (1.0 - alpha) * q_cnt:
      break
    last_k[cur_q] = cur_k
    if cur_k < eff_k - 2:
      nxt = cur_k + 1
      heapq.heappush(heap, (cnt + 1.0 / top_kdes[cur_q][nxt], nxt, cur_q))
      w_sum[cur_q] += top_dists[cur_q][nxt] / top_kdes[cur_q][nxt]
  probs = np.zeros(c_cnt, dtype=np.float64)
  for q_i in range(q_cnt):
    assigned = 0.0
    for nbr in range(last_k[q_i] + 1):
      val = 1.0 / q_cnt / mass / top_kdes[q_i][nbr]
      probs[top_idx[q_i][nbr]] += val
      assigned += val
    probs[top_idx[q_i][last_k[q_i] + 1]] += max(1.0 / q_cnt - assigned, 0.0)
  total = probs.sum()
  return probs / total if total > 0 else np.full(c_cnt, 1.0 / c_cnt)


class TSDSRetriever:
  """Embedding similarity neighbor ranker backed by `tsds_probabilities`."""

  def __init__(self, embeddings: Mapping[str, np.ndarray]):
    self.embeddings = embeddings

  def rank(
      self, candidate_ids: Sequence[str], anchor_vec: np.ndarray, *, limit: int
  ) -> list[str]:
    ids = list(dict.fromkeys(candidate_ids))
    if not ids:
      return []
    mat = np.stack([self.embeddings[pid] for pid in ids], axis=0)
    probs = tsds_probabilities(anchor_vec[None, :], mat)
    order = np.argsort(-probs, kind="stable")
    return [ids[int(i)] for i in order[:limit] if probs[int(i)] > 0]


class ADSBoundaryScheduler:
  """Learnability-weighted ADS difficulty boundary scheduler over fixed pool."""

  def __init__(
      self,
      tasks: Sequence[scoring.R2OrchTask],
      *,
      seed: int = 42,
      active_clusters: int = 2,
      mini_cluster_size: int = 8,
      boundary_eps: float = 0.17,
      alpha: float = 0.30,
      cooldown: int = 64,
      cluster_floor: float = 0.20,
      cluster_ema: float = 0.30,
      saturated_min_obs: int = 2,
      hopeless_min_obs: int = 3,
      history_window: int = 48,
  ):
    self.tasks = {t.task_id: t for t in tasks}
    self.ids = tuple(t.task_id for t in tasks)
    self.active_clusters = active_clusters
    self.mini_cluster_size = mini_cluster_size
    self.band_low = 0.5 - boundary_eps
    self.band_high = 0.5 + boundary_eps
    self.alpha = alpha
    self.cooldown = cooldown
    self.cluster_floor = cluster_floor
    self.cluster_ema = cluster_ema
    self.saturated_min_obs = saturated_min_obs
    self.hopeless_min_obs = hopeless_min_obs
    self.history_window = history_window
    self.rng = np.random.default_rng(seed)
    self.update_index = 0
    self.recent: list[str] = []
    self.reserved: set[str] = set()
    self.selection_count: collections.Counter[str] = collections.Counter()
    self.task_success: dict[str, float] = {}
    self.task_history: dict[str, list[list[float]]] = {}
    grouped: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    self.cluster_dataset: dict[str, str] = {}
    for idx, t in enumerate(tasks):
      cid = str(t.curriculum.get("cluster_id") or f"{t.dataset}:ads:0")
      diff = float(t.curriculum.get("difficulty_score", idx))  # pytype: disable=wrong-arg-types
      grouped[cid].append((diff, t.task_id))
      self.cluster_dataset[cid] = t.dataset
    self.dataset_clusters = {
        dom: tuple(sorted(c for c, d in self.cluster_dataset.items() if d == dom))
        for dom in sorted(set(self.cluster_dataset.values()))
    }
    self.cluster_sorted_ids = {
        cid: tuple(tid for _, tid in sorted(items))
        for cid, items in grouped.items()
    }
    self.cluster_states = {
        cid: ADSClusterState(
            cluster_id=cid,
            cluster_size=len(tids),
            prob=1.0 / len(self.dataset_clusters[self.cluster_dataset[cid]]),
            mini_positions=spread_positions(len(tids), mini_cluster_size),
        )
        for cid, tids in self.cluster_sorted_ids.items()
    }

  def order_key(self, pool_id: str) -> tuple[int, int, str]:
    hist = self.task_history.get(pool_id)
    if not hist:
      tier = 1
    else:
      _, rho, cnt = hist[-1]
      tier = 0 if cnt >= MIN_EVIDENCE_ROLLOUTS and 0.0 < rho < 1.0 else 2
    return (tier, self.selection_count[pool_id], pool_id)

  def recent_ids(self, dataset: str) -> set[str]:
    if self.cooldown <= 0:
      return set()
    seen: list[str] = []
    for pid in reversed(self.recent):
      if self.tasks[pid].dataset == dataset:
        seen.append(pid)
        if len(seen) >= self.cooldown:
          break
    return set(seen)

  def excluded_ids(self, dataset: str) -> set[str]:
    horizon = self.update_index - self.history_window
    out: set[str] = set()
    for pid, hist in self.task_history.items():
      if self.tasks[pid].dataset != dataset:
        continue
      recent_h = [r for idx, r, c in hist if idx >= horizon and c >= MIN_EVIDENCE_ROLLOUTS]
      sat = self.saturated_min_obs > 0 and sum(r >= 1.0 for r in recent_h) >= self.saturated_min_obs
      hope = self.hopeless_min_obs > 0 and sum(r <= 0.0 for r in recent_h) >= self.hopeless_min_obs
      if sat or hope:
        out.add(pid)
    return out

  def draw_single_anchor(
      self,
      rng: random.Random,
      clusters: Sequence[str],
      recent: set[str],
      skipped: set[str],
      used: set[str],
  ) -> str | None:
    """Draws one coverage-weighted anchor task from `clusters`."""
    rem = list(range(len(clusters)))
    while rem:
      weights = [max(self.cluster_states[clusters[i]].prob, 0.0) for i in rem]
      tot = sum(weights)
      chosen_pos = rng.choices(rem, weights=weights if tot > 0 else None, k=1)[0]
      rem.remove(chosen_pos)
      members = self.cluster_sorted_ids[clusters[chosen_pos]]
      for relax in (0, 1, 2):
        elig = [
            p
            for p in members
            if p not in used
            and (relax >= 1 or p not in skipped)
            and (relax >= 2 or (p not in recent and p not in self.reserved))
        ]
        if elig:
          best_cnt = min(self.selection_count[p] for p in elig)
          return rng.choice([p for p in elig if self.selection_count[p] == best_cnt])
    return None

  def anchor_ids(self, *, per_dataset: int, cycle: int, seed: int) -> list[str]:
    """Draws `per_dataset` anchors per dataset, interleaved across domains."""
    drawn: dict[str, list[str]] = {}
    for dom, clusters in self.dataset_clusters.items():
      rng = random.Random(f"anchors:{seed}:{cycle}:{dom}")
      recent = self.recent_ids(dom)
      skipped = self.excluded_ids(dom)
      used: set[str] = set()
      picks: list[str] = []
      for _ in range(per_dataset):
        pick = self.draw_single_anchor(rng, clusters, recent, skipped, used)
        if pick is not None:
          picks.append(pick)
          used.add(pick)
      drawn[dom] = picks
    return [
        drawn[dom][slot]
        for slot in range(per_dataset)
        for dom in sorted(drawn)
        if slot < len(drawn[dom])
    ]

  def revisit_learnable_ids(
      self, dataset: str, blocked: set[str]
  ) -> list[str]:
    """Returns previously mixed-outcome tasks in `dataset` outside `blocked`."""
    found = [
        pid
        for pid, h in self.task_history.items()
        if self.tasks[pid].dataset == dataset
        and pid not in blocked
        and bool(h)
        and h[-1][2] >= MIN_EVIDENCE_ROLLOUTS
        and 0.0 < h[-1][1] < 1.0
    ]
    return sorted(found, key=self.order_key)

  def candidates(self, dataset: str) -> list[str]:
    """Returns learnability-ordered ADS boundary candidates in `dataset`."""
    clusters = self.dataset_clusters[dataset]
    raw_p = np.asarray([self.cluster_states[c].prob for c in clusters], dtype=np.float64)
    probs = raw_p / raw_p.sum() if raw_p.sum() > 0 else None
    chosen = self.rng.choice(
        len(clusters), size=min(self.active_clusters, len(clusters)), replace=False, p=probs
    )
    blocked = self.recent_ids(dataset) | self.reserved
    skipped = self.excluded_ids(dataset)
    buckets: list[list[str]] = []
    held_back: list[str] = []
    for c_idx in chosen.tolist():
      cid = clusters[int(c_idx)]
      tids = self.cluster_sorted_ids[cid]
      elig = [tids[p] for p in sorted(self.cluster_states[cid].mini_positions) if tids[p] not in blocked]
      held_back.extend(set(elig) & skipped)
      buckets.append(sorted(set(elig) - skipped, key=self.order_key))
    in_b = {p for b in buckets for p in b}
    buckets.insert(0, self.revisit_learnable_ids(dataset, blocked | skipped | in_b))
    max_len = max((len(b) for b in buckets), default=0)
    selected = [b[i] for i in range(max_len) for b in buckets if i < len(b)]
    if len(selected) < THIN_CANDIDATES and held_back:
      selected.extend(sorted(held_back, key=self.order_key))
    return selected or [p for p in self.ids if self.tasks[p].dataset == dataset]

  def shift_mini_positions(self, cid: str, directions: Mapping[int, int]) -> None:
    """Shifts observed boundary positions easier (`-1`) or harder (`+1`)."""
    st = self.cluster_states[cid]
    if st.cluster_size <= len(st.mini_positions):
      return
    kept = {pos for pos in st.mini_positions if directions.get(pos, 0) == 0}
    easier = sorted(pos for pos in st.mini_positions if directions.get(pos, 0) < 0)
    harder = sorted(
        (pos for pos in st.mini_positions if directions.get(pos, 0) > 0),
        reverse=True,
    )
    for pos in easier:
      kept.add(nearest_free_position(pos - 1, -1, kept, st.cluster_size, pos))
    for pos in harder:
      kept.add(nearest_free_position(pos + 1, 1, kept, st.cluster_size, pos))
    st.mini_positions = kept

  def refresh_dataset_probs(self, dataset: str) -> None:
    """Updates inter-cluster selection probabilities with uniform floor."""
    cids = self.dataset_clusters[dataset]
    scores = np.asarray([self.cluster_states[c].learnability for c in cids], dtype=np.float64)
    shares = scores / scores.sum() if scores.sum() > 0 else np.full(len(scores), 1.0 / len(scores))
    targets = self.cluster_floor / len(cids) + (1.0 - self.cluster_floor) * shares
    for cid, tgt in zip(cids, targets.tolist(), strict=True):
      st = self.cluster_states[cid]
      st.prob = (1.0 - self.alpha) * st.prob + self.alpha * float(tgt)
    tot = sum(self.cluster_states[c].prob for c in cids)
    for cid in cids:
      self.cluster_states[cid].prob /= tot

  def observe_single_task(
      self, pid: str, rewards: Sequence[float]
  ) -> tuple[str, float, int] | None:
    """Records one task's rollout outcomes and returns `(cid, rho, move_dir)`."""
    if pid not in self.tasks or not rewards:
      return None
    rho = float(np.mean(np.clip(np.asarray(rewards, dtype=np.float64), 0.0, 1.0)))
    self.task_success[pid] = rho
    self.selection_count[pid] += 1
    hist = self.task_history.setdefault(pid, [])
    hist.append([float(self.update_index), rho, float(len(rewards))])
    del hist[:-HISTORY_KEEP]
    self.reserved.discard(pid)
    self.recent.append(pid)
    dom = self.tasks[pid].dataset
    cid = next(c for c in self.dataset_clusters[dom] if pid in self.cluster_sorted_ids[c])
    direction = -1 if rho < self.band_low else (1 if rho > self.band_high else 0)
    return cid, rho, direction

  def record_batch(self, observations: Sequence[tuple[str, Sequence[float]]]) -> None:
    """Updates task history, boundary positions, and cluster learnabilities."""
    cluster_rhos: dict[str, list[float]] = collections.defaultdict(list)
    moves: dict[str, dict[int, int]] = collections.defaultdict(dict)
    touched_domains: set[str] = set()
    for pid, rewards in observations:
      obs = self.observe_single_task(pid, rewards)
      if obs is None:
        continue
      cid, rho, direction = obs
      touched_domains.add(self.tasks[pid].dataset)
      cluster_rhos[cid].append(rho)
      pos = self.cluster_sorted_ids[cid].index(pid)
      if pos in self.cluster_states[cid].mini_positions:
        moves[cid][pos] = direction
    cap = self.cooldown * max(1, len(self.dataset_clusters))
    if cap and len(self.recent) > cap:
      self.recent = self.recent[-cap:]
    for cid, vals in cluster_rhos.items():
      st = self.cluster_states[cid]
      w = 1.0 if st.observations == 0 else self.cluster_ema
      st.success_rate = (1.0 - w) * st.success_rate + w * float(np.mean(vals))
      mean_l = float(np.mean([skill_ledger.bell(v) for v in vals]))
      st.learnability = (1.0 - w) * st.learnability + w * mean_l
      st.observations += 1
      self.shift_mini_positions(cid, moves[cid])
    for dom in sorted(touched_domains):
      self.refresh_dataset_probs(dom)
    if touched_domains:
      self.update_index += 1

  def state_dict(self) -> dict[str, object]:
    return {
        "cluster_states": {
            cid: {
                "success_rate": st.success_rate,
                "prob": st.prob,
                "mini_positions": sorted(st.mini_positions),
                "learnability": st.learnability,
                "observations": st.observations,
            }
            for cid, st in self.cluster_states.items()
        },
        "task_success": dict(self.task_success),
        "task_history": dict(self.task_history),
        "update_index": self.update_index,
        "selection_count": dict(self.selection_count),
        "recent": list(self.recent),
    }

  def load_state_dict(self, payload: Mapping[str, object]) -> None:
    for cid, vals in dict(payload.get("cluster_states") or {}).items():  # pytype: disable=wrong-arg-types
      if cid in self.cluster_states:
        st = self.cluster_states[cid]
        st.success_rate = float(vals["success_rate"])
        st.prob = float(vals["prob"])
        st.mini_positions = {int(x) for x in vals["mini_positions"]}
        st.learnability = float(vals["learnability"])
        st.observations = int(vals["observations"])
    self.task_success = {
        str(k): float(v)
        for k, v in dict(payload.get("task_success") or {}).items()  # pytype: disable=wrong-arg-types
    }
    self.task_history = {
        str(k): [[float(a), float(b), float(c)] for a, b, c in rows]
        for k, rows in dict(payload.get("task_history") or {}).items()  # pytype: disable=wrong-arg-types
    }
    self.update_index = int(payload.get("update_index", 0))  # pytype: disable=wrong-arg-types
    self.selection_count = collections.Counter({
        str(k): int(v)
        for k, v in dict(payload.get("selection_count") or {}).items()  # pytype: disable=wrong-arg-types
    })
    self.recent = [str(x) for x in (payload.get("recent") or ())]  # pytype: disable=wrong-arg-types


def arms_bell_reward(
    hidden: Sequence[float],
    visible: Sequence[float],
    *,
    prior: float = 0.5,
    gamma: float = 0.5,
    endpoint_penalty: float = 0.2,
    shrink_strength: float = 2.0,
    siblings: Sequence[float] | None = None,
    time_factor: float = 1.0,
) -> dict[str, object]:
  """Recursive Learnability Proposer reward `R_P(x)` (`[r2orch.arms]`)."""
  hid = [float(v) for v in hidden]
  vis = [float(v) for v in visible]
  arm_data = bool(hid and vis)
  all_vals = [*hid, *vis] if arm_data or siblings is None else [float(v) for v in siblings]
  if len(all_vals) < 2:
    return {"case": "insufficient", "value": None}
  if all(v == all_vals[0] for v in all_vals):
    return {"case": "endpoint", "value": -float(endpoint_penalty)}
  if not arm_data:
    p_all = skill_ledger.shrunk_rate(
        sum(all_vals), len(all_vals), prior=prior, strength=shrink_strength
    )
    return {"case": "fallback_bell", "value": float(time_factor) * skill_ledger.bell(p_all)}
  s_h = sum(hid)
  if s_h > 0.0:
    p_h = skill_ledger.shrunk_rate(s_h, len(hid), prior=prior, strength=shrink_strength)
    return {"case": "hidden_bell", "value": float(time_factor) * skill_ledger.bell(p_h)}
  p_v = skill_ledger.shrunk_rate(sum(vis), len(vis), prior=prior, strength=shrink_strength)
  return {
      "case": "assisted_bell",
      "value": float(time_factor) * float(gamma) * skill_ledger.bell(p_v),
  }


def sample_categorical_label(
    labels: Sequence[str],
    logprobs: Sequence[float],
    *,
    seed: int,
    temperature: float = 1.0,
    exclude: str | None = None,
) -> str | None:
  """Samples one label from temperature-scaled `logprobs` (`!= exclude`)."""
  items = [
      (lbl, float(lp))
      for lbl, lp in zip(labels, logprobs, strict=True)
      if lbl != exclude and math.isfinite(float(lp))
  ]
  if not items:
    return None
  scale = 1.0 / max(float(temperature), 1e-6)
  top = max(lp for _, lp in items)
  weights = [math.exp((lp - top) * scale) for _, lp in items]
  digest = hashlib.sha256(f"{seed}\0proposer_pick".encode()).digest()
  draw = random.Random(int.from_bytes(digest[:8], "big")).random() * sum(weights)
  acc = 0.0
  for (lbl, _), w in zip(items, weights, strict=True):
    acc += w
    if draw < acc:
      return lbl
  return items[-1][0]


def proposer_fwd_label_logps(
    model: nnx.Module, payload: Mapping[str, jax.Array]
) -> jax.Array:
  """Evaluates final-position log-softmax over `label_ids` (`[B, K]`)."""
  tokens = payload["tokens"]
  mask = payload["mask"]
  label_ids = payload["label_ids"]
  label_mask = payload["label_mask"]
  positions = sft_utils.build_positions_from_mask(mask)
  attn_mask = sft_utils.make_causal_attn_mask(mask)
  if hasattr(model, "compute_final_logits"):
    hidden, _ = model(tokens, positions, None, attn_mask, skip_lm_head=True)  # pytype: disable=wrong-keyword-args
    last_logits = model.compute_final_logits(hidden[:, -1:, :])[:, 0, :]
  else:
    logits, _ = model(tokens, positions, None, attn_mask)
    last_logits = jnp.asarray(logits[:, -1, :], dtype=jnp.float32)
  gathered = jnp.take_along_axis(last_logits, label_ids, axis=-1)
  masked = jnp.where(label_mask, gathered, -1e30)
  return jax.nn.log_softmax(masked, axis=-1)


def make_proposer_loss_fn(kl_beta: float, gap_clip: float):
  """Builds the pairwise REINFORCE + reference-KL loss with trust-region clip."""

  def loss_fn(
      model: nnx.Module,
      tokens: jax.Array,
      mask: jax.Array,
      label_ids: jax.Array,
      label_mask: jax.Array,
      ref_logps: jax.Array,
      pair_a: jax.Array,
      pair_b: jax.Array,
      advantages: jax.Array,
      pair_valid: jax.Array,
  ) -> jax.Array:
    curr_lp = proposer_fwd_label_logps(
        model,
        {
            "tokens": tokens,
            "mask": mask,
            "label_ids": label_ids,
            "label_mask": label_mask,
        },
    )
    probs = jnp.where(label_mask, jnp.exp(curr_lp), 0.0)
    kl = jnp.sum(
        jnp.where(label_mask, probs * (curr_lp - ref_logps), 0.0), axis=-1
    )
    idx_a = pair_a[:, None]
    idx_b = pair_b[:, None]
    curr_diff = (
        jnp.take_along_axis(curr_lp, idx_a, -1)[:, 0]
        - jnp.take_along_axis(curr_lp, idx_b, -1)[:, 0]
    )
    ref_diff = (
        jnp.take_along_axis(ref_logps, idx_a, -1)[:, 0]
        - jnp.take_along_axis(ref_logps, idx_b, -1)[:, 0]
    )
    sg_gap = jax.lax.stop_gradient(curr_diff - ref_diff)
    clipped = (
        (gap_clip > 0.0)
        & ((advantages * sg_gap) > 0.0)
        & (jnp.abs(sg_gap) >= gap_clip)
    )
    pg_term = jnp.where(clipped, 0.0, -advantages * curr_diff)
    denom = jnp.maximum(jnp.sum(pair_valid.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(pair_valid, pg_term + kl_beta * kl, 0.0)) / denom

  return loss_fn


class ProposerRuntime:
  """Coordinates ADS+TSDS candidate lists, JAX label picks, and pair updates."""

  def __init__(
      self,
      tasks: Sequence[scoring.R2OrchTask],
      proposer_model: nnx.Module,
      proposer_trainer: peft_trainer_v2.PeftTrainer,
      encode_proposer: Callable[[str], Sequence[int]],
      label_token_ids: Mapping[str, int],
      *,
      pad_id: int,
      batch_multiple: int,
      config: Mapping[str, object],
  ):
    self.tasks = {t.task_id: t for t in tasks}
    self.model = proposer_model
    self.trainer = proposer_trainer
    self.encode = encode_proposer
    self.label_ids_map = dict(label_token_ids)
    self.labels = tuple(label_token_ids)
    self.pad_id = int(pad_id)
    self.batch_multiple = max(1, int(batch_multiple))
    self.config = config
    self.embeddings = {t.task_id: task_embedding(t) for t in tasks}
    self.scheduler = ADSBoundaryScheduler(
        tasks,
        seed=int(config["seed"]),  # pytype: disable=wrong-arg-types
        active_clusters=int(config.get("ads_active_clusters", 2)),  # pytype: disable=wrong-arg-types
        mini_cluster_size=int(config.get("ads_mini_cluster_size", 8)),  # pytype: disable=wrong-arg-types
    )
    self.retriever = TSDSRetriever(self.embeddings)
    self.ref_graphdef, ref_state = nnx.split(proposer_model)
    self.ref_state = jax.tree.map(lambda x: x.copy(), ref_state)
    self._jitted_ref = jax.jit(
        lambda gd, st, p: proposer_fwd_label_logps(
            nnx.merge(gd, st, copy=True), p
        )
    )
    self.trainer.is_managed_externally = True
    self.trainer.with_fwd_only_fn(proposer_fwd_label_logps)
    self.trainer.with_loss_fn(
        make_proposer_loss_fn(
            float(config.get("proposer_kl_beta", 1.0)),  # pytype: disable=wrong-arg-types
            float(config.get("proposer_gap_clip", 4.0)),  # pytype: disable=wrong-arg-types
        ),
        has_aux=False,
    )

  def rank_anchor_candidates(self, anchor_id: str) -> list[tuple[str, str]]:
    """Selects up to `len(self.labels)` `(pool_id, source)` candidates."""
    dom = self.tasks[anchor_id].dataset
    k_max = len(self.labels)
    ads_k = min(int(self.config.get("proposer_ads_candidates", 4)), k_max)  # pytype: disable=wrong-arg-types
    ads_picks = [
        p for p in self.scheduler.candidates(dom) if p != anchor_id
    ][:ads_k]
    ranked = [(p, "ads") for p in ads_picks]
    taken = {anchor_id, *ads_picks}
    blocked = self.scheduler.recent_ids(dom) | self.scheduler.reserved | taken
    dom_ids = [p for p, t in self.tasks.items() if t.dataset == dom]
    repo = [p for p in dom_ids if p not in blocked] or [
        p for p in dom_ids if p not in taken
    ]
    anchor_vec = self.embeddings[anchor_id]
    tsds_ids = self.retriever.rank(repo, anchor_vec, limit=k_max - len(ranked))
    nearest = sorted(
        repo, key=lambda p: float(np.sum((self.embeddings[p] - anchor_vec) ** 2))
    )
    for pid in dict.fromkeys((*tsds_ids, *nearest, *dom_ids)):
      if len(ranked) >= k_max:
        break
      if pid not in taken or len(ranked) < 2:
        taken.add(pid)
        ranked.append((pid, "tsds"))
    return ranked[:k_max]

  def build_candidate_list(
      self,
      anchor_id: str,
      *,
      cycle: int,
      scaffold: skill_ledger.SkillScaffold,
  ) -> CandidateList:
    """Constructs one `A..H` (`4` ADS + `4` TSDS) candidate list for `anchor_id`."""
    anchor = self.tasks[anchor_id]
    dom = anchor.dataset
    scope = scoring.task_scope(anchor)
    ranked = self.rank_anchor_candidates(anchor_id)
    sc_info = scaffold.scope_summary(scope)
    cands = [
        {
            "label": lbl,
            "pool_id": pid,
            "source": src,
            "dataset": dom,
            "scope_cards": sc_info["card_count"],
            "scope_card_tokens": sc_info["card_tokens"],
            "previous_success": self.scheduler.task_success.get(pid),
            "selection_count": int(self.scheduler.selection_count[pid]),
            "prompt_preview": self.tasks[pid].prompt[:420],
        }
        for lbl, (pid, src) in zip(self.labels, ranked)
    ]
    public = [
        {k: v for k, v in c.items() if k not in ("pool_id", "source")}
        for c in cands
    ]
    ahash = hashlib.sha256(anchor_id.encode("utf-8")).hexdigest()[:10]
    list_id = f"c{cycle:04d}-{dom}-{ahash}"
    user_json = json.dumps(
        {
            "anchor": anchor.prompt[:420],
            "candidates": public,
            "answer_with_one_of": [c["label"] for c in cands],
        },
        ensure_ascii=False,
    )
    return CandidateList(
        list_id=list_id,
        cycle=cycle,
        dataset=dom,
        scope=scope,
        anchor_id=anchor_id,
        candidates=tuple(cands),
        prompt_text=user_json,
    )

  def pack_fwd_payload(
      self, lists: Sequence[CandidateList]
  ) -> dict[str, np.ndarray]:
    """Left-pads `lists` into fixed `[B_pad, L_pad]` arrays for JIT scoring."""
    b_raw = len(lists)
    rem = b_raw % self.batch_multiple
    b_pad = b_raw if rem == 0 else b_raw + (self.batch_multiple - rem)
    l_pad = int(self.config.get("proposer_max_prompt_length", 2048))  # pytype: disable=wrong-arg-types
    k_max = len(self.labels)
    tokens = np.full((b_pad, l_pad), self.pad_id, dtype=np.int32)
    mask = np.zeros((b_pad, l_pad), dtype=bool)
    default_ids = [self.label_ids_map[lbl] for lbl in self.labels]
    label_ids = np.tile(np.asarray(default_ids, dtype=np.int32), (b_pad, 1))
    label_mask = np.zeros((b_pad, k_max), dtype=bool)
    label_mask[:, 0] = True
    for idx, clist in enumerate(lists):
      ids = list(self.encode(clist.prompt_text))[-l_pad:]
      n = len(ids)
      tokens[idx, l_pad - n :] = np.asarray(ids, dtype=np.int32)
      mask[idx, l_pad - n :] = True
      valid_k = len(clist.candidates)
      label_mask[idx, :valid_k] = True
    return {
        "tokens": tokens,
        "mask": mask,
        "label_ids": label_ids,
        "label_mask": label_mask,
    }

  def select_cycle_tasks(
      self, cycle: int, scaffold: skill_ledger.SkillScaffold
  ) -> tuple[
      list[scoring.R2OrchTask], list[PairSelection], dict[str, np.ndarray]
  ]:
    """Builds candidate lists, scores labels on `train_mesh`, and picks pairs."""
    per_dom = int(self.config.get("proposer_lists_per_dataset", 2))  # pytype: disable=wrong-arg-types
    base_seed = int(self.config["seed"])  # pytype: disable=wrong-arg-types
    temp = float(self.config.get("proposer_temperature", 1.0))  # pytype: disable=wrong-arg-types
    anchors = self.scheduler.anchor_ids(
        per_dataset=per_dom, cycle=cycle, seed=base_seed
    )
    lists = [
        self.build_candidate_list(aid, cycle=cycle, scaffold=scaffold)
        for aid in anchors
    ]
    fwd_payload = self.pack_fwd_payload(lists)
    logps = np.asarray(self.trainer.fwd_only(fwd_payload), dtype=np.float64)
    chosen_tasks: list[scoring.R2OrchTask] = []
    selections: list[PairSelection] = []
    for idx, clist in enumerate(lists):
      valid_k = len(clist.candidates)
      lbls = [str(c["label"]) for c in clist.candidates]
      row_lps = [float(x) for x in logps[idx, :valid_k]]
      by_lbl = {str(c["label"]): c for c in clist.candidates}
      s0 = base_seed + cycle * 1009 + idx * 17
      s1 = s0 + 1
      pick0 = sample_categorical_label(lbls, row_lps, seed=s0, temperature=temp)
      pick1 = sample_categorical_label(
          lbls, row_lps, seed=s1, temperature=temp, exclude=pick0
      ) or pick0
      assert pick0 is not None and pick1 is not None
      same = by_lbl[pick0]["pool_id"] == by_lbl[pick1]["pool_id"]
      choices = []
      for k_i, lbl in enumerate((pick0, pick1)):
        pid = str(by_lbl[lbl]["pool_id"])
        slot_id = f"{clist.list_id}-k0" if (same and k_i == 1) else f"{clist.list_id}-k{k_i}"
        choices.append({
            "k": k_i,
            "label": lbl,
            "index": lbls.index(lbl),
            "pool_id": pid,
            "task_id": slot_id,
        })
        self.scheduler.reserved.add(pid)
        if not (same and k_i == 1):
          base_t = self.tasks[pid]
          cur = {**base_t.curriculum, "pool_id": pid}
          chosen_tasks.append(
              dataclasses.replace(base_t, task_id=slot_id, curriculum=cur)
          )
      selections.append(
          PairSelection(
              cand_list=clist,
              choices=tuple(choices),
              same_choice=same,
              behavior_logprobs=tuple(row_lps),
          )
      )
    return chosen_tasks, selections, fwd_payload

  def score_slot_learnability(
      self,
      outs: Sequence[workflow.R2OrchOutcome],
      prior_snapshot: Mapping[str, float],
      scaffold: skill_ledger.SkillScaffold,
  ) -> tuple[str, float | None]:
    """Computes `(case, F_tilde)` for one task slot's sibling outcomes."""
    dom = outs[0].dataset
    scope = scoring.DATASET_SCOPES.get(dom, dom)
    hid = [o.correctness for o in outs if o.skill_arm == "hidden"]
    vis = [o.correctness for o in outs if o.skill_arm == "visible"]
    t_q = sum(o.duration_s for o in outs) / len(outs)
    tfac = skill_ledger.compute_factor(
        t_q,
        scaffold.dataset_durations.get(dom),
        float(self.config.get("arms_time_exponent", 0.5)),  # pytype: disable=wrong-arg-types
    )
    res = arms_bell_reward(
        hid,
        vis,
        prior=prior_snapshot.get(scope, skill_ledger.DEFAULT_SCOPE_PRIOR),
        gamma=float(self.config.get("arms_gamma", 0.5)),  # pytype: disable=wrong-arg-types
        endpoint_penalty=float(self.config.get("arms_endpoint_penalty", 0.2)),  # pytype: disable=wrong-arg-types
        shrink_strength=float(self.config.get("arms_shrink_strength", 2.0)),  # pytype: disable=wrong-arg-types
        siblings=[o.correctness for o in outs],
        time_factor=tfac,
    )
    val = None if res["value"] is None else float(res["value"])  # pytype: disable=wrong-arg-types
    return str(res["case"]), val

  def score_frontier_targets(
      self,
      selections: Sequence[PairSelection],
      outcomes_by_slot: Mapping[str, Sequence[workflow.R2OrchOutcome]],
      prior_snapshot: Mapping[str, float],
      scaffold: skill_ledger.SkillScaffold,
  ) -> tuple[list[float], list[tuple[str, list[float]]], dict[str, int]]:
    """Computes `arms_bell_reward` targets and ADS pool observations."""
    f_by_slot: dict[str, float | None] = {}
    case_counts: dict[str, int] = collections.Counter()
    for slot_id, outs in outcomes_by_slot.items():
      if outs:
        case_name, f_val = self.score_slot_learnability(
            outs, prior_snapshot, scaffold
        )
        case_counts[case_name] += 1
        f_by_slot[slot_id] = f_val
    slot_pool = {
        str(ch["task_id"]): str(ch["pool_id"])
        for sel in selections
        for ch in sel.choices
    }
    ads_obs = [
        (pid, [o.correctness for o in outcomes_by_slot[sid]])
        for sid, pid in slot_pool.items()
        if sid in outcomes_by_slot
    ]
    targets: list[float] = []
    for sel in selections:
      f_a = f_by_slot.get(str(sel.choices[0]["task_id"]))
      f_b = f_by_slot.get(str(sel.choices[1]["task_id"]))
      ok = (not sel.same_choice) and (f_a is not None) and (f_b is not None)
      targets.append(float(f_a - f_b) if ok else 0.0)
    return targets, ads_obs, dict(case_counts)

  def update_from_cycle(
      self,
      selections: Sequence[PairSelection],
      fwd_payload: Mapping[str, np.ndarray],
      outcomes_by_slot: Mapping[str, Sequence[workflow.R2OrchOutcome]],
      prior_snapshot: Mapping[str, float],
      scaffold: skill_ledger.SkillScaffold,
  ) -> dict[str, object]:
    """Updates ADS state and applies 1 Proposer pairwise RL+KL optimizer step."""
    targets, ads_obs, case_counts = self.score_frontier_targets(
        selections, outcomes_by_slot, prior_snapshot, scaffold
    )
    self.scheduler.record_batch(ads_obs)
    b_pad = fwd_payload["tokens"].shape[0]
    sharded_fwd = self.trainer._prepare_payload(dict(fwd_payload))  # pylint: disable=protected-access
    ref_lps = np.asarray(
        self._jitted_ref(self.ref_graphdef, self.ref_state, sharded_fwd),
        dtype=np.float32,
    )
    curr_lps = np.asarray(self.trainer.fwd_only(dict(fwd_payload)), dtype=np.float32)
    valid = np.zeros(b_pad, dtype=bool)
    pair_a = np.zeros(b_pad, dtype=np.int32)
    pair_b = np.zeros(b_pad, dtype=np.int32)
    half_valid: list[float] = []
    kls: list[float] = []
    ads_mass: list[float] = []
    for idx, (sel, tgt) in enumerate(zip(selections, targets, strict=True)):
      k_n = len(sel.cand_list.candidates)
      c_row = curr_lps[idx, :k_n]
      r_row = ref_lps[idx, :k_n]
      probs = np.exp(c_row)
      kls.append(float(np.sum(probs * (c_row - r_row))))
      ads_mass.append(
          float(
              sum(
                  probs[j]
                  for j, c in enumerate(sel.cand_list.candidates)
                  if c["source"] == "ads"
              )
          )
      )
      if not sel.same_choice:
        valid[idx] = True
        pair_a[idx] = int(sel.choices[0]["index"])
        pair_b[idx] = int(sel.choices[1]["index"])
        half_valid.append(0.5 * tgt)
    rms = (
        math.sqrt(sum(v * v for v in half_valid) / len(half_valid))
        if half_valid
        else 0.0
    )
    advantages = np.zeros(b_pad, dtype=np.float32)
    for idx, (sel, tgt) in enumerate(zip(selections, targets, strict=True)):
      if not sel.same_choice and rms > 1e-12:
        advantages[idx] = float((0.5 * tgt) / rms)
    loss_val = 0.0
    grad_norm = 0.0
    if bool(valid.any()):
      train_payload = {
          **fwd_payload,
          "ref_logps": ref_lps,
          "pair_a": pair_a,
          "pair_b": pair_b,
          "advantages": advantages,
          "pair_valid": valid,
      }
      self.trainer.train_step(train_payload)
      loss_val = float(np.asarray(self.trainer._last_fwd_bwd_loss))  # pylint: disable=protected-access
      grad_norm = float(np.asarray(self.trainer._last_update_grad_norm))  # pylint: disable=protected-access
    return {
        "proposer_loss": loss_val,
        "proposer_grad_norm": grad_norm,
        "proposer_pairs": int(valid.sum()),
        "proposer_nonzero_pairs": sum(abs(t) > 1e-12 for t in targets),
        "proposer_target_abs_mean": (
            float(np.mean(np.abs(targets))) if targets else 0.0
        ),
        "proposer_kl_to_ref_mean": float(np.mean(kls)) if kls else 0.0,
        "proposer_p_ads_policy": float(np.mean(ads_mass)) if ads_mass else 0.0,
        "proposer_cases": case_counts,
    }
