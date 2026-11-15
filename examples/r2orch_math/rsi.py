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

"""R²Orch RSI mathematics and resumable skill evidence.

Independent implementation of the published recursive-learnability estimator,
paired Proposer advantages, and Bayesian-bootstrap skill decisions. Reference:
https://arxiv.org/abs/2610.22740v2 and R2Orch source revision
12ceb7563aa50439687f21ba5da60ad74a75eb89. This module does not implement the
two-model optimizer loop, ADS/TSDS curriculum, or the skill-distillation agent.
"""

from collections import Counter
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np


def _outcomes(values):
  values = tuple(float(v) for v in values)
  if any(v not in (0.0, 1.0) for v in values):
    raise ValueError("RSI evidence must contain raw binary correctness.")
  return values


def recursive_learnability(
    hidden: Sequence[float],
    visible: Sequence[float],
    *,
    prior: float = 0.5,
    gamma: float = 0.5,
    endpoint_penalty: float = 0.2,
    shrink_strength: float = 2.0,
    siblings: Sequence[float] | None = None,
    time_factor: float = 1.0,
) -> float | None:
  """Returns the original arm-coupled reward, or None for insufficient evidence.

  The compute multiplier affects only positive learnability, never the penalty
  for an all-right/all-wrong group. Missing arms fall back to sibling outcomes.
  """
  h, v = _outcomes(hidden), _outcomes(visible)
  if not math.isfinite(prior) or not 0 <= prior <= 1:
    raise ValueError("prior must be a probability.")
  for name, value in (
      ("gamma", gamma),
      ("endpoint_penalty", endpoint_penalty),
      ("shrink_strength", shrink_strength),
      ("time_factor", time_factor),
  ):
    if not math.isfinite(value) or value < 0:
      raise ValueError(f"{name} must be finite and nonnegative.")
  paired = bool(h and v)
  outcomes = h + v if paired or siblings is None else _outcomes(siblings)
  if len(outcomes) < 2:
    return None
  if min(outcomes) == max(outcomes):
    return -endpoint_penalty
  arm = h if paired and any(h) else v if paired else outcomes
  rate = (math.fsum(arm) + shrink_strength * prior) / (
      len(arm) + shrink_strength
  )
  discount = gamma if paired and not any(h) else 1.0
  return time_factor * discount * 4.0 * rate * (1.0 - rate)


def proposer_pair_advantages(
    reward_pairs: Sequence[Sequence[float]],
) -> np.ndarray:
  """Half reward differences with opposite signs, normalized to batch RMS one."""
  values = np.asarray(reward_pairs, dtype=np.float64)
  if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
    raise ValueError("Expected finite reward pairs with shape [pairs, 2].")
  half = (values[:, 0] - values[:, 1]) / 2.0
  rms = math.sqrt(float(np.mean(half**2))) if len(half) else 0.0
  normalized = half / rms if rms > 0 else np.zeros_like(half)
  return np.stack((normalized, -normalized), axis=1)


def arm_assignment(cycle: int, task_id: str, *, hidden=3, siblings=5):
  """Reproduces the source's deterministic randomized arm assignment."""
  if not task_id or cycle < 0 or not 0 <= hidden <= siblings:
    raise ValueError("Invalid cycle, task identity, or arm sizes.")
  digest = hashlib.sha256(f"{cycle}\0{task_id}\0skill_arms".encode()).digest()
  order = list(range(siblings))
  random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
  chosen = set(order[:hidden])
  return tuple("hidden" if i in chosen else "visible" for i in range(siblings))


def candidate_list(
    ads_ids, similarity_ids, *, allowed_training_ids, per_source=4
):
  """Combines existing ADS and similarity rankings without admitting test IDs.

  Ranking computation belongs to the curriculum. This function validates the
  list boundary and takes distinct candidates; it does not replace ADS/TSDS.
  """
  if per_source < 1:
    raise ValueError("per_source must be positive.")
  allowed = set(allowed_training_ids)
  selected = []
  for source in (ads_ids, similarity_ids):
    added = 0
    for task_id in source:
      if task_id not in allowed:
        raise ValueError("Candidate list contains a non-training task.")
      if task_id not in selected:
        selected.append(task_id)
        added += 1
      if added == per_source:
        break
    if added != per_source:
      raise ValueError("Insufficient distinct candidates in a source ranking.")
  return tuple(selected)


@dataclasses.dataclass(frozen=True)
class CandidatePair:
  indices: tuple[int, int]
  logprobs: tuple[float, float]


def sample_candidate_pair(logits, *, seed: int, temperature: float = 1.0):
  """Samples two labels, restricting the second distribution to unused labels."""
  scores = np.asarray(logits, dtype=np.float64)
  if (
      scores.ndim != 1
      or len(scores) < 2
      or not np.isfinite(scores).all()
      or not math.isfinite(temperature)
      or temperature <= 0
  ):
    raise ValueError("Need finite label logits and a positive temperature.")
  rng = np.random.default_rng(seed)
  available = np.ones(len(scores), dtype=bool)
  indices, logprobs = [], []
  for _ in range(2):
    labels = np.flatnonzero(available)
    shifted = scores[labels] / temperature
    shifted -= shifted.max()
    log_weights = shifted - np.log(np.exp(shifted).sum())
    offset = int(rng.choice(len(labels), p=np.exp(log_weights)))
    selected = int(labels[offset])
    indices.append(selected)
    logprobs.append(float(log_weights[offset]))
    available[selected] = False
  return CandidatePair(tuple(indices), tuple(logprobs))


@dataclasses.dataclass(frozen=True)
class LedgerConfig:
  min_tasks: int = 12
  epsilon: float = 0.02
  window_cycles: int = 30
  bootstrap_draws: int = 2000
  admit_quantile: float = 0.05
  reinstate_quantile: float = 0.05
  audit_share: float = 0.3
  prior_sd: float = 0.25

  def __post_init__(self):
    if min(self.min_tasks, self.window_cycles, self.bootstrap_draws) < 1:
      raise ValueError("Ledger counts must be positive.")
    if (
        not 0 <= self.epsilon <= 1
        or not math.isfinite(self.prior_sd)
        or self.prior_sd < 0
    ):
      raise ValueError("Invalid effect threshold or prior.")
    if any(
        not 0 <= p <= 1
        for p in (
            self.admit_quantile,
            self.reinstate_quantile,
            self.audit_share,
        )
    ):
      raise ValueError("Ledger probabilities must be in [0, 1].")


def _rng(seed, cycle, purpose, key):
  digest = hashlib.sha256(
      f"skill_ledger\0{seed}\0{cycle}\0{purpose}\0{key}".encode()
  ).digest()
  return random.Random(int.from_bytes(digest[:8], "big"))


def _bootstrap(contrasts, *, draws, rng):
  groups = sorted(Counter(round(v, 12) for v in contrasts).items())
  if not groups:
    return np.empty(0, dtype=np.float64)
  if len(groups) == 1:
    return np.full(draws, groups[0][0], dtype=np.float64)
  samples = []
  for _ in range(draws):
    weights = [rng.gammavariate(float(count), 1.0) for _, count in groups]
    samples.append(
        sum(w * pair[0] for w, pair in zip(weights, groups)) / sum(weights)
    )
  return np.asarray(samples)


class SkillLedger:
  """Versioned evidence with idempotent replay and atomic local snapshots.

  A saved ledger is a component of a committed training-cycle checkpoint. The
  caller must commit model/optimizer/ledger snapshots together before resuming.
  There is intentionally no concurrent-writer or distributed transaction claim.
  """

  SCHEMA = "tunix_r2orch_skill_ledger_v1"

  def __init__(self, config=LedgerConfig(), *, seed=0):
    self.config = config
    self.seed = int(seed)
    self.skills = {}
    self.records = {}
    self.last_decision_cycle = -1

  def register(self, skill_id: str, *, version: int, content: str) -> str:
    if not skill_id or version < 1 or not content:
      raise ValueError("A skill requires an ID, version and content.")
    digest = hashlib.sha256(content.encode()).hexdigest()
    key = f"{skill_id}@{version}#{digest}"
    self.skills.setdefault(
        key,
        {
            "skill_id": skill_id,
            "version": version,
            "content": content,
            "content_sha256": digest,
            "status": "trial",
        },
    )
    return key

  def record(
      self, *, cycle: int, task_id: str, skill_key: str, hidden, visible
  ):
    h, v = _outcomes(hidden), _outcomes(visible)
    if (
        cycle < 0
        or not task_id
        or skill_key not in self.skills
        or not h
        or not v
    ):
      raise ValueError(
          "Paired evidence requires known skill, task, cycle and both arms."
      )
    key = json.dumps([cycle, task_id, skill_key], separators=(",", ":"))
    item = {
        "cycle": cycle,
        "task_id": task_id,
        "skill_key": skill_key,
        "hidden": list(h),
        "visible": list(v),
        "contrast": math.fsum(v) / len(v) - math.fsum(h) / len(h),
    }
    if key in self.records:
      if self.records[key] != item:
        raise ValueError("Conflicting replay of skill evidence.")
      return
    if cycle <= self.last_decision_cycle:
      raise ValueError("Cannot append evidence to a committed decision cycle.")
    self.records[key] = item

  def contrasts(self, skill_key, *, cycle):
    return [
        row["contrast"]
        for _, row in sorted(self.records.items())
        if row["skill_key"] == skill_key
        and cycle - self.config.window_cycles < row["cycle"] <= cycle
    ]

  def summary(self, skill_key, *, cycle):
    if skill_key not in self.skills:
      raise ValueError("Unknown skill.")
    values = self.contrasts(skill_key, cycle=cycle)
    if not values:
      return {
          "n": 0,
          "mean": None,
          "positive_effect": None,
          "q_admit": None,
          "q_reinstate": None,
      }
    samples = _bootstrap(
        values,
        draws=self.config.bootstrap_draws,
        rng=_rng(self.seed, cycle, "posterior", skill_key),
    )
    return {
        "n": len(values),
        "mean": math.fsum(values) / len(values),
        "positive_effect": float(np.maximum(samples, 0).mean()),
        "q_admit": float(np.quantile(samples, self.config.admit_quantile)),
        "q_reinstate": float(
            np.quantile(samples, self.config.reinstate_quantile)
        ),
    }

  def decide(self, *, cycle: int):
    if cycle < self.last_decision_cycle:
      raise ValueError("Ledger decisions cannot move backwards.")
    decisions = {}
    for key, skill in sorted(self.skills.items()):
      if skill["status"] == "archived":
        continue
      s = self.summary(key, cycle=cycle)
      if s["n"] < self.config.min_tasks:
        decisions[key] = "undecided"
      elif skill["status"] == "retired":
        reinstate = (
            s["positive_effect"] > self.config.epsilon and s["q_reinstate"] > 0
        )
        decisions[key] = "reinstate" if reinstate else "retired"
        if reinstate:
          skill["status"] = "active"
      elif s["positive_effect"] <= self.config.epsilon:
        decisions[key] = "retire"
        skill["status"] = "retired"
      elif s["q_admit"] > 0:
        decisions[key] = "admit"
        skill["status"] = "active"
      else:
        decisions[key] = "undecided"
    self.last_decision_cycle = cycle
    return decisions

  def choose(self, candidate_keys, *, cycle, task_id):
    """Selects a candidate by posterior sampling, sometimes auditing a retired skill."""
    keys = sorted(set(candidate_keys))
    if not keys or any(k not in self.skills for k in keys):
      raise ValueError("Provide known candidate skills.")
    rng = _rng(self.seed, cycle, "audit", task_id)
    retired = [k for k in keys if self.skills[k]["status"] == "retired"]
    active = [
        k for k in keys if self.skills[k]["status"] in ("trial", "active")
    ]
    if retired and rng.random() < self.config.audit_share:
      return rng.choice(retired), True
    if not active:
      return None, False
    draws = {}
    sampler = _rng(self.seed, cycle, "thompson", task_id)
    # The source draws for every active card in the scope before selecting the
    # lowest draw among the query's retrieved cards.
    for key in sorted(
        k for k, s in self.skills.items() if s["status"] in ("trial", "active")
    ):
      prior = sampler.gauss(0.0, self.config.prior_sd)
      groups = sorted(
          Counter(
              round(v, 12) for v in self.contrasts(key, cycle=cycle)
          ).items()
      )
      if not groups:
        draws[key] = prior
        continue
      weights = [sampler.gammavariate(1.0, 1.0)]
      values = [prior]
      for value, count in groups:
        weights.append(sampler.gammavariate(float(count), 1.0))
        values.append(value)
      draws[key] = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    return min(active, key=lambda k: (draws[k], k)), False

  def save(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": self.SCHEMA,
        "config": dataclasses.asdict(self.config),
        "seed": self.seed,
        "skills": self.skills,
        "records": self.records,
        "last_decision_cycle": self.last_decision_cycle,
    }
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as file:
      temporary = file.name
      json.dump(state, file, sort_keys=True, allow_nan=False)
      file.flush()
      os.fsync(file.fileno())
    try:
      os.replace(temporary, path)
    finally:
      if os.path.exists(temporary):
        os.unlink(temporary)

  @classmethod
  def load(cls, path):
    state = json.loads(Path(path).read_text())
    if state.get("schema") != cls.SCHEMA:
      raise ValueError("Unknown skill ledger schema.")
    ledger = cls(LedgerConfig(**state["config"]), seed=state["seed"])
    for key, skill in state["skills"].items():
      actual = ledger.register(
          skill["skill_id"], version=skill["version"], content=skill["content"]
      )
      if (
          actual != key
          or skill["content_sha256"] != ledger.skills[key]["content_sha256"]
      ):
        raise ValueError("Skill content identity mismatch.")
      if skill["status"] not in ("trial", "active", "retired", "archived"):
        raise ValueError("Unknown skill state.")
      ledger.skills[key]["status"] = skill["status"]
    for row in state["records"].values():
      ledger.record(**{
          k: row[k]
          for k in ("cycle", "task_id", "skill_key", "hidden", "visible")
      })
    if ledger.records != state["records"]:
      raise ValueError("Skill evidence identity or contrast mismatch.")
    ledger.last_decision_cycle = int(state["last_decision_cycle"])
    if ledger.last_decision_cycle < -1:
      raise ValueError("Invalid committed cycle.")
    return ledger
