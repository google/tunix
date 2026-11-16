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

"""Arm-coupled R2Orch Skill Ledger, Thompson scaffold planner, and withdrawal."""

import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import re

from examples.swarm_r2orch import scoring
from examples.swarm_r2orch import workflow

SKILL_KINDS = frozenset({
    "orchestration",
    "decomposition",
    "communication",
    "verification",
    "tool-use",
    "bug-repair",
})
DEFAULT_PRIOR_SD = 0.25
DEFAULT_SCOPE_PRIOR = 0.5
_CONTRAST_DIGITS = 12
_WORD_RE = re.compile(r"\w+")


@dataclasses.dataclass
class SkillCard:
  """One scoped Director orchestration skill card managed by `SkillScaffold`."""

  skill_id: str
  name: str
  description: str
  trigger: str
  plan: str
  pitfall: str = ""
  constraint: str = ""
  kind: str = "orchestration"
  task_types: tuple[str, ...] = ()
  version: int = 1
  status: str = "active"
  is_seed: bool = False

  def content_dict(self) -> dict[str, object]:
    """Canonical content dictionary hashed for ledger identity."""
    return {
        "skill_id": self.skill_id,
        "name": self.name,
        "description": self.description,
        "trigger": self.trigger,
        "plan": self.plan,
        "pitfall": self.pitfall,
        "constraint": self.constraint,
        "kind": self.kind,
        "task_types": list(self.task_types),
    }

  def to_dict(self) -> dict[str, object]:
    return {
        **self.content_dict(),
        "version": self.version,
        "status": self.status,
        "is_seed": self.is_seed,
    }

  @classmethod
  def from_dict(cls, payload: Mapping[str, object]) -> "SkillCard":
    stats = payload.get("stats")
    is_seed = bool(
        payload.get("is_seed")
        or (isinstance(stats, Mapping) and stats.get("is_seed"))
    )
    raw_types = payload.get("task_types") or ()
    return cls(
        skill_id=str(payload["skill_id"]),
        name=str(payload["name"]),
        description=str(payload["description"]),
        trigger=str(payload["trigger"]),
        plan=str(payload["plan"]),
        pitfall=str(payload.get("pitfall", "")),
        constraint=str(payload.get("constraint", "")),
        kind=str(payload.get("kind", "orchestration")),
        task_types=tuple(str(x) for x in raw_types),  # pytype: disable=wrong-arg-types
        version=int(payload.get("version", 1)),  # pytype: disable=wrong-arg-types
        status=str(payload.get("status", "active")),
        is_seed=is_seed,
    )


@dataclasses.dataclass(frozen=True)
class EffectPosterior:
  """Bayesian-bootstrap posterior summary of a card's paired contrast Delta."""

  n: int
  mean: float | None
  e_delta_plus: float | None
  q05: float | None
  q95: float | None
  q_admit: float | None


@dataclasses.dataclass(frozen=True)
class TaskArmPlan:
  """Per-task hidden/visible sibling assignment and rendered skill contexts."""

  task_id: str
  scope: str
  arms: tuple[str, ...]
  hidden_context: str
  visible_context: str
  arm_card: SkillCard | None
  arm_card_key: str
  is_audit: bool


def card_key(card: SkillCard) -> str:
  """Ledger identity combining `skill_id`, `version`, and content digest."""
  encoded = json.dumps(
      card.content_dict(),
      sort_keys=True,
      ensure_ascii=False,
      separators=(",", ":"),
  ).encode("utf-8")
  sha = hashlib.sha256(encoded).hexdigest()[:12]
  return f"{card.skill_id}@{card.version}#{sha}"


def ledger_rng(base_seed: int, cycle: int, *salt: object) -> random.Random:
  """Deterministic `random.Random` seeded from `(base_seed, cycle, *salt)`."""
  joined = "\0".join(str(p) for p in ("skill_ledger", base_seed, cycle, *salt))
  seed_int = int.from_bytes(hashlib.sha256(joined.encode()).digest()[:8], "big")
  return random.Random(seed_int)


def token_jaccard(left: str, right: str) -> float:
  """Case-folded word-set Jaccard similarity in `[0.0, 1.0]`."""
  l_toks = set(_WORD_RE.findall(left.casefold()))
  r_toks = set(_WORD_RE.findall(right.casefold()))
  union = l_toks | r_toks
  return len(l_toks & r_toks) / len(union) if union else 0.0


def render_skill_context(cards: Sequence[SkillCard]) -> str:
  """Formats retrieved skill cards into the Director prompt scaffold block."""
  blocks = [
      f"[{idx}. {c.name} ({c.kind})]\n"
      f"When: {c.trigger}\n"
      f"Plan: {c.plan}\n"
      f"Avoid: {c.pitfall}\n"
      f"Constraint: {c.constraint}"
      for idx, c in enumerate(cards, 1)
  ]
  return "\n\n".join(blocks)


def arm_assignment(
    cycle: int, task_id: str, hidden: int, *, siblings: int
) -> tuple[str, ...]:
  """Deterministically permutes `siblings` slots into `hidden` and `visible`."""
  if not 0 <= hidden <= siblings:
    raise ValueError("hidden count must lie in [0, siblings].")
  digest = hashlib.sha256(f"{cycle}\0{task_id}\0skill_arms".encode()).digest()
  order = list(range(siblings))
  random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
  chosen = set(order[:hidden])
  return tuple("hidden" if idx in chosen else "visible" for idx in range(siblings))


def dirichlet_mean_draw(
    groups: Sequence[tuple[float, int]],
    rng: random.Random,
    prior_atom: float | None,
) -> float:
  """Draws one Dirichlet-weighted contrast mean with optional prior atom."""
  total = 0.0
  weighted = 0.0
  if prior_atom is not None:
    w_prior = rng.gammavariate(1.0, 1.0)
    total += w_prior
    weighted += w_prior * prior_atom
  for val, cnt in groups:
    weight = rng.gammavariate(float(cnt), 1.0)
    total += weight
    weighted += weight * val
  return weighted / total


def sample_quantile(sorted_values: Sequence[float], q: float) -> float:
  """Linear order-statistic interpolation matching `numpy.quantile`."""
  pos = (len(sorted_values) - 1) * float(q)
  lo = math.floor(pos)
  hi = min(lo + 1, len(sorted_values) - 1)
  frac = pos - lo
  return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def effect_posterior(
    contrasts: Sequence[float],
    *,
    draws: int = 2000,
    rng: random.Random,
    admit_quantile: float = 0.05,
) -> EffectPosterior:
  """Computes the Rubin (1981) Bayesian-bootstrap posterior over `contrasts`."""
  vals = [float(v) for v in contrasts]
  if not vals:
    return EffectPosterior(0, None, None, None, None, None)
  counts = collections.Counter(round(v, _CONTRAST_DIGITS) for v in vals)
  groups = sorted(counts.items())
  sample = (
      [groups[0][0]] * draws
      if len(groups) == 1
      else sorted(dirichlet_mean_draw(groups, rng, None) for _ in range(draws))
  )
  return EffectPosterior(
      n=len(vals),
      mean=math.fsum(vals) / len(vals),
      e_delta_plus=math.fsum(max(v, 0.0) for v in sample) / len(sample),
      q05=sample_quantile(sample, 0.05),
      q95=sample_quantile(sample, 0.95),
      q_admit=sample_quantile(sample, admit_quantile),
  )


def decide(
    posterior: EffectPosterior, *, min_tasks: int = 12, epsilon: float = 0.02
) -> str:
  """Returns `'retire'`, `'admit'`, or `'undecided'` from `posterior`."""
  if posterior.n < min_tasks or posterior.e_delta_plus is None:
    return "undecided"
  if posterior.e_delta_plus <= epsilon:
    return "retire"
  if posterior.q_admit is not None and posterior.q_admit > 0.0:
    return "admit"
  return "undecided"


def thompson_pick(
    candidates: Mapping[str, Sequence[float]],
    rng: random.Random,
    *,
    prior_sd: float = DEFAULT_PRIOR_SD,
) -> tuple[str | None, dict[str, float]]:
  """Picks the candidate key minimizing one Dirichlet+Normal Thompson draw."""
  draws: dict[str, float] = {}
  for key in sorted(candidates):
    atom = rng.gauss(0.0, prior_sd)
    counts = collections.Counter(
        round(float(v), _CONTRAST_DIGITS) for v in candidates[key]
    )
    groups = sorted(counts.items())
    draws[key] = (
        dirichlet_mean_draw(groups, rng, atom) if groups else atom
    )
  if not draws:
    return None, {}
  best = min(draws, key=lambda k: (draws[k], k))
  return best, draws


def audit_choice(
    retired_keys: Sequence[str], share: float, rng: random.Random
) -> str | None:
  """With probability `share`, samples one retired card key uniformly."""
  if not retired_keys or rng.random() >= share:
    return None
  ordered = sorted(retired_keys)
  return ordered[rng.randrange(len(ordered))]


def bell(p: float) -> float:
  """Recursive learnability bell curve `4 * p * (1 - p)`."""
  return 4.0 * float(p) * (1.0 - float(p))


def shrunk_rate(
    successes: float, n: int, *, prior: float, strength: float = 2.0
) -> float:
  """Beta-shrunk success rate `(s + m * prior) / (n + m)`."""
  denom = n + strength
  return (successes + strength * prior) / denom if denom > 0 else prior


def compute_factor(
    task_seconds: float | None, dataset_seconds: float | None, exponent: float
) -> float:
  """Compute normalization `(Tbar_d / T_q) ** exponent`."""
  if exponent <= 0.0 or not task_seconds or not dataset_seconds:
    return 1.0
  return (dataset_seconds / task_seconds) ** exponent


class SkillScaffold:
  """Scoped skill bank coupled with the sliding-window Bayesian ledger."""

  def __init__(
      self,
      cards_by_scope: Mapping[str, Iterable[SkillCard]],
      *,
      base_seed: int = 42,
      top_k: int = 3,
      min_tasks: int = 12,
      epsilon: float = 0.02,
      admit_quantile: float = 0.05,
      reinstate_quantile: float = 0.05,
      window_cycles: int = 30,
      audit_share: float = 0.30,
      bootstrap_draws: int = 2000,
      max_skills_per_scope: int = 30,
      scope_prior_decay: float = 0.90,
  ):
    self.cards: dict[str, dict[str, SkillCard]] = {
        scope: {c.skill_id: c for c in cards}
        for scope, cards in cards_by_scope.items()
    }
    self.base_seed = base_seed
    self.top_k = top_k
    self.min_tasks = min_tasks
    self.epsilon = epsilon
    self.admit_quantile = admit_quantile
    self.reinstate_quantile = reinstate_quantile
    self.window_cycles = window_cycles
    self.audit_share = audit_share
    self.bootstrap_draws = bootstrap_draws
    self.max_skills_per_scope = max_skills_per_scope
    self.scope_prior_decay = scope_prior_decay
    self.records: list[dict[str, object]] = []
    self.scope_priors: dict[str, float] = {}
    self.dataset_durations: dict[str, float] = {}
    self.counters = {
        "retirements": 0,
        "admissions": 0,
        "reinstatements": 0,
        "revisions": 0,
        "distilled": 0,
    }

  @classmethod
  def from_seed_file(
      cls,
      seed_path: str | Path | None,
      *,
      allowed_scopes: Iterable[str] = (
          "math",
          "multi_hop_qa",
          "medical_mcq",
          "code_function",
      ),
      **kwargs,
  ) -> "SkillScaffold":
    """Loads scoped seed skill cards from `seed_cards.json`."""
    scopes = set(allowed_scopes)
    grouped: dict[str, list[SkillCard]] = {s: [] for s in sorted(scopes)}
    if seed_path is not None and Path(seed_path).is_file():
      raw_items = json.loads(Path(seed_path).read_text(encoding="utf-8"))
      for entry in raw_items:
        card_dict = entry["card"] if "card" in entry else entry
        card = SkillCard.from_dict(card_dict)
        for scope in card.task_types:
          if scope in scopes:
            grouped[scope].append(dataclasses.replace(card))
    return cls(grouped, **kwargs)

  def retrieve(self, scope: str, query_text: str) -> list[SkillCard]:
    """Retrieves up to `top_k` non-retired cards in `scope` by Jaccard match."""
    pool = [
        c
        for c in self.cards.get(scope, {}).values()
        if c.status in ("active", "trial")
    ]
    ranked = sorted(
        pool,
        key=lambda c: (
            -token_jaccard(query_text, f"{c.name} {c.trigger} {c.description}"),
            c.skill_id,
        ),
    )
    return ranked[: self.top_k]

  def windowed_contrasts(self, through_cycle: int) -> dict[str, list[float]]:
    """Returns per-`card_key` contrasts within the last `window_cycles`."""
    start = through_cycle - self.window_cycles + 1
    out: dict[str, list[float]] = collections.defaultdict(list)
    for rec in self.records:
      if start <= int(rec["cycle"]) <= through_cycle:  # pytype: disable=wrong-arg-types
        out[str(rec["card_key"])].append(float(rec["z"]))  # pytype: disable=wrong-arg-types
    return dict(out)

  def plan_task_arms(
      self,
      cycle: int,
      task: scoring.R2OrchTask,
      *,
      n_hidden: int = 3,
      n_visible: int = 2,
  ) -> TaskArmPlan:
    """Plans `(hidden, visible)` contexts and candidate card `k` for `task`."""
    scope = scoring.task_scope(task)
    retrieved = self.retrieve(scope, task.prompt)
    contrasts = self.windowed_contrasts(cycle - 1)
    retired_map = {
        card_key(c): c
        for c in self.cards.get(scope, {}).values()
        if c.status == "retired"
    }
    audit_key = audit_choice(
        list(retired_map),
        self.audit_share,
        ledger_rng(self.base_seed, cycle, "audit", task.task_id),
    )
    if audit_key is not None:
      chosen = retired_map[audit_key]
      hid_cards, vis_cards, is_audit = retrieved, [*retrieved, chosen], True
    elif retrieved:
      cand_map = {card_key(c): contrasts.get(card_key(c), ()) for c in retrieved}
      picked_key, _ = thompson_pick(
          cand_map, ledger_rng(self.base_seed, cycle, "plan", task.task_id)
      )
      chosen = next(c for c in retrieved if card_key(c) == picked_key)
      hid_cards = [c for c in retrieved if c.skill_id != chosen.skill_id]
      vis_cards, is_audit = retrieved, False
    else:
      total = n_hidden + n_visible
      return TaskArmPlan(
          task_id=task.task_id,
          scope=scope,
          arms=("off",) * total,
          hidden_context="",
          visible_context="",
          arm_card=None,
          arm_card_key="",
          is_audit=False,
      )
    arms = arm_assignment(cycle, task.task_id, n_hidden, siblings=n_hidden + n_visible)
    return TaskArmPlan(
        task_id=task.task_id,
        scope=scope,
        arms=arms,
        hidden_context=render_skill_context(hid_cards),
        visible_context=render_skill_context(vis_cards),
        arm_card=chosen,
        arm_card_key=card_key(chosen),
        is_audit=is_audit,
    )

  def apply_card_verdict(self, card: SkillCard, verdict: str) -> str | None:
    """Applies `'retire'`/`'admit'` to `card` and returns event key if changed."""
    if card.status in ("active", "trial") and verdict == "retire":
      card.status = "retired"
      self.counters["retirements"] += 1
      return "retired"
    if card.status == "trial" and verdict == "admit":
      card.status = "active"
      self.counters["admissions"] += 1
      return "admitted"
    if card.status == "retired" and verdict == "admit":
      card.status = "active"
      self.counters["reinstatements"] += 1
      return "reinstated"
    return None

  def apply_ledger_decisions(self, cycle: int) -> dict[str, int]:
    """Updates card statuses (`retire`, `admit`, `reinstate`) after `cycle`."""
    contrasts = self.windowed_contrasts(cycle)
    events = {"retired": 0, "admitted": 0, "reinstated": 0}
    all_cards = [c for bucket in self.cards.values() for c in bucket.values()]
    for card in all_cards:
      ckey = card_key(card)
      vals = contrasts.get(ckey, ())
      if not vals:
        continue
      q = (
          self.reinstate_quantile
          if card.status == "retired"
          else self.admit_quantile
      )
      post = effect_posterior(
          vals,
          draws=self.bootstrap_draws,
          rng=ledger_rng(self.base_seed, cycle, "posterior", ckey),
          admit_quantile=q,
      )
      verdict = decide(post, min_tasks=self.min_tasks, epsilon=self.epsilon)
      changed = self.apply_card_verdict(card, verdict)
      if changed is not None:
        events[changed] += 1
    return events

  def update_cycle_emas(
      self,
      hidden_by_scope: Mapping[str, list[float]],
      durations_by_dataset: Mapping[str, list[float]],
  ) -> None:
    """Updates exponential moving averages for scope priors and task durations."""
    decay = self.scope_prior_decay
    for scope, hits in hidden_by_scope.items():
      if hits:
        rate = float(sum(hits) / len(hits))
        prev = self.scope_priors.get(scope)
        self.scope_priors[scope] = (
            rate if prev is None else decay * prev + (1.0 - decay) * rate
        )
    for dataset, secs in durations_by_dataset.items():
      if secs:
        mean_s = float(sum(secs) / len(secs))
        prev_s = self.dataset_durations.get(dataset)
        self.dataset_durations[dataset] = (
            mean_s if prev_s is None else decay * prev_s + (1.0 - decay) * mean_s
        )

  def record_cycle(
      self,
      cycle: int,
      task_groups: Sequence[
          tuple[scoring.R2OrchTask, TaskArmPlan, Sequence[workflow.R2OrchOutcome]]
      ],
  ) -> dict[str, object]:
    """Appends cycle contrasts, updates EMAs, and runs Bayesian decisions."""
    self.records = [r for r in self.records if int(r["cycle"]) != cycle]  # pytype: disable=wrong-arg-types
    new_contrasts: list[float] = []
    audits = 0
    hidden_by_scope: dict[str, list[float]] = collections.defaultdict(list)
    durations_by_dataset: dict[str, list[float]] = collections.defaultdict(list)
    for task, plan, outs in task_groups:
      pos_durs = [o.duration_s for o in outs if o.duration_s > 0]
      if pos_durs:
        durations_by_dataset[task.dataset].append(sum(pos_durs) / len(pos_durs))
      hid = [o.correctness for o in outs if o.skill_arm == "hidden"]
      vis = [o.correctness for o in outs if o.skill_arm == "visible"]
      hidden_by_scope[plan.scope].extend(hid)
      if plan.arm_card is None or not hid or not vis:
        continue
      z_val = float(sum(vis) / len(vis) - sum(hid) / len(hid))
      new_contrasts.append(z_val)
      audits += int(plan.is_audit)
      self.records.append({
          "cycle": cycle,
          "task_id": task.task_id,
          "scope": plan.scope,
          "card_key": plan.arm_card_key,
          "skill_id": plan.arm_card.skill_id,
          "n_h": len(hid),
          "n_v": len(vis),
          "s_h": float(sum(hid)),
          "s_v": float(sum(vis)),
          "z": z_val,
          "audit": plan.is_audit,
      })
    self.update_cycle_emas(hidden_by_scope, durations_by_dataset)
    events = self.apply_ledger_decisions(cycle)
    return {
        **self.summary(),
        "cycle_records": len(new_contrasts),
        "cycle_audits": audits,
        "mean_contrast_z": (
            float(sum(new_contrasts) / len(new_contrasts))
            if new_contrasts
            else 0.0
        ),
        **events,
    }

  def scope_summary(self, scope: str) -> dict[str, int]:
    """Returns active/trial card count and approximate token budget for `scope`."""
    live = [
        c
        for c in self.cards.get(scope, {}).values()
        if c.status in ("active", "trial")
    ]
    chars = sum(len(render_skill_context([c])) for c in live)
    return {"card_count": len(live), "card_tokens": chars // 4}

  def summary(self) -> dict[str, int]:
    """Aggregate counts across all scopes for monitoring and W&B telemetry."""
    all_cards = [c for bucket in self.cards.values() for c in bucket.values()]
    return {
        "active_cards": sum(c.status == "active" for c in all_cards),
        "trial_cards": sum(c.status == "trial" for c in all_cards),
        "retired_cards": sum(c.status == "retired" for c in all_cards),
        "seed_active_cards": sum(
            c.is_seed and c.status == "active" for c in all_cards
        ),
        "distilled_live_cards": sum(
            (not c.is_seed) and c.status in ("active", "trial")
            for c in all_cards
        ),
        **{f"{k}_total": v for k, v in self.counters.items()},
    }

  def state_dict(self) -> dict[str, object]:
    """Serializes bank cards, contrasts, EMAs, and counters for checkpointing."""
    return {
        "cards": {
            scope: [c.to_dict() for c in bucket.values()]
            for scope, bucket in self.cards.items()
        },
        "records": list(self.records),
        "scope_priors": dict(self.scope_priors),
        "dataset_durations": dict(self.dataset_durations),
        "counters": dict(self.counters),
    }

  def load_state_dict(self, payload: Mapping[str, object]) -> None:
    """Restores bank cards, contrasts, EMAs, and counters from `state_dict`."""
    raw_cards = payload.get("cards")
    if isinstance(raw_cards, Mapping):
      self.cards = {
          str(scope): {
              c.skill_id: c for c in (SkillCard.from_dict(x) for x in items)
          }
          for scope, items in raw_cards.items()
      }
    self.records = [dict(r) for r in (payload.get("records") or ())]  # pytype: disable=wrong-arg-types
    self.scope_priors = {
        str(k): float(v)
        for k, v in dict(payload.get("scope_priors") or {}).items()  # pytype: disable=wrong-arg-types
    }
    self.dataset_durations = {
        str(k): float(v)
        for k, v in dict(payload.get("dataset_durations") or {}).items()  # pytype: disable=wrong-arg-types
    }
    for k, v in dict(payload.get("counters") or {}).items():  # pytype: disable=wrong-arg-types
      if k in self.counters:
        self.counters[str(k)] = int(v)
