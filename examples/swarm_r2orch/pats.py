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

"""Outer-loop PATS skill revision and distillation on the frozen Worker base."""

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import re

import numpy as np
from examples.swarm_r2orch import scoring
from examples.swarm_r2orch import skill_ledger
from examples.swarm_r2orch import workflow
from tunix.dist_rl.rollout import sampler as sampler_lib

_THINK_TAG_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
_LEAK_RE = re.compile(
    r"\\boxed\s*\{|```|\bfinal\s+answer\s*[:=]|answer\s+is\b", re.IGNORECASE
)
_GENERIC_RE = re.compile(
    r"\b(?:be\s+careful|think\s+step\s+by\s+step|do\s+your\s+best)\b",
    re.IGNORECASE,
)

PATS_SYSTEM_PROMPT = (
    "You review public Director orchestration traces to maintain concise,"
    " reusable graph-orchestration skill cards. Return strictly JSON"
    ' {"operations": [...]} with no markdown prose. Each card requires fields'
    " name, description, trigger, plan, pitfall, constraint, kind (one of"
    " orchestration/decomposition/communication/verification/tool-use/bug-repair)"
    " with <= 280 chars per field and zero task-specific answers or code blocks."
    " Reference at least 2 distinct supplied evidence IDs (E1, E2, ...)."
)


def scrub_public_group(
    cycle: int,
    task: scoring.R2OrchTask,
    plan: skill_ledger.TaskArmPlan,
    outcomes: Sequence[workflow.R2OrchOutcome],
) -> dict[str, object]:
  """Builds a strictly public trace summary (never inspects gold answers/tests)."""
  hid = [o for o in outcomes if o.skill_arm == "hidden"]
  vis = [o for o in outcomes if o.skill_arm == "visible"]
  s_h = sum(o.correctness for o in hid)
  s_v = sum(o.correctness for o in vis)
  s_all = sum(o.correctness for o in outcomes)
  attempts = [
      {
          "arm": o.skill_arm or o.scaffold_condition,
          "passed": bool(o.correctness >= 1.0),
          "finished": o.finished,
          "nodes": o.num_nodes,
          "max_layer": o.max_layer,
          "parallel": o.parallel_wave,
          "actions": list(o.action_log[:4]),
      }
      for o in outcomes[:4]
  ]
  return {
      "cycle": cycle,
      "task_id": task.task_id,
      "scope": plan.scope,
      "prompt_preview": task.prompt[:360],
      "arm_card_id": plan.arm_card.skill_id if plan.arm_card else "",
      "arm_card_key": plan.arm_card_key,
      "both_fail": bool(outcomes and s_all == 0.0),
      "z": (
          float(s_v / len(vis) - s_h / len(hid))
          if hid and vis
          else 0.0
      ),
      "attempts": attempts,
  }


def validate_card_fields(
    raw: Mapping[str, object], *, field_limit: int = 280
) -> dict[str, str] | None:
  """Validates a proposed PATS card dictionary and rejects leakage/filler."""
  required = (
      "name",
      "description",
      "trigger",
      "plan",
      "pitfall",
      "constraint",
      "kind",
  )
  cleaned: dict[str, str] = {}
  for key in required:
    val = str(raw.get(key, "")).strip()
    if not val or (key != "kind" and len(val) > field_limit):
      return None
    if _LEAK_RE.search(val) or _GENERIC_RE.search(val):
      return None
    cleaned[key] = val
  return cleaned if cleaned["kind"] in skill_ledger.SKILL_KINDS else None


def parse_pats_operations(
    raw_text: str,
    evidence_task_by_alias: Mapping[str, str],
    *,
    min_distinct_tasks: int = 2,
) -> list[dict[str, object]]:
  """Parses JSON `operations` and verifies distinct-task evidence citations."""
  text = _THINK_TAG_RE.sub("", raw_text).strip()
  if text.startswith("```"):
    text = _FENCE_RE.sub("", text).strip()
  if not text.startswith("{") or not text.endswith("}"):
    return []
  try:
    payload = json.loads(text)
  except json.JSONDecodeError:
    return []
  ops = payload.get("operations") if isinstance(payload, dict) else None
  if not isinstance(ops, list):
    return []
  valid_ops: list[dict[str, object]] = []
  for item in ops[:4]:
    if not isinstance(item, dict):
      continue
    ev_ids = item.get("evidence_ids")
    if not isinstance(ev_ids, list):
      continue
    cited_tasks = {
        evidence_task_by_alias[str(eid)]
        for eid in ev_ids
        if str(eid) in evidence_task_by_alias
    }
    if len(cited_tasks) >= min_distinct_tasks:
      valid_ops.append(item)
  return valid_ops


def build_editor_prompt(
    *,
    editor_step: str,
    scope: str,
    cards: Sequence[skill_ledger.SkillCard],
    evidence_groups: Sequence[Mapping[str, object]],
    revise_target: str = "",
) -> tuple[str, dict[str, str]]:
  """Renders a bounded PATS editor request and returns `(prompt, alias_map)`."""
  aliased = []
  alias_to_task: dict[str, str] = {}
  for idx, group in enumerate(evidence_groups[:8], 1):
    alias = f"E{idx}"
    alias_to_task[alias] = str(group["task_id"])
    aliased.append({"id": alias, **group})
  body = {
      "editor_step": editor_step,
      "scope": scope,
      "revise_target": revise_target,
      "allowed_evidence_ids": list(alias_to_task),
      "cards": [c.content_dict() for c in cards],
      "evidence": aliased,
      "instruction": (
          f"UPDATE or DELETE only skill_id={revise_target!r}."
          if editor_step == "REVISE"
          else "ADD at most 2 reusable orchestration cards for recurring failures."
      ),
  }
  rendered = (
      f"{PATS_SYSTEM_PROMPT}\n\nReview Payload:\n"
      f"{json.dumps(body, ensure_ascii=False)}"
  )
  return rendered, alias_to_task


class PatsOuterReviewer:
  """Bounded outer-loop skill editor backed by the frozen Worker sampler."""

  def __init__(
      self,
      *,
      max_groups_per_scope: int = 36,
      dedup_jaccard: float = 0.70,
      min_revise_tasks: int = 4,
      max_reviews_per_cycle: int = 2,
  ):
    self.max_groups_per_scope = max_groups_per_scope
    self.dedup_jaccard = dedup_jaccard
    self.min_revise_tasks = min_revise_tasks
    self.max_reviews_per_cycle = max_reviews_per_cycle
    self.evidence_by_scope: dict[str, list[dict[str, object]]] = (
        collections.defaultdict(list)
    )
    self._next_skill_seq = 1

  def ingest_cycle(
      self,
      cycle: int,
      task_groups: Sequence[
          tuple[
              scoring.R2OrchTask,
              skill_ledger.TaskArmPlan,
              Sequence[workflow.R2OrchOutcome],
          ]
      ],
  ) -> None:
    """Scrubs and buffers public process evidence from one completed cycle."""
    for task, plan, outs in task_groups:
      bucket = self.evidence_by_scope[plan.scope]
      bucket.append(scrub_public_group(cycle, task, plan, outs))
      del bucket[: -self.max_groups_per_scope]

  def pick_revise_target(
      self, scope: str, cycle: int, scaffold: skill_ledger.SkillScaffold
  ) -> skill_ledger.SkillCard | None:
    """Finds the lowest-mean live card in `scope` with negative contrast."""
    contrasts = scaffold.windowed_contrasts(cycle)
    worst: tuple[float, skill_ledger.SkillCard] | None = None
    for card in scaffold.cards.get(scope, {}).values():
      if card.status not in ("active", "trial"):
        continue
      vals = contrasts.get(skill_ledger.card_key(card), ())
      if len(vals) < self.min_revise_tasks:
        continue
      mean_z = float(sum(vals) / len(vals))
      if mean_z < -0.02 and (worst is None or mean_z < worst[0]):
        worst = (mean_z, card)
    return worst[1] if worst is not None else None

  def apply_revise_ops(
      self,
      scope: str,
      target: skill_ledger.SkillCard,
      ops: Sequence[Mapping[str, object]],
      scaffold: skill_ledger.SkillScaffold,
  ) -> int:
    """Applies at most one validated UPDATE or DELETE to `target`."""
    for op in ops:
      kind = str(op.get("op", "")).upper()
      if str(op.get("skill_id", "")) != target.skill_id:
        continue
      if kind == "DELETE":
        target.status = "retired"
        scaffold.counters["retirements"] += 1
        return 1
      if kind == "UPDATE" and isinstance(op.get("card"), Mapping):
        fields = validate_card_fields(op["card"])  # pytype: disable=wrong-arg-types
        if fields is None:
          continue
        scaffold.cards[scope][target.skill_id] = dataclasses.replace(
            target,
            **fields,
            version=target.version + 1,
            status="trial",
        )
        scaffold.counters["revisions"] += 1
        return 1
    return 0

  def apply_distill_ops(
      self,
      scope: str,
      cycle: int,
      ops: Sequence[Mapping[str, object]],
      scaffold: skill_ledger.SkillScaffold,
  ) -> int:
    """Admits up to two deduplicated `trial` cards into `scaffold`."""
    bucket = scaffold.cards.setdefault(scope, {})
    added = 0
    for op in ops:
      if added >= 2 or str(op.get("op", "")).upper() != "ADD":
        continue
      live = [c for c in bucket.values() if c.status in ("active", "trial")]
      if len(live) >= scaffold.max_skills_per_scope:
        break
      raw_card = op.get("card")
      fields = (
          validate_card_fields(raw_card)
          if isinstance(raw_card, Mapping)
          else None
      )
      if fields is None:
        continue
      cand_sig = f"{fields['name']} {fields['trigger']} {fields['plan']}"
      if any(
          skill_ledger.token_jaccard(
              cand_sig, f"{c.name} {c.trigger} {c.plan}"
          )
          >= self.dedup_jaccard
          for c in live
      ):
        continue
      sid = f"pats_{scope}_c{cycle}_{self._next_skill_seq}"
      self._next_skill_seq += 1
      bucket[sid] = skill_ledger.SkillCard(
          skill_id=sid,
          task_types=(scope,),
          version=1,
          status="trial",
          is_seed=False,
          **fields,
      )
      scaffold.counters["distilled"] += 1
      added += 1
    return added

  async def query_frozen_reviewer(
      self,
      prompt: str,
      worker_sampler: object,
      encode_worker: Callable[[str], Sequence[int]],
      *,
      req_id: str,
      seed: int,
      max_tokens: int,
  ) -> str:
    """Queries the frozen Worker sampler with `return_logprobs=False`."""
    req = sampler_lib.SamplingRequest(
        request_id=req_id,
        prompt=np.asarray(encode_worker(prompt), dtype=np.int32),
        sampling_params=sampler_lib.SamplingParams(
            max_tokens=max_tokens,
            seed=seed,
            temperature=0.2,
            top_p=0.95,
            return_logprobs=False,
        ),
    )
    resp = await worker_sampler.sample(req)  # pytype: disable=attribute-error
    return str(resp.text)

  async def maybe_review(
      self,
      cycle: int,
      scaffold: skill_ledger.SkillScaffold,
      worker_sampler: object,
      encode_worker: Callable[[str], Sequence[int]],
      config: Mapping[str, object],
  ) -> dict[str, int]:
    """Runs bounded REVISE and DISTILL steps when `cycle` hits interval."""
    interval = int(config.get("pats_review_every_steps", 0))  # pytype: disable=wrong-arg-types
    if interval <= 0 or cycle <= 0 or cycle % interval != 0:
      return {"pats_revised": 0, "pats_distilled": 0, "pats_calls": 0}
    max_toks = int(config.get("pats_max_response_length", 512))  # pytype: disable=wrong-arg-types
    base_seed = int(config["seed"])  # pytype: disable=wrong-arg-types
    revised = 0
    distilled = 0
    calls = 0
    for scope in sorted(scaffold.cards)[: self.max_reviews_per_cycle]:
      groups = list(reversed(self.evidence_by_scope.get(scope, ())))
      live_cards = [
          c
          for c in scaffold.cards[scope].values()
          if c.status in ("active", "trial")
      ]
      target = self.pick_revise_target(scope, cycle, scaffold)
      if target is not None and len({g["task_id"] for g in groups}) >= 2:
        prompt, aliases = build_editor_prompt(
            editor_step="REVISE",
            scope=scope,
            cards=live_cards,
            evidence_groups=groups,
            revise_target=target.skill_id,
        )
        raw = await self.query_frozen_reviewer(
            prompt,
            worker_sampler,
            encode_worker,
            req_id=f"pats/revise/{cycle}/{scope}",
            seed=base_seed + cycle * 31 + calls,
            max_tokens=max_toks,
        )
        calls += 1
        ops = parse_pats_operations(raw, aliases)
        revised += self.apply_revise_ops(scope, target, ops, scaffold)
      failed = [g for g in groups if g.get("both_fail")]
      if len({g["task_id"] for g in failed}) >= 2:
        prompt, aliases = build_editor_prompt(
            editor_step="DISTILL",
            scope=scope,
            cards=live_cards,
            evidence_groups=failed,
        )
        raw = await self.query_frozen_reviewer(
            prompt,
            worker_sampler,
            encode_worker,
            req_id=f"pats/distill/{cycle}/{scope}",
            seed=base_seed + cycle * 97 + calls,
            max_tokens=max_toks,
        )
        calls += 1
        ops = parse_pats_operations(raw, aliases)
        distilled += self.apply_distill_ops(scope, cycle, ops, scaffold)
    return {
        "pats_revised": revised,
        "pats_distilled": distilled,
        "pats_calls": calls,
    }

  def state_dict(self) -> dict[str, object]:
    return {
        "next_skill_seq": self._next_skill_seq,
        "evidence_by_scope": {
            k: list(v) for k, v in self.evidence_by_scope.items()
        },
    }

  def load_state_dict(self, payload: Mapping[str, object]) -> None:
    self._next_skill_seq = int(payload.get("next_skill_seq", 1))  # pytype: disable=wrong-arg-types
    raw_ev = payload.get("evidence_by_scope")
    if isinstance(raw_ev, Mapping):
      self.evidence_by_scope = collections.defaultdict(
          list, {str(k): [dict(x) for x in v] for k, v in raw_ev.items()}
      )
