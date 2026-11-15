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

"""Frozen-model PATS REVISE/DISTILL using the pinned source's edit contract.

The transport is Tunix native sampling. Prompt packing, alias normalization,
rule checking, deduplication, token capacity and versioned edits execute the
original functions. Evidence contains public process traces and correctness,
never the grader's private targets.
"""

import dataclasses
import json

import jax
from examples.r2orch_math.source_adapter import load_source_module
from examples.swarm_math.train import NativeMathSampler
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import runtime


def _active_records(ledger):
  records = []
  for key, skill in sorted(ledger.skills.items()):
    if skill["status"] not in ("trial", "active"):
      continue
    card = json.loads(skill["content"])
    card["skill_id"] = skill["skill_id"]
    records.append({
        "card": card,
        "version": skill["version"],
        "status": "active",
        "required_tools": [],
        "excluded_task_types": [],
        "ledger_key": key,
    })
  return records


def _publish(ledger, before, after):
  """Archive rewritten/deleted versions; retired versions remain audit eligible."""
  identities = {(row["card"]["skill_id"], row["version"]) for row in after}
  for row in before:
    if (row["card"]["skill_id"], row["version"]) not in identities:
      ledger.skills[row["ledger_key"]]["status"] = "archived"
  old = {(row["card"]["skill_id"], row["version"]) for row in before}
  added = []
  for row in after:
    if (row["card"]["skill_id"], row["version"]) not in old:
      added.append(
          ledger.register(
              row["card"]["skill_id"],
              version=row["version"],
              content=json.dumps(row["card"], sort_keys=True),
          )
      )
  return added


def retrieve(
    ledger,
    problem,
    *,
    embedder,
    tokenizer,
    source,
    top_k=3,
    prompt_token_budget=3072,
):
  """Use the original E5 text, positive-score cutoff and rendered token budget."""
  module = load_source_module(source, "skill_store")
  records = _active_records(ledger)
  bank = module.DirectorSkillBank(
      {
          "schema": module.SCHEMA,
          "snapshot_id": "cycle-frozen",
          "cards": records,
      },
      embedder=embedder,
      top_k=top_k,
      prompt_token_budget=prompt_token_budget,
  )
  selected, _, _ = bank.select_context(
      problem, task_type="math", tokenizer=tokenizer, tools=("python",)
  )
  keys = {row["card"]["skill_id"]: row["ledger_key"] for row in records}
  return [keys[card.skill_id] for card in selected]


def render(ledger, keys, *, source):
  module = load_source_module(source, "skill_store")
  cards = load_source_module(source, "skills")
  entries = []
  for key in keys:
    skill = ledger.skills[key]
    card = json.loads(skill["content"])
    card["skill_id"] = skill["skill_id"]
    entries.append((cards.SkillCard.from_dict(card), skill["version"]))
  return module.render_skill_context(entries)


async def review(
    ledger,
    groups,
    previous_evidence,
    *,
    frozen_sampler,
    tokenizer,
    config,
    cycle,
    policy_version,
    source,
):
  """Run the original two editor stages and return persistent evidence + audit."""
  pats = load_source_module(source, "pats")
  refiner = load_source_module(source, "pats_refiner")
  scope = pats.resolve_scope("math", {"dataset": "aime"})
  pats_config = pats.PatsConfig(
      enabled=True,
      max_skills=config["max_skills"],
      max_tokens=config.get("skill_max_tokens", 4000),
      max_reviews_per_cycle=3,
      max_review_input_tokens=min(
          20480,
          config.get("reviewer_max_prompt_length", config["max_prompt_length"])
          - 64,
      ),
  )
  pats_config.validate()
  evidence = [
      e
      for e in previous_evidence
      if cycle - e["step"] <= pats_config.max_policy_lag
  ]
  # Each task is one evidence group even if two candidate lists selected it.
  by_task = {}
  for task_id, episodes in groups:
    by_task.setdefault(task_id, []).extend(episodes)
  for task_id, episodes in sorted(by_task.items()):
    outcomes = [e.reward for e in episodes]
    evidence.append({
        "id": f"cycle/{cycle}/{task_id}",
        "step": cycle,
        "scope": scope,
        "task_id": task_id,
        "policy_snapshot": f"director:{policy_version}",
        "reward_mean": sum(outcomes) / len(outcomes),
        "all_failed": not any(outcomes),
        "attempts": [
            {
                "episode_id": e.episode_id,
                "reward": e.reward,
                "scaffold_condition": e.scaffold_condition,
                "calls": [
                    {
                        "role": c.role,
                        "policy_id": c.policy_id,
                        "policy_version": c.policy_version,
                        "text": c.text,
                    }
                    for c in e.calls
                ],
            }
            for e in episodes
        ],
    })
  evidence.sort(key=lambda e: (-e["step"], e["id"]))
  evidence = evidence[: pats_config.max_evidence_groups]
  count_tokens = lambda text: len(
      tokenizer.encode(text, add_special_tokens=False)
  )
  encode = lambda messages: tokenizer.apply_chat_template(
      messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
  )
  if isinstance(frozen_sampler, NativeMathSampler):
    frozen_sampler.sampler._rng = jax.random.PRNGKey(
        config["seed"] + 100000 + cycle
    )
  reviewer = runtime.RecordingRuntime(
      frozen_sampler,
      encode,
      sampler_lib.SamplingParams(
          max_tokens=config["reviewer_max_tokens"],
          temperature=0.0,
          top_p=None,
          seed=config["seed"] + 100000 + cycle,
      ),
      policy_id="worker",
      policy_version=0,
      episode_id=f"review/{cycle}",
      max_prompt_length=config.get(
          "reviewer_max_prompt_length", config["max_prompt_length"]
      ),
  )
  reports = []
  for stage in ("REVISE", "DISTILL"):
    records = _active_records(ledger)
    supplied = (
        evidence
        if stage == "REVISE"
        else [e for e in evidence if e["all_failed"]]
    )
    report = {"stage": stage, "status": "insufficient_evidence", "calls": 0}
    reports.append(report)
    if len({e["task_id"] for e in supplied}) < pats_config.min_groups:
      continue
    editable, target = None, None
    if stage == "REVISE":
      eligible = []
      for row in records:
        summary = ledger.summary(row["ledger_key"], cycle=cycle)
        if summary["n"]:
          eligible.append((summary["mean"], row["ledger_key"], row))
      if not eligible:
        report["status"] = "no_revision_candidate"
        continue
      target = min(eligible, key=lambda item: item[:2])[2]
      editable = {target["card"]["skill_id"]}
    elif (
        len(records) >= pats_config.max_skills
        or count_tokens(pats.render_cards(records)) >= pats_config.max_tokens
    ):
      report["status"] = "scaffold_full"
      continue
    try:
      messages, admitted, audit = refiner.build_review_messages(
          scope=scope,
          editor_step=stage,
          policy_snapshot=f"director:{policy_version}",
          record={"cards": records, "evidence": supplied},
          config=pats_config,
          token_counter=count_tokens,
          extra={"revise_target": target},
      )
      if len(encode(messages)) > reviewer.max_prompt_length:
        report["status"] = "context_budget"
        continue
      call = await reviewer.generate("skill_reviewer", messages)
      report.update(calls=1, trace=dataclasses.asdict(call), request=audit)
      payload = refiner.normalize_evidence_references(
          refiner.parse_review_response(call.text), audit["evidence_aliases"]
      )
      revised, edit_audit = pats.apply_operations(
          records,
          payload,
          editor_step=stage,
          evidence=admitted,
          config=pats_config,
          token_counter=count_tokens,
          scope=scope,
          step=cycle,
          editable=editable,
      )
      report.update(
          status="reviewed",
          added=_publish(ledger, records, revised),
          edits=edit_audit,
      )
    except (ValueError, KeyError, TypeError) as exc:
      report.update(status="rejected", error=str(exc)[:300])
  return {
      "status": "reviewed",
      "calls": sum(r["calls"] for r in reports),
      "stages": reports,
  }, evidence
