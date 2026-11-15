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

"""Pinned PATS edit validation, versioning and evidence admission tests."""

import asyncio
import json
import os

import numpy as np
import pytest
from examples.r2orch_math import rsi
from examples.r2orch_math import skill_review
from examples.r2orch_math import train
from tunix.dist_rl.rollout import sampler
from tunix.dist_rl.swarm import trace


def test_source_revise_then_distill_archives_old_version_and_checks_rules(
    tmp_path,
):
  source = os.environ.get("TUNIX_R2ORCH_SOURCE")
  tokenizer_path = os.environ.get("TUNIX_SWARM_QWEN_TOKENIZER")
  if not source or not tokenizer_path:
    pytest.skip("Set pinned source and tokenizer environment variables.")
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_path, local_files_only=True
  )
  cfg = json.loads((train.ROOT / "configs/selfplay_qwen3_4b.json").read_text())
  ledger = rsi.SkillLedger()
  card = {
      "name": "Check signs",
      "description": "Check the sign of the derived quantity.",
      "trigger": "A derivation introduces subtraction.",
      "plan": "Ask a verifier to check every sign.",
      "pitfall": "A copied derivation may repeat its sign error.",
      "constraint": "Keep the verifier independent.",
      "kind": "verification",
  }
  key = ledger.register("check-signs", version=1, content=json.dumps(card))
  ledger.record(
      cycle=0,
      task_id="old-task",
      skill_key=key,
      hidden=[0, 1, 0],
      visible=[1, 1],
  )
  ledger.skills[key]["status"] = "active"

  class Embedder:
    model_path = "fixture-source-retrieval"

    def __init__(self):
      self.passages = []

    def encode(self, texts, *, query):
      if not query:
        self.passages.extend(texts)
      return [(1.0, 0.0)] * len(texts)

  embedder = Embedder()
  assert skill_review.retrieve(
      ledger,
      "Check a signed value",
      embedder=embedder,
      tokenizer=tokenizer,
      source=source,
  ) == [key]
  assert embedder.passages == [
      f"{card['description']} || {card['trigger']} || {card['plan']}"
  ]
  assert (
      skill_review.retrieve(
          ledger,
          "Check a signed value",
          embedder=embedder,
          tokenizer=tokenizer,
          source=source,
          prompt_token_budget=1,
      )
      == []
  )
  assert "[check-signs@1]" in skill_review.render(ledger, [key], source=source)
  revised = {
      **card,
      "plan": "Ask an independent verifier to recompute the signed quantity.",
  }
  added = {
      "name": "Reduce case branches",
      "description": "Merge overlapping cases before exploring branches.",
      "trigger": "A counting problem produces overlapping cases.",
      "plan": "Partition cases into disjoint sets before adding agents.",
      "pitfall": "Overlapping branches can double count solutions.",
      "constraint": "Assign each branch a unique condition.",
      "kind": "decomposition",
  }

  class ScriptedSampler:

    def __init__(self):
      self.calls = 0

    async def sample(self, request):
      self.calls += 1
      assert request.sampling_params.temperature == 0.0
      assert request.sampling_params.top_p is None
      operation = (
          {"op": "UPDATE", "skill_id": "check-signs", "card": revised}
          if self.calls == 1
          else {"op": "ADD", "card": added}
      )
      text = json.dumps(
          {"operations": [{**operation, "evidence_ids": ["E1", "E2"]}]}
      )
      ids = np.asarray(
          tokenizer.encode(text, add_special_tokens=False), np.int32
      )
      return sampler.SamplingResponse(
          request_id=request.request_id,
          text=text,
          prompt_token_ids=np.asarray(request.prompt, np.int32),
          token_ids=ids,
          logprobs=np.full(len(ids), -0.1),
      )

  def episode(task_id):
    return trace.SwarmEpisode(
        task_id=task_id,
        episode_id=task_id,
        calls=(
            trace.CallTrace(
                call_id=task_id,
                role="director",
                policy_id="director",
                policy_version=0,
                prompt_ids=(1,),
                completion_ids=(2,),
                rollout_logprobs=(-0.1,),
                loss_mask=(1,),
                text="The independently computed cases did not agree.",
            ),
        ),
        reward=0,
        final_text="Failed",
    )

  groups = [(key, [episode(key)]) for key in ("task-a", "task-b")]
  transport = ScriptedSampler()
  report, evidence = asyncio.run(
      skill_review.review(
          ledger,
          groups,
          [],
          frozen_sampler=transport,
          tokenizer=tokenizer,
          config=cfg,
          cycle=1,
          policy_version=2,
          source=source,
      )
  )
  assert report["calls"] == 2
  assert [stage["status"] for stage in report["stages"]] == [
      "reviewed",
      "reviewed",
  ]
  assert all(len(stage["edits"]["applied"]) == 1 for stage in report["stages"])
  assert ledger.skills[key]["status"] == "archived"
  active = [s for s in ledger.skills.values() if s["status"] == "trial"]
  assert len(active) == 2 and any(
      s["skill_id"] == "check-signs" and s["version"] == 2 for s in active
  )
  assert len(evidence) == 2 and all("answer" not in e for e in evidence)
  ledger.save(tmp_path / "ledger.json")
  loaded = rsi.SkillLedger.load(tmp_path / "ledger.json")
  assert loaded.skills == ledger.skills
  assert key not in loaded.decide(cycle=1)


def test_archived_version_is_not_retrieved_or_audited():
  ledger = rsi.SkillLedger(rsi.LedgerConfig(audit_share=1))
  key = ledger.register("deleted", version=1, content="Deleted by the reviewer")
  ledger.skills[key]["status"] = "archived"
  assert ledger.choose([key], cycle=1, task_id="q") == (None, False)


def test_real_tiny_qwen_reviewer_is_greedy_and_keeps_frozen_parameters():
  from types import SimpleNamespace
  from flax import nnx
  import jax
  import jax.numpy as jnp
  from tokenizers import Tokenizer, models
  from transformers import PreTrainedTokenizerFast
  from examples.r2orch_math import selfplay
  from tunix.dist_rl.swarm import runtime
  from tunix.models.qwen3 import model as qwen_model

  tokenizer = PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(
          models.WordLevel(
              {
                  "[EOS]": 0,
                  "[UNK]": 1,
                  "a": 2,
                  "b": 3,
                  "c": 4,
                  "d": 5,
                  "e": 6,
                  "f": 7,
              },
              unk_token="[UNK]",
          )
      ),
      eos_token="[EOS]",
      pad_token="[EOS]",
      unk_token="[UNK]",
  )
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("fsdp", "tp")
  )
  with mesh:
    frozen = qwen_model.Qwen3(
        qwen_model.ModelConfig(
            num_layers=1,
            vocab_size=8,
            embed_dim=16,
            hidden_dim=32,
            num_heads=2,
            head_dim=8,
            num_kv_heads=1,
            norm_eps=1e-6,
            rope_theta=1000000,
        ),
        rngs=nnx.Rngs(42),
    )
    before = [
        np.asarray(x).copy()
        for x in jax.tree.leaves(nnx.state(frozen, nnx.Param))
    ]
    logits, _ = frozen(
        input_tokens=jnp.asarray([[2, 3]]),
        positions=jnp.asarray([[0, 1]]),
        cache=None,
        attention_mask=jnp.asarray([[[True, False], [True, True]]]),
    )
    expected_first = int(np.argmax(np.asarray(logits[0, -1])))
    cfg = {
        "max_prompt_length": 4,
        "reviewer_max_prompt_length": 4,
        "reviewer_max_tokens": 2,
        "rollout_batch_size": 1,
        "seed": 42,
    }
    reviewer = selfplay._reviewer_sampler(
        "qwen",
        frozen,
        SimpleNamespace(config=SimpleNamespace(eos_tokens=[0])),
        cfg,
        tokenizer,
    )
    assert reviewer.config.temperature == 0 and reviewer.config.top_p is None

    async def collect():
      calls = []
      try:
        for seed in (7, 999):
          reviewer.sampler._rng = jax.random.PRNGKey(seed)
          recorded = runtime.RecordingRuntime(
              reviewer,
              lambda _: [2, 3],
              sampler.SamplingParams(
                  max_tokens=2, temperature=0, top_p=None, seed=seed
              ),
              policy_id="worker",
              policy_version=0,
              episode_id=f"review/{seed}",
              max_prompt_length=4,
          )
          calls.append(
              await recorded.generate(
                  "skill_reviewer", "Check process evidence"
              )
          )
        return calls
      finally:
        if hasattr(reviewer, "close"):
          await reviewer.close()

    first, second = asyncio.run(collect())
    assert first.completion_ids == second.completion_ids
    assert first.completion_ids[0] == expected_first
    np.testing.assert_array_equal(
        first.rollout_logprobs, second.rollout_logprobs
    )
    assert np.isfinite(first.rollout_logprobs).all()
    after = jax.tree.leaves(nnx.state(frozen, nnx.Param))
    for original, current in zip(before, after, strict=True):
      np.testing.assert_array_equal(original, current)
