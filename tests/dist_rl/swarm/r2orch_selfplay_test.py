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

"""Real alternating learner updates and atomic outer-cycle recovery."""

import asyncio
import dataclasses
import json

import numpy as np
import pytest
from examples.r2orch_math import cpu_fixture
from examples.r2orch_math import rsi
from examples.r2orch_math import selfplay
from examples.r2orch_math import train
from tunix.dist_rl.swarm import trace


def inputs():
  config = json.loads((train.ROOT / "configs/selfplay_cpu.json").read_text())
  return (
      config,
      selfplay.fixture_tasks(),
      train.load_tasks(train.ROOT / "data/cpu_eval.jsonl"),
  )


def parameters(path):
  with np.load(path / "parameters.npz") as data:
    return {name: data[name].copy() for name in data.files}


def test_two_native_learners_update_and_resume_whole_cycle(tmp_path):
  config, tasks, evaluation = inputs()
  full = selfplay.run("cpu", config, tasks, evaluation, tmp_path / "full")
  first = selfplay.run(
      "cpu", {**config, "num_steps": 1}, tasks, evaluation, tmp_path / "resume"
  )
  resumed = selfplay.run(
      "cpu", config, tasks, evaluation, tmp_path / "resume", resume=True
  )
  assert first["director_optimizer_steps"] == 2
  assert first["proposer_optimizer_steps"] == 1
  assert (
      full["director_optimizer_steps"]
      == resumed["director_optimizer_steps"]
      == 4
  )
  assert (
      full["proposer_optimizer_steps"]
      == resumed["proposer_optimizer_steps"]
      == 2
  )
  assert (
      full["frozen_parameters_unchanged"]
      and resumed["frozen_parameters_unchanged"]
  )
  assert full["history"] == resumed["history"]
  a, b = parameters(tmp_path / "full"), parameters(tmp_path / "resume")
  for name in a:
    np.testing.assert_array_equal(a[name], b[name])
  assert np.linalg.norm(a["proposer"]) > 0.01
  for cycle in full["history"]:
    assert any(
        any(value != 0 for value in pair)
        for pair in cycle["proposer_advantages"]
    )
    assert all(
        len(set(item["selected_ids"])) == 2 for item in cycle["selections"]
    )
  assert not np.array_equal(a["director"], a["frozen"])
  committed = (tmp_path / "resume/checkpoints/CURRENT.json").read_bytes()
  locked_test = [
      dataclasses.replace(
          evaluation[0], task_id="locked/test", problem="What is 3 - 1?"
      )
  ]
  measured = selfplay.run(
      "cpu",
      config,
      tasks,
      locked_test,
      tmp_path / "resume",
      resume=True,
      mode="eval",
  )
  assert measured["mode"] == "eval"
  assert measured["director_optimizer_steps"] == 4
  assert measured["proposer_optimizer_steps"] == 2
  assert (
      tmp_path / "resume/checkpoints/CURRENT.json"
  ).read_bytes() == committed
  with pytest.raises(ValueError, match="development split"):
    selfplay.run(
        "cpu", config, tasks, locked_test, tmp_path / "resume", resume=True
    )
  with pytest.raises(ValueError, match="identity"):
    selfplay.run(
        "cpu",
        {**config, "proposer_learning_rate": 0.3},
        tasks,
        evaluation,
        tmp_path / "resume",
        resume=True,
    )


@pytest.mark.parametrize(
    "phase",
    ["after_director_update", "after_director_checkpoint", "before_commit"],
)
def test_failure_does_not_commit_one_learner_or_advance_curriculum(
    tmp_path, phase
):
  config, tasks, evaluation = inputs()
  config["num_steps"] = 1

  def fault(observed, cycle):
    if observed == phase and cycle == 1:
      raise RuntimeError("Injected incomplete cycle")

  interrupted = tmp_path / "interrupted"
  with pytest.raises(RuntimeError, match="Injected"):
    selfplay.run("cpu", config, tasks, evaluation, interrupted, fault=fault)
  pointer = json.loads((interrupted / "checkpoints/CURRENT.json").read_text())
  state = json.loads(
      (
          interrupted / "checkpoints" / pointer["directory"] / "state.json"
      ).read_text()
  )
  assert state["cycle"] == 0
  assert state["optimizer_steps"] == {"director": 0, "proposer": 0}
  assert state["curriculum"]["observations"] == {}
  resumed = selfplay.run(
      "cpu", config, tasks, evaluation, interrupted, resume=True
  )
  full = selfplay.run("cpu", config, tasks, evaluation, tmp_path / "full")
  assert resumed["history"] == full["history"]
  a, b = parameters(interrupted), parameters(tmp_path / "full")
  for name in a:
    np.testing.assert_array_equal(a[name], b[name])


def test_frozen_model_distills_two_distinct_failure_traces():
  config, _, _ = inputs()
  frozen = cpu_fixture.make_policy()
  before = np.asarray(frozen.logits[...]).copy()
  ledger = rsi.SkillLedger()

  def episode(task):
    return trace.SwarmEpisode(
        task_id=task,
        episode_id=task,
        calls=(
            trace.CallTrace(
                call_id=task + "/call",
                role="director",
                policy_id="director",
                policy_version=0,
                prompt_ids=(1,),
                completion_ids=(8,),
                rollout_logprobs=(-0.7,),
                loss_mask=(1,),
                text="A solution was submitted without independent checking.",
            ),
        ),
        reward=0.0,
        final_text="Incorrect attempt",
    )

  groups = [("a", [episode("a")]), ("b", [episode("b")])]
  result = asyncio.run(
      selfplay._review(
          ledger,
          groups,
          cpu_fixture.TinySampler(frozen),
          cpu_fixture.encode,
          config,
          0,
          "cpu",
          None,
      )
  )
  assert result["status"] == "reviewed" and result["calls"] == 1
  assert len(result["added"]) == 1
  assert ledger.skills[result["added"][0]]["status"] == "trial"
  np.testing.assert_array_equal(frozen.logits[...], before)
  duplicate = asyncio.run(
      selfplay._review(
          ledger,
          [groups[0], groups[0]],
          cpu_fixture.TinySampler(frozen),
          cpu_fixture.encode,
          config,
          1,
          "cpu",
          None,
      )
  )
  assert duplicate["status"] == "insufficient_evidence"


def test_retired_skills_are_not_always_reintroduced():
  ledger = rsi.SkillLedger(rsi.LedgerConfig(audit_share=0.0))
  key = ledger.register("k", version=1, content="Optional prior skill")
  ledger.skills[key]["status"] = "retired"
  assert ledger.choose([key], cycle=1, task_id="a") == (None, False)


def test_selfplay_rejects_test_task_contamination(tmp_path):
  config, tasks, evaluation = inputs()
  overlap = dataclasses.replace(
      evaluation[0], task_id="different-id", problem=tasks[0].problem
  )
  with pytest.raises(ValueError, match="disjoint"):
    selfplay.run("cpu", config, tasks, [overlap], tmp_path)


def test_active_retrieval_is_not_crowded_out_by_retired_cards():
  ledger = rsi.SkillLedger()
  retired = ledger.register("a-retired", version=1, content="Old hint")
  active = ledger.register("z-active", version=1, content="Useful hint")
  ledger.skills[retired]["status"] = "retired"
  assert selfplay._skills_for(ledger, "Problem", 1) == [active]


def test_pinned_source_curriculum_and_reviewer_contract(tmp_path):
  import os
  from pathlib import Path
  from transformers import AutoTokenizer
  from examples.r2orch_math import prepare_data
  from tunix.dist_rl.rollout import sampler

  source = os.environ.get("TUNIX_R2ORCH_SOURCE")
  tokenizer_path = os.environ.get("TUNIX_SWARM_QWEN_TOKENIZER")
  if not source or not tokenizer_path:
    pytest.skip("Set pinned source and tokenizer environment variables.")
  prepare_data.prepare(source, tmp_path / "data")
  rows = [
      json.loads(line)
      for line in (tmp_path / "data/train.jsonl").read_text().splitlines()
  ]
  tasks = train.load_tasks(tmp_path / "data/train.jsonl")
  config = json.loads(
      (train.ROOT / "configs/selfplay_qwen3_4b.json").read_text()
  )
  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_path, local_files_only=True
  )
  curriculum = selfplay.CandidateCurriculum(
      tasks,
      config,
      source=source,
      rows=rows,
      tokenizer=tokenizer,
      output=tmp_path,
  )
  lists = []
  for ids, messages in curriculum.lists(0, 1):
    assert len(ids) == len(set(ids)) == 8
    assert set(ids) <= {t.task_id for t in tasks}
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert 0 < len(tokens) < config["max_prompt_length"]
    assert len(set(curriculum.builder.label_token_ids.values())) == 8
    assert all(
        c["scope_cards"] == 1
        for c in json.loads(messages[1]["content"])["candidates"]
    )
    curriculum.reserve(ids[:2])
    lists.append((ids, messages))
  assert len(lists) == config["lists_per_cycle"]
  curriculum.record([(lists[0][0][0], [1, 0, 1, 0, 1])])
  snapshot = json.loads(json.dumps(curriculum.state()))
  before = list(curriculum.lists(1, 1))
  curriculum.restore(snapshot)
  assert list(curriculum.lists(1, 1)) == before

  card = {
      k: "Verify the computation before submission"
      for k in (
          "name",
          "description",
          "trigger",
          "plan",
          "pitfall",
          "constraint",
      )
  }
  card["kind"] = "verification"

  class ScriptedSampler:

    async def sample(self, request):
      assert request.sampling_params.temperature == 0.0
      assert request.sampling_params.top_p is None
      return sampler.SamplingResponse(
          request_id=request.request_id,
          text=json.dumps({
              "operations": [
                  {"op": "ADD", "card": card, "evidence_ids": ["E1", "E2"]}
              ]
          }),
          prompt_token_ids=np.asarray(request.prompt, np.int32),
          token_ids=np.asarray([42]),
          logprobs=np.asarray([-0.1]),
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
                text="Failed calculation",
            ),
        ),
        reward=0,
        final_text="Failed",
    )

  ledger = rsi.SkillLedger()
  result = asyncio.run(
      selfplay._review(
          ledger,
          [("a", [episode("a")]), ("b", [episode("b")])],
          ScriptedSampler(),
          lambda _: [1, 2],
          config,
          0,
          "qwen",
          source,
      )
  )
  assert result["status"] == "reviewed" and len(result["added"]) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("proposer_learning_rate", float("nan")),
        ("proposer_gap_clip", 0),
        ("time_exponent", float("inf")),
        ("hidden_siblings", 2),
    ],
)
def test_invalid_selfplay_configuration_fails_before_model_loading(
    tmp_path, key, value
):
  config, tasks, evaluation = inputs()
  config[key] = value
  with pytest.raises(ValueError):
    selfplay.run("cpu", config, tasks, evaluation, tmp_path)


def test_too_few_pairs_skip_proposer_but_commit_curriculum_and_global_role_step(
    tmp_path,
):
  config, tasks, evaluation = inputs()
  config.update(min_proposer_pair_samples=4, num_steps=1)
  report = selfplay.run("cpu", config, tasks, evaluation, tmp_path / "run")
  assert report["proposer_optimizer_steps"] == 0
  assert report["director_optimizer_steps"] == 2
  assert report["history"][0]["global_role_step"] == 1
  config["num_steps"] = 2
  resumed = selfplay.run(
      "cpu", config, tasks, evaluation, tmp_path / "run", resume=True
  )
  assert resumed["cycles"] == 2 and resumed["proposer_optimizer_steps"] == 0
  assert resumed["history"][-1]["global_role_step"] == 2
  pointer = json.loads((tmp_path / "run/checkpoints/CURRENT.json").read_text())
  state = json.loads(
      (
          tmp_path / "run/checkpoints" / pointer["directory"] / "state.json"
      ).read_text()
  )
  assert state["cycle"] == 2 and state["optimizer_steps"]["proposer"] == 0
  assert state["curriculum"]["observations"]
