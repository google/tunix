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

"""Unit and CPU integration tests for SW3 R2Orch (Director + Frozen Workers)."""

import asyncio
import json

import numpy as np
import pytest
from examples.swarm_r2orch import canvas as canvas_lib
from examples.swarm_r2orch import scoring
from examples.swarm_r2orch import train as r2orch_train
from examples.swarm_r2orch import workflow
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import runtime as runtime_lib
from tunix.dist_rl.swarm import trace


class _ScriptedSampler:
  """Deterministic in-memory fake sampler for protocol and concurrency tests."""

  def __init__(self, reply_fn):
    self._reply_fn = reply_fn

  async def sample(self, request: sampler_lib.SamplingRequest):
    await asyncio.sleep(0)
    text, tok_ids = self._reply_fn(request)
    lps = (
        np.full(len(tok_ids), -0.5, dtype=np.float32)
        if request.sampling_params.return_logprobs
        else None
    )
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt, dtype=np.int32),
        token_ids=np.asarray(tok_ids, dtype=np.int32),
        logprobs=lps,
        text=text,
        finish_reason="stop",
    )


def test_frozen_calls_excluded_from_credit_and_enforce_zero_loss_mask():
  with pytest.raises(ValueError, match="all-zero loss mask"):
    trace.CallTrace(
        call_id="bad/0",
        role="Frozen Worker",
        policy_id="worker",
        policy_version=0,
        prompt_ids=(1, 2),
        completion_ids=(3, 4),
        rollout_logprobs=(),
        loss_mask=(1, 0),
        text="x",
        trainable=False,
    )

  episodes = []
  for idx, reward in enumerate((1.0, 0.0)):
    d_call = trace.CallTrace(
        call_id=f"ep{idx}/0",
        role="Graph Director",
        policy_id="director",
        policy_version=3,
        prompt_ids=(1, 2),
        completion_ids=(5, 6),
        rollout_logprobs=(-0.4, -0.6),
        loss_mask=(1, 1),
        text='{"action":"finish","target":"agent_1"}',
        trainable=True,
    )
    w_call = trace.CallTrace(
        call_id=f"ep{idx}/1",
        role="Frozen Worker",
        policy_id="worker",
        policy_version=0,
        prompt_ids=(1, 2, 3),
        completion_ids=(7, 8, 9),
        rollout_logprobs=(),
        loss_mask=(0, 0, 0),
        text="Answer: 42",
        trainable=False,
    )
    episodes.append(
        trace.SwarmEpisode(
            task_id="t1",
            episode_id=f"ep{idx}",
            calls=(d_call, w_call),
            reward=reward,
            final_text="Answer: 42",
        )
    )

  batch = credit.build_credit(episodes, normalization="episode")
  assert len(batch.payloads) == 2
  assert batch.roles == ("Graph Director", "Graph Director")
  assert all(p.metadata["policy_id"] == "director" for p in batch.payloads)
  assert batch.advantages[0] > 0 and batch.advantages[1] < 0


def test_concurrent_from_bindings_assigns_unique_call_ids():
  d_sampler = _ScriptedSampler(lambda _: ('{"action":"run_graph"}', [2, 0]))
  w_sampler = _ScriptedSampler(lambda req: (f"out:{req.request_id}", [3, 4, 0]))
  rt = runtime_lib.RecordingRuntime.from_bindings(
      {
          "director": runtime_lib.PolicyBinding(
              policy_id="dir",
              policy_version=2,
              sampler=d_sampler,
              encode=lambda s: [1, len(s) % 5 + 1],
              sampling_params=sampler_lib.SamplingParams(
                  max_tokens=4, seed=10, temperature=1.0
              ),
              max_prompt_length=16,
              trainable=True,
          ),
          "worker": runtime_lib.PolicyBinding(
              policy_id="wrk",
              policy_version=0,
              sampler=w_sampler,
              encode=lambda s: [1, len(s) % 5 + 1],
              sampling_params=sampler_lib.SamplingParams(
                  max_tokens=4, seed=20, temperature=0.3, return_logprobs=False
              ),
              max_prompt_length=16,
              trainable=False,
          ),
      },
      episode_id="ep_conc",
  )

  async def _gather_calls():
    d = await rt.generate("director", "plan")
    w1, w2, w3 = await asyncio.gather(
        rt.generate("worker", "subtask a"),
        rt.generate("worker", "subtask b"),
        rt.generate("worker", "subtask c"),
    )
    return d, w1, w2, w3

  d, w1, w2, w3 = asyncio.run(_gather_calls())
  assert [c.call_id for c in (d, w1, w2, w3)] == [
      "ep_conc/0",
      "ep_conc/1",
      "ep_conc/2",
      "ep_conc/3",
  ]
  assert d.trainable and d.loss_mask == (1, 1)
  assert not w1.trainable and w1.loss_mask == (0, 0, 0)
  assert w1.rollout_logprobs == ()


def test_multi_domain_verifiers_and_worker_cost():
  aime_task = scoring.R2OrchTask(
      "aime/1", "aime", "integer_answer", "p", answers=("200",)
  )
  assert scoring.score_task(aime_task, r"Step 1... \boxed{200}") == 1.0
  assert scoring.score_task(aime_task, "Maybe 200 or 201") == 0.0

  ood_math = scoring.R2OrchTask(
      "math_hard/1", "math_hard", "math_symbolic", "p", answers=("-6.5",)
  )
  assert scoring.score_task(ood_math, r"Thus \boxed{-\frac{13}{2}}.") == 1.0

  qa_task = scoring.R2OrchTask(
      "hotpot/1",
      "hotpotqa",
      "qa_exact_match",
      "p",
      answers=("March 24, 2009",),
  )
  assert scoring.score_task(qa_task, "Reasoning...\nAnswer: march 24 2009.") == 1.0

  yes_task = scoring.R2OrchTask(
      "hotpot/2", "hotpotqa", "qa_exact_match", "p", answers=("yes",)
  )
  assert scoring.score_task(yes_task, "Answer: Yes, both are directors.") == 1.0
  assert scoring.score_task(yes_task, "Answer: No, they differ.") == 0.0

  mcq_task = scoring.R2OrchTask(
      "medqa/1",
      "medqa",
      "letter_choice",
      "p",
      answers=("C",),
      options=("A", "B", "C", "D"),
  )
  assert scoring.score_task(mcq_task, " exclude A and B. Answer: (C)") == 1.0
  assert scoring.score_task(mcq_task, "patient has a fever") == 0.0

  mbpp_task = scoring.R2OrchTask(
      "mbpp/1",
      "mbppplus",
      "mbppplus_tests",
      "Write double_nums(xs).",
      code_spec={
          "test_list": ["assert double_nums([1, 2]) == [2, 4]"],
          "test_imports": [],
          "plus_test": (
              "assert double_nums([1, 2, -3]) == [2, 4, -6]\n"
              "assert double_nums([]) == []\n"
          ),
      },
  )
  good_code = (
      "Here is the solution:\n```python\n"
      "def double_nums(xs):\n  return [x * 2 for x in xs]\n"
      "print(double_nums([1]))\n```"
  )
  early_exit = (
      "```python\nimport os\ndef double_nums(xs):\n  os._exit(0)\n```"
  )
  assert scoring.score_task(mbpp_task, good_code) == 1.0
  assert scoring.score_task(mbpp_task, early_exit) == 0.0

  # Reference solve on AIME (`[14000, 8400]`) has normalized cost exactly 1.0.
  assert scoring.compute_worker_cost(14000, 8400, "aime") == pytest.approx(1.0)


def test_canvas_parallel_layer_guardrails_and_workflow_cost_reward():
  canvas = canvas_lib.GraphCanvas(max_agents=4, relay_max_chars=200)

  async def fake_worker(node, upstream):
    up_ids = ",".join(a.agent_id for a in upstream)
    return canvas_lib.WorkerArtifact(
        agent_id=node.agent_id,
        answer=f"ans({node.agent_id};up={up_ids})",
        summary=f"sum({node.agent_id};up={up_ids})",
        token_in=100,
        token_out=50,
    )

  async def _exercise_canvas():
    leak = await canvas.step(
        '{"action":"add_agent","role":"S","objective":"Output \\\\boxed{5}"}',
        fake_worker,
    )
    assert not leak.accepted
    s1 = await canvas.step(
        '{"action":"add_agent","role":"ExtractorA","objective":"Hop 1","layer":0,"result_scope":"subtask","defer_execution":true}',
        fake_worker,
    )
    assert s1.accepted and s1.executed_agents == ()
    s2 = await canvas.step(
        '{"action":"add_agent","role":"ExtractorB","objective":"Hop 2","layer":0,"result_scope":"subtask"}',
        fake_worker,
    )
    assert s2.accepted and s2.executed_agents == ("agent_1", "agent_2")
    assert canvas.had_parallel_wave
    # Finish must reject disconnected nodes before synthesis connects them.
    bad_fin = await canvas.step(
        '{"action":"finish","target":"agent_1"}', fake_worker
    )
    assert not bad_fin.accepted
    s3 = await canvas.step(
        '{"action":"add_agent","role":"Synthesizer","objective":"Combine hops","layer":1,"inputs":["agent_1","agent_2"],"result_scope":"task_result"}',
        fake_worker,
    )
    assert s3.accepted and s3.executed_agents == ("agent_3",)
    assert canvas.artifacts["agent_3"].summary == "sum(agent_3;up=agent_1,agent_2)"
    fin = await canvas.step(
        '{"action":"finish","target":"agent_3"}', fake_worker
    )
    assert fin.accepted and fin.finished
    assert canvas.submitted_answer() == "ans(agent_3;up=agent_1,agent_2)"

  asyncio.run(_exercise_canvas())


def test_two_step_cpu_training_updates_director_and_keeps_worker_frozen(tmp_path):
  cfg = json.loads(
      (r2orch_train.ROOT / "configs/cpu_smoke.json").read_text(encoding="utf-8")
  )
  train_tasks = r2orch_train.load_tasks(
      r2orch_train.ROOT / "data/cpu_train.jsonl"
  )
  eval_tasks = r2orch_train.load_tasks(
      r2orch_train.ROOT / "data/cpu_eval.jsonl"
  )
  out_dir = tmp_path / "run"
  report = r2orch_train.run(
      "cpu", cfg, train_tasks, eval_tasks, out_dir
  )
  assert report["director_norm_delta_l2"] > 0.0
  assert report["frozen_worker_max_norm_drift"] == 0.0
  assert len(report["history"]) == 2
  assert all(h["optimizer_applied"] for h in report["history"])
  assert all(h["frozen_calls"] > 0 for h in report["history"])

  eval_out = tmp_path / "eval_only"
  eval_report = r2orch_train.run(
      "cpu",
      cfg,
      train_tasks,
      eval_tasks,
      eval_out,
      checkpoint_dir=str(out_dir / "checkpoints"),
      mode="eval",
      restore_step=2,
  )
  assert eval_report["restored_step"] == 2
  assert eval_report["frozen_worker_max_norm_drift"] == 0.0
