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

"""Execute the actual pinned source controller against an explicit test sampler."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
from examples.r2orch_math import source_workflow
from examples.r2orch_math import workflow
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import runtime


@pytest.fixture
def source():
  value = os.environ.get("TUNIX_R2ORCH_SOURCE")
  if not value:
    pytest.skip(
        "Set TUNIX_R2ORCH_SOURCE to execute the pinned original Canvas."
    )
  return Path(value)


class Tokenizer:

  def __init__(self):
    self.messages = []

  def apply_chat_template(self, messages, **kwargs):
    del kwargs
    self.messages.append(messages)
    return [1, 2, len(self.messages) + 2]

  def encode(self, text, **kwargs):
    del kwargs
    return [{"off": 8, "on": 9}[text]]

  def decode(self, ids, **kwargs):
    del kwargs
    if len(ids) > 1:
      return "<|im_start|>assistant\n"
    return {8: "off", 9: "on"}.get(ids[0], "fixture action")


class ScriptedSampler:

  def __init__(self, responses):
    self.responses = iter(responses)
    self.requests = []

  async def sample(self, request):
    self.requests.append(request)
    await asyncio.sleep(0)
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.array(request.prompt),
        token_ids=np.array([4]),
        logprobs=np.array([-0.5]),
        text=next(self.responses),
        finish_reason="stop",
    )


def make_runtime(sampler, policy_id):
  return runtime.RecordingRuntime(
      sampler,
      lambda _: [1],
      sampler_lib.SamplingParams(
          max_tokens=4096,
          temperature=1.0 if policy_id == "director" else 0.3,
          top_p=1.0,
          seed=7,
      ),
      policy_id=policy_id,
      policy_version=2 if policy_id == "director" else 0,
      episode_id="e/" + policy_id,
      max_prompt_length=8192,
  )


async def binary(prompt, support, seed):
  assert prompt and support == (8, 9) and seed >= 7
  return np.log([1e-20, 1.0])


def prompt_action(target, result_scope="task_result"):
  return {
      "action": "set_prompt",
      "target": target,
      "role": "solver",
      "objective": "Solve the complete mathematics problem",
      "scope": "whole task",
      "expected_output": "A justified integer answer",
      "result_scope": result_scope,
  }


def test_original_canvas_mandatory_configuration_and_submission(source):
  async def run():
    tokenizer = Tokenizer()
    director = ScriptedSampler(
        map(
            json.dumps,
            [
                {"action": "add_agent"},
                # This action must be rejected by the source's mandatory prompt state.
                {"action": "add_agent"},
                prompt_action("agent_1"),
                {
                    "action": "set_model",
                    "target": "agent_1",
                    "runtime_route": "worker",
                },
                {"action": "finish", "target": "agent_1"},
            ],
        )
    )
    worker = ScriptedSampler(["The answer is \\boxed{2}."] * 8)
    flow = source_workflow.SourceDirectorWorkflow(
        source,
        tokenizer=tokenizer,
        binary_sampler=binary,
        grader=workflow.grade_aime,
    )
    result = await flow.run(
        workflow.MathTask("t", "Compute 1+1.", "2"),
        make_runtime(director, "director"),
        make_runtime(worker, "worker"),
        episode_id="e",
        skills=("VISIBLE_ORCHESTRATION_SKILL",),
        scaffold_condition="on",
    )
    assert result.correct and result.termination == "finished"
    assert result.graph["executor"] == "original_canvas"
    assert result.graph["director_context_mode"] == "append_only"
    assert result.graph["submission_receipt"] is not None
    assert len(result.graph["nodes"]) == 1
    assert any(not step["accepted"] for step in result.graph["history"])
    assert result.episode.scaffold_condition == "on"
    calls = result.episode.calls
    assert len({c.call_id for c in calls}) == len(calls)
    assert {c.policy_version for c in calls if c.policy_id == "director"} == {2}
    assert {c.policy_version for c in calls if c.policy_id == "worker"} == {0}
    worker_inputs = [
        tokenizer.messages[int(r.prompt[-1]) - 3] for r in worker.requests
    ]
    assert all(
        "VISIBLE_ORCHESTRATION_SKILL" not in json.dumps(m)
        for m in worker_inputs
    )
    assert result.episode.worker_cost > 0

  asyncio.run(run())


def test_original_relation_uses_recorded_restricted_distribution(source):
  async def run():
    tokenizer = Tokenizer()
    actions = [
        {"action": "add_agent"},
        prompt_action("agent_1", "subtask"),
        {"action": "set_model", "target": "agent_1", "runtime_route": "worker"},
        {"action": "add_agent"},
        prompt_action("agent_2"),
        {"action": "set_model", "target": "agent_2", "runtime_route": "worker"},
        {
            "action": "consider_relation",
            "source": "agent_1",
            "target": "agent_2",
        },
        {"action": "finish", "target": "agent_2"},
    ]
    director = ScriptedSampler(map(json.dumps, actions))
    worker = ScriptedSampler(["The answer is \\boxed{2}."] * 20)
    flow = source_workflow.SourceDirectorWorkflow(
        source,
        tokenizer=tokenizer,
        binary_sampler=binary,
        grader=workflow.grade_aime,
    )
    result = await flow.run(
        workflow.MathTask("t", "Compute 1+1.", "2"),
        make_runtime(director, "director"),
        make_runtime(worker, "worker"),
        episode_id="e",
    )
    selected = [
        c for c in result.episode.calls if c.action_support_ids is not None
    ]
    assert len(selected) == 1
    assert selected[0].action_support_ids == (8, 9)
    assert selected[0].completion_ids == (9,)
    assert selected[0].rollout_logprobs == (0.0,)
    assert any(
        t.get("action_diagnostics", {}).get("binary_policy_audit")
        for t in result.graph["turns"]
    )
    assert result.correct

  asyncio.run(run())


def test_cancellation_drains_thread_and_pending_sampler(source):
  async def run():
    started, cancelled = asyncio.Event(), asyncio.Event()

    class HangingSampler:

      async def sample(self, request):
        del request
        started.set()
        try:
          await asyncio.Event().wait()
        finally:
          cancelled.set()

    flow = source_workflow.SourceDirectorWorkflow(
        source,
        tokenizer=Tokenizer(),
        binary_sampler=binary,
        grader=workflow.grade_aime,
    )
    task = asyncio.create_task(
        flow.run(
            workflow.MathTask("t", "Compute 1+1.", "2"),
            make_runtime(HangingSampler(), "director"),
            make_runtime(ScriptedSampler([]), "worker"),
            episode_id="e",
        )
    )
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, 3)
    assert cancelled.is_set()

  asyncio.run(run())


def test_worker_transport_failure_never_becomes_a_zero_reward(source):
  async def run():
    class FailedWorker:

      async def sample(self, request):
        del request
        # The source Canvas catches ValueError as an execution failure, so
        # this exercises reward admission rather than exception propagation.
        raise ValueError("transport returned invalid response provenance")

    actions = [
        {"action": "add_agent"},
        prompt_action("agent_1"),
        {"action": "set_model", "target": "agent_1", "runtime_route": "worker"},
    ]
    flow = source_workflow.SourceDirectorWorkflow(
        source,
        tokenizer=Tokenizer(),
        binary_sampler=binary,
        grader=workflow.grade_aime,
    )
    with pytest.raises(source_workflow.SourceOutcomeUnavailable) as caught:
      await flow.run(
          workflow.MathTask("t", "Compute 1+1.", "2"),
          make_runtime(ScriptedSampler(map(json.dumps, actions)), "director"),
          make_runtime(FailedWorker(), "worker"),
          episode_id="e",
      )
    assert caught.value.calls
    assert any(
        step["rejection_code"] == "execution_failure"
        for step in caught.value.graph["history"]
    )
    assert caught.value.graph["typed_policy_failure"] is None

  asyncio.run(run())
