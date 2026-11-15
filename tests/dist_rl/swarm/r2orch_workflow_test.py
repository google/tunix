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

"""Graph validity, bounded execution and frozen-call provenance."""

import asyncio
import json

import numpy as np
import pytest
from examples.r2orch_math import workflow
from tunix.dist_rl.rollout import sampler
from tunix.dist_rl.swarm import runtime


def edit(graph, action, **kwargs):
  return graph.apply(json.dumps({"action": action, **kwargs}))


def configure(graph, key):
  edit(
      graph,
      "set_prompt",
      target=key,
      role="solver",
      objective="Solve",
      scope="whole task",
      expected_output="boxed integer",
  )


def test_graph_rejects_reverse_edges_transactionally_and_deletes_dependents():
  graph = workflow.Graph(2)
  edit(graph, "add_agent")
  edit(graph, "add_agent")
  for key in graph.nodes:
    configure(graph, key)
  edit(graph, "set_layer", target="agent_1", layer=1)
  edit(graph, "consider_relation", source="agent_0", target="agent_1")
  edit(graph, "set_output", target="agent_1")
  with pytest.raises(ValueError, match="reverse"):
    edit(graph, "set_layer", target="agent_1", layer=0)
  assert graph.nodes["agent_1"]["layer"] == 1
  with pytest.raises(ValueError, match="lower"):
    edit(graph, "consider_relation", source="agent_1", target="agent_0")
  assert edit(graph, "finish") == "finished"
  edit(graph, "delete_agent", target="agent_1")
  assert not graph.edges and graph.output is None
  with pytest.raises(ValueError, match="output"):
    edit(graph, "finish")


def test_parser_rejects_multiple_actions_duplicate_fields_and_bad_routes():
  graph = workflow.Graph(1)
  with pytest.raises(ValueError):
    graph.apply('{"action":"add_agent"}{"action":"finish"}')
  with pytest.raises(ValueError, match="Duplicate"):
    graph.apply('{"action":"add_agent","action":"finish"}')
  with pytest.raises(ValueError, match="schema"):
    graph.apply('{"action":"add_agent","unexpected":"argument"}')
  edit(graph, "add_agent")
  with pytest.raises(ValueError, match="frozen"):
    edit(graph, "set_model", target="agent_0", runtime_route="director")
  with pytest.raises(ValueError, match="budget"):
    edit(graph, "add_agent")


class Scripted:

  def __init__(self, texts):
    self.texts, self.requests = iter(texts), []
    self.inflight, self.maximum_inflight = 0, 0

  async def sample(self, request):
    self.requests.append(request)
    text = next(self.texts)
    self.inflight += 1
    self.maximum_inflight = max(self.maximum_inflight, self.inflight)
    await asyncio.sleep(0)
    self.inflight -= 1
    return sampler.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt),
        token_ids=np.array([4, 0]),
        logprobs=np.array([-1.0, -2.0]),
        text=text,
        finish_reason="stop",
    )


def make_runtime(backend, policy_id, observed):
  def encode(prompt):
    observed.append(prompt)
    return [1, 0, 2]

  return runtime.RecordingRuntime(
      backend,
      encode,
      sampler.SamplingParams(max_tokens=2, seed=42),
      policy_id=policy_id,
      policy_version=3 if policy_id == "director" else 0,
      episode_id=f"episode/{policy_id}",
      max_prompt_length=8,
  )


def test_parallel_nodes_have_unique_ids_and_do_not_receive_reference_answer():
  actions = []
  for i in range(2):
    actions.extend([
        {"action": "add_agent"},
        {
            "action": "set_prompt",
            "target": f"agent_{i}",
            "role": "solver",
            "objective": "Solve",
            "scope": "task",
            "expected_output": "boxed integer",
        },
    ])
  actions.extend(
      [{"action": "set_output", "target": "agent_1"}, {"action": "finish"}]
  )
  director = Scripted([json.dumps(a) for a in actions])
  workers = Scripted([r"\boxed{2}", r"\boxed{731}"])
  seen = []
  result = asyncio.run(
      workflow.DirectorWorkflow(incremental_workers=False).run(
          workflow.MathTask(
              "task", "An unspecified mathematical problem.", "731"
          ),
          make_runtime(director, "director", seen),
          make_runtime(workers, "worker", seen),
          episode_id="episode",
      )
  )
  assert result.correct and result.episode.reward == 1
  assert workers.maximum_inflight == 2
  assert len({c.call_id for c in result.episode.calls}) == 8
  assert {
      c.policy_version for c in result.episode.calls if c.policy_id == "worker"
  } == {0}
  assert all("731" not in prompt for prompt in seen)
  assert len(result.episode.calls) == 8


def test_token_budget_stops_before_next_call_and_never_fabricates_success():
  director = Scripted(['{"action":"add_agent"}'] * 3)
  workers = Scripted([])
  result = asyncio.run(
      workflow.DirectorWorkflow(max_total_tokens=7).run(
          workflow.MathTask("task", "Compute 1+1.", "2"),
          make_runtime(director, "director", []),
          make_runtime(workers, "worker", []),
          episode_id="episode",
      )
  )
  assert result.termination == "token_budget"
  assert len(director.requests) == 1 and not workers.requests
  assert result.episode.reward == 0 and not result.episode.final_text


def test_invalid_action_is_retained_in_training_trace():
  director = Scripted(["not json"])
  result = asyncio.run(
      workflow.DirectorWorkflow(max_rounds=1).run(
          workflow.MathTask("task", "Compute 1+1.", "2"),
          make_runtime(director, "director", []),
          make_runtime(Scripted([]), "worker", []),
          episode_id="episode",
      )
  )
  assert not result.episode.calls[0].valid_action
  assert result.episode.calls[0].completion_ids == (4, 0)
  assert result.episode.calls[0].loss_mask == (1, 1)


def _prompt_action(target, objective="Solve"):
  return {
      "action": "set_prompt",
      "target": target,
      "role": "solver",
      "objective": objective,
      "scope": "task",
      "expected_output": "boxed integer",
  }


def test_incremental_feedback_and_transitive_artifact_invalidation():
  actions = [
      {"action": "add_agent"},
      _prompt_action("agent_0"),
      {"action": "add_agent"},
      _prompt_action("agent_1"),
      {"action": "set_layer", "target": "agent_1", "layer": 1},
      {"action": "consider_relation", "source": "agent_0", "target": "agent_1"},
      {"action": "set_output", "target": "agent_1"},
      _prompt_action("agent_0", "Recheck the arithmetic"),
      {"action": "finish"},
  ]
  director = Scripted([json.dumps(a) for a in actions])
  workers = Scripted([
      "upstream initial",
      "independent",
      "new layer",
      "combined",
      "upstream revised",
      r"\boxed{731}",
  ])
  director_seen, worker_seen = [], []
  result = asyncio.run(
      workflow.DirectorWorkflow().run(
          workflow.MathTask("t", "Public mathematical problem", "731"),
          make_runtime(director, "director", director_seen),
          make_runtime(workers, "worker", worker_seen),
          episode_id="e",
          skills=("Always recheck arithmetic",),
          scaffold_condition="on",
      )
  )
  assert result.correct and result.episode.scaffold_condition == "on"
  assert len(workers.requests) == 6
  assert result.graph["artifact_revisions"] == {"agent_0": 2, "agent_1": 4}
  assert any("upstream initial" in prompt for prompt in director_seen)
  assert any("upstream revised" in prompt for prompt in worker_seen)
  assert all("Always recheck arithmetic" in prompt for prompt in director_seen)
  assert all(
      "Always recheck arithmetic" not in prompt for prompt in worker_seen
  )
  calls = result.episode.calls
  assert len({c.call_id for c in calls}) == len(calls)
  # Every recomputation is charged, rather than only the last surviving output.
  assert result.episode.worker_cost == pytest.approx(6 * (0.25 * 3 + 2) / 11900)


def test_same_output_revision_still_invalidates_downstream():
  actions = [
      {"action": "add_agent"},
      _prompt_action("agent_0"),
      {"action": "add_agent"},
      _prompt_action("agent_1"),
      {"action": "set_layer", "target": "agent_1", "layer": 1},
      {"action": "consider_relation", "source": "agent_0", "target": "agent_1"},
      _prompt_action("agent_0", "Different reasoning, same conclusion"),
      {"action": "set_output", "target": "agent_1"},
      {"action": "finish"},
  ]
  workers = Scripted([r"\boxed{2}"] * 6)
  result = asyncio.run(
      workflow.DirectorWorkflow().run(
          workflow.MathTask("t", "Compute 1+1", "2"),
          make_runtime(
              Scripted([json.dumps(a) for a in actions]), "director", []
          ),
          make_runtime(workers, "worker", []),
          episode_id="e",
      )
  )
  assert result.correct and len(workers.requests) == 6
  assert result.graph["artifact_revisions"]["agent_1"] == 4


def test_deleting_upstream_node_recomputes_surviving_consumer():
  actions = [
      {"action": "add_agent"},
      _prompt_action("agent_0"),
      {"action": "add_agent"},
      _prompt_action("agent_1"),
      {"action": "set_layer", "target": "agent_1", "layer": 1},
      {"action": "consider_relation", "source": "agent_0", "target": "agent_1"},
      {"action": "set_output", "target": "agent_1"},
      {"action": "delete_agent", "target": "agent_0"},
      {"action": "finish"},
  ]
  workers = Scripted([r"\boxed{2}"] * 5)
  result = asyncio.run(
      workflow.DirectorWorkflow().run(
          workflow.MathTask("t", "Compute 1+1", "2"),
          make_runtime(
              Scripted([json.dumps(a) for a in actions]), "director", []
          ),
          make_runtime(workers, "worker", []),
          episode_id="e",
      )
  )
  assert result.correct and len(workers.requests) == 5
  assert "agent_0" not in result.graph["artifacts"]
  assert result.graph["artifact_revisions"] == {"agent_1": 4}
