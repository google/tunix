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

"""Bounded, auditable graph editing with frozen Workers.

This is an independently implemented Director-only adaptation. In particular,
ConsiderRelation inserts a directed handoff; upstream uses an additional
relation-choice protocol. Workers execute changed nodes and their downstream dependency closure after
each accepted edit. The full upstream repair controller and constrained binary
relation head remain separate source-adapter requirements. No generated Python
is executed.
"""

import asyncio
from collections.abc import Callable
import dataclasses
import json
import re

from tunix.dist_rl.swarm import trace
from tunix.dist_rl.swarm import runtime as runtime_lib

ACTIONS = (
    "add_agent",
    "set_prompt",
    "set_model",
    "set_layer",
    "consider_relation",
    "delete_agent",
    "set_output",
    "finish",
)
INSTRUCTIONS = """You are the Director of a mathematical team. Return exactly one JSON
object per turn. Edit a directed acyclic graph. Frozen Workers execute configured
nodes after edits; their artifacts appear in the next State. Editing an upstream
node refreshes its downstream dependents. finish submits the selected output. Available actions and fields:
{"action":"add_agent"}
{"action":"set_prompt","target":"agent_0","role":"solver","objective":"Solve the problem","scope":"whole task","expected_output":"An explanation and boxed answer"}
{"action":"set_model","target":"agent_0","runtime_route":"worker"}
{"action":"set_layer","target":"agent_0","layer":0}
{"action":"consider_relation","source":"agent_0","target":"agent_1"}
{"action":"delete_agent","target":"agent_0"}
{"action":"set_output","target":"agent_0"}
{"action":"finish"}
Agents receive the problem and their predecessors' outputs. Every agent needs a
structured prompt. Edges must point to a strictly higher layer. Select one output
agent. Use the available call budget economically. Do not solve the problem in
the action channel. Do not put reference answers into prompts.
"""


@dataclasses.dataclass(frozen=True)
class MathTask:
  task_id: str
  problem: str
  answer: str
  benchmark: str = "aime"


def grade_aime(text: str, answer: str) -> bool:
  """Conservative boxed/scalar subset of the upstream final-answer parser."""
  text = text.strip()
  if re.fullmatch(r"[0-9]{1,3}", text):
    values = [text]
  else:
    values = re.findall(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}", text)
    if len(re.findall(r"\\boxed", text)) != len(values):
      return False
  return (
      bool(values)
      and len({int(x) for x in values}) == 1
      and int(values[0]) == int(answer)
  )


def _unique(pairs):
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError("Duplicate action field.")
    result[key] = value
  return result


class Graph:
  """A small graph with transactional edits and explicit resource bounds."""

  def __init__(self, max_agents=6):
    if max_agents < 1:
      raise ValueError("max_agents must be positive.")
    self.max_agents = max_agents
    self.nodes = {}
    self.edges = set()
    self.output = None
    self.next_id = 0

  def state(self):
    return {
        "nodes": self.nodes,
        "edges": sorted(self.edges),
        "output": self.output,
    }

  def apply(self, text):
    action = json.loads(text, object_pairs_hook=_unique)
    if not isinstance(action, dict) or action.get("action") not in ACTIONS:
      raise ValueError("Expected one supported action object.")
    name = action["action"]
    fields = {
        "add_agent": (),
        "set_prompt": (
            "target",
            "role",
            "objective",
            "scope",
            "expected_output",
        ),
        "set_model": ("target", "runtime_route"),
        "set_layer": ("target", "layer"),
        "consider_relation": ("source", "target"),
        "delete_agent": ("target",),
        "set_output": ("target",),
        "finish": (),
    }
    if set(action) != {"action", *fields[name]}:
      raise ValueError("Action fields do not match the advertised schema.")
    target = action.get("target")
    if name == "add_agent":
      if len(self.nodes) >= self.max_agents:
        raise ValueError("Agent budget exhausted.")
      key = f"agent_{self.next_id}"
      self.nodes[key] = {"prompt": None, "layer": 0, "runtime_route": "worker"}
      self.next_id += 1
      return f"Created {key}."
    if name == "finish":
      if self.output not in self.nodes:
        raise ValueError("Select an output agent before finishing.")
      if any(node["prompt"] is None for node in self.nodes.values()):
        raise ValueError("Every agent requires a prompt.")
      return "finished"
    if target not in self.nodes:
      raise ValueError("Unknown target agent.")
    if name == "set_prompt":
      fields = ("role", "objective", "scope", "expected_output")
      if any(
          not isinstance(action.get(k), str) or not action[k].strip()
          for k in fields
      ):
        raise ValueError(
            "SetPrompt requires role, objective, scope and expected_output."
        )
      self.nodes[target]["prompt"] = {k: action[k] for k in fields}
    elif name == "set_model":
      if action.get("runtime_route") != "worker":
        raise ValueError("Only the frozen worker route is available.")
    elif name == "set_layer":
      layer = action.get("layer")
      if type(layer) is not int or not 0 <= layer < self.max_agents:
        raise ValueError("Layer must be an integer within the graph budget.")
      layers = {key: node["layer"] for key, node in self.nodes.items()}
      layers[target] = layer
      if any(layers[a] >= layers[b] for a, b in self.edges):
        raise ValueError("Layer edit would reverse an existing edge.")
      self.nodes[target]["layer"] = layer
    elif name == "consider_relation":
      source = action.get("source")
      if (
          source not in self.nodes
          or self.nodes[source]["layer"] >= self.nodes[target]["layer"]
      ):
        raise ValueError("Handoffs must point from lower to higher layers.")
      self.edges.add((source, target))
    elif name == "delete_agent":
      del self.nodes[target]
      self.edges = {(a, b) for a, b in self.edges if target not in (a, b)}
      if self.output == target:
        self.output = None
    elif name == "set_output":
      self.output = target
    return f"Applied {name}."


@dataclasses.dataclass(frozen=True)
class Result:
  episode: trace.SwarmEpisode
  graph: dict
  termination: str
  correct: bool


class DirectorWorkflow:

  def __init__(
      self,
      *,
      max_agents=6,
      max_rounds=24,
      parallel_workers=4,
      max_total_tokens=240000,
      grader: Callable = grade_aime,
      incremental_workers: bool = True,
  ):
    if min(max_agents, max_rounds, parallel_workers, max_total_tokens) < 1:
      raise ValueError("All workflow budgets must be positive.")
    self.max_agents, self.max_rounds = max_agents, max_rounds
    self.parallel_workers = parallel_workers
    self.max_total_tokens = max_total_tokens
    self.grader = grader
    self.incremental_workers = incremental_workers

  async def run(
      self,
      task,
      director,
      worker,
      *,
      episode_id,
      skills: tuple[str, ...] = (),
      scaffold_condition: str = "none",
  ):
    if not isinstance(skills, tuple) or any(
        not isinstance(s, str) for s in skills
    ):
      raise ValueError("Skills must be an immutable tuple of text cards.")
    graph, history, calls = Graph(self.max_agents), [], []
    outputs, signatures, revisions, node_runtimes = {}, {}, {}, {}
    termination, finished = "action_budget", False
    semaphore = asyncio.Semaphore(self.parallel_workers)

    def used_tokens():
      return sum(len(c.prompt_ids) + len(c.completion_ids) for c in calls)

    def worker_prompt(key):
      node = graph.nodes[key]
      predecessors = {a: outputs[a] for a, b in sorted(graph.edges) if b == key}
      prompt = (
          "Solve this task according to your assigned role.\nProblem:\n"
          + task.problem
      )
      prompt += "\nRole:\n" + json.dumps(node["prompt"])
      prompt += "\nPredecessor outputs:\n" + json.dumps(predecessors)
      prompt += "\nEnd with one unambiguous boxed final answer."
      return prompt

    def signature(key):
      return json.dumps(
          {
              "node": graph.nodes[key],
              # Artifact revisions, rather than text alone, invalidate descendants
              # even when an upstream recomputation happens to repeat its answer.
              "predecessors": [
                  (a, revisions[a]) for a, b in sorted(graph.edges) if b == key
              ],
          },
          sort_keys=True,
      )

    async def run_node(key, prompt):
      async with semaphore:
        if key not in node_runtimes:
          node_runtimes[key] = runtime_lib.RecordingRuntime(
              worker.sampler,
              worker.encode,
              dataclasses.replace(
                  worker.params,
                  seed=None
                  if worker.params.seed is None
                  else worker.params.seed
                  + int(key.split("_")[-1]) * self.max_rounds,
              ),
              policy_id=worker.policy_id,
              policy_version=worker.policy_version,
              episode_id=f"{worker.episode_id}/{key}",
              max_prompt_length=worker.max_prompt_length,
          )
        return key, await node_runtimes[key].generate(key, prompt)

    async def execute_changed():
      """Execute only changed configured nodes, then their dirty closure."""
      emitted = {}
      for key in tuple(outputs):
        if key not in graph.nodes:
          outputs.pop(key)
          signatures.pop(key, None)
          revisions.pop(key, None)
      for layer in sorted({node["layer"] for node in graph.nodes.values()}):
        keys = [
            key
            for key, node in sorted(graph.nodes.items())
            if node["layer"] == layer
            and node["prompt"] is not None
            and all(a in outputs for a, b in graph.edges if b == key)
        ]
        changed = [key for key in keys if signatures.get(key) != signature(key)]
        if not changed:
          continue
        prompts = [worker_prompt(key) for key in changed]
        lengths = [len(worker.encode(prompt)) for prompt in prompts]
        if any(length > worker.max_prompt_length for length in lengths):
          return emitted, "context_budget"
        reserve = sum(lengths) + len(changed) * worker.params.max_tokens
        if used_tokens() + reserve > self.max_total_tokens:
          return emitted, "token_budget"
        # Each layer reserves its full potential charge before any request.
        tasks = [
            asyncio.create_task(run_node(key, prompt))
            for key, prompt in zip(changed, prompts, strict=True)
        ]
        try:
          results = await asyncio.gather(*tasks)
        except BaseException:
          for pending in tasks:
            pending.cancel()
          await asyncio.gather(*tasks, return_exceptions=True)
          raise
        for key, call in results:
          calls.append(call)
          signatures[key] = signature(key)
          revisions[key] = revisions.get(key, 0) + 1
          outputs[key] = call.text
          emitted[key] = {"text": call.text, "revision": revisions[key]}
      return emitted, None

    for turn in range(self.max_rounds):
      prompt = INSTRUCTIONS
      if skills:
        prompt += "\nOptional Orchestration Knowledge:\n" + json.dumps(skills)
      prompt += "\nProblem:\n" + task.problem + "\nState:\n"
      state = {
          **graph.state(),
          "artifacts": {
              key: {"text": text, "revision": revisions[key]}
              for key, text in sorted(outputs.items())
          },
      }
      prompt += json.dumps(state, sort_keys=True)
      prompt += "\nHistory:\n" + json.dumps(history)
      prompt += f"\nRemaining edits: {self.max_rounds - turn}"
      prompt_length = len(director.encode(prompt))
      if (
          prompt_length > director.max_prompt_length
          or used_tokens() + prompt_length + director.params.max_tokens
          > self.max_total_tokens
      ):
        if not calls:
          raise ValueError(
              "The initial Director prompt cannot fit the configured budget."
          )
        termination = (
            "context_budget"
            if prompt_length > director.max_prompt_length
            else "token_budget"
        )
        break
      # Ground-truth answers enter only the grader, after all model calls.
      call = await director.generate("director", prompt)
      calls.append(call)
      accepted = True
      try:
        feedback = graph.apply(call.text)
      except (ValueError, TypeError) as exc:
        calls[-1] = dataclasses.replace(call, valid_action=False)
        feedback, accepted = str(exc), False
      event = {"action": call.text, "feedback": feedback}
      if accepted and (self.incremental_workers or feedback == "finished"):
        emitted, limit = await execute_changed()
        event["worker_outputs"] = emitted
        if limit is not None:
          termination = limit
          history.append(event)
          break
      history.append(event)
      if feedback == "finished":
        finished = graph.output in outputs
        if not finished:
          termination = "missing_output"
        break
    final = outputs.get(graph.output, "") if finished else ""
    correct = bool(final) and self.grader(final, task.answer)
    worker_calls = [
        call for call in calls if call.policy_id == worker.policy_id
    ]
    cost = (
        sum(
            0.25 * len(c.prompt_ids) + len(c.completion_ids)
            for c in worker_calls
        )
        / 11900.0
    )
    episode = trace.SwarmEpisode(
        task_id=task.task_id,
        episode_id=episode_id,
        calls=tuple(calls),
        reward=float(correct),
        final_text=final,
        worker_cost=cost,
        scaffold_condition=scaffold_condition,
    )
    return Result(
        episode,
        {
            **graph.state(),
            "artifacts": outputs,
            "artifact_revisions": revisions,
        },
        "finished" if finished else termination,
        bool(correct),
    )
