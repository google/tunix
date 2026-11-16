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

"""Layered DAG Agent Canvas and atomic JSON Director mutations for R2Orch."""

import asyncio
import collections
from collections.abc import Awaitable, Callable
import dataclasses
import json
import re

VALID_SCOPES = frozenset({"subtask", "task_result"})
VALID_ACTIONS = frozenset({
    "add_agent",
    "set_prompt",
    "set_layer",
    "add_edge",
    "consider_relation",
    "delete_agent",
    "run_graph",
    "finish",
})
_LEAKAGE_PATTERNS = (
    re.compile(r"\\boxed\s*\{"),
    re.compile(r"```"),
    re.compile(r"\bfinal\s+answer\s*[:=]", re.IGNORECASE),
)
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


@dataclasses.dataclass
class AgentNode:
  """One worker slot on the layered DAG canvas."""

  agent_id: str
  role: str = ""
  objective: str = ""
  layer: int = 0
  result_scope: str = "task_result"
  dirty: bool = True

  @property
  def configured(self) -> bool:
    """Whether this node has a non-empty role and objective."""
    return bool(self.role.strip() and self.objective.strip())


@dataclasses.dataclass(frozen=True)
class WorkerArtifact:
  """Bounded execution record produced by one frozen Worker invocation."""

  agent_id: str
  answer: str
  summary: str
  token_in: int
  token_out: int


@dataclasses.dataclass(frozen=True)
class CanvasStepResult:
  """Environment transition feedback returned to the Director after one turn."""

  accepted: bool
  finished: bool
  feedback: str
  executed_agents: tuple[str, ...] = ()


def parse_director_action(raw_text: str) -> dict[str, object] | None:
  """Parses a single JSON object from Director output; returns None if invalid."""
  cleaned = _THINK_RE.sub("", raw_text).strip()
  if cleaned.startswith("```"):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
  if not cleaned.startswith("{") or not cleaned.endswith("}"):
    return None
  try:
    parsed = json.loads(cleaned)
  except json.JSONDecodeError:
    return None
  if not isinstance(parsed, dict) or not isinstance(parsed.get("action"), str):
    return None
  action = parsed["action"].strip().lower()
  if action not in VALID_ACTIONS:
    return None
  parsed["action"] = "add_edge" if action == "consider_relation" else action
  return parsed


def _has_solution_leakage(text: str) -> bool:
  return any(pattern.search(text) for pattern in _LEAKAGE_PATTERNS)


class GraphCanvas:
  """Stateful layered DAG canvas mutated by the Director and run by Workers."""

  def __init__(
      self,
      *,
      max_agents: int = 4,
      max_layers: int = 4,
      relay_max_chars: int = 1000,
  ):
    if min(max_agents, max_layers, relay_max_chars) < 1:
      raise ValueError("Canvas budgets must be positive.")
    self.max_agents = max_agents
    self.max_layers = max_layers
    self.relay_max_chars = relay_max_chars
    self.nodes: dict[str, AgentNode] = {}
    self.edges: set[tuple[str, str]] = set()
    self.artifacts: dict[str, WorkerArtifact] = {}
    self.output_agent: str | None = None
    self.version = 0
    self.worker_token_in = 0
    self.worker_token_out = 0
    self.worker_calls = 0
    self.had_parallel_wave = False
    self._next_index = 1

  def predecessors(self, agent_id: str) -> list[str]:
    """Sorted upstream agent IDs feeding `agent_id`."""
    return sorted(src for src, dst in self.edges if dst == agent_id)

  def _mark_downstream_dirty(self, seeds: set[str]) -> None:
    queue = collections.deque(a for a in seeds if a in self.nodes)
    visited: set[str] = set()
    while queue:
      curr = queue.popleft()
      if curr in visited:
        continue
      visited.add(curr)
      self.nodes[curr].dirty = True
      for src, dst in self.edges:
        if src == curr and dst not in visited:
          queue.append(dst)

  def _all_reach_target(self, target_id: str) -> list[str]:
    """Returns retained agent IDs that have no directed path to `target_id`."""
    reverse_adj: dict[str, list[str]] = collections.defaultdict(list)
    for src, dst in self.edges:
      reverse_adj[dst].append(src)
    reachable = {target_id}
    queue = collections.deque([target_id])
    while queue:
      curr = queue.popleft()
      for prev in reverse_adj[curr]:
        if prev not in reachable:
          reachable.add(prev)
          queue.append(prev)
    return sorted(aid for aid in self.nodes if aid not in reachable)

  def _validate_delegation(
      self, role: str, objective: str, scope: str
  ) -> str | None:
    if not role or len(role) > 80:
      return "Provide a concise non-empty 'role' (<= 80 chars)."
    if not objective or len(objective) > 600:
      return "Provide a clear non-empty 'objective' (<= 600 chars)."
    if scope not in VALID_SCOPES:
      return "Field 'result_scope' must be 'subtask' or 'task_result'."
    if _has_solution_leakage(role) or _has_solution_leakage(objective):
      return (
          "Delegation rejected: describe WHAT the Worker must determine, never"
          " supply boxed answers, code blocks, or 'Final Answer:' text."
      )
    return None

  def _apply_add_agent(self, cmd: dict[str, object]) -> tuple[bool, str, bool]:
    if len(self.nodes) >= self.max_agents:
      return False, f"Agent budget ({self.max_agents}) reached.", False
    layer = int(cmd.get("layer", 0))
    if not 0 <= layer < self.max_layers:
      return False, f"Layer must be in [0, {self.max_layers - 1}].", False
    inputs = [str(x) for x in (cmd.get("inputs") or ())]
    for src in inputs:
      if src not in self.nodes:
        return False, f"Unknown input agent {src!r}.", False
      if self.nodes[src].layer >= layer:
        return (
            False,
            f"Edge {src} -> new agent requires layer({src}) < {layer}.",
            False,
        )
    role = str(cmd.get("role", "")).strip()
    objective = str(cmd.get("objective", "")).strip()
    scope = str(cmd.get("result_scope", "task_result")).strip()
    if role or objective:
      err = self._validate_delegation(role, objective, scope)
      if err is not None:
        return False, err, False
    agent_id = f"agent_{self._next_index}"
    self._next_index += 1
    self.nodes[agent_id] = AgentNode(
        agent_id=agent_id,
        role=role,
        objective=objective,
        layer=layer,
        result_scope=scope,
        dirty=True,
    )
    for src in inputs:
      self.edges.add((src, agent_id))
    self.version += 1
    defer = bool(cmd.get("defer_execution", False)) or not bool(objective)
    return True, f"Added {agent_id} at layer {layer}.", not defer

  def _apply_set_prompt(self, cmd: dict[str, object]) -> tuple[bool, str, bool]:
    target = str(cmd.get("target", ""))
    if target not in self.nodes:
      return False, f"Unknown target agent {target!r}.", False
    node = self.nodes[target]
    role = str(cmd.get("role", node.role)).strip()
    objective = str(cmd.get("objective", "")).strip()
    scope = str(cmd.get("result_scope", node.result_scope)).strip()
    err = self._validate_delegation(role, objective, scope)
    if err is not None:
      return False, err, False
    node.role = role
    node.objective = objective
    node.result_scope = scope
    self._mark_downstream_dirty({target})
    self.version += 1
    defer = bool(cmd.get("defer_execution", False))
    return True, f"Updated prompt for {target}.", not defer

  def _apply_set_layer(self, cmd: dict[str, object]) -> tuple[bool, str, bool]:
    target = str(cmd.get("target", ""))
    if target not in self.nodes:
      return False, f"Unknown target agent {target!r}.", False
    layer = int(cmd.get("layer", -1))
    if not 0 <= layer < self.max_layers:
      return False, f"Layer must be in [0, {self.max_layers - 1}].", False
    for src, dst in self.edges:
      src_layer = layer if src == target else self.nodes[src].layer
      dst_layer = layer if dst == target else self.nodes[dst].layer
      if src_layer >= dst_layer:
        return (
            False,
            f"Layer {layer} violates directed edge {src} -> {dst}.",
            False,
        )
    self.nodes[target].layer = layer
    self._mark_downstream_dirty({target})
    self.version += 1
    return True, f"Moved {target} to layer {layer}.", False

  def _apply_add_edge(self, cmd: dict[str, object]) -> tuple[bool, str, bool]:
    src = str(cmd.get("source", ""))
    dst = str(cmd.get("target", ""))
    if src not in self.nodes or dst not in self.nodes or src == dst:
      return False, "Edge requires two distinct existing agents.", False
    if self.nodes[src].layer >= self.nodes[dst].layer:
      return (
          False,
          f"Directed edge {src} -> {dst} requires layer({src}) < layer({dst}).",
          False,
      )
    self.edges.add((src, dst))
    self._mark_downstream_dirty({dst})
    self.version += 1
    defer = bool(cmd.get("defer_execution", False))
    return True, f"Connected {src} -> {dst}.", not defer

  def _apply_delete_agent(
      self, cmd: dict[str, object]
  ) -> tuple[bool, str, bool]:
    target = str(cmd.get("target", ""))
    if target not in self.nodes:
      return False, f"Unknown target agent {target!r}.", False
    downstream = {dst for src, dst in self.edges if src == target}
    self.edges = {edge for edge in self.edges if target not in edge}
    del self.nodes[target]
    self.artifacts.pop(target, None)
    if self.output_agent == target:
      self.output_agent = None
    self._mark_downstream_dirty(downstream)
    self.version += 1
    return True, f"Deleted {target}.", False

  def _apply_finish(self, cmd: dict[str, object]) -> CanvasStepResult:
    target = str(cmd.get("target", ""))
    if target not in self.nodes:
      return CanvasStepResult(False, False, f"Unknown finish target {target!r}.")
    node = self.nodes[target]
    if not node.configured or node.result_scope != "task_result":
      return CanvasStepResult(
          False,
          False,
          f"Target {target} must be configured with result_scope='task_result'.",
      )
    if node.dirty or target not in self.artifacts:
      return CanvasStepResult(
          False,
          False,
          f"Target {target} has unexecuted changes; run_graph before finish.",
      )
    disconnected = self._all_reach_target(target)
    if disconnected:
      return CanvasStepResult(
          False,
          False,
          f"Connect or delete disconnected nodes before finish: {disconnected}.",
      )
    self.output_agent = target
    return CanvasStepResult(True, True, f"Submitted {target}.")

  async def step(
      self,
      raw_action_text: str,
      worker_fn: Callable[
          [AgentNode, list[WorkerArtifact]], Awaitable[WorkerArtifact]
      ],
  ) -> CanvasStepResult:
    """Applies one Director JSON command and runs dirty Workers when triggered."""
    cmd = parse_director_action(raw_action_text)
    if cmd is None:
      return CanvasStepResult(
          False,
          False,
          "Invalid JSON action. Return exactly one legal JSON action object.",
      )
    action = str(cmd["action"])
    if action == "finish":
      return self._apply_finish(cmd)
    try:
      if action == "add_agent":
        ok, msg, should_run = self._apply_add_agent(cmd)
      elif action == "set_prompt":
        ok, msg, should_run = self._apply_set_prompt(cmd)
      elif action == "set_layer":
        ok, msg, should_run = self._apply_set_layer(cmd)
      elif action == "add_edge":
        ok, msg, should_run = self._apply_add_edge(cmd)
      elif action == "delete_agent":
        ok, msg, should_run = self._apply_delete_agent(cmd)
      else:
        ok, msg, should_run = True, "Executing dirty graph layers.", True
    except (TypeError, ValueError) as exc:
      return CanvasStepResult(False, False, f"Parameter error: {exc}")
    if not ok:
      return CanvasStepResult(False, False, msg)
    executed = await self.execute_dirty(worker_fn) if should_run else ()
    if executed:
      msg = f"{msg} Executed: {', '.join(executed)}."
    return CanvasStepResult(True, False, msg, executed)

  async def execute_dirty(
      self,
      worker_fn: Callable[
          [AgentNode, list[WorkerArtifact]], Awaitable[WorkerArtifact]
      ],
  ) -> tuple[str, ...]:
    """Executes configured dirty nodes layer-by-layer (`asyncio.gather` per layer)."""
    executed: list[str] = []
    for layer in range(self.max_layers):
      ready = [
          node
          for aid, node in sorted(self.nodes.items())
          if node.layer == layer
          and node.configured
          and node.dirty
          and all(
              not self.nodes[p].dirty and p in self.artifacts
              for p in self.predecessors(aid)
          )
      ]
      if not ready:
        continue
      if len(ready) > 1:
        self.had_parallel_wave = True
      results = await asyncio.gather(
          *(
              worker_fn(
                  node, [self.artifacts[p] for p in self.predecessors(node.agent_id)]
              )
              for node in ready
          )
      )
      for node, artifact in zip(ready, results, strict=True):
        trimmed = dataclasses.replace(
            artifact, summary=artifact.summary[: self.relay_max_chars]
        )
        self.artifacts[node.agent_id] = trimmed
        node.dirty = False
        self.worker_token_in += trimmed.token_in
        self.worker_token_out += trimmed.token_out
        self.worker_calls += 1
        executed.append(node.agent_id)
    return tuple(executed)

  def is_submit_ready(self, agent_id: str) -> bool:
    """Returns True if `agent_id` is an executed `task_result` reached by all nodes."""
    node = self.nodes.get(agent_id)
    if node is None or not node.configured or node.result_scope != "task_result":
      return False
    if node.dirty or agent_id not in self.artifacts:
      return False
    return not self._all_reach_target(agent_id)

  def _node_status(self, node: AgentNode) -> str:
    if not node.configured:
      return "awaiting_prompt"
    return "dirty" if node.dirty else "ready"

  def snapshot(self, *, remaining_rounds: int) -> dict[str, object]:
    """Compact authoritative Canvas state shown to the Director."""
    sorted_ids = sorted(self.nodes)
    submit_ready = [aid for aid in sorted_ids if self.is_submit_ready(aid)]
    nodes_view = [
        {
            "id": aid,
            "layer": self.nodes[aid].layer,
            "role": self.nodes[aid].role,
            "scope": self.nodes[aid].result_scope,
            "status": self._node_status(self.nodes[aid]),
            "inputs": self.predecessors(aid),
            "artifact_summary": (
                self.artifacts[aid].summary if aid in self.artifacts else ""
            ),
        }
        for aid in sorted_ids
    ]
    allowed: list[str] = []
    if len(self.nodes) < self.max_agents:
      allowed.append("add_agent")
    if self.nodes:
      allowed.extend(["set_prompt", "set_layer", "delete_agent"])
    if len(self.nodes) >= 2:
      allowed.append("add_edge")
    if any(n.configured and n.dirty for n in self.nodes.values()):
      allowed.append("run_graph")
    if submit_ready:
      allowed.append("finish")
    return {
        "version": self.version,
        "remaining_rounds": remaining_rounds,
        "max_agents": self.max_agents,
        "nodes": nodes_view,
        "edges": [f"{u}->{v}" for u, v in sorted(self.edges)],
        "submit_ready": submit_ready,
        "allowed_actions": allowed,
    }

  def submitted_answer(self) -> str:
    """Returns the submitted Worker answer, or empty string if unfinished."""
    if self.output_agent is None:
      return ""
    artifact = self.artifacts.get(self.output_agent)
    return artifact.answer if artifact is not None else ""
