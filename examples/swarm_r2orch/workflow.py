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

"""R2Orch Inner-Loop workflow: trainable Graph Director + frozen Workers."""

import dataclasses
import json

from examples.swarm_r2orch import canvas as canvas_lib
from examples.swarm_r2orch import scoring
from tunix.dist_rl.swarm import runtime as runtime_lib
from tunix.dist_rl.swarm import trace
from tunix.utils import token_sanitization

DIRECTOR_ROLE = "Graph Director"
WORKER_ROLE = "Frozen Worker"

DIRECTOR_SYSTEM_PROMPT = """You are the R2Orch Graph Director. You build and adapt a layered DAG workflow of frozen Workers to solve the task accurately and efficiently.
Never solve the task yourself and never include candidate answers, \\boxed{}, or code blocks in delegation prompts.
Return EXACTLY ONE strict JSON object per turn with no surrounding prose or markdown fences.

Available JSON actions:
1. Add and run a Worker:
   {"action":"add_agent","role":"...","objective":"...","layer":0,"inputs":[],"result_scope":"subtask"|"task_result","defer_execution":false}
   - Set "defer_execution":true when staging multiple parallel Workers in the same layer, then run them together on the final node or with {"action":"run_graph"}.
   - Every edge u -> v requires layer(u) < layer(v).
2. Revise a Worker's responsibility:
   {"action":"set_prompt","target":"agent_1","role":"...","objective":"...","result_scope":"subtask"|"task_result"}
3. Move a Worker's layer or connect nodes:
   {"action":"set_layer","target":"agent_2","layer":1}
   {"action":"add_edge","source":"agent_1","target":"agent_2"}
4. Prune an unhelpful or disconnected Worker:
   {"action":"delete_agent","target":"agent_1"}
5. Execute deferred dirty Workers:
   {"action":"run_graph"}
6. Submit a completed task_result Worker (all retained nodes must reach target):
   {"action":"finish","target":"agent_1"}

Match graph topology to task complexity: use a single task_result Worker on straightforward problems, or compose decomposition / parallel verification / synthesis layers when cross-checking or multi-hop evidence aggregation improves reliability. Always reserve your final turn for {"action":"finish","target":"..."}."""

DOMAIN_DIRECTOR_HINTS = {
    "aime": (
        "Competition math (integer 0..999): consider independent derivation or"
        " algebraic/modular verification before finishing."
    ),
    "math_hard": (
        "Competition math: delegate symbolic/algebraic derivation and verify"
        " edge cases or sign/fraction simplification before finishing."
    ),
    "hotpotqa": (
        "Multi-hop passage QA: extract bridge entities across passages before"
        " synthesizing the exact short answer."
    ),
    "musique": (
        "Multi-hop passage QA: resolve hop-by-hop facts from the passages and"
        " verify entity links before submitting."
    ),
    "medqa": (
        "Clinical USMLE MCQ: analyze clinical findings and rule out distractors"
        " before selecting the option letter."
    ),
    "mbppplus": (
        "Python synthesis: implement the exact function signature from the"
        " prompt assertion and audit boundary/type edge cases."
    ),
}

WORKER_FORMAT_INSTRUCTIONS = {
    "integer_answer": (
        "Show concise mathematical reasoning and put the final integer answer"
        " inside \\boxed{}."
    ),
    "math_symbolic": (
        "Show concise mathematical reasoning and put the final simplified"
        " expression or number inside \\boxed{}."
    ),
    "qa_exact_match": (
        "Use only the provided passages. End your response with a final line"
        " 'Answer: <concise exact answer span>'."
    ),
    "letter_choice": (
        "Evaluate the clinical vignette and options carefully. End your"
        " response with 'Answer: <option letter>'."
    ),
    "mbppplus_tests": (
        "Return a complete, self-contained Python implementation inside one"
        " ```python ... ``` block defining the exact function name tested."
    ),
}


@dataclasses.dataclass(frozen=True)
class R2OrchOutcome:
  """Episode trace paired with R2Orch accuracy, cost, and topology telemetry."""

  episode: trace.SwarmEpisode
  dataset: str
  correctness: float
  worker_cost: float
  finished: bool
  director_turns: int
  worker_calls: int
  num_nodes: int
  max_layer: int
  parallel_wave: bool


def _summarize_worker_output(text: str, limit: int) -> str:
  stripped = text.strip()
  if len(stripped) <= limit:
    return stripped
  head = limit // 3
  tail = limit - head - 5
  return f"{stripped[:head]}\n...\n{stripped[-tail:]}"


def build_worker_prompt(
    task: scoring.R2OrchTask,
    node: canvas_lib.AgentNode,
    upstream: list[canvas_lib.WorkerArtifact],
) -> str:
  """Renders the stateless prompt sent to one frozen Worker node."""
  upstream_block = (
      "\n\n".join(
          f"[{art.agent_id} output]\n{art.summary}" for art in upstream
      )
      if upstream
      else "None (you are an entry-layer Worker)."
  )
  scope_rule = (
      "You are a `task_result` Worker: deliver the complete final answer to the"
      " public task."
      if node.result_scope == "task_result"
      else (
          "You are a `subtask` Worker: deliver focused intermediate findings"
          " and evidence for downstream Workers."
      )
  )
  fmt = (
      WORKER_FORMAT_INSTRUCTIONS.get(task.verifier, "")
      if node.result_scope == "task_result"
      else "Present clear, verifiable findings concisely."
  )
  return (
      f"## Assigned Role: {node.role}\n"
      f"## Delegated Objective\n{node.objective}\n\n"
      f"## Scope & Output Contract\n{scope_rule}\n{fmt}\n\n"
      f"## Upstream Worker Reports\n{upstream_block}\n\n"
      f"## Public Task\n{task.prompt}"
  )


def build_director_turn_prompt(
    task: scoring.R2OrchTask,
    history: list[tuple[str, str]],
    snapshot: dict[str, object],
    feedback: str,
    *,
    skill_context: str = "",
) -> str:
  """Renders the Director turn prompt with task, compact history, and state."""
  hint = DOMAIN_DIRECTOR_HINTS.get(task.dataset, "")
  skill_block = (
      f"Optional Orchestration Knowledge:\n{skill_context.strip()}\n\n"
      if skill_context.strip()
      else ""
  )
  history_lines = [
      f"Turn {idx + 1} action: {act}\nTurn {idx + 1} feedback: {fb}"
      for idx, (act, fb) in enumerate(history[-6:])
  ]
  history_block = "\n".join(history_lines) if history_lines else "None yet."
  snap_json = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
  return (
      f"Domain guidance: {hint}\n\n"
      f"{skill_block}"
      f"Public Task ({task.dataset}):\n{task.prompt}\n\n"
      f"Recent Director History:\n{history_block}\n\n"
      f"Latest Feedback: {feedback}\n"
      f"Canvas Snapshot:\n{snap_json}\n\n"
      "Return the next single JSON action:"
  )


class GraphDirectorWorkflow:
  """Executes one multi-turn R2Orch episode over `RecordingRuntime`."""

  def __init__(
      self,
      *,
      max_rounds: int = 6,
      max_agents: int = 4,
      relay_max_chars: int = 1000,
      cost_lambda: float = 0.005,
      grader=scoring.score_task,
  ):
    if min(max_rounds, max_agents, relay_max_chars) < 1 or cost_lambda < 0:
      raise ValueError("Invalid GraphDirectorWorkflow configuration.")
    self.max_rounds = max_rounds
    self.max_agents = max_agents
    self.relay_max_chars = relay_max_chars
    self.cost_lambda = cost_lambda
    self.grader = grader

  async def run_detailed(
      self,
      task: scoring.R2OrchTask,
      rt: runtime_lib.RecordingRuntime,
      *,
      evaluate: bool = False,
      skill_context: str = "",
  ) -> R2OrchOutcome:
    """Runs one Director+Frozen-Workers episode and returns full telemetry."""
    canvas = canvas_lib.GraphCanvas(
        max_agents=self.max_agents,
        relay_max_chars=self.relay_max_chars,
    )
    recorded_calls: list[trace.CallTrace] = []
    history: list[tuple[str, str]] = []
    feedback = "Canvas is empty. Add your first Worker."

    async def run_worker(
        node: canvas_lib.AgentNode,
        upstream: list[canvas_lib.WorkerArtifact],
    ) -> canvas_lib.WorkerArtifact:
      w_prompt = build_worker_prompt(task, node, upstream)
      w_call = await rt.generate(WORKER_ROLE, w_prompt)
      recorded_calls.append(w_call)
      return canvas_lib.WorkerArtifact(
          agent_id=node.agent_id,
          answer=w_call.text,
          summary=_summarize_worker_output(w_call.text, self.relay_max_chars),
          token_in=len(w_call.prompt_ids),
          token_out=len(w_call.completion_ids),
      )

    director_turns = 0
    finished = False
    for turn in range(self.max_rounds):
      snap = canvas.snapshot(remaining_rounds=self.max_rounds - turn)
      d_prompt = build_director_turn_prompt(
          task, history, snap, feedback, skill_context=skill_context
      )
      d_call = await rt.generate(DIRECTOR_ROLE, d_prompt)
      director_turns += 1
      step_res = await canvas.step(d_call.text, run_worker)
      recorded_calls.append(
          dataclasses.replace(d_call, valid_action=step_res.accepted)
      )
      history.append((d_call.text.strip()[:240], step_res.feedback))
      feedback = step_res.feedback
      if step_res.finished:
        finished = True
        break

    final_answer = canvas.submitted_answer() if finished else ""
    correctness = float(self.grader(task, final_answer)) if final_answer else 0.0
    w_cost = scoring.compute_worker_cost(
        canvas.worker_token_in, canvas.worker_token_out, task.dataset
    )
    reward = (
        correctness
        if evaluate
        else correctness - self.cost_lambda * w_cost
    )
    episode = trace.SwarmEpisode(
        task_id=task.task_id,
        episode_id=rt.episode_id,
        calls=tuple(recorded_calls),
        reward=float(reward),
        final_text=final_answer,
    )
    max_layer = max((n.layer for n in canvas.nodes.values()), default=0)
    return R2OrchOutcome(
        episode=episode,
        dataset=task.dataset,
        correctness=correctness,
        worker_cost=w_cost,
        finished=finished,
        director_turns=director_turns,
        worker_calls=canvas.worker_calls,
        num_nodes=len(canvas.nodes),
        max_layer=max_layer,
        parallel_wave=canvas.had_parallel_wave,
    )


def role_chat_encoder(tokenizer, system_prompt: str, max_prompt_length: int):
  """Builds a middle-truncating non-thinking chat encoder for a given role."""

  def encode(prompt: str) -> list[int]:
    clean = token_sanitization.sanitize_control_tokens(prompt)
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean},
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) > max_prompt_length:
      left = max_prompt_length // 2
      ids = ids[:left] + ids[-(max_prompt_length - left) :]
    return ids

  return encode
