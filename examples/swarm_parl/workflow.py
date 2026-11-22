# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Option 5 PARL (Kimi K2.5 §3) parallel orchestrator & frozen search workers."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from typing import Any
import urllib.request

from tunix.dist_rl.swarm import episode_runtime

_PARALLEL_RE = re.compile(r"<parallel>(.*?)</parallel>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)

def director_system_prompt(max_wave_width: int = 4) -> str:
  return (
      "You are a Parallel Orchestrator (PARL) solving complex information-seeking "
      "questions under a strict CriticalSteps latency budget. You have NO direct "
      "search tool, but you can dispatch parallel search subagents in a single wave "
      "(all subagents in one wave run concurrently and only count as the slowest "
      "branch!).\n"
      "Output EXACTLY ONE action tag per turn:\n"
      "1. Dispatch parallel workers: <parallel>[\"subtask 1\", \"subtask 2\", ...]</parallel> "
      f"(up to {int(max_wave_width)} concise independent sub-questions)\n"
      "2. Give final answer once you have enough evidence: <answer>concise answer</answer>"
  )


DIRECTOR_SYSTEM_PROMPT = director_system_prompt(4)

WORKER_SYSTEM_PROMPT = (
    "You are a fast Wikipedia research subagent. You can run Wikipedia search "
    "using <search>concise query</search> and once you find the exact factual "
    "answer, return <report>concise factual answer with key entity names</report>."
)

SERIAL_SYSTEM_PROMPT = (
    "You are a single-agent Wikipedia researcher under a strict step budget. "
    "Use <search>concise query</search> to look up facts, or return "
    "<answer>concise answer</answer> when ready."
)


class HttpWikiSearchEnv:
  """Public retrieval environment calling the TPU Wiki-18 `POST /retrieve` server."""

  def __init__(self, endpoint: str = "http://127.0.0.1:18610/retrieve", top_k: int = 3) -> None:
    self.endpoint = endpoint
    self.top_k = int(top_k)

  def _search_sync(self, query: str) -> str:
    payload = json.dumps({"queries": [query], "topk": self.top_k, "return_scores": True}).encode("utf-8")
    req = urllib.request.Request(
        self.endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
      data = json.loads(resp.read().decode("utf-8"))
    hits = (data.get("result") or [[]])[0]
    snippets = []
    for i, item in enumerate(hits[: self.top_k], 1):
      doc = item.get("document", item)
      title = str(doc.get("title", "")).strip()
      text = str(doc.get("contents", doc.get("text", ""))).strip()[:420]
      snippets.append(f"[{i}] {title}: {text}" if title else f"[{i}] {text}")
    return "\n".join(snippets) if snippets else "No relevant Wikipedia passages found."

  async def search(self, query: str) -> str:
    return await asyncio.to_thread(self._search_sync, query)


@dataclasses.dataclass(frozen=True)
class DirectorCommand:
  is_final: bool
  is_valid: bool
  answer: str = ""
  jobs: tuple[str, ...] = ()


def parse_director_action(text: str, *, max_wave_width: int = 4) -> DirectorCommand:
  """Parses bounded Director `<parallel>` or `<answer>` action tags."""
  ans_m = _ANSWER_RE.search(text or "")
  if ans_m:
    return DirectorCommand(is_final=True, is_valid=True, answer=ans_m.group(1).strip())
  par_m = _PARALLEL_RE.search(text or "")
  if par_m:
    raw = par_m.group(1).strip()
    jobs: list[str] = []
    try:
      parsed = json.loads(raw)
      if isinstance(parsed, list):
        jobs = [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
      jobs = [ln.strip(" -*\t") for ln in raw.splitlines() if ln.strip()]
    if jobs:
      return DirectorCommand(
          is_final=False, is_valid=True, jobs=tuple(jobs[:max_wave_width])
      )
  return DirectorCommand(is_final=True, is_valid=False, answer=(text or "").strip())


def _fit_user_body(user: str, max_chars: int = 9000) -> str:
  if len(user) <= max_chars:
    return user
  head = max_chars // 2
  tail = max_chars - head - 32
  return f"{user[:head]}\n...[truncated]...\n{user[-tail:]}"


def format_chat_prompt(system: str, user: str) -> str:
  return (
      f"<|im_start|>system\n{system}<|im_end|>\n"
      f"<|im_start|>user\n{_fit_user_body(user)}<|im_end|>\n"
      f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
  )


async def solve_subtask_workflow(
    job: str,
    ctx: episode_runtime.EpisodeRuntime,
    *,
    max_worker_steps: int = 2,
) -> str:
  """Frozen multi-step search subagent executing inside `ctx.parallel`."""
  history: list[str] = [f"Subtask: {job}"]
  last_text = ""
  for turn in range(max_worker_steps):
    prompt = format_chat_prompt(WORKER_SYSTEM_PROMPT, "\n\n".join(history))
    act = await ctx.generate("worker", prompt)
    last_text = act.text.strip()
    rep_m = _REPORT_RE.search(last_text)
    if rep_m:
      return rep_m.group(1).strip()
    srch_m = _SEARCH_RE.search(last_text)
    if srch_m and ctx.env is not None and turn + 1 < max_worker_steps:
      query = srch_m.group(1).strip()
      obs = await ctx.env.search(query)
      history.append(f"Search({query}) results:\n{obs}")
      continue
    if ctx.env is not None and turn == 0 and max_worker_steps > 1:
      obs = await ctx.env.search(job[:160])
      history.append(f"Search({job[:160]}) results:\n{obs}")
      continue
    return last_text
  return last_text


async def parl_episode_workflow(
    task: dict[str, Any],
    ctx: episode_runtime.EpisodeRuntime,
    *,
    max_waves: int = 3,
    max_wave_width: int = 4,
    max_worker_steps: int = 2,
) -> str:
  """Kimi K2.5 PARL orchestrator over Option 5 `ctx.parallel` + `CriticalSteps`."""
  question = str(task["question"])
  reports: list[str] = []
  last_answer = ""
  sys_prompt = director_system_prompt(max_wave_width)
  for wave_idx in range(max_waves):
    body = f"Question:\n{question}"
    if reports:
      body += "\n\nSubagent Reports:\n" + "\n".join(reports)
    if wave_idx + 1 == max_waves or (
        ctx.remaining_critical_steps is not None and ctx.remaining_critical_steps <= 1
    ):
      body += "\n\nYou must output <answer>...</answer> now."
    decision = await ctx.generate(
        "director", format_chat_prompt(sys_prompt, body)
    )
    cmd = parse_director_action(decision.text, max_wave_width=max_wave_width)
    ctx.annotate(decision, valid=cmd.is_valid, n_jobs=len(cmd.jobs))
    if cmd.is_final:
      return cmd.answer
    last_answer = decision.text
    worker_fn = lambda job, child_ctx: solve_subtask_workflow(
        job, child_ctx, max_worker_steps=max_worker_steps
    )
    wave_outputs = await ctx.parallel(cmd.jobs, worker_fn, parent=decision)
    for sub_q, sub_a in zip(cmd.jobs, wave_outputs):
      reports.append(f"- [{sub_q[:160]}] => {str(sub_a or '').strip()[:300]}")
  return last_answer


async def serial_baseline_workflow(
    task: dict[str, Any],
    ctx: episode_runtime.EpisodeRuntime,
    *,
    max_steps: int = 6,
) -> str:
  """Serial single-agent baseline under identical `CriticalSteps` & Wiki-18 env."""
  history = [f"Question:\n{task['question']}"]
  last_text = ""
  for step_i in range(max_steps):
    if step_i + 1 == max_steps:
      history.append("Return your final <answer>...</answer> now.")
    act = await ctx.generate(
        "director", format_chat_prompt(SERIAL_SYSTEM_PROMPT, "\n\n".join(history))
    )
    last_text = act.text.strip()
    ans_m = _ANSWER_RE.search(last_text)
    if ans_m:
      return ans_m.group(1).strip()
    srch_m = _SEARCH_RE.search(last_text)
    if srch_m and ctx.env is not None:
      q = srch_m.group(1).strip()
      obs = await ctx.env.search(q)
      history.append(f"Search({q}) results:\n{obs}")
  return last_text
