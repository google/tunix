# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Option 5 PARL (Kimi K2.5 §3) dual-mode orchestrator & dynamic subagents."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from typing import Any
import urllib.request

from tunix.dist_rl.swarm import episode_runtime

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPAWN_RE = re.compile(r"<spawn_wave>(.*?)</spawn_wave>", re.DOTALL | re.IGNORECASE)
_PARALLEL_RE = re.compile(r"<parallel>(.*?)</parallel>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_OPEN_DOC_RE = re.compile(r"<open_doc>(.*?)</open_doc>", re.DOTALL | re.IGNORECASE)
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)


def director_system_prompt(
    max_wave_width: int = 4,
    *,
    dual_mode: bool = False,
    allow_open_doc: bool = False,
) -> str:
  if not dual_mode:
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
  open_doc_line = (
      "2. Open full text of a retrieved docid directly: <open_doc>docid</open_doc>\n"
      if allow_open_doc
      else ""
  )
  idx_spawn = 3 if allow_open_doc else 2
  idx_ans = idx_spawn + 1
  return (
      "You are a Kimi-style Parallel Agent Orchestrator (PARL) solving complex "
      "information-seeking tasks under a strict CriticalSteps latency budget. "
      "You may think briefly inside <think>...</think>, then output EXACTLY ONE "
      "action tag:\n"
      "1. Direct serial search: <search>concise query</search>\n"
      f"{open_doc_line}"
      f"{idx_spawn}. Dynamically instantiate specialized parallel subagents in one wave "
      f"(all run concurrently and count only as the slowest branch toward CriticalSteps; "
      f"up to {int(max_wave_width)} subagents):\n"
      "   <spawn_wave>[{\"role\": \"Entity_Researcher\", \"persona\": \"Verify exact facts\", "
      "\"task\": \"sub-question 1\"}, ...]</spawn_wave>\n"
      f"{idx_ans}. Final answer once sufficient evidence is gathered: "
      "<answer>concise answer</answer>"
  )


DIRECTOR_SYSTEM_PROMPT = director_system_prompt(4, dual_mode=False)

WORKER_SYSTEM_PROMPT = (
    "You are a fast Wikipedia research subagent. You can run Wikipedia search "
    "using <search>concise query</search> and once you find the exact factual "
    "answer, return <report>concise factual answer with key entity names</report>."
)

SERIAL_SYSTEM_PROMPT = (
    "You are a single-agent researcher under a strict CriticalSteps budget. "
    "You may think briefly inside <think>...</think>, then use "
    "<search>concise query</search> to look up facts, or return "
    "<answer>concise answer</answer> when ready."
)


def _worker_prompt_for(spec: SubagentSpec, allow_open_doc: bool) -> str:
  extra = (
      " You can also read a full document using <open_doc>docid</open_doc>."
      if allow_open_doc
      else ""
  )
  persona = f" {spec.persona}" if spec.persona else ""
  return (
      f"You are {spec.role}, a focused research subagent.{persona} "
      f"Use <search>concise query</search> to search.{extra} "
      "Once you find the exact factual evidence, return "
      "<report>concise factual answer with key entities and docids</report>."
  )


def _serial_prompt_for(allow_open_doc: bool) -> str:
  if not allow_open_doc:
    return SERIAL_SYSTEM_PROMPT
  return (
      "You are a single-agent deep researcher under a strict CriticalSteps budget. "
      "You may think briefly inside <think>...</think>, then output ONE tag: "
      "<search>concise query</search>, <open_doc>docid</open_doc>, or "
      "<answer>concise final answer</answer>."
  )


class HttpWikiSearchEnv:
  """Public retrieval environment calling multi-corpus `/retrieve` & `/open_doc`."""

  def __init__(
      self,
      endpoint: str = "http://127.0.0.1:18610/retrieve",
      top_k: int = 3,
  ) -> None:
    self.endpoint = endpoint
    self.open_doc_endpoint = endpoint.rsplit("/", 1)[0] + "/open_doc"
    self.top_k = int(top_k)

  def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=raw, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
      return json.loads(resp.read().decode("utf-8"))

  def _search_sync(self, query: str, corpus: str = "wiki18") -> str:
    topk = max(self.top_k, 5) if corpus == "browsecomp" else self.top_k
    data = self._post_json(
        self.endpoint,
        {"queries": [query], "topk": topk, "corpus": corpus, "return_scores": True},
    )
    hits = (data.get("result") or [[]])[0]
    snippets = []
    for i, item in enumerate(hits[:topk], 1):
      doc = item.get("document", item)
      docid = str(doc.get("docid", "")).strip()
      title = str(doc.get("title", "")).strip()
      text = str(doc.get("contents", doc.get("text", ""))).strip()[:450]
      tag = f"[{i}] (docid={docid}) " if docid else f"[{i}] "
      snippets.append(f"{tag}{title}: {text}" if title else f"{tag}{text}")
    return "\n".join(snippets) if snippets else "No relevant passages found."

  def _open_doc_sync(self, docid: str, corpus: str = "browsecomp") -> str:
    data = self._post_json(
        self.open_doc_endpoint,
        {"docid": docid, "corpus": corpus, "max_chars": 2600},
    )
    if not data.get("found"):
      return f"Document {docid} not found."
    title = str(data.get("title", "")).strip()
    url = str(data.get("url", "")).strip()
    body = str(data.get("contents", "")).strip()
    hdr = f"Doc {data.get('docid', docid)}"
    if title:
      hdr += f" | {title}"
    if url:
      hdr += f" ({url})"
    return f"{hdr}\n{body}"

  async def search(self, query: str, *, corpus: str = "wiki18") -> str:
    return await asyncio.to_thread(self._search_sync, query, corpus)

  async def open_doc(self, docid: str, *, corpus: str = "browsecomp") -> str:
    return await asyncio.to_thread(self._open_doc_sync, docid, corpus)


@dataclasses.dataclass(frozen=True)
class SubagentSpec:
  role: str
  persona: str
  task: str


@dataclasses.dataclass(frozen=True)
class DirectorCommand:
  is_final: bool
  is_valid: bool
  answer: str = ""
  jobs: tuple[str, ...] = ()
  action_kind: str = "answer"
  query: str = ""
  docid: str = ""
  subagents: tuple[SubagentSpec, ...] = ()


def _strip_thinking(text: str) -> str:
  cleaned = _THINK_RE.sub("", text or "")
  if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
    # If thinking was truncated before closing tag, inspect full text for any tags.
    return text or ""
  return cleaned.strip()


def _coerce_subagent(idx: int, item: Any) -> SubagentSpec | None:
  default_role = f"Researcher_{idx + 1}"
  if isinstance(item, dict):
    task = str(item.get("task", item.get("subtask", item.get("query", "")))).strip()
    if not task:
      return None
    role = str(item.get("role", item.get("name", default_role))).strip()[:48]
    persona = str(item.get("persona", item.get("system_prompt", ""))).strip()[:200]
    return SubagentSpec(role=role or default_role, persona=persona, task=task)
  txt = str(item).strip()
  return SubagentSpec(role=default_role, persona="", task=txt) if txt else None


def _parse_subagent_list(raw: str, max_wave_width: int) -> tuple[SubagentSpec, ...]:
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError:
    parsed = None
  items = (
      parsed
      if isinstance(parsed, list)
      else [ln.strip(" -*\t") for ln in raw.splitlines() if ln.strip()]
  )
  specs = [s for idx, it in enumerate(items) if (s := _coerce_subagent(idx, it)) is not None]
  return tuple(specs[:max_wave_width])


def parse_director_action(
    text: str,
    *,
    max_wave_width: int = 4,
    allow_direct_tools: bool = True,
) -> DirectorCommand:
  """Parses Director `<answer>`, `<spawn_wave>`, `<parallel>`, `<search>`, or `<open_doc>`."""
  body = _strip_thinking(text)
  ans_m = _ANSWER_RE.search(body)
  if ans_m:
    return DirectorCommand(
        is_final=True, is_valid=True, answer=ans_m.group(1).strip(), action_kind="answer"
    )
  for regex in (_SPAWN_RE, _PARALLEL_RE):
    m = regex.search(body)
    if m:
      specs = _parse_subagent_list(m.group(1).strip(), max_wave_width)
      if specs:
        return DirectorCommand(
            is_final=False,
            is_valid=True,
            jobs=tuple(s.task for s in specs),
            action_kind="spawn_wave",
            subagents=specs,
        )
  if allow_direct_tools:
    open_m = _OPEN_DOC_RE.search(body)
    if open_m and open_m.group(1).strip():
      return DirectorCommand(
          is_final=False,
          is_valid=True,
          action_kind="open_doc",
          docid=open_m.group(1).strip(),
      )
    srch_m = _SEARCH_RE.search(body)
    if srch_m and srch_m.group(1).strip():
      return DirectorCommand(
          is_final=False,
          is_valid=True,
          action_kind="search",
          query=srch_m.group(1).strip(),
      )
  return DirectorCommand(
      is_final=True, is_valid=False, answer=body.strip(), action_kind="invalid"
  )


def _fit_user_body(user: str, max_chars: int = 16000) -> str:
  if len(user) <= max_chars:
    return user
  head = max_chars // 2
  tail = max_chars - head - 32
  return f"{user[:head]}\n...[truncated]...\n{user[-tail:]}"


def format_chat_prompt(
    system: str,
    user: str,
    *,
    enable_thinking: bool = False,
) -> str:
  prefix = "" if enable_thinking else "<think>\n\n</think>\n\n"
  return (
      f"<|im_start|>system\n{system}<|im_end|>\n"
      f"<|im_start|>user\n{_fit_user_body(user)}<|im_end|>\n"
      f"<|im_start|>assistant\n{prefix}"
  )


async def _env_search(env: Any, query: str, corpus: str) -> str:
  try:
    return await env.search(query, corpus=corpus)
  except TypeError:
    return await env.search(query)


async def _env_open_doc(env: Any, docid: str, corpus: str) -> str:
  if hasattr(env, "open_doc"):
    return await env.open_doc(docid, corpus=corpus)
  return f"open_doc unavailable for {docid}"


async def _maybe_exec_tool(
    env: Any,
    text: str,
    corpus: str,
    *,
    allow_open: bool,
    fallback_query: str = "",
) -> str | None:
  if env is None:
    return None
  open_m = _OPEN_DOC_RE.search(text) if allow_open else None
  if open_m:
    did = open_m.group(1).strip()
    return f"OpenDoc({did}):\n{await _env_open_doc(env, did, corpus)}"
  srch_m = _SEARCH_RE.search(text)
  if srch_m:
    q = srch_m.group(1).strip()
    return f"Search({q}) results:\n{await _env_search(env, q, corpus)}"
  if fallback_query:
    return f"Search({fallback_query}) results:\n{await _env_search(env, fallback_query, corpus)}"
  return None


async def solve_subtask_workflow(
    job: str | SubagentSpec,
    ctx: episode_runtime.EpisodeRuntime,
    *,
    max_worker_steps: int = 2,
    corpus: str = "wiki18",
    enable_thinking: bool = False,
) -> str:
  """Frozen multi-step specialized search subagent executing inside `ctx.parallel`."""
  spec = (
      job
      if isinstance(job, SubagentSpec)
      else SubagentSpec(role="Wikipedia_Researcher", persona="", task=str(job))
  )
  allow_open = corpus == "browsecomp"
  sys_prompt = (
      _worker_prompt_for(spec, allow_open)
      if isinstance(job, SubagentSpec) or allow_open
      else WORKER_SYSTEM_PROMPT
  )
  history: list[str] = [f"Subtask: {spec.task}"]
  last_text = ""
  for turn in range(max_worker_steps):
    if turn + 1 == max_worker_steps:
      history.append("Return your concise <report>...</report> now.")
    prompt = format_chat_prompt(
        sys_prompt, "\n\n".join(history), enable_thinking=enable_thinking
    )
    last_text = _strip_thinking((await ctx.generate("worker", prompt)).text).strip()
    rep_m = _REPORT_RE.search(last_text)
    if rep_m:
      return rep_m.group(1).strip()
    if turn + 1 >= max_worker_steps:
      return last_text
    obs = await _maybe_exec_tool(
        ctx.env, last_text, corpus, allow_open=allow_open,
        fallback_query=spec.task[:160] if turn == 0 else "",
    )
    if obs is None:
      return last_text
    history.append(obs)
  return last_text


async def _exec_director_step(
    cmd: DirectorCommand,
    decision: Any,
    ctx: episode_runtime.EpisodeRuntime,
    *,
    corpus: str,
    dual_mode: bool,
    max_worker_steps: int,
) -> list[str]:
  if cmd.action_kind == "search" and ctx.env is not None:
    obs = await _env_search(ctx.env, cmd.query, corpus)
    return [f"[Direct Search: {cmd.query[:120]}]\n{obs}"]
  if cmd.action_kind == "open_doc" and ctx.env is not None:
    obs = await _env_open_doc(ctx.env, cmd.docid, corpus)
    return [f"[Direct OpenDoc: {cmd.docid}]\n{obs}"]
  worker_items = cmd.subagents if dual_mode else cmd.jobs
  worker_fn = lambda job, child_ctx: solve_subtask_workflow(
      job, child_ctx, max_worker_steps=max_worker_steps, corpus=corpus
  )
  wave_outputs = await ctx.parallel(worker_items, worker_fn, parent=decision)
  return [
      f"- [{spec.role} | {spec.task[:140]}] => {str(sub_a or '').strip()[:360]}"
      for spec, sub_a in zip(cmd.subagents, wave_outputs)
  ]


async def parl_episode_workflow(
    task: dict[str, Any],
    ctx: episode_runtime.EpisodeRuntime,
    *,
    max_waves: int = 3,
    max_wave_width: int = 4,
    max_worker_steps: int = 2,
    dual_mode: bool = False,
    enable_thinking: bool = False,
) -> str:
  """Kimi K2.5 PARL orchestrator over Option 5 `ctx.parallel` + `CriticalSteps`."""
  question = str(task["question"])
  corpus = str(task.get("corpus", "wiki18"))
  sys_prompt = director_system_prompt(
      max_wave_width, dual_mode=dual_mode, allow_open_doc=(corpus == "browsecomp")
  )
  notes: list[str] = []
  last_answer = ""
  for wave_idx in range(max_waves):
    body = f"Question:\n{question}"
    if notes:
      body += "\n\nEvidence & Subagent Reports:\n" + "\n".join(notes)
    near_end = wave_idx + 1 == max_waves or (
        ctx.remaining_critical_steps is not None and ctx.remaining_critical_steps <= 1
    )
    if near_end:
      body += "\n\nYou must output <answer>...</answer> now."
    decision = await ctx.generate(
        "director",
        format_chat_prompt(sys_prompt, body, enable_thinking=enable_thinking),
    )
    cmd = parse_director_action(
        decision.text, max_wave_width=max_wave_width, allow_direct_tools=dual_mode
    )
    ctx.annotate(
        decision, valid=cmd.is_valid, action_kind=cmd.action_kind, n_jobs=len(cmd.jobs)
    )
    if cmd.is_final:
      return cmd.answer
    last_answer = _strip_thinking(decision.text)
    notes.extend(
        await _exec_director_step(
            cmd, decision, ctx,
            corpus=corpus, dual_mode=dual_mode, max_worker_steps=max_worker_steps,
        )
    )
  return last_answer


async def serial_baseline_workflow(
    task: dict[str, Any],
    ctx: episode_runtime.EpisodeRuntime,
    *,
    max_steps: int = 6,
    role: str = "director",
    enable_thinking: bool = False,
) -> str:
  """Serial single-agent baseline under identical `CriticalSteps` & corpus env."""
  corpus = str(task.get("corpus", "wiki18"))
  allow_open = corpus == "browsecomp"
  sys_prompt = _serial_prompt_for(allow_open)
  history = [f"Question:\n{task['question']}"]
  last_text = ""
  for step_i in range(max_steps):
    near_end = step_i + 1 == max_steps or (
        ctx.remaining_critical_steps is not None and ctx.remaining_critical_steps <= 1
    )
    if near_end:
      history.append("Return your final <answer>...</answer> now.")
    prompt = format_chat_prompt(
        sys_prompt, "\n\n".join(history), enable_thinking=enable_thinking
    )
    last_text = _strip_thinking((await ctx.generate(role, prompt)).text).strip()
    ans_m = _ANSWER_RE.search(last_text)
    if ans_m:
      return ans_m.group(1).strip()
    if step_i + 1 < max_steps:
      obs = await _maybe_exec_tool(ctx.env, last_text, corpus, allow_open=allow_open)
      if obs is not None:
        history.append(obs)
  return last_text
