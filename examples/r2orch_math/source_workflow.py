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

"""Execute the pinned original Canvas with native Tunix policy transports.

The original Director, Worker executor, graph, artifact repair and submission
controller execute unchanged. Only ChatBackend is replaced. Generated tools
must be explicitly supplied by an isolated adapter; no upstream host executor
is instantiated here. The compact workflow remains a separate CPU fixture.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import concurrent.futures
import dataclasses
import math
from pathlib import Path
import tempfile
import threading
import tomllib

import numpy as np
from examples.r2orch_math import source_adapter
from examples.r2orch_math import workflow
from examples.r2orch_math.prepare_data import SOURCE_COMMIT
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import trace

BinarySampler = Callable[
    [tuple[int, ...], tuple[int, ...], int], Awaitable[Sequence[float]]
]


class SourceOutcomeUnavailable(RuntimeError):
  """A source rollout has no trustworthy reward and cannot enter a PPO group."""

  def __init__(self, reason, *, graph, calls):
    super().__init__(f"Original rollout outcome is unavailable: {reason}")
    self.graph, self.calls = graph, calls


class _Transport:
  """Marshal source threads onto the sampler's owning event loop."""

  def __init__(self, *, loop, tokenizer, binary_sampler, deadline):
    self.loop, self.tokenizer = loop, tokenizer
    self.binary_sampler, self.deadline = binary_sampler, deadline
    self.calls = {}
    self.pending = set()
    self.next_call = 0
    self.closed = False

  def request(
      self,
      runtime,
      messages,
      *,
      role,
      max_tokens=None,
      support=None,
      enable_thinking=False,
      temperature=None,
  ):
    self.deadline.check("tunix_request_start")
    future = asyncio.run_coroutine_threadsafe(
        self._request(
            runtime,
            messages,
            role=role,
            max_tokens=max_tokens,
            support=support,
            enable_thinking=enable_thinking,
            temperature=temperature,
        ),
        self.loop,
    )
    try:
      # The source deadline remains authoritative. A finite host wait also
      # interrupts an unresponsive transport and allows thread cleanup.
      return future.result(timeout=self.deadline.request_timeout_s)
    except concurrent.futures.TimeoutError:
      future.cancel()
      raise TimeoutError(
          "The original executor's model request timed out."
      ) from None

  async def _request(
      self,
      runtime,
      messages,
      *,
      role,
      max_tokens,
      support,
      enable_thinking,
      temperature,
  ):
    if self.closed:
      raise RuntimeError("The source workflow has closed its transport.")
    current = asyncio.current_task()
    self.pending.add(current)
    ordinal = self.next_call
    self.next_call += 1
    local_index = runtime._calls  # pylint: disable=protected-access
    runtime._calls += 1  # pylint: disable=protected-access
    call_id = f"{runtime.episode_id}/{local_index}"
    seed = (
        0 if runtime.params.seed is None else runtime.params.seed
    ) + local_index
    try:
      self.deadline.check("tunix_request_admission")
      ids = tuple(
          int(x)
          for x in self.tokenizer.apply_chat_template(
              messages,
              tokenize=True,
              add_generation_prompt=True,
              enable_thinking=bool(enable_thinking),
          )
      )
      if not ids or len(ids) > runtime.max_prompt_length:
        raise ValueError(
            "Original chat prompt exceeds the configured context budget."
        )
      if support is not None:
        logps = np.asarray(
            await self.binary_sampler(ids, support, seed), dtype=np.float64
        )
        if logps.shape != (2,) or not np.all(np.isfinite(logps)):
          raise ValueError(
              "Binary sampler must return two finite restricted log"
              " probabilities."
          )
        probabilities = np.exp(logps)
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
          raise ValueError(
              "Binary log probabilities must be normalized over off/on."
          )
        choice = int(
            np.random.default_rng(seed).choice(
                2, p=probabilities / probabilities.sum()
            )
        )
        completion, selected_logps = (support[choice],), (float(logps[choice]),)
        text, finish_reason = ("off", "on")[choice], "stop"
      else:
        if (
            temperature is not None
            and temperature != runtime.params.temperature
        ):
          raise ValueError(
              "Source requested a temperature different from the sampler"
              " profile."
          )
        params = dataclasses.replace(
            runtime.params,
            seed=seed,
            max_tokens=min(
                runtime.params.max_tokens,
                max_tokens or runtime.params.max_tokens,
            ),
        )
        response = await runtime.sampler.sample(
            sampler_lib.SamplingRequest(
                request_id=call_id,
                prompt=np.asarray(ids, dtype=np.int32),
                sampling_params=params,
                metadata={
                    "role": role,
                    "policy_id": runtime.policy_id,
                    "policy_version": runtime.policy_version,
                    "executor": "r2orch-source",
                },
            )
        )
        if not isinstance(response, sampler_lib.SamplingResponse):
          raise TypeError(
              "Source transport requires one SamplingResponse per request."
          )
        if (
            response.error is not None
            or response.finish_reason not in ("stop", "length")
            or response.request_id != call_id
            or tuple(response.prompt_token_ids) != ids
            or response.logprobs is None
            or len(response.token_ids) > params.max_tokens
        ):
          raise ValueError(
              "Source transport received an invalid exact-token sampling"
              " response."
          )
        completion = tuple(int(x) for x in response.token_ids)
        selected_logps = tuple(float(x) for x in response.logprobs)
        text, finish_reason = response.text, response.finish_reason
        logps = None
      call = trace.CallTrace(
          call_id=call_id,
          role=role,
          policy_id=runtime.policy_id,
          policy_version=runtime.policy_version,
          prompt_ids=ids,
          completion_ids=completion,
          rollout_logprobs=selected_logps,
          loss_mask=(1,) * len(completion),
          text=text,
          **({"action_support_ids": support} if support is not None else {}),
      )
      self.calls[ordinal] = call
      return call, logps, finish_reason
    finally:
      self.pending.discard(current)

  async def close(self):
    self.closed = True
    pending = tuple(self.pending)
    for task in pending:
      task.cancel()
    if pending:
      await asyncio.gather(*pending, return_exceptions=True)


class _ChatBackend:
  """The official ChatBackend contract, retaining exact sampled token records."""

  def __init__(self, transport, runtime, llm, *, enable_thinking=False):
    self.transport, self.runtime, self.llm = transport, runtime, llm
    self.enable_thinking = enable_thinking

  def generate(
      self,
      messages,
      *,
      role,
      actions=(),
      temperature=None,
      max_tokens=None,
      enable_thinking=None,
  ):
    thinking = (
        self.enable_thinking if enable_thinking is None else enable_thinking
    )
    if actions:
      converted = self.llm._text_action_messages(messages, actions)  # pylint: disable=protected-access
      systems = [
          str(m.get("content") or "")
          for m in converted
          if m.get("role") in ("system", "developer")
      ]
      messages = [
          {"role": "system", "content": "\n\n".join(systems)},
          *[
              m
              for m in converted
              if m.get("role") not in ("system", "developer")
          ],
      ]
    else:
      messages = self.llm._openai_messages(messages)  # pylint: disable=protected-access
    call, _, finish = self.transport.request(
        self.runtime,
        messages,
        role=role,
        max_tokens=max_tokens,
        enable_thinking=thinking,
        temperature=temperature,
    )
    raw_reasoning, raw_action = "", call.text
    if "</think>" in raw_action:
      prefix, raw_action = raw_action.split("</think>", 1)
      raw_reasoning = prefix + "</think>"
    return self.llm.LLMResponse(
        text=call.text,
        model=self.runtime.policy_id,
        assistant_message={"role": "assistant", "content": call.text},
        token_in=len(call.prompt_ids),
        token_out=len(call.completion_ids),
        metadata={
            "route": (
                "worker" if self.runtime.policy_id == "worker" else "director"
            ),
            "finish_reason": finish,
            "qwen_text_actions": bool(actions),
            "enable_thinking": bool(thinking),
            "executor": "tunix-native",
        },
        raw_reasoning_text=raw_reasoning,
        raw_action_text=raw_action,
        prompt_token_ids=call.prompt_ids,
        completion_token_ids=call.completion_ids,
        behavior_log_probs=call.rollout_logprobs,
        token_provenance=(
            "provider_prompt_and_completion_token_ids_and_logprobs"
        ),
        training_eligible=True,
    )

  def choose_binary(self, messages, *, role, token_ids):
    support = tuple(int(token_ids[key]) for key in ("off", "on"))
    call, logps, _ = self.transport.request(
        self.runtime,
        self.llm._openai_messages(messages),
        role=role,  # pylint: disable=protected-access
        support=support,
        enable_thinking=False,
    )
    return self.llm.BinaryChoiceResponse(
        choice=call.text,
        model=self.runtime.policy_id,
        probabilities={
            key: float(np.exp(logps[i])) for i, key in enumerate(("off", "on"))
        },
        log_probabilities=dict(
            zip(("off", "on"), map(float, logps), strict=True)
        ),
        token_ids=dict(token_ids),
        prompt_token_ids=call.prompt_ids,
        token_in=len(call.prompt_ids),
        token_out=1,
        metadata={"route": "director", "executor": "tunix-native"},
    )


class SourceDirectorWorkflow:
  """Use the official graph/controller with separate trainable/frozen policies."""

  def __init__(
      self,
      source,
      *,
      tokenizer,
      binary_sampler: BinarySampler,
      grader,
      tools=None,
      max_agents=6,
      max_rounds=24,
      parallel_workers=4,
      max_total_tokens=240000,
      timeout_seconds=300,
      enable_thinking=False,
      director_context_mode="append_only",
  ):
    if any(
        type(x) is not int or x < 1
        for x in (max_agents, max_rounds, parallel_workers, max_total_tokens)
    ):
      raise ValueError("Source workflow budgets must be positive integers.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
      raise ValueError("Source workflow timeout must be finite and positive.")
    if not callable(binary_sampler):
      raise ValueError("Source workflow requires a constrained binary sampler.")
    self.source, self.tokenizer, self.binary_sampler = (
        source,
        tokenizer,
        binary_sampler,
    )
    self.grader, self.tools = grader, dict(tools or {})
    self.timeout_seconds, self.enable_thinking = (
        timeout_seconds,
        enable_thinking,
    )
    if director_context_mode not in ("append_only", "snapshot_dedup"):
      raise ValueError("Unsupported original Director context mode.")
    self.director_context_mode = director_context_mode
    self.modules = {
        name: source_adapter.load_source_module(source, name)
        for name in (
            "llm",
            "canvas",
            "config",
            "director",
            "runtime",
            "dataset_actions",
            "deadline",
            "worker_cost",
            "run_config",
            "submission_contract",
            "outcome_admission",
        )
    }
    config_path = (
        Path(self.modules["config"].__file__).parents[2] / "configs/r2orch.toml"
    )
    original = tomllib.loads(config_path.read_text())
    self.canvas_settings = dict(original["canvas"])
    self.canvas_settings.update(
        max_agents=max_agents,
        max_rounds=max_rounds,
        parallel_components_max=parallel_workers,
        max_total_tokens=max_total_tokens,
        max_total_tokens_by_dataset={"aime": max_total_tokens},
    )
    cost = dict(original["r2orch"]["cost"])
    cost["lambda_"] = cost.pop("lambda")
    self.cost = self.modules["run_config"].CostConfig(**cost)
    self.max_tool_rounds = original["retrieval"]["max_tool_rounds"]

  async def run(
      self,
      task,
      director,
      worker,
      *,
      episode_id,
      skills=(),
      scaffold_condition="none",
  ):
    if task.benchmark != "aime":
      raise ValueError(
          "The source bridge currently supports the AIME environment only."
      )
    if not isinstance(skills, tuple) or any(
        not isinstance(s, str) for s in skills
    ):
      raise ValueError("Skills must be an immutable tuple of text cards.")
    cancellation = threading.Event()
    deadline = self.modules["deadline"].RolloutDeadline(
        total_timeout_s=self.timeout_seconds,
        no_progress_timeout_s=self.timeout_seconds,
        request_timeout_s=self.timeout_seconds,
        cancellation_event=cancellation,
        cancellation_reason="tunix_workflow_cancelled",
    )
    transport = _Transport(
        loop=asyncio.get_running_loop(),
        tokenizer=self.tokenizer,
        binary_sampler=self.binary_sampler,
        deadline=deadline,
    )

    def execute():
      with tempfile.TemporaryDirectory(
          prefix="tunix-r2orch-submission-"
      ) as journals:
        config = self.modules["config"].CanvasConfig(
            **{**self.canvas_settings, "submission_journal_dir": journals}
        )
        registry = self.modules[
            "dataset_actions"
        ].default_dataset_action_registry(self.tools)
        backend = _ChatBackend(
            transport,
            worker,
            self.modules["llm"],
            enable_thinking=self.enable_thinking,
        )
        executor = self.modules["runtime"].ModelAgentExecutor(
            backend,
            tools=self.tools,
            action_registry=registry,
            max_tool_rounds=self.max_tool_rounds,
            rollout_deadline=deadline,
        )
        runtime = self.modules["runtime"].MultiAgentRuntime(
            executor,
            relay_max_chars=config.relay_max_chars,
            seed=director.params.seed or 0,
            bidirectional_revision_confidence_threshold=config.bidirectional_revision_confidence_threshold,
            parallel_components_max=config.parallel_components_max,
        )
        canvas = self.modules["canvas"].GraphCanvas(
            task=task.problem,
            worker_task=task.problem,
            director_task=task.problem,
            runtime=runtime,
            config=config,
            runtime_routes=("worker",),
            task_type="math",
            dataset="aime",
            action_adapter=registry.get("aime"),
            rollout_deadline=deadline,
        )
        controller = self.modules["director"].GraphDirector(
            backend=_ChatBackend(
                transport,
                director,
                self.modules["llm"],
                enable_thinking=self.enable_thinking,
            ),
            canvas=canvas,
            solver_skill_context="\n\n".join(skills),
            enable_thinking=self.enable_thinking,
            tokenizer=self.tokenizer,
            call_namespace=episode_id,
        )
        # Match the paper launcher's explicit setting without changing a
        # process-global variable while other episodes execute.
        controller.context_mode = self.director_context_mode
        controller.context_schema = (
            "append_only_action_feedback_history_with_thinking"
            if self.director_context_mode == "append_only"
            else self.modules["director"].DIRECTOR_CONTEXT_SCHEMA
        )
        run = controller.run()
        binding_error = self.modules["submission_contract"].receipt_error(
            run.submission_receipt,
            run=run,
            events=canvas.history,
            run_id=episode_id,
            dataset="aime",
        )
        artifacts = {
            key: {
                "integrity_risks": list(artifact.integrity_risks),
                "runtime_tool_evidence": dict(artifact.runtime_tool_evidence),
                "backend_failure": (
                    artifact.answer
                    == self.modules["runtime"].WORKER_BACKEND_FAILURE_SENTINEL
                ),
            }
            for key, artifact in runtime.artifacts.items()
        }
        historical = {}
        for step in canvas.history:
          if step.execution is None:
            continue
          for artifact in (
              *step.execution.artifacts.values(),
              *step.execution.attempt_artifacts,
          ):
            historical[artifact.artifact_id] = {
                "runtime_tool_evidence": dict(artifact.runtime_tool_evidence),
                "backend_failure": (
                    artifact.answer
                    == self.modules["runtime"].WORKER_BACKEND_FAILURE_SENTINEL
                ),
            }
        terminal_failure = self.modules[
            "outcome_admission"
        ].terminal_policy_failure(
            "aime",
            terminal=not canvas.active,
            rejection_codes=[
                step.rejection_code
                for step in canvas.history
                if step.rejection_code
            ],
            artifacts=artifacts,
            historical_artifacts=historical,
            output_agent=canvas.graph.output_agent,
            rounds=canvas.round_index,
            max_rounds=canvas.director_round_limit,
            worker_tokens=canvas.total_tokens,
            worker_token_limit=config.max_total_tokens,
            infrastructure_failure=any(
                item["backend_failure"] for item in artifacts.values()
            ),
        )
        cost = self.modules["worker_cost"].worker_cost_record(
            self.modules["worker_cost"].worker_usage_from_history(
                canvas.history
            ),
            dataset="aime",
            cost=self.cost,
            canvas_total_tokens=canvas.total_tokens,
        )
        graph = {
            **run.graph,
            "source_commit": SOURCE_COMMIT,
            "executor": "original_canvas",
            "director_context_mode": self.director_context_mode,
            "tools": sorted(self.tools),
            "history": [step.to_dict() for step in canvas.history],
            "turns": [turn.to_dict() for turn in run.turns],
            "submission_receipt": (
                run.submission_receipt.to_dict()
                if run.submission_receipt
                else None
            ),
            "worker_cost_record": cost,
            "submission_binding_error": binding_error,
            "typed_policy_failure": terminal_failure,
        }
        return run, graph, canvas.state.value, cost["worker_cost"]

    thread = asyncio.create_task(asyncio.to_thread(execute))
    try:
      run, graph, termination, cost = await asyncio.wait_for(
          asyncio.shield(thread), timeout=self.timeout_seconds
      )
    except BaseException:
      cancellation.set()
      await transport.close()
      # Cancelling to_thread alone leaves the Canvas mutating in the background.
      # Source checks + cancelled model requests make the thread exit cooperatively.
      await asyncio.gather(thread, return_exceptions=True)
      raise
    finally:
      await transport.close()
    calls = tuple(transport.calls[key] for key in sorted(transport.calls))
    if graph["submission_binding_error"] and not graph["typed_policy_failure"]:
      raise SourceOutcomeUnavailable(
          graph["submission_binding_error"],
          graph=graph,
          calls=calls,
      )
    submitted = not graph["submission_binding_error"]
    final_text = (
        run.submission_receipt.submitted_answer_snapshot if submitted else ""
    )
    correct = bool(submitted and self.grader(final_text, task.answer))
    episode = trace.SwarmEpisode(
        task_id=task.task_id,
        episode_id=episode_id,
        calls=calls,
        reward=float(correct),
        final_text=final_text,
        worker_cost=cost,
        scaffold_condition=scaffold_condition,
    )
    return workflow.Result(
        episode, graph, "finished" if submitted else termination, correct
    )
