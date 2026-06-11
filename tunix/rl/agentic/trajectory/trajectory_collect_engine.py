# Copyright 2025 Google LLC
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

"""Engine for collecting trajectories from agent-environment interactions.

This module defines the `TrajectoryCollectEngine`, which facilitates the
asynchronous collection of rollouts by managing the interaction loop between
an LLM-based agent and an environment. It supports single and concurrent
multi-pair trajectory collection.
"""
import asyncio
import json
import time
from typing import Any, AsyncGenerator, Callable, Concatenate, Dict, List, Optional, ParamSpec, Set, Tuple

from absl import logging
import numpy as np
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.rollout import base_rollout

P = ParamSpec("P")

BaseTaskEnv = base_environment.BaseTaskEnv
ConversationAgentBase = base_agent.ConversationAgentBase


class TrajectoryCollectEngine:
  """Asynchronous trajectory collection engine for agent-env interactions.

  This engine orchestrates complete rollout episodes by managing the interaction
  loop between LLM-based agents and environments. It handles model inference,
  environment stepping, reward computation, and trajectory storage with support
  for concurrent multi-pair execution and streaming results.

  The engine implements the standard RL rollout pattern: reset → step* → final
  reward computation → return calculation, while providing flexible callback
  integration for custom model calls and reward functions.
  """

  def __init__(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      model_call_kwargs: Optional[Dict[str, Any]] = None,
      gamma: float = 1.0,
      max_response_length: Optional[int] = None,
      timeout: float = 600.0,
      tokenizer=None,
      chat_parser=None,
      filter_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
      overlong_filter: bool = False,
      perf_v2: Optional[perf_tracer_v2.Tracer] = None,
  ):
    """Initialize the trajectory collection engine.

    Args:
        agent (ConversationAgentBase): The agent that will interact with the
          environment
        env (BaseTaskEnv): The environment providing tasks and feedback
        model_call (Callable): Function that takes chat completions as first
          argument with optional kwargs and returns model response string.
          Handles the actual LLM inference.
        model_call_kwargs (Optional[Dict[str, Any]]): Optional kwargs to pass to
          model_call.
        final_reward_fn (Optional[Callable]): Optional function to compute
          additional reward at episode end. Takes (task, response) and returns
          float. Defaults to zero if not provided.
        gamma (float): Discount factor for MC reward calculation (1.0 = no
          discounting).
        max_response_length (Optional[int]): Maximum number of context tokens to
          use before forced termination.
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs. This
          is required if we want to track down token counts.
        chat_parser: Optional chat parser for formatting messages
        filter_statuses (Set[TrajectoryStatus]): A set of statuses that are
          masked out for overlong filtering.
        overlong_filter: Whether to filter overlong trajectories.
        perf_v2 (Optional[perf_tracer_v2.Tracer]): Optional performance tracer
          to use for performance measurements. Defaults to a no-op tracer.
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = None
    self.model_call_kwargs = model_call_kwargs or {}
    self.perf_v2 = (
        perf_v2 if perf_v2 is not None else perf_tracer_v2.NoopTracer()
    )
    self.max_steps = getattr(self.env, "max_steps", 1)
    self.gamma = gamma
    self.max_response_length = max_response_length
    self._response_token_count = 0
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self._start_ts: float = 0.0
    self.filter_statuses = filter_statuses or {
        agent_types.TrajectoryStatus.MAX_STEPS_REACHED,
        agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
        agent_types.TrajectoryStatus.TIMEOUT,
        agent_types.TrajectoryStatus.ENV_TIMEOUT,
    }

    self.overlong_filter = overlong_filter
    self.perf_v2 = perf_v2 or perf_tracer_v2.NoopTracer()
    self.env_time = {
        "reset_latency": 0.0,  # Wall-clock time (Total real-world time elapsed)
        "reset_cpu_time": (
            0.0
        ),  # Thread/CPU time (Actual processing time on the worker thread)
        "step_latency": 0.0,  # Wall-clock time (Total real-world time elapsed)
        "step_cpu_time": (
            0.0
        ),  # Thread/CPU time (Actual processing time on the worker thread)
    }
    self.reward_time = {
        "reward_latency": (
            0.0
        ),  # Wall-clock time (Total real-world time elapsed)
        "reward_cpu_time": (
            0.0
        ),  # Thread/CPU time (Actual processing time on the worker thread)
    }

    if self.max_response_length is not None and not (
        self.tokenizer and self.chat_parser
    ):
      logging.warning(
          "max_response_length is set to %d, but no tokenizer or chat_parser is"
          " provided. response length limits will not be enforced.",
          self.max_response_length,
      )

  async def _run_with_timing(
      self, func: Callable[..., Any], *args, timeout: Optional[float] = None
  ) -> Tuple[Any, float, float]:
    """Runs a sync function in an executor and returns (result, wall_time, cpu_time).

    Args:
      func: Synchronous callable to run in the executor.
      *args: Positional arguments forwarded to func.
      timeout: Optional deadline in seconds. If provided and the executor future
        does not finish in time, asyncio.TimeoutError is re-raised to the
        caller.

    Raises:
      asyncio.TimeoutError: When timeout is not None and is exceeded.
    """

    def _clocked_wrapper():
      t_start = time.thread_time()
      res = func(*args)
      t_delta = time.thread_time() - t_start
      return res, t_delta

    loop = asyncio.get_running_loop()
    wall_start = time.perf_counter()

    fut = loop.run_in_executor(None, _clocked_wrapper)
    if timeout is not None:
      result, cpu_delta = await asyncio.wait_for(fut, timeout=timeout)
    else:
      result, cpu_delta = await fut

    wall_delta = time.perf_counter() - wall_start
    return result, wall_delta, cpu_delta

  async def collect(self, mode: str = "Conversation") -> Any:
    """Execute a complete rollout episode and return the resulting trajectory.

    Orchestrates the full interaction sequence: environment reset, iterative
    agent-environment steps, final reward computation, Monte Carlo return
    calculation, and resource cleanup.

    Args:
        mode (str): Output format. Options: 
          - "Trajectory": return full Trajectory object.
          - "Token": return flattened tokenized dict for training.
          - "Steps": return stepwise tokenized data only.
          - "Conversation": return raw conversation messages (default).

    Returns:
        Trajectory | dict | list: Depending on mode.
    """  # fmt: skip
    await self._reset()

    self.agent.trajectory.status = agent_types.TrajectoryStatus.RUNNING

    while True:
      if len(self.agent.trajectory.steps) >= self.max_steps:
        self.agent.trajectory.status = (
            agent_types.TrajectoryStatus.MAX_STEPS_REACHED
        )
        break

      done = await self._one_step()

      if done:
        if self.agent.trajectory.status == agent_types.TrajectoryStatus.RUNNING:
          self.agent.trajectory.status = agent_types.TrajectoryStatus.SUCCEEDED
        break

    masked_out = (
        self.overlong_filter
        and self.agent.trajectory.status in self.filter_statuses
    )
    try:
      if not masked_out:
        await self._append_final_reward()
      else:
        logging.debug(
            "%s mask out trajectory due to status %s",
            self._debug_prefix,
            self.agent.trajectory.status.name,
        )
      self.compute_mc_reward()
      self.compute_trajectory_reward()
    finally:
      await self._close()

    if mode not in ["Trajectory", "Steps", "Token", "Conversation"]:
      raise ValueError(
          f"Unsupported mode: {mode}, currently supported modes: "
          f" {['Trajectory', 'Steps', 'Token', 'Conversation']}",
      )

    if mode == "Trajectory":
      self.agent.trajectory.env_time = self.env_time
      self.agent.trajectory.reward_time = self.reward_time
      return self.agent.trajectory
    elif mode == "Steps":
      return [
          {
              "assistant_text": getattr(step, "model_response", ""),
              "env_text": getattr(step, "observation", ""),
              "done": getattr(step, "done", False),
              "assistant_tokens": getattr(step, "assistant_tokens", []),
              "assistant_masks": getattr(step, "assistant_masks", []),
              "env_tokens": getattr(step, "env_tokens", []),
              "env_masks": getattr(step, "env_masks", []),
              "reward": step.reward,
              "mc_return": step.mc_return,
              "env_time": self.env_time,
              "reward_time": self.reward_time,
          }
          for step in self.agent.trajectory.steps
      ]
    elif mode == "Token":
      self._strip_past_thinking = True
      # flatten all steps into single batch dict
      conversation_tokens, conversation_masks, logprobs = [], [], []
      prompt_tokens = getattr(self.agent.trajectory, "prompt_tokens", [])
      # TODO: need to remove `"<|channel>thought\n<channel|>"` from prompt_tokens if there is turns before the last turn.
      if isinstance(prompt_tokens, (int, np.integer)):
        prompt_tokens = np.array([prompt_tokens], dtype=np.int32)
      elif isinstance(prompt_tokens, np.ndarray) and prompt_tokens.ndim == 0:
        prompt_tokens = np.array([prompt_tokens.item()], dtype=np.int32)

      # Unpad prompt_tokens to get the real unpadded prompt length
      pad_id = self.tokenizer.pad_id() if self.tokenizer else 0
      prompt_tokens_np = np.asarray(prompt_tokens)
      non_pad_indices = np.where(prompt_tokens_np != pad_id)[0]
      if len(non_pad_indices) > 0:
        prompt_tokens = prompt_tokens_np[non_pad_indices[0]:]
      else:
        prompt_tokens = prompt_tokens_np

      suffix_tokens = self.chat_parser.thought_suffix_tokens if self.chat_parser else []

      last_step_has_env = False
      if self.agent.trajectory.steps:
        last_step_has_env = getattr(self.agent.trajectory.steps[-1], "env_tokens", None) is not None

      # if (
      #     self._strip_past_thinking
      #     and len(self.agent.trajectory.steps) > 1
      # ):
      #   prompt_tokens, _ = self._strip_suffix(prompt_tokens, None, suffix_tokens)

      for idx, step in enumerate(self.agent.trajectory.steps):
        # Keep tokens/masks/logprobs appended in lockstep — a step with env_tokens
        # but no vllm logprobs (initial observation, empty completion) would
        # otherwise leave the logprobs array short by `len(env_tokens)` and offset
        # every subsequent step.
        assistant_tokens = getattr(step, "assistant_tokens", None)
        env_tokens = getattr(step, "env_tokens", None)
        step_logprobs = getattr(step, "logprobs", None)
        if assistant_tokens is not None:
          assistant_masks = step.assistant_masks
          # If this is a past turn, make sure assistant_tokens ends with "<turn|>\n"
          if (
              self.chat_parser is not None
              and idx < len(self.agent.trajectory.steps) - 1
          ):
            eot_tokens = self.chat_parser.eot_tokens
            assistant_tokens_list = list(assistant_tokens)
            n_eot = len(eot_tokens)
            if eot_tokens and (
                len(assistant_tokens_list) < n_eot 
                or assistant_tokens_list[-n_eot:] != list(eot_tokens)
            ):
              turn_only_tokens = self.chat_parser.turn_only_tokens
              n_turn_only = len(turn_only_tokens)
              
              if turn_only_tokens and len(assistant_tokens_list) >= n_turn_only and assistant_tokens_list[-n_turn_only:] == list(turn_only_tokens):
                to_append = self.chat_parser.newline_tokens
              else:
                to_append = eot_tokens
              
              if to_append:
                assistant_tokens = np.concatenate([assistant_tokens, np.array(to_append, dtype=np.int32)], axis=0)
                assistant_masks = np.concatenate([assistant_masks, np.zeros(len(to_append), dtype=np.int32)], axis=0)
                if step_logprobs is not None:
                  step_logprobs = np.concatenate([step_logprobs, np.zeros(len(to_append), dtype=np.float32)], axis=0)

          conversation_tokens.append(assistant_tokens)
          conversation_masks.append(assistant_masks)
          if step_logprobs is not None:
            assert len(step_logprobs) == len(assistant_tokens), (
                f"Logprobs length {len(step_logprobs)} does not match assistant"
                f" tokens length {len(assistant_tokens)}"
            )
            logprobs.append(step_logprobs)
          else:
            logprobs.append(np.zeros(len(assistant_tokens)))
        if env_tokens is not None and idx < len(self.agent.trajectory.steps) - 1:
          # Skip appending the final step's env_tokens to conversation_tokens
          env_masks = step.env_masks
          # if self._strip_past_thinking:
          #   if idx < len(self.agent.trajectory.steps) - 2 or (
          #       idx == len(self.agent.trajectory.steps) - 1 and last_step_has_env
          #   ):
          #     env_tokens, env_masks = self._strip_suffix(env_tokens, env_masks, suffix_tokens)

          conversation_tokens.append(env_tokens)
          conversation_masks.append(env_masks)
          logprobs.append(np.zeros(len(env_tokens)))

      # Reconstruct last-turn fields
      last_step = self.agent.trajectory.steps[-1] if self.agent.trajectory.steps else None
      # last_turn_prompt_tokens = np.array([], dtype=np.int32)
      # last_turn_tokens = np.array([], dtype=np.int32)
      # last_turn_prompt_logprobs = np.array([], dtype=np.float32)
      # last_turn_logprobs = np.array([], dtype=np.float32)

      # if last_step is not None:
      #   if last_step.assistant_tokens is not None:
      #     last_turn_tokens = np.asarray(last_step.assistant_tokens)
      #   if last_step.logprobs is not None:
      #     last_turn_logprobs = np.asarray(last_step.logprobs)
      #   if last_step.info:
      #     unpadded_p = last_step.info.get('prompt_tokens_unpadded')
      #     unpadded_lp = last_step.info.get('prompt_logprobs_unpadded')
      #     if unpadded_p is not None:
      #       last_turn_prompt_tokens = np.asarray(unpadded_p)
      #     if unpadded_lp is not None:
      #       last_turn_prompt_logprobs = np.asarray(unpadded_lp)

      conversation_tokens = [
          np.asarray(tokens)
          for tokens in conversation_tokens
          if len(tokens) > 0
      ]
      conversation_masks = [
          np.asarray(masks) for masks in conversation_masks if len(masks) > 0
      ]
      logprobs = [
          np.asarray(step_logprobs)
          for step_logprobs in logprobs
          if len(step_logprobs) > 0
      ]
      conversation_masks = (
          np.concatenate(conversation_masks, axis=0)
          if conversation_masks
          else np.array([], dtype=np.int32)
      )
      conversation_tokens = (
          np.concatenate(conversation_tokens, axis=0)
          if conversation_tokens
          else np.array([], dtype=np.int32)
      )
      logprobs = (
          np.concatenate(logprobs, axis=0)
          if logprobs
          else np.array([], dtype=np.float32)
      )
      final_masks = (
          np.zeros_like(conversation_masks)
          if masked_out
          else conversation_masks
      )
      # # Comparison: concatenate prompt_tokens with conversation_tokens and compare with concatenated last_turn_prompt_tokens and last_turn_tokens.
      # if self.tokenizer is not None and len(last_turn_prompt_tokens) > 0:
      #   seq1 = np.concatenate([prompt_tokens, conversation_tokens], axis=0)
      #   seq2 = np.concatenate([last_turn_prompt_tokens, last_turn_tokens], axis=0)
        
      #   # Unpad any trailing zeros/padding from seq1/seq2 just in case
      #   pad_id = self.tokenizer.pad_id() if self.tokenizer else 0
      #   if len(seq1) > 0:
      #     non_pad1 = np.where(seq1 != pad_id)[0]
      #     if len(non_pad1) > 0:
      #       seq1 = seq1[:non_pad1[-1] + 1]
      #   if len(seq2) > 0:
      #     non_pad2 = np.where(seq2 != pad_id)[0]
      #     if len(non_pad2) > 0:
      #       seq2 = seq2[:non_pad2[-1] + 1]
            
      #   arrays_equal = np.array_equal(seq1, seq2)
      #   logging.info(
      #       "%s [Comparison] prompt+conv length: %d, last_turn_prompt+last_turn length: %d, equal: %s",
      #       self._debug_prefix,
      #       len(seq1),
      #       len(seq2),
      #       arrays_equal,
      #   )

      #   # 1. Compare lengths of masks, tokens, and logprobs for compiled trajectory
      #   compiled_conv_len = len(conversation_tokens)
      #   compiled_mask_len = len(final_masks)
      #   compiled_logp_raw = logprobs
      #   compiled_logp_len = len(compiled_logp_raw)
        
      #   logging.info(
      #       "%s [Compiled Lengths] conversation_tokens: %d, final_masks: %d, logprobs: %d",
      #       self._debug_prefix,
      #       compiled_conv_len,
      #       compiled_mask_len,
      #       compiled_logp_len,
      #   )
      #   if compiled_conv_len != compiled_mask_len or compiled_conv_len != compiled_logp_len:
      #     logging.error(
      #         "%s [Compiled Length Mismatch!] conversation_tokens=%d, masks=%d, logprobs=%d",
      #         self._debug_prefix,
      #         compiled_conv_len,
      #         compiled_mask_len,
      #         compiled_logp_len,
      #     )

      #   # 2. Compare masked logprobs
      #   last_turn_prompt_logprobs_np = np.asarray(last_turn_prompt_logprobs) if last_turn_prompt_logprobs is not None else np.array([], dtype=np.float32)
      #   last_turn_logprobs_np = np.asarray(last_turn_logprobs) if last_turn_logprobs is not None else np.array([], dtype=np.float32)
        
      #   n_initial_prompt = len(prompt_tokens)
      #   if len(last_turn_prompt_logprobs_np) >= n_initial_prompt:
      #     sliced_prompt_logprobs = last_turn_prompt_logprobs_np[n_initial_prompt:]
      #     logprobs_rollout_full = np.concatenate([sliced_prompt_logprobs, last_turn_logprobs_np], axis=0)
          
      #     logging.info(
      #         "%s [Logprobs Lengths] compiled: %d, rollout: %d",
      #         self._debug_prefix,
      #         compiled_logp_len,
      #         len(logprobs_rollout_full),
      #     )
          
      #     if len(compiled_logp_raw) == len(logprobs_rollout_full):
      #       # Compare active masked logprobs
      #       active_indices = np.where(final_masks == 1)[0]
      #       if len(active_indices) > 0:
      #         compiled_active_lp = compiled_logp_raw[active_indices]
      #         rollout_active_lp = logprobs_rollout_full[active_indices]
              
      #         lp_diff = np.abs(compiled_active_lp - rollout_active_lp)
      #         max_diff = np.max(lp_diff)
      #         mean_diff = np.mean(lp_diff)
      #         logging.info(
      #             "%s [Active Logprobs Agreement] max_diff: %.6f, mean_diff: %.6f on %d active tokens",
      #             self._debug_prefix,
      #             max_diff,
      #             mean_diff,
      #             len(active_indices),
      #         )
      #         if max_diff > 1e-4:
      #           first_diff_idx = np.where(lp_diff > 1e-4)[0][0]
      #           actual_idx = active_indices[first_diff_idx]
      #           logging.warning(
      #               "%s [Active Logprobs Mismatch] First mismatch at seq index %d (active index %d). "
      #               "Compiled logprob: %.6f, Rollout logprob: %.6f, token ID: %d (%r)",
      #               self._debug_prefix,
      #               actual_idx,
      #               first_diff_idx,
      #               compiled_active_lp[first_diff_idx],
      #               rollout_active_lp[first_diff_idx],
      #               conversation_tokens[actual_idx],
      #               self.tokenizer.decode([int(conversation_tokens[actual_idx])]) if self.tokenizer else "N/A"
      #           )
      #           if self.tokenizer is not None:
      #             decoded_tokens = [self.tokenizer.decode([int(t)]) for t in conversation_tokens]
      #             details_list = []
      #             for i, (tok_str, c_lp, r_lp) in enumerate(zip(decoded_tokens, compiled_logp_raw, logprobs_rollout_full)):
      #               is_active = "*" if final_masks[i] == 1 else " "
      #               details_list.append(f"{is_active} [{i:03d}] {tok_str!r} ({conversation_tokens[i]}): {c_lp:.4f} vs {r_lp:.4f}")
      #             logging.warning(
      #                 "%s [Active Logprobs Mismatch Details]\n%s",
      #                 self._debug_prefix,
      #                 "\n".join(details_list),
      #             )
      #             decoded_seq1 = self.tokenizer.decode(list(seq1))
      #             decoded_seq2 = self.tokenizer.decode(list(seq2))
      #             logging.warning(
      #                 "%s [Active Logprobs Mismatch Sequences]\n=== compiled seq (seq1) ===\n%s\n=========================\n=== rollout seq (seq2) ===\n%s\n=========================",
      #                 self._debug_prefix,
      #                 decoded_seq1,
      #                 decoded_seq2,
      #             )
      #           else:
      #             logging.warning(
      #                 "%s [Active Logprobs Mismatch Details]\n=== compiled_logp_raw ===\n%s\n=========================\n=== logprobs_rollout_full ===\n%s\n=========================",
      #                 self._debug_prefix,
      #                 compiled_logp_raw.tolist(),
      #                 logprobs_rollout_full.tolist(),
      #             )
      #       else:
      #         logging.info("%s [Active Logprobs Agreement] No active tokens in final_masks.", self._debug_prefix)
      #     else:
      #       logging.error(
      #           "%s [Logprobs Length Mismatch!] cannot compare active logprobs. compiled=%d, rollout=%d",
      #           self._debug_prefix,
      #           compiled_logp_len,
      #           len(logprobs_rollout_full),
      #       )
      #   else:
      #     logging.error(
      #         "%s [Logprobs Slicing Mismatch] last_turn_prompt_logprobs len (%d) is shorter than initial prompt len (%d)",
      #         self._debug_prefix,
      #         len(last_turn_prompt_logprobs_np),
      #         n_initial_prompt,
      #     )
      #   if not arrays_equal:
      #     decoded_seq1 = self.tokenizer.decode(list(seq1)) if self.tokenizer else "N/A"
      #     decoded_seq2 = self.tokenizer.decode(list(seq2)) if self.tokenizer else "N/A"
          
      #     logging.warning(
      #         "%s [Comparison Mismatch]\n=== seq1 (compiled) ===\n%s\n=======================\n=== seq2 (rollout) ===\n%s\n=======================",
      #         self._debug_prefix,
      #         decoded_seq1,
      #         decoded_seq2,
      #     )
          
      #     min_len = min(len(seq1), len(seq2))
      #     mismatch_idx = -1
      #     for idx in range(min_len):
      #       if seq1[idx] != seq2[idx]:
      #         mismatch_idx = idx
      #         break
      #     if mismatch_idx != -1:
      #       t1_decoded = self.tokenizer.decode([int(seq1[mismatch_idx])]) if self.tokenizer else "N/A"
      #       t2_decoded = self.tokenizer.decode([int(seq2[mismatch_idx])]) if self.tokenizer else "N/A"
            
      #       context_start = max(0, mismatch_idx - 10)
      #       context_end_1 = min(len(seq1), mismatch_idx + 10)
      #       context_end_2 = min(len(seq2), mismatch_idx + 10)
            
      #       context1 = self.tokenizer.decode(list(seq1[context_start:context_end_1])) if self.tokenizer else "N/A"
      #       context2 = self.tokenizer.decode(list(seq2[context_start:context_end_2])) if self.tokenizer else "N/A"
            
      #       logging.warning(
      #           "%s [Comparison Mismatch Details] First mismatch at index %d. seq1 token: %d (%r), seq2 token: %d (%r)\nseq1 context: %r\nseq2 context: %r",
      #           self._debug_prefix,
      #           mismatch_idx,
      #           seq1[mismatch_idx],
      #           t1_decoded,
      #           seq2[mismatch_idx],
      #           t2_decoded,
      #           context1,
      #           context2,
      #       )
      #     else:
      #       logging.warning(
      #           "%s [Comparison Mismatch] One sequence is prefix of another. seq1 len: %d, seq2 len: %d",
      #           self._debug_prefix,
      #           len(seq1),
      #           len(seq2),
      #       )
      return {
          "conversation_text": self.agent.chat_completions,
          "prompt_tokens": prompt_tokens,
          "conversation_tokens": conversation_tokens,
          "conversation_masks": final_masks,
          "status": self.agent.trajectory.status.name,
          "steps_count": len(self.agent.trajectory.steps),
          "trajectory_reward": self.agent.trajectory.reward,
          "env_time": self.env_time,
          "reward_time": self.reward_time,
          "old_logprobs": logprobs,
          # "last_turn_prompt_tokens": last_turn_prompt_tokens,
          # "last_turn_tokens": last_turn_tokens,
          # "last_turn_prompt_logprobs": last_turn_prompt_logprobs,
          # "last_turn_logprobs": last_turn_logprobs,
          "policy_version": self.env.task.get("policy_version"),
          "original_input": self.agent.trajectory.task,
          "group_id": self.env.extra_kwargs.get("group_id"),
      }
    elif mode == "Conversation":
      # return raw conversation history
      return self.agent.chat_completions

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[ConversationAgentBase, BaseTaskEnv]],
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      gamma: float = 1.0,
      timeout: float = 30.0,
      max_response_length: Optional[int] = None,
      mode: str = "Trajectory",
      filter_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
      overlong_filter: bool = True,
      perf_v2: Optional[perf_tracer_v2.Tracer] = None,
  ) -> AsyncGenerator[Tuple[int, Any], None]:
    """Execute multiple agent-environment pairs concurrently.

    Runs multiple rollouts in parallel and yields completed trajectories
    as they finish, enabling efficient batch processing with streaming
    results. Useful for distributed training or large-scale evaluation.

    Args:
        pairs (List[Tuple[ConversationAgentBase, BaseTaskEnv]]): List of (agent,
          environment) pairs
        model_call (Callable): Shared model inference function for all pairs
        gamma (float): Discount factor for return calculation
        max_response_length (Optional[int]): Maximum context limit per episode
        timeout (float): Per-episode timeout in seconds
        mode (str): Output format. See `collect` method for options.
        filter_statuses (Optional[Set[TrajectoryStatus]]): A set of statuses
          that are masked out for filtering.
        overlong_filter (bool): Whether to filter overlong trajectories.
        perf_v2 (Optional[perf_tracer_v2.Tracer]): Optional performance tracer
          to use for performance measurements.

    Yields:
        Tuple[int, Any]: `(pair_index, result)`. The type of `result`
          depends on the `mode` argument. See the `collect` method for details.
    """

    async def _run_one(i: int, agent: ConversationAgentBase, env: BaseTaskEnv):
      """Execute a single agent-env pair with the given configuration."""
      engine = TrajectoryCollectEngine(
          agent,
          env,
          model_call=model_call,
          gamma=gamma,
          max_response_length=max_response_length,
          timeout=timeout,
          filter_statuses=filter_statuses,
          overlong_filter=overlong_filter,
          perf_v2=perf_v2,
      )
      traj = await engine.collect(mode=mode)
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, agent, env) for i, (agent, env) in enumerate(pairs)]
    for coro in asyncio.as_completed(tasks):
      yield await coro

  async def _reset(self):
    """Resets the environment and agent at the beginning of a new episode.

    This involves calling the environment's reset method, updating the agent's
    state, and optionally tokenizing the initial prompt messages.
    """
    logging.debug("%s env.reset starting", self._debug_prefix)
    (obs, _), wall_time, cpu_time = await self._run_with_timing(self.env.reset)
    logging.debug(
        "%s env.reset done in %.1fs",
        self._debug_prefix,
        wall_time,
    )

    self.env_time["reset_latency"] += wall_time
    self.env_time["reset_cpu_time"] += cpu_time
    self.final_reward_fn = (
        self.env.final_reward_fn
        if hasattr(self.env, "final_reward_fn")
        else None
    )
    self.agent.reset()
    self.agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

    if self.tokenizer is not None and self.chat_parser is not None:
      # Get the current messages (usually System + User)
      init_messages = self.agent.chat_completions
      prompt_tokens, _ = utils.tokenize_and_generate_masks(
          init_messages,
          tokenizer=self.tokenizer,
          parser=self.chat_parser,
          contains_first_msg=True,
          contains_generation_msg=True,
      )
      self.agent.trajectory.prompt_tokens = prompt_tokens

    self._start_ts = time.perf_counter()
    self._response_token_count = 0

  def _strip_suffix(
      self,
      tokens: np.ndarray,
      masks: Optional[np.ndarray],
      suffix_tokens: list[int],
  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Strips suffix_tokens from the end of tokens and masks if present."""
    if not suffix_tokens or len(tokens) < len(suffix_tokens):
      return tokens, masks

    n_tokens = len(suffix_tokens)
    tokens_list = list(tokens)
    if tokens_list[-n_tokens:] == list(suffix_tokens):
      tokens = np.array(tokens_list[:-n_tokens], dtype=np.int32)
      if masks is not None:
        masks = masks[:-n_tokens]
    return tokens, masks

  @property
  def _debug_prefix(self) -> str:
    """Returns a consistent log prefix with step_idx, pair_index, and group_id."""
    extra = getattr(self.env, "extra_kwargs", {}) or {}
    step_idx = len(self.agent.trajectory.steps)
    pair_index = extra.get("pair_index")
    group_id = extra.get("group_id")
    return (
        f"[step_idx={step_idx}, pair_index={pair_index}, group_id={group_id}]"
    )

  def _get_perf_tags(self) -> Dict[str, Any]:
    """Extracts performance tracing tags from the environment."""
    tags = {}
    if hasattr(self.env, "extra_kwargs"):
      group_id = self.env.extra_kwargs.get("group_id")
      if group_id is not None:
        tags[perf_constants.GROUP_ID] = group_id
      pair_index = self.env.extra_kwargs.get("pair_index")
      if pair_index is not None:
        tags[perf_constants.PAIR_INDEX] = pair_index
    if hasattr(self.env, "task"):
      policy_version = self.env.task.get("policy_version")
      if policy_version is not None:
        tags[perf_constants.STEP] = policy_version
    return tags

  def _check_and_set_context_limit_reached(self) -> bool:
    """Returns True and updates trajectory status if response budget is exhausted."""
    if (
        self.max_response_length is not None
        and self._response_token_count >= self.max_response_length
    ):
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      logging.debug("%s MAX_CONTEXT_LIMIT_REACHED", self._debug_prefix)
      return True
    return False

  async def _one_step(self) -> bool:
    """Executes a single step and returns the Step object and Done status.

    This involves calling the model, updating the agent with the response,
    stepping the environment with the agent's action, and updating the agent
    with the environment's feedback.

    Returns:
        bool: True if the episode is done (either by environment or timeout),
          False otherwise.
    """
    if self._check_and_set_context_limit_reached():
      return True
    max_generation_steps = (
        self.max_response_length - self._response_token_count
        if self.max_response_length is not None
        else None
    )
    logging.debug("%s model_call starting", self._debug_prefix)

    def _safe_model_call():
      try:
        return self.model_call(
            self.agent.chat_completions,
            self.env,
            max_generation_steps=max_generation_steps,
            **self.model_call_kwargs,
        )
      except Exception as e:
        logging.exception("Caught exception inside model_call: %s", e)
        raise

    rollout_output = await asyncio.get_event_loop().run_in_executor(
        None,
        _safe_model_call,
    )
    logging.debug("%s model_call done", self._debug_prefix)

    # temp_prompt_tokens_unpadded = None
    # temp_prompt_logprobs_unpadded = None
    # if (
    #     rollout_output.left_padded_prompt_tokens is not None
    #     and rollout_output.prompt_logprobs is not None
    # ):
    #   pad_id = self.tokenizer.pad_id() if self.tokenizer else 0
    #   padded_prompt = rollout_output.left_padded_prompt_tokens[0]
    #   non_pad_indices = np.where(padded_prompt != pad_id)[0]
    #   if len(non_pad_indices) > 0:
    #     prompt_tokens_unpadded = padded_prompt[non_pad_indices[0]:]
    #   else:
    #     prompt_tokens_unpadded = padded_prompt
    #   temp_prompt_tokens_unpadded = list(prompt_tokens_unpadded)
    #   prompt_logprobs_raw = rollout_output.prompt_logprobs[0]
    #   extracted_lp = [0.0]
    #   for idx in range(1, len(prompt_tokens_unpadded)):
    #     tok_id = int(prompt_tokens_unpadded[idx])
    #     tok_logprobs = prompt_logprobs_raw[idx]
    #     if tok_logprobs is not None and tok_id in tok_logprobs:
    #       extracted_lp.append(tok_logprobs[tok_id].logprob)
    #     else:
    #       extracted_lp.append(0.0)
    #   temp_prompt_logprobs_unpadded = extracted_lp

    # Align trajectory prompt tokens with the rollout worker's actual
    # tokenization on the first turn to prevent prompt token desync.
    if (
        not self.agent.trajectory.steps
        and rollout_output.left_padded_prompt_tokens is not None
    ):
      self.agent.trajectory.prompt_tokens = (
          rollout_output.left_padded_prompt_tokens[0]
      )

    if rollout_output.tokens:
      self._response_token_count += len(rollout_output.tokens[0])

    action = self.agent.update_from_model(rollout_output.text[0]).action
    logging.debug(
        "%s Agent Action:\n%s",
        self._debug_prefix,
        json.dumps(action, default=str, indent=2),
    )
    if action is None:
      logging.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    step_idx = len(self.agent.trajectory.steps)
    remaining_time = self.timeout - (time.perf_counter() - self._start_ts)

    if self._check_and_set_context_limit_reached():
      done = True
    else:
      tags = self._get_perf_tags()
      try:
        with self.perf_v2.span(
            perf_constants.ENVIRONMENT,
            tags=tags,
        ):
          (obs, rew, done, info), wall_time, cpu_time = (
              await self._run_with_timing(
                  self.env.step, action, timeout=remaining_time
              )
          )
      except asyncio.TimeoutError:
        self.agent.trajectory.status = agent_types.TrajectoryStatus.ENV_TIMEOUT
        if step_idx == 0:
          logging.error(
              "%s env.step hung at step 0 (first action) and was killed after"
              " %.1f s remaining timeout. This trajectory produced no usable"
              " data. Consider investigating the environment.",
              self._debug_prefix,
              remaining_time,
          )
        else:
          logging.error(
              "%s env.step hung at step %d and was killed after %.1f s"
              " remaining timeout.",
              self._debug_prefix,
              step_idx,
              remaining_time,
          )
        cur_step = self.agent.get_current_step()
        if cur_step is not None:
          cur_step.done = True
        return True

      self.env_time["step_latency"] += wall_time
      self.env_time["step_cpu_time"] += cpu_time

      logging.debug(
          "%s Env Observation (Rew: %s, Done: %s):\n%s",
          self._debug_prefix,
          rew,
          done,
          json.dumps(obs, default=str, indent=2),
      )
      logging.debug(
          "%s Env Info:\n%s",
          self._debug_prefix,
          json.dumps(info, default=str, indent=2),
      )
      self.agent.update_from_env(obs, rew, done, info)

    cur_step = self.agent.get_current_step()
    # if cur_step is not None:
    #   if temp_prompt_tokens_unpadded is not None:
    #     cur_step.info['prompt_tokens_unpadded'] = temp_prompt_tokens_unpadded
    #   if temp_prompt_logprobs_unpadded is not None:
    #     cur_step.info['prompt_logprobs_unpadded'] = temp_prompt_logprobs_unpadded

    if cur_step is not None and rollout_output.logprobs is not None:
      cur_step.logprobs = rollout_output.logprobs[0]

    step_timed_out = time.perf_counter() - self._start_ts > self.timeout
    if cur_step is not None and self.tokenizer and self.chat_parser:
      assistant_message, env_messages = (
          utils.get_recent_assistant_user_messages(self.agent.chat_completions)
      )

      # Assistant tokens/masks
      if assistant_message:
        cur_step.assistant_tokens = rollout_output.tokens[0]
        cur_step.assistant_masks = np.ones_like(rollout_output.tokens[0])

      # Environment tokens/masks
      # Terminal-step environment messages are not appended to the response
      # token stream when the step ends the trajectory.
      if env_messages and not done and not step_timed_out:
        e_tokens, e_masks = utils.tokenize_and_generate_masks(
            env_messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=True,
        )
        cur_step.env_tokens = np.array(e_tokens)
        cur_step.env_masks = np.array(e_masks)
        self._response_token_count += len(e_tokens)

    if step_timed_out:
      self.agent.trajectory.status = agent_types.TrajectoryStatus.TIMEOUT
      logging.warning("Episode timed out after %d seconds.", self.timeout)
      self.agent.get_current_step().done = True
      return True

    return done

  async def _append_final_reward(self):
    """Compute and add final reward to the last step of the episode.

    Applies the final reward function (if provided) to the episode's
    final response and adds it to the last step's reward. This enables
    additional reward signals based on overall episode performance.
    """
    last_step = self.agent.get_current_step()
    if (
        last_step is None
        or self.final_reward_fn is None
        or not callable(self.final_reward_fn)
    ):
      # Skip reward computation in trajectory collection if no reward function
      # is provided or no step is taken.
      logging.debug("%s Final reward function is skipped", self._debug_prefix)
      return
    final_reward, wall_time, cpu_time = await self._run_with_timing(
        self.final_reward_fn
    )

    self.reward_time["reward_latency"] += wall_time
    self.reward_time["reward_cpu_time"] += cpu_time
    last_step.reward += final_reward
    logging.debug(
        "%s Final reward computed: %s", self._debug_prefix, final_reward
    )

  def compute_trajectory_reward(self):
    """Computes and stores the total reward for the trajectory.

    The trajectory reward is the undiscounted sum of rewards from all steps and
    is stored in `trajectory.reward`.

    Returns:
        The updated trajectory with the `reward` attribute populated.
    """
    trajectory = self.agent.trajectory
    if not trajectory:
      return None
    trajectory.reward = float(
        np.sum(np.array([s.reward for s in trajectory.steps]))
    )
    return trajectory

  def compute_mc_reward(self):
    """Compute Monte Carlo rewards for all steps in the trajectory.

    Calculates discounted rewards working backwards from the final step.
    Each step's Monte Carlo reward (return) is its immediate reward plus the
    discounted reward of subsequent steps. The result is stored in
    `step.mc_return`.
    """
    trajectory = self.agent.trajectory
    g = 0.0
    for step in reversed(trajectory.steps):
      g = step.reward + self.gamma * g
      step.mc_return = g

  async def _close(self):
    """Clean up resources by closing the environment.

    Ensures proper cleanup of environment resources such as network
    connections, file handles, or external processes.
    """
    logging.debug("%s Closing environment.", self._debug_prefix)
    for k, v in self.env_time.items():
      logging.debug("%s k=%s v=%s", self._debug_prefix, k, v)
    for k, v in self.reward_time.items():
      logging.debug("%s k=%s v=%s", self._debug_prefix, k, v)

    try:
      await asyncio.wait_for(
          asyncio.get_event_loop().run_in_executor(
              None, self.env.close
          ),
          timeout=150.0,
      )
    except asyncio.TimeoutError:
      logging.error(
          "%s env.close() timed out after 150s — executor thread may be"
          " leaked. This will starve the thread pool over time.",
          self._debug_prefix,
      )
    logging.debug("%s Environment closed.", self._debug_prefix)