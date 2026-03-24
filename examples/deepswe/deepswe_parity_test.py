#!/usr/bin/env python3
"""Parity test for tunix vs rllm DeepSWE agent preprocessing.

This script mocks the inference layer and simulates a multi-turn SWE task
pipeline. It drives both tunix and rllm DeepSWE agents through the same
sequence of:

1. initial task observation
2. mocked model response
3. mocked environment observation

At each turn it compares the training-facing intermediate data before any real
model inference or optimizer step:

- final cumulative prompt token ids
- final cumulative response/conversation token ids
- cumulative response mask
- mocked cumulative old logprobs
- final chat completions
- final reward / termination-relevant fields

Usage:
  python3 examples/deepswe/deepswe_parity_test.py
  python3 examples/deepswe/deepswe_parity_test.py --tokenizer-path /path/to/local/qwen-tokenizer --dump-json /tmp/deepswe_parity.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import numpy as np
import os
from pathlib import Path
import sys
import types
from typing import Any

from transformers import AutoTokenizer


LOGGER = logging.getLogger("deepswe_parity")
MAX_DIFF_CHARS = 1500
DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-32B"


def _install_fake_absl_logging() -> None:
  """Install a minimal absl.logging shim when absl is unavailable."""
  try:
    import absl.logging as _  # noqa: F401
    return
  except ImportError:
    pass

  absl_module = types.ModuleType("absl")
  logging_module = types.ModuleType("absl.logging")

  logging_module.debug = logging.debug
  logging_module.info = logging.info
  logging_module.warning = logging.warning
  logging_module.error = logging.error
  logging_module.fatal = logging.critical
  logging_module.exception = logging.exception

  absl_module.logging = logging_module

  sys.modules.setdefault("absl", absl_module)
  sys.modules["absl.logging"] = logging_module


def _install_fake_r2egym_action() -> None:
  """Install a tiny in-memory r2egym Action replacement for agent imports."""

  class FakeAction:
    def __init__(self, function_name: str = "", parameters: dict[str, Any] | None = None):
      self.function_name = function_name or ""
      self.parameters = parameters or {}

    @classmethod
    def from_string(cls, action_str: str):
      import re

      if not action_str:
        return cls("", {})

      fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
      function_name = fn_match.group(1).strip() if fn_match else ""
      param_matches = re.findall(
          r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>",
          action_str,
          flags=re.DOTALL,
      )
      parameters = {key.strip(): value.strip() for key, value in param_matches}
      return cls(function_name=function_name, parameters=parameters)

    def to_xml_string(self) -> str:
      if not self.function_name:
        return ""
      lines = [f"<function={self.function_name}>"]
      for key, value in self.parameters.items():
        lines.append(f"<parameter={key}>{value}</parameter>")
      lines.append("</function>")
      return "\n".join(lines)

  r2egym_module = types.ModuleType("r2egym")
  agenthub_module = types.ModuleType("r2egym.agenthub")
  action_module = types.ModuleType("r2egym.agenthub.action")
  action_module.Action = FakeAction

  sys.modules.setdefault("r2egym", r2egym_module)
  sys.modules.setdefault("r2egym.agenthub", agenthub_module)
  sys.modules["r2egym.agenthub.action"] = action_module


def _ensure_package(module_name: str) -> None:
  if module_name in sys.modules:
    return
  module = types.ModuleType(module_name)
  module.__path__ = []
  sys.modules[module_name] = module


def _load_module(module_name: str, file_path: Path):
  if module_name in sys.modules:
    return sys.modules[module_name]
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load module {module_name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def _load_tunix_modules(repo_root: Path):
  _ensure_package("tunix")
  _ensure_package("tunix.rl")
  _ensure_package("tunix.rl.agentic")
  _ensure_package("tunix.rl.agentic.agents")
  _ensure_package("tunix.rl.agentic.parser")
  _ensure_package("tunix.rl.agentic.parser.chat_template_parser")
  _ensure_package("examples")
  _ensure_package("examples.deepswe")

  _load_module(
      "tunix.rl.agentic.agents.agent_types",
      repo_root / "tunix/rl/agentic/agents/agent_types.py",
  )
  _load_module(
      "tunix.rl.agentic.agents.base_agent",
      repo_root / "tunix/rl/agentic/agents/base_agent.py",
  )
  _load_module(
      "tunix.rl.agentic.parser.chat_template_parser.parser",
      repo_root / "tunix/rl/agentic/parser/chat_template_parser/parser.py",
  )
  utils_module = _load_module(
      "tunix.rl.agentic.utils",
      repo_root / "tunix/rl/agentic/utils.py",
  )
  swe_agent_module = _load_module(
      "examples.deepswe.swe_agent",
      repo_root / "examples/deepswe/swe_agent.py",
  )
  parser_module = sys.modules["tunix.rl.agentic.parser.chat_template_parser.parser"]
  return swe_agent_module, parser_module, utils_module


def _load_rllm_modules(repo_root: Path):
  _ensure_package("rllm")
  _ensure_package("rllm.agents")
  _ensure_package("rllm.tools")
  _ensure_package("rllm.parser")

  _load_module(
      "rllm.agents.agent",
      repo_root / "rllm/rllm/agents/agent.py",
  )
  _load_module(
      "rllm.agents.system_prompts",
      repo_root / "rllm/rllm/agents/system_prompts.py",
  )
  _load_module(
      "rllm.tools.utils",
      repo_root / "rllm/rllm/tools/utils.py",
  )
  _load_module(
      "rllm.tools.tool_base",
      repo_root / "rllm/rllm/tools/tool_base.py",
  )
  _load_module(
      "rllm.parser.utils",
      repo_root / "rllm/rllm/parser/utils.py",
  )
  _load_module(
      "rllm.parser.tool_parser",
      repo_root / "rllm/rllm/parser/tool_parser.py",
  )
  _load_module(
      "rllm.parser.chat_template_parser",
      repo_root / "rllm/rllm/parser/chat_template_parser.py",
  )
  swe_agent_module = _load_module(
      "rllm.agents.swe_agent",
      repo_root / "rllm/rllm/agents/swe_agent.py",
  )
  parser_module = sys.modules["rllm.parser.chat_template_parser"]
  return swe_agent_module, parser_module


def _normalize_step(step: Any) -> dict[str, Any] | None:
  if step is None:
    return None
  action = step.action.action if hasattr(step.action, "action") else step.action
  normalized = {
      "observation": step.observation,
      "thought": step.thought,
      "action": action,
      "model_response": step.model_response,
      "reward": getattr(step, "reward", 0.0),
      "done": getattr(step, "done", False),
      "info": getattr(step, "info", {}) or {},
  }
  chat_completions = getattr(step, "chat_completions", None)
  if chat_completions:
    normalized["chat_completions"] = chat_completions
  return normalized


def _mock_logprobs(length: int) -> np.ndarray:
  return np.asarray([-(idx + 1) / 100.0 for idx in range(length)], dtype=np.float32)


def _scenario() -> dict[str, Any]:
  return {
      "instance_id": "astropy__astropy-13033",
      "initial_observation": (
          "TimeSeries: misleading exception when required column check fails.\n"
          "For a TimeSeries object that has additional required columns, "
          "removing a required column raises a misleading error."
      ),
      "max_steps": 4,
      "turns": [
          {
              "response": (
                  "I will locate the TimeSeries implementation first.\n"
                  "<function=search>\n"
                  "<parameter=search_term>class TimeSeries</parameter>\n"
                  "<parameter=path>/testbed/astropy/timeseries</parameter>\n"
                  "</function>"
              ),
              "observation": (
                  "Found 1 matches for \"class TimeSeries\" in "
                  "/testbed/astropy/timeseries/sampled.py"
              ),
              "reward": 0.0,
              "done": False,
              "info": {"tool": "search", "match_count": 1},
          },
          {
              "response": (
                  "I should inspect the required column validation.\n"
                  "<function=file_editor>\n"
                  "<parameter=command>view</parameter>\n"
                  "<parameter=path>/testbed/astropy/timeseries/core.py</parameter>\n"
                  "</function>"
              ),
              "observation": (
                  "Here's the result of running cat -n on "
                  "/testbed/astropy/timeseries/core.py:\n"
                  "57 def _check_required_columns(self):\n"
                  "77 elif self.colnames[:len(required_columns)] != required_columns:"
              ),
              "reward": 0.0,
              "done": False,
              "info": {"tool": "file_editor", "command": "view"},
          },
          {
              "response": (
                  "The error message should report missing required columns.\n"
                  "<function=file_editor>\n"
                  "<parameter=command>str_replace</parameter>\n"
                  "<parameter=path>/testbed/astropy/timeseries/core.py</parameter>\n"
                  "<parameter=old_str>raise ValueError(\"{} object is invalid - expected '{}' \"\n"
                  "                             \"as the first column{} but found '{}'\"</parameter>\n"
                  "<parameter=new_str>raise ValueError(\"{} object is invalid - required columns are missing\")</parameter>\n"
                  "</function>"
              ),
              "observation": "Updated file: /testbed/astropy/timeseries/core.py",
              "reward": 0.0,
              "done": False,
              "info": {"tool": "file_editor", "command": "str_replace"},
          },
          {
              "response": (
                  "The fix is in place and the behavior is corrected.\n"
                  "<function=finish>\n"
                  "<parameter=command>submit</parameter>\n"
                  "<parameter=result>Fixed the misleading required-column error message.</parameter>\n"
                  "</function>"
              ),
              "observation": "Submission accepted.",
              "reward": 1.0,
              "done": True,
              "info": {"tool": "finish", "status": "submitted"},
          },
      ],
  }


def _compare(label: str, lhs: Any, rhs: Any, failures: list[str]) -> None:
  if lhs != rhs:
    failures.append(label)
    print(f"[FAIL] {label}")
    print("  tunix:", _render_diff_value(lhs))
    print("  rllm :", _render_diff_value(rhs))
  else:
    print(f"[ OK ] {label}")


def _render_diff_value(value: Any) -> str:
  rendered = json.dumps(value, ensure_ascii=False, indent=2)
  if len(rendered) <= MAX_DIFF_CHARS:
    return rendered
  return rendered[:MAX_DIFF_CHARS] + "\n...<truncated>..."


def _build_tunix_training_sample(agent, trajectory_reward: float) -> dict[str, Any]:
  conversation_tokens, conversation_masks, logprobs = [], [], []
  prompt_tokens = getattr(agent.trajectory, "prompt_tokens", [])

  for step in agent.trajectory.steps:
    if getattr(step, "assistant_tokens", None) is not None:
      conversation_tokens.append(np.asarray(step.assistant_tokens))
      conversation_masks.append(np.asarray(step.assistant_masks))
    if getattr(step, "env_tokens", None) is not None:
      conversation_tokens.append(np.asarray(step.env_tokens))
      conversation_masks.append(np.asarray(step.env_masks))
    if getattr(step, "logprobs", None) is not None:
      logprobs.append(np.asarray(step.logprobs))
      if getattr(step, "env_tokens", None) is not None:
        logprobs.append(np.zeros(len(step.env_tokens), dtype=np.float32))

  return {
      "final_chat_completions": agent.chat_completions,
      "prompt_ids": np.asarray(prompt_tokens).tolist(),
      "response_ids": (
          np.concatenate(conversation_tokens, axis=0).tolist()
          if conversation_tokens
          else []
      ),
      "response_mask": (
          np.concatenate(conversation_masks, axis=0).tolist()
          if conversation_masks
          else []
      ),
      "old_logprobs": (
          np.concatenate(logprobs, axis=0).tolist() if logprobs else None
      ),
      "trajectory_reward": trajectory_reward,
      "status": getattr(agent.trajectory.status, "name", str(agent.trajectory.status)),
  }


def _build_rllm_cumulative_logprobs(messages, tokenizer, chat_parser) -> np.ndarray:
  response_logprobs = []
  first_assistant_idx = next(
      i for i, msg in enumerate(messages) if msg["role"] == "assistant"
  )
  for idx in range(first_assistant_idx, len(messages)):
    message = messages[idx]
    if message["role"] == "assistant":
      response = chat_parser.parse(
          [message],
          is_first_msg=False,
          add_generation_prompt=False,
          accumulate_reasoning=True,
      )
      response = response[len(chat_parser.generation_prompt):]
      ids = tokenizer.encode(response, add_special_tokens=False)
      response_logprobs.extend(_mock_logprobs(len(ids)).tolist())
    else:
      response = chat_parser.parse(
          [message],
          is_first_msg=False,
          add_generation_prompt=True,
          accumulate_reasoning=False,
      )
      ids = tokenizer.encode(response, add_special_tokens=False)
      response_logprobs.extend([0.0] * len(ids))
  return np.asarray(response_logprobs, dtype=np.float32)


def _build_rllm_training_sample(agent, chat_parser, trajectory_reward: float, termination_reason: str) -> dict[str, Any]:
  prompt_ids, response_ids, response_mask = chat_parser.tokenize_and_mask_cumulative(
      agent.chat_completions
  )
  old_logprobs = _build_rllm_cumulative_logprobs(
      agent.chat_completions,
      chat_parser.tokenizer,
      chat_parser,
  )
  return {
      "final_chat_completions": agent.chat_completions,
      "prompt_ids": prompt_ids.tolist(),
      "response_ids": response_ids.tolist(),
      "response_mask": response_mask.tolist(),
      "old_logprobs": old_logprobs.tolist(),
      "trajectory_reward": trajectory_reward,
      "termination_reason": termination_reason,
  }


def _load_real_tokenizer(tokenizer_path: str):
  try:
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
  except Exception:
    try:
      return AutoTokenizer.from_pretrained(
          tokenizer_path,
          trust_remote_code=True,
      )
    except Exception as exc:
      raise RuntimeError(
          "Failed to load a real tokenizer from local cache or remote source. "
          f"Attempted: {tokenizer_path}"
      ) from exc


def _run_parity_scenario(tokenizer_path: str, dump_json: Path | None = None) -> int:
  _install_fake_absl_logging()
  _install_fake_r2egym_action()

  repo_root = Path(__file__).resolve().parents[2]
  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
  if str(repo_root / "rllm") not in sys.path:
    sys.path.insert(0, str(repo_root / "rllm"))

  tunix_swe_agent_module, tunix_parser_module, tunix_utils_module = _load_tunix_modules(repo_root)
  rllm_swe_agent_module, rllm_parser_module = _load_rllm_modules(repo_root)

  TunixSWEAgent = tunix_swe_agent_module.SWEAgent
  RllmSWEAgent = rllm_swe_agent_module.SWEAgent

  tokenizer = _load_real_tokenizer(tokenizer_path)
  tunix_chat_parser = tunix_parser_module.QwenChatTemplateParser(tokenizer)
  rllm_chat_parser = rllm_parser_module.QwenChatTemplateParser(tokenizer)
  scenario = _scenario()

  tunix_agent = TunixSWEAgent()
  rllm_agent = RllmSWEAgent()

  initial_info = {"max_steps": scenario["max_steps"]}
  tunix_agent.reset()
  rllm_agent.reset()
  tunix_agent.update_from_env(
      scenario["initial_observation"], reward=0.0, done=False, info=initial_info
  )
  rllm_agent.update_from_env(
      scenario["initial_observation"], reward=0.0, done=False, info=initial_info
  )

  tunix_prompt_tokens, _ = tunix_utils_module.tokenize_and_generate_masks(
      tunix_agent.chat_completions,
      tokenizer=tokenizer,
      parser=tunix_chat_parser,
      contains_first_msg=True,
      contains_generation_msg=True,
  )
  tunix_agent.trajectory.prompt_tokens = tunix_prompt_tokens

  failures: list[str] = []
  trajectory_reward = 0.0
  termination_reason = "UNKNOWN"
  parity_dump: dict[str, Any] = {
      "instance_id": scenario["instance_id"],
      "tokenizer_path": tokenizer_path,
      "initial_chat_completions": {
          "tunix": tunix_agent.chat_completions,
          "rllm": rllm_agent.chat_completions,
      },
  }

  for turn_idx, turn in enumerate(scenario["turns"], start=1):
    mocked_response = turn["response"]
    tunix_action = tunix_agent.update_from_model(mocked_response).action
    rllm_action = rllm_agent.update_from_model(mocked_response).action

    _compare(f"turn {turn_idx}: action_xml", tunix_action, rllm_action, failures)

    tunix_step = tunix_agent.get_current_step()
    if tunix_step is not None:
      assistant_tokens = tokenizer.encode(mocked_response, add_special_tokens=False)
      tunix_step.assistant_tokens = np.asarray(assistant_tokens, dtype=np.int32)
      tunix_step.assistant_masks = np.ones_like(tunix_step.assistant_tokens)
      tunix_step.logprobs = _mock_logprobs(len(assistant_tokens))

    next_info = {"max_steps": scenario["max_steps"], **turn.get("info", {})}
    tunix_agent.update_from_env(
        turn["observation"],
        reward=turn["reward"],
        done=turn["done"],
        info=next_info,
    )
    rllm_agent.update_from_env(
        turn["observation"],
        reward=turn["reward"],
        done=turn["done"],
        info=next_info,
    )

    tunix_completed_step = tunix_agent.get_current_step()
    if tunix_completed_step is not None:
      _, env_messages = tunix_utils_module.get_recent_assistant_user_messages(
          tunix_agent.chat_completions
      )
      if env_messages:
        env_tokens, env_masks = tunix_utils_module.tokenize_and_generate_masks(
            env_messages,
            tokenizer=tokenizer,
            parser=tunix_chat_parser,
            contains_first_msg=False,
            contains_generation_msg=True,
        )
        tunix_completed_step.env_tokens = np.asarray(env_tokens, dtype=np.int32)
        tunix_completed_step.env_masks = np.asarray(env_masks, dtype=np.int32)

    trajectory_reward += turn["reward"]
    if turn["done"]:
      termination_reason = "ENV_DONE"
    elif turn_idx == scenario["max_steps"]:
      termination_reason = "MAX_STEPS"

  tunix_sample = _build_tunix_training_sample(
      tunix_agent, trajectory_reward=trajectory_reward
  )
  rllm_sample = _build_rllm_training_sample(
      rllm_agent,
      rllm_chat_parser,
      trajectory_reward=trajectory_reward,
      termination_reason=termination_reason,
  )

  parity_dump["cumulative_training_sample"] = {
      "tunix": tunix_sample,
      "rllm": rllm_sample,
  }

  print("\n=== CUMULATIVE TRAINING SAMPLE ===")
  _compare(
      "final_chat_completions",
      tunix_sample["final_chat_completions"],
      rllm_sample["final_chat_completions"],
      failures,
  )
  _compare("prompt_ids", tunix_sample["prompt_ids"], rllm_sample["prompt_ids"], failures)
  _compare(
      "response_ids",
      tunix_sample["response_ids"],
      rllm_sample["response_ids"],
      failures,
  )
  _compare(
      "response_mask",
      tunix_sample["response_mask"],
      rllm_sample["response_mask"],
      failures,
  )
  _compare(
      "old_logprobs",
      tunix_sample["old_logprobs"],
      rllm_sample["old_logprobs"],
      failures,
  )
  _compare(
      "trajectory_reward",
      tunix_sample["trajectory_reward"],
      rllm_sample["trajectory_reward"],
      failures,
  )

  if dump_json is not None:
    dump_json.parent.mkdir(parents=True, exist_ok=True)
    dump_json.write_text(
        json.dumps(parity_dump, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote parity dump to {dump_json}")

  if failures:
    print("\nParity FAILED.")
    print("Mismatched fields:")
    for failure in failures:
      print(" -", failure)
    return 1

  print("\nParity PASSED.")
  return 0


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--tokenizer-path",
      type=str,
      default=(
          os.getenv("DEEPSWE_TOKENIZER_PATH")
          or os.getenv("MODEL_PATH")
          or os.getenv("MODEL_VERSION")
          or DEFAULT_TOKENIZER_PATH
      ),
      help=(
          "Local tokenizer/model directory for AutoTokenizer.from_pretrained(..., "
          "local_files_only=True). Defaults to Qwen/Qwen3-32B."
      ),
  )
  parser.add_argument(
      "--dump-json",
      type=Path,
      default=None,
      help="Optional path to write the normalized cumulative training sample as JSON.",
  )
  return parser.parse_args()


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = _parse_args()
  return _run_parity_scenario(
      tokenizer_path=args.tokenizer_path,
      dump_json=args.dump_json,
  )


if __name__ == "__main__":
  raise SystemExit(main())
