#!/usr/bin/env python3
"""Audit full-chat retokenization against an exact incremental token ledger."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from tunix.rl.agentic.parser.chat_template_parser import parser as parser_lib  # pylint: disable=g-import-not-at-top


def _tokens(value: Sequence[int] | np.ndarray, *, field: str) -> np.ndarray:
  result = np.asarray(value)
  if result.ndim != 1 or result.dtype.kind not in "iu":
    raise ValueError(f"{field} must be a one-dimensional integer vector")
  if np.any(result < 0):
    raise ValueError(f"{field} contains a negative token id")
  return np.asarray(result, dtype=np.int64)


def _encode(tokenizer: Any, text: str) -> np.ndarray:
  return _tokens(
      tokenizer.encode(text, add_special_tokens=False), field="encoded text"
  )


def _digest(tokens: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(tokens).tobytes()).hexdigest()


def compare_tokens(
    legacy: Sequence[int] | np.ndarray,
    exact: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
  """Return a bounded equality receipt without exposing transcript tokens."""
  left = _tokens(legacy, field="legacy tokens")
  right = _tokens(exact, field="exact tokens")
  common = min(left.size, right.size)
  unequal = np.flatnonzero(left[:common] != right[:common])
  if unequal.size:
    first = int(unequal[0])
  elif left.size != right.size:
    first = common
  else:
    first = -1
  return {
      "verdict": "TOKEN_STREAM_EQUAL" if first == -1 else "TOKEN_STREAM_DIFFERENT",
      "legacy_tokens": int(left.size),
      "exact_tokens": int(right.size),
      "legacy_sha256": _digest(left),
      "exact_sha256": _digest(right),
      "first_mismatch": first,
  }


def _load_first_trajectory(path: Path) -> dict[str, Any]:
  opener = gzip.open if path.suffix == ".gz" else open
  with opener(path, "rt", encoding="utf-8") as source:
    for line in source:
      if line.strip():
        record = json.loads(line)
        trajectory = record.get("trajectory", record)
        if not isinstance(trajectory, dict):
          raise ValueError("trajectory record must be a JSON object")
        return trajectory
  raise ValueError(f"trajectory file is empty: {path}")


def audit_saved_trajectory(
    trajectory: dict[str, Any],
    *,
    tokenizer: Any,
    chat_parser: parser_lib.BaseChatTemplateParser,
) -> list[dict[str, Any]]:
  """Audit every later assistant turn in a persisted real trajectory."""
  messages = trajectory.get("conversation_text")
  conversation = _tokens(
      trajectory.get("conversation_tokens"), field="conversation tokens"
  )
  masks = _tokens(trajectory.get("conversation_masks"), field="conversation masks")
  prompt = _tokens(trajectory.get("prompt_tokens"), field="prompt tokens")
  prompt_length = trajectory.get("prompt_length")
  if not isinstance(messages, list) or not messages:
    raise ValueError("trajectory has no conversation_text messages")
  if conversation.size != masks.size:
    raise ValueError("conversation token/mask lengths differ")
  if not isinstance(prompt_length, int) or not 0 < prompt_length <= prompt.size:
    raise ValueError("trajectory prompt_length is invalid")
  initial_prompt = prompt[-prompt_length:]

  assistant_message_indices = [
      index
      for index, message in enumerate(messages)
      if isinstance(message, dict) and message.get("role") == "assistant"
  ]
  assistant_run_starts = np.flatnonzero(
      (masks == 1) & np.concatenate(([True], masks[:-1] != 1))
  )
  if len(assistant_message_indices) != assistant_run_starts.size:
    raise ValueError(
        "assistant message/run count differs: "
        f"{len(assistant_message_indices)} vs {assistant_run_starts.size}"
    )

  receipts = []
  for turn in range(1, len(assistant_message_indices)):
    message_index = assistant_message_indices[turn]
    legacy_text = chat_parser.parse(
        messages=messages[:message_index],
        add_generation_prompt=True,
        is_first_msg=True,
    )
    legacy = _encode(tokenizer, legacy_text)
    exact = np.concatenate(
        [initial_prompt, conversation[: int(assistant_run_starts[turn])]]
    )
    receipt = compare_tokens(legacy, exact)
    receipt.update({
        "turn": turn,
        "assistant_message_index": message_index,
        "preceding_role": messages[message_index - 1].get("role", "unknown"),
        "exact_history_tokens": int(assistant_run_starts[turn]),
    })
    receipts.append(receipt)
  if not receipts:
    raise ValueError("trajectory has no later assistant turn to audit")
  return receipts


def frozenlake_fixture_receipts(
    *, tokenizer: Any, chat_parser: parser_lib.BaseChatTemplateParser
) -> list[dict[str, Any]]:
  """Build a realistic FrozenLake user-turn transcript with sampled IDs."""
  messages = [
      {"role": "system", "content": "Navigate FrozenLake and emit one move."},
      {"role": "user", "content": "Current Observation: SFFF / FHFH."},
      {"role": "assistant", "content": "I should move right.\n```Right```"},
      {"role": "user", "content": "Current Observation: FSFF / FHFH."},
      {"role": "assistant", "content": "Now move down.\n```Down```"},
      {"role": "user", "content": "Current Observation: FHFF / FSFH."},
      {"role": "assistant", "content": "Move right.\n```Right```"},
  ]
  initial_text = chat_parser.parse(
      messages=messages[:2], add_generation_prompt=True, is_first_msg=True
  )
  exact = _encode(tokenizer, initial_text)
  receipts = []
  for turn, assistant_index in enumerate((2, 4, 6)):
    if turn:
      legacy_text = chat_parser.parse(
          messages=messages[:assistant_index],
          add_generation_prompt=True,
          is_first_msg=True,
      )
      receipt = compare_tokens(_encode(tokenizer, legacy_text), exact)
      receipt.update({
          "turn": turn,
          "assistant_message_index": assistant_index,
          "preceding_role": "user",
          "exact_history_tokens": int(exact.size),
      })
      receipts.append(receipt)

    # Qwen generation starts after the assistant header and normally includes
    # the im_end stop token in the sampled token ledger.
    assistant_tail = messages[assistant_index]["content"] + chat_parser.tokens.eot_token
    exact = np.concatenate([exact, _encode(tokenizer, assistant_tail)])
    if assistant_index + 1 < len(messages):
      environment_text = chat_parser.parse(
          messages=[messages[assistant_index + 1]],
          add_generation_prompt=True,
          is_first_msg=False,
      )
      exact = np.concatenate([exact, _encode(tokenizer, environment_text)])
  return receipts


def _poison_receipt(receipt_source: np.ndarray) -> dict[str, Any]:
  clean = _tokens(receipt_source, field="poison source")
  poison = clean.copy()
  index = clean.size // 2
  poison[index] = poison[index] + 1
  receipt = compare_tokens(clean, poison)
  receipt["injected_mismatch"] = index
  if receipt["first_mismatch"] != index:
    raise AssertionError("poison negative did not trip at the injected token")
  return receipt


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--tokenizer", required=True)
  parser.add_argument("--workload", choices=("deepswe", "frozenlake"), required=True)
  parser.add_argument("--trajectory")
  parser.add_argument("--output")
  parser.add_argument("--enable-thinking", choices=("0", "1"), required=True)
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(
      args.tokenizer, local_files_only=True, trust_remote_code=False
  )
  chat_parser = parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=args.enable_thinking == "1"
  )
  if args.workload == "deepswe":
    if not args.trajectory:
      parser.error("--trajectory is required for DeepSWE")
    receipts = audit_saved_trajectory(
        _load_first_trajectory(Path(args.trajectory)),
        tokenizer=tokenizer,
        chat_parser=chat_parser,
    )
  else:
    if args.trajectory:
      receipts = audit_saved_trajectory(
          _load_first_trajectory(Path(args.trajectory)),
          tokenizer=tokenizer,
          chat_parser=chat_parser,
      )
    else:
      receipts = frozenlake_fixture_receipts(
          tokenizer=tokenizer, chat_parser=chat_parser
      )

  first_exact = np.arange(8, dtype=np.int64)
  result = {
      "schema": "canon.multiturn-tito-tokenizer-audit.v1",
      "workload": args.workload,
      "enable_thinking": args.enable_thinking == "1",
      "tokenizer_path_name": Path(args.tokenizer).name,
      "turns": receipts,
      "different_turns": sum(
          item["verdict"] == "TOKEN_STREAM_DIFFERENT" for item in receipts
      ),
      "poison_negative": _poison_receipt(first_exact),
  }
  rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.output:
    Path(args.output).write_text(rendered, encoding="utf-8")
  sys.stdout.write(rendered)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
