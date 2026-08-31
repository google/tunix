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

"""Probes a rollout worker directly over Tunix remote execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Sequence

from tunix.experimental.worker import remote_execution


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Probe a rollout worker")
  parser.add_argument("--address", required=True)
  parser.add_argument(
      "--prompt",
      action="append",
      dest="prompts",
      help="Prompt to send. May be specified multiple times.",
  )
  parser.add_argument("--startup-timeout-s", type=float, default=300.0)
  parser.add_argument("--poll-interval-s", type=float, default=5.0)
  parser.add_argument("--max-generation-steps", type=int, default=32)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--top-p", type=float, default=1.0)
  parser.add_argument("--top-k", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--return-logprobs", action="store_true")
  return parser.parse_args(list(argv))


def _as_jsonable(value: Any) -> Any:
  if hasattr(value, "tolist"):
    return value.tolist()
  if isinstance(value, dict):
    return {k: _as_jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_as_jsonable(v) for v in value]
  if hasattr(value, "__dict__"):
    return {
        k: _as_jsonable(v)
        for k, v in vars(value).items()
        if not k.startswith("_")
    }
  return value


async def _wait_until_ready(
    handle: remote_execution.ActorHandle,
    timeout_s: float,
    poll_interval_s: float,
) -> Any:
  deadline = time.time() + timeout_s
  last_error: Exception | None = None
  while time.time() < deadline:
    try:
      return await handle.asubmit("heartbeat")
    except Exception as exc:  # pylint: disable=broad-except
      last_error = exc
      await asyncio.sleep(poll_interval_s)
  raise RuntimeError(f"rollout worker did not become ready: {last_error}")


async def _run(args: argparse.Namespace) -> None:
  prompts = args.prompts or [
      "Reply with exactly OK.",
      "What is 2 + 3? Reply with digits only.",
  ]
  chat_prompts = [
      [{"role": "user", "content": prompt}] for prompt in prompts
  ]
  handle = remote_execution.ActorHandle.from_address(args.address)

  heartbeat = await _wait_until_ready(
      handle,
      timeout_s=args.startup_timeout_s,
      poll_interval_s=args.poll_interval_s,
  )
  print("Heartbeat:")
  print(json.dumps(_as_jsonable(heartbeat), indent=2, sort_keys=True))

  output = await handle.asubmit(
      "sample_prompts",
      prompts=chat_prompts,
      max_generation_steps=args.max_generation_steps,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=args.top_k,
      seed=args.seed,
      return_logprobs=args.return_logprobs,
  )

  texts = list(getattr(output, "text", []) or [])
  tokens = list(getattr(output, "tokens", []) or [])
  logprobs = getattr(output, "logprobs", None)

  for index, prompt in enumerate(prompts, start=1):
    print(f"\nPrompt {index}: {prompt}")
    if index - 1 < len(texts):
      print(f"Text {index}: {texts[index - 1]}")
    if index - 1 < len(tokens):
      print(
          f"Tokens {index}: "
          f"{json.dumps(_as_jsonable(tokens[index - 1]), ensure_ascii=False)}"
      )
    if logprobs is not None and index - 1 < len(logprobs):
      print(
          f"Logprobs {index}: "
          f"{json.dumps(_as_jsonable(logprobs[index - 1]), ensure_ascii=False)}"
      )


def main(argv: Sequence[str]) -> None:
  asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
  import sys

  main(sys.argv[1:])