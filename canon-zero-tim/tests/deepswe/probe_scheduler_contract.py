#!/usr/bin/env python3
"""Checks the P34 scheduler contract with the pinned engine implementation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable


DP_SIZE = 16
LOCAL_M = 256
MAX_NUM_SEQS_PER_DP = 4
MAX_BATCHED_TOKENS_PER_DP = 256
LEGACY_MAX_NUM_SEQS_PER_DP = 64
LEGACY_MAX_BATCHED_TOKENS_PER_DP = 8192


@dataclass(frozen=True)
class SchedulerResult:
  current_paddings: tuple[int, ...]
  current_num_reqs: int
  legacy_paddings: tuple[int, ...]
  legacy_num_reqs: int


def evaluate(
    get_token_paddings: Callable[..., list[int]], *, padding_gap: int = 0
) -> SchedulerResult:
  global_min = DP_SIZE * LOCAL_M
  current = tuple(get_token_paddings(
      min_token_size=global_min,
      max_token_size=DP_SIZE * MAX_BATCHED_TOKENS_PER_DP,
      padding_gap=padding_gap,
  ))
  legacy = tuple(get_token_paddings(
      min_token_size=global_min,
      max_token_size=DP_SIZE * LEGACY_MAX_BATCHED_TOKENS_PER_DP,
      padding_gap=padding_gap,
  ))
  result = SchedulerResult(
      current_paddings=current,
      current_num_reqs=DP_SIZE * MAX_NUM_SEQS_PER_DP,
      legacy_paddings=legacy,
      legacy_num_reqs=DP_SIZE * LEGACY_MAX_NUM_SEQS_PER_DP,
  )
  if result.current_paddings != (4096,):
    raise ValueError(
        f"P34 expected exactly one global M4096 bucket, got {current}"
    )
  if result.current_num_reqs != 64:
    raise ValueError(
        f"P34 expected 64 global request slots, got {result.current_num_reqs}"
    )
  if (
      result.legacy_paddings == result.current_paddings
      or max(result.legacy_paddings, default=0) <= 4096
      or result.legacy_num_reqs != 1024
  ):
    raise ValueError("P34 historical global-as-local negative control did not reject")
  return result


def _load_engine_function() -> tuple[Callable[..., list[int]], str]:
  try:
    from tpu_inference.runner import utils as runner_utils  # type: ignore
  except ImportError:
    from tpu_inference.runner.utils import get_token_paddings  # type: ignore

    return get_token_paddings, "tpu_inference.runner.utils"
  return runner_utils.get_token_paddings, "tpu_inference.runner.utils"


def main() -> None:
  get_token_paddings, implementation = _load_engine_function()
  padding_gap = int(os.environ.get("VLLM_TPU_BUCKET_PADDING_GAP", "0"))
  result = evaluate(get_token_paddings, padding_gap=padding_gap)
  print(f"[P34.SCHEDULER] implementation={implementation}", flush=True)
  print(
      "[P34.SCHEDULER] current "
      f"global_paddings={list(result.current_paddings)} "
      f"global_num_reqs={result.current_num_reqs}",
      flush=True,
  )
  print(
      "[P34.SCHEDULER] legacy_negative "
      f"global_paddings={list(result.legacy_paddings)} "
      f"global_num_reqs={result.legacy_num_reqs} rejected=1",
      flush=True,
  )
  print("[P34.SCHEDULER] VERDICT PASS", flush=True)


if __name__ == "__main__":
  main()
