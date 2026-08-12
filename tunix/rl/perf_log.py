"""Lightweight stage-timing markers for RL training.

Prints one ``[PERF] stage=<name> seconds=<dt>`` line when a stage ends, and
optionally forwards the duration to a metrics sink (W&B via
``buffer_metrics_async``). Enabled by default; set ``CANON_PERF_LOG=0`` to
silence.

Observer-safety contract: a ``phase`` block must wrap an EXISTING host
barrier (a call that already returns host data or blocks on device work).
Never add a new device sync just to time it -- that changes pipelining.
"""

from __future__ import annotations

import contextlib
import functools
import os
import time
from typing import Any, Callable, Iterator


def enabled() -> bool:
  return os.environ.get("CANON_PERF_LOG", "1") != "0"


@contextlib.contextmanager
def phase(
    stage: str,
    *,
    step: int | None = None,
    sink: Callable[[str, float], None] | None = None,
) -> Iterator[dict[str, Any]]:
  """Times a stage; yields a dict the caller may fill with extra fields."""
  info: dict[str, Any] = {}
  if not enabled():
    yield info
    return
  t0 = time.perf_counter()
  try:
    yield info
  finally:
    dt = time.perf_counter() - t0
    parts = ["[PERF]"]
    if step is not None:
      parts.append(f"step={int(step)}")
    parts.append(f"stage={stage}")
    parts.append(f"seconds={dt:.3f}")
    for key, value in info.items():
      parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)
    if sink is not None:
      try:
        sink(stage, dt)
      except Exception as exc:  # pylint: disable=broad-except
        print(f"[PERF] WARN metric sink failed for {stage}: {exc}", flush=True)


def timed(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Decorator form of :func:`phase` for whole functions (stdout only)."""

  def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
      with phase(stage):
        return fn(*args, **kwargs)

    return inner

  return wrap
