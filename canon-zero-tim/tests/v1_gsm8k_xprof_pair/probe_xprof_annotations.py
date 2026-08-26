#!/usr/bin/env python3
"""Pinned-image API probe for P60-2 JAX host annotations."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import jax
import jax.numpy as jnp

from tunix.rl import gsm8k_xprof


def _load_trace_census():
  path = (
      Path(__file__).resolve().parents[2]
      / "tasks/v1-gsm8k-onehost-xprof-pair/scripts"
      / "census_gsm8k_xprof_trace.py"
  )
  spec = importlib.util.spec_from_file_location(
      "p60_exact_image_trace_census", path
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load trace census: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main() -> None:
  os.environ["CANON_XPROF_LABELS"] = "1"
  with tempfile.TemporaryDirectory(prefix="p60-xprof-annotation-") as directory:
    add_one = jax.jit(lambda value: value + 1)
    add_one(jnp.arange(4, dtype=jnp.int32)).block_until_ready()
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    jax.profiler.start_trace(directory, profiler_options=options)
    with gsm8k_xprof.trace_annotation("zero_tim_update", update_step=2):
      with gsm8k_xprof.trace_annotation("forward_groups"):
        for index in range(16):
          with gsm8k_xprof.trace_annotation(
              "forward_group", group_index=index
          ):
            if index == 0:
              add_one(jnp.arange(4, dtype=jnp.int32)).block_until_ready()
      with gsm8k_xprof.trace_annotation("loss_pullback"):
        pass
      schedule = gsm8k_xprof.ZeroHpTrainMicrostepSchedule(update_step=2)
      with schedule:
        for index in range(16):
          with schedule.transaction(index):
            with gsm8k_xprof.trace_annotation(
                "reverse_group", group_index=index
            ):
              for stage in (
                  "replay_forward",
                  "model_backward",
                  "report_adjoint",
                  "fixed_dp_reduce",
              ):
                with gsm8k_xprof.trace_annotation(
                    stage, group_index=index
                ):
                  pass
              with gsm8k_xprof.trace_annotation(
                  "gradient_accumulate",
                  group_index=index,
                  micro_step=index,
                  is_last_accumulate=int(index == 15),
              ):
                pass
        with schedule.optimizer_commit():
          pass
    jax.profiler.stop_trace()
    root = Path(directory)
    xplanes = [
        path for path in root.rglob("*.xplane.pb") if path.stat().st_size
    ]
    traces = [
        path for path in root.rglob("*.trace.json.gz") if path.stat().st_size
    ]
    if len(xplanes) != 1 or len(traces) != 1:
      raise RuntimeError(
          f"annotation probe artifacts changed: xplanes={len(xplanes)} "
          f"traces={len(traces)}"
      )
    trace_census = _load_trace_census()
    spans, compiler_counts, event_count = trace_census.read_trace(traces[0])
    reasons = trace_census.validate_trace(
        spans, compiler_counts=compiler_counts, expected_update_step=2
    )
    if reasons:
      raise RuntimeError(f"annotation trace contract changed: {reasons}")
  print(
      "P60_XPROF_ANNOTATION_API_PASS "
      "train_steps=32..47 micro_steps=0..15 last_accumulate=15 "
      "optimizer_update=2 optimizer_owned_by_last=1 compiler_events=0 "
      f"trace_events={event_count} metadata=integer "
      "host_plane=/host:CPU host_line=python3 xplane=1 trace=1"
  )


if __name__ == "__main__":
  main()
