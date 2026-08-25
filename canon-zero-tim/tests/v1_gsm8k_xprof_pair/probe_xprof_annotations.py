#!/usr/bin/env python3
"""Pinned-image API probe for P60-2 JAX host annotations."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp

from tunix.rl import gsm8k_xprof


def main() -> None:
  os.environ["CANON_XPROF_LABELS"] = "1"
  with tempfile.TemporaryDirectory(prefix="p60-xprof-annotation-") as directory:
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    jax.profiler.start_trace(directory, profiler_options=options)
    with gsm8k_xprof.train_step_annotation(step_num=1):
      with gsm8k_xprof.trace_annotation("zero_tim_update"):
        with gsm8k_xprof.trace_annotation("forward_group", group_index=3):
          result = jax.jit(lambda value: value + 1)(
              jnp.arange(4, dtype=jnp.int32)
          )
          result.block_until_ready()
        for index in range(16):
          with gsm8k_xprof.trace_annotation(
              "gradient_accumulate",
              group_index=index,
              micro_step=index,
              is_last_accumulate=int(index == 15),
          ):
            pass
        with gsm8k_xprof.trace_annotation("optimizer_commit", update_step=1):
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
    with gzip.open(traces[0], "rt", encoding="utf-8") as stream:
      events = json.load(stream)["traceEvents"]
    selected_names = (
        "train",
        "zero_tim_update",
        "forward_group",
        "gradient_accumulate",
        "optimizer_commit",
    )
    selected_events = [
        event for event in events if event.get("name") in selected_names
    ]
    singleton_names = (
        "train",
        "zero_tim_update",
        "forward_group",
        "optimizer_commit",
    )
    singleton_events = {
        name: [
            event.get("args") or {}
            for event in selected_events
            if event.get("name") == name
        ]
        for name in singleton_names
    }
    expected_singletons = {
        "train": [{"step_num": "1"}],
        "zero_tim_update": [{}],
        "forward_group": [{"group_index": "3"}],
        "optimizer_commit": [{"update_step": "1"}],
    }
    if singleton_events != expected_singletons:
      raise RuntimeError(
          "annotation singleton contract changed: "
          f"{singleton_events}"
      )
    accumulators = [
        event.get("args") or {}
        for event in selected_events
        if event.get("name") == "gradient_accumulate"
    ]
    try:
      accumulators.sort(key=lambda item: int(item["group_index"]))
    except (KeyError, TypeError, ValueError) as error:
      raise RuntimeError(
          "annotation accumulator metadata is not integer-valued: "
          f"{accumulators}"
      ) from error
    expected_accumulators = [
        {
            "group_index": str(index),
            "micro_step": str(index),
            "is_last_accumulate": str(int(index == 15)),
        }
        for index in range(16)
    ]
    if accumulators != expected_accumulators:
      raise RuntimeError(
          "annotation accumulator contract changed: "
          f"{accumulators}"
      )
    locations = {
        (event.get("pid"), event.get("tid")) for event in selected_events
    }
    process_names = {
        event.get("pid"): (event.get("args") or {}).get("name")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    thread_names = {
        (event.get("pid"), event.get("tid")):
            (event.get("args") or {}).get("name")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    if len(locations) != 1:
      raise RuntimeError(f"annotation events span multiple tracks: {locations}")
    pid, tid = next(iter(locations))
    if process_names.get(pid) != "/host:CPU" or thread_names.get(
        (pid, tid)
    ) != "python3":
      raise RuntimeError(
          "annotation host track changed: "
          f"process={process_names.get(pid)} "
          f"thread={thread_names.get((pid, tid))}"
      )
  print(
      "P60_XPROF_ANNOTATION_API_PASS "
      "step=train step_num=1 micro_steps=0..15 last_accumulate=15 "
      "optimizer_update=1 metadata=integer "
      "host_plane=/host:CPU host_line=python3 xplane=1 trace=1"
  )


if __name__ == "__main__":
  main()
