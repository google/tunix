# Copyright 2025 Google LLC
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

"""Helper functions for metrics export."""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Callable

import numpy as np
from tunix.perf import metrics
from tunix.perf import span
from tunix.perf import trace
from tunix.rl import rl_cluster


ClusterConfig = rl_cluster.ClusterConfig
MetricsT = metrics.MetricsT
partial = functools.partial
PerfSpanQuery = metrics.PerfSpanQuery
Span = span.Span
SpanGroup = span.SpanGroup

MetricsExportFn = Callable[[PerfSpanQuery], MetricsT]


class PerfMetricsExport:
  """Provides helper functions to create metrics export functions.

  1. from role to devices mapping

    role_to_devices = {
        "rollout": ["tpu0", "tpu1"],
        "actor": ["tpu2", "tpu3"],
        "refer": ["tpu4", "tpu5"],
    }
    export_fn = PerfMetricsExport.from_role_to_devices(role_to_devices)

  2. from cluster config

   export_fn = PerfMetricsExport.from_cluster_config(cluster_config)

   # DEPRECATED: use from_cluster_config instead.
   export_fn = PerfMetricsExport.create_metrics_export_fn(cluster_config)
  """

  _DEFAULT_EXPORT_DIR = "/tmp/perf_traces"

  @staticmethod
  def from_role_to_devices(
      role_to_devices: dict[str, list[str]],
      export_dir: str | None = None,
  ) -> MetricsExportFn:
    """Creates a metrics export function based on the role to devices mapping."""
    r2d = role_to_devices
    if r2d["rollout"] == r2d["actor"] == r2d["refer"]:
      return partial(PerfMetricsExport._grpo_metrics_colocated, r2d, export_dir)
    elif r2d["rollout"] != r2d["actor"] == r2d["refer"]:
      return partial(
          PerfMetricsExport._grpo_metrics_rollout_1_actor_2_reference_2,
          r2d,
          export_dir,
      )
    elif r2d["rollout"] != r2d["actor"] != r2d["refer"]:
      return partial(
          PerfMetricsExport._grpo_metrics_fully_disaggregated, r2d, export_dir
      )
    else:
      raise ValueError("Unsupported mesh configuration.")

  @staticmethod
  def from_cluster_config(cluster_config: ClusterConfig) -> MetricsExportFn:
    """Creates a metrics export function based on the mesh topology in cluster config."""

    rollo_mesh = cluster_config.role_to_mesh[rl_cluster.Role.ROLLOUT]
    actor_mesh = cluster_config.role_to_mesh[rl_cluster.Role.ACTOR]
    refer_mesh = cluster_config.role_to_mesh[rl_cluster.Role.REFERENCE]

    rollo_devices = map(
        trace.create_device_timeline_id, rollo_mesh.devices.flatten().tolist()
    )
    actor_devices = map(
        trace.create_device_timeline_id, actor_mesh.devices.flatten().tolist()
    )
    refer_devices = map(
        trace.create_device_timeline_id, refer_mesh.devices.flatten().tolist()
    )

    return PerfMetricsExport.from_role_to_devices(
        role_to_devices={
            "rollout": list(rollo_devices),
            "actor": list(actor_devices),
            "refer": list(refer_devices),
        },
        export_dir=cluster_config.training_config.perf_metrics_options.log_dir,
    )

  # TODO(yangmu): DEPRECATED: remove after all users use the new API.
  @staticmethod
  def create_metrics_export_fn(
      cluster_config: ClusterConfig,
  ) -> MetricsExportFn:
    return PerfMetricsExport.from_cluster_config(cluster_config)

  @staticmethod
  def _grpo_metrics_colocated(
      role_to_devices: dict[str, list[str]],
      export_dir: str | None,
      query: PerfSpanQuery,
  ) -> MetricsT:
    """GRPO workflow: rollout, actor and reference are colocated on the same mesh."""
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
        actor_train_step_spans,
    ) = PerfMetricsExport._grpo_extract_spans_and_groups(role_to_devices, query)
    if not ok:
      return {}

    weight_sync_span = global_step_group.find_last_inner_span("weight_sync")
    # If weight sync is skipped (due to shared model), create a zero duration
    # span for metrics computation.
    if weight_sync_span is None:
      weight_sync_span = Span("weight_sync", global_step_group.end)
      weight_sync_span.end = global_step_group.end

    # Step 2: compute metrics from spans and span groups

    global_step_time: float = global_step_group.duration
    weight_sync_time: float = weight_sync_span.duration

    rollout_time: list[float] = [span.duration for span in rollout_spans]

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference_spans
    ]

    # train time includes gradient update and eval
    actor_train_time: list[float] = [
        group.duration for group in actor_train_groups
    ]
    actor_train_step_time: list[float] = [
        span.duration for span in actor_train_step_spans
    ]

    PerfMetricsExport._log_perfetto_trace(
        export_dir,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
    )

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/actor_train_time": (np.sum(actor_train_time), None),
        "perf/sum/actor_train_step_time": (np.sum(actor_train_step_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/actor_train_time": (np.mean(actor_train_time), None),
        "perf/mean/actor_train_step_time": (np.mean(actor_train_step_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_metrics_rollout_1_actor_2_reference_2(
      role_to_devices: dict[str, list[str]],
      export_dir: str | None,
      query: PerfSpanQuery,
  ) -> MetricsT:
    """GRPO workflow: actor and reference are on the same mesh,rollout is on a different mesh."""
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
        actor_train_step_spans,
    ) = PerfMetricsExport._grpo_extract_spans_and_groups(role_to_devices, query)
    if not ok:
      return {}

    weight_sync_span = global_step_group.find_last_inner_span("weight_sync")
    # If weight sync is skipped (due to shared model), create a zero duration
    # span for metrics computation.
    if weight_sync_span is None:
      weight_sync_span = Span("weight_sync", global_step_group.end)
      weight_sync_span.end = global_step_group.end

    # Step 2: compute metrics from spans and span groups

    global_step_time: float = global_step_group.duration
    weight_sync_time: float = weight_sync_span.duration

    rollout_time: list[float] = [span.duration for span in rollout_spans]
    rollout_idle_time: float = weight_sync_span.begin - rollout_spans[-1].end

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference_spans
    ]

    # train time includes gradient update and eval
    actor_train_time: list[float] = [
        group.duration for group in actor_train_groups
    ]
    actor_train_step_time: list[float] = [
        span.duration for span in actor_train_step_spans
    ]

    first_micro_batch_rollout_time: float = (
        rollout_spans[0].end - global_step_group.begin
    )

    # append [0.0] to make size equal to micro batch
    between_micro_batch_gap_time: list[float] = [
        b.begin - a.end
        for a, b in zip(actor_train_groups[:-1], refer_inference_spans[1:])
    ] + [0.0]

    PerfMetricsExport._log_perfetto_trace(
        export_dir,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
    )

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/first_micro_batch_rollout_time": (first_micro_batch_rollout_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/actor_train_time": (np.sum(actor_train_time), None),
        "perf/sum/actor_train_step_time": (np.sum(actor_train_step_time), None),
        "perf/sum/between_micro_batch_gap_time": (np.sum(between_micro_batch_gap_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/actor_train_time": (np.mean(actor_train_time), None),
        "perf/mean/actor_train_step_time": (np.mean(actor_train_step_time), None),
        "perf/mean/between_micro_batch_gap_time": (np.mean(between_micro_batch_gap_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_metrics_fully_disaggregated(
      role_to_devices: dict[str, list[str]],
      export_dir: str | None,
      query: PerfSpanQuery,
  ) -> MetricsT:
    """GRPO workflow: rollout, actor and reference are all on different meshes."""
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
        actor_train_step_spans,
    ) = PerfMetricsExport._grpo_extract_spans_and_groups(role_to_devices, query)
    if not ok:
      return {}

    weight_sync_span = global_step_group.find_last_inner_span("weight_sync")
    if weight_sync_span is None:
      logging.warning("weight_sync is None")
      return {}

    # Step 2: compute metrics from spans and span groups

    global_step_time: float = global_step_group.duration
    weight_sync_time: float = weight_sync_span.duration

    rollout_time: list[float] = [span.duration for span in rollout_spans]
    rollout_idle_time: float = weight_sync_span.begin - rollout_spans[-1].end

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference_spans
    ]
    # append [0.0] to make size equal to micro batch
    refer_gap_time: list[float] = [
        b.end - a.begin
        for a, b in zip(refer_inference_spans[:-1], refer_inference_spans[1:])
    ] + [0.0]

    # train time includes gradient update and eval
    actor_train_time: list[float] = [
        group.duration for group in actor_train_groups
    ]
    actor_train_step_time: list[float] = [
        span.duration for span in actor_train_step_spans
    ]

    first_micro_batch_rollout_time: float = (
        rollout_spans[0].end - global_step_group.begin
    )

    # append [0.0] to make size equal to micro batch
    actor_gap_time: list[float] = [
        b.end - a.begin
        for a, b in zip(actor_train_groups[:-1], actor_train_groups[1:])
    ] + [0.0]

    PerfMetricsExport._log_perfetto_trace(
        export_dir,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
    )

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/first_micro_batch_rollout_time": (first_micro_batch_rollout_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/refer_gap_time": (np.sum(refer_gap_time), None),
        "perf/sum/actor_train_time": (np.sum(actor_train_time), None),
        "perf/sum/actor_train_step_time": (np.sum(actor_train_step_time), None),
        "perf/sum/actor_gap_time": (np.sum(actor_gap_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/refer_gap_time": (np.mean(refer_gap_time), None),
        "perf/mean/actor_train_time": (np.mean(actor_train_time), None),
        "perf/mean/actor_train_step_time": (np.mean(actor_train_step_time), None),
        "perf/mean/actor_gap_time": (np.mean(actor_gap_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _log_perfetto_trace(
      export_dir: str | None,
      global_step_group: SpanGroup,
      rollout_spans: list[Span],
      refer_inference_spans: list[Span],
      actor_train_groups: list[SpanGroup],
  ) -> None:
    """Generates and writes Perfetto trace events to a file."""
    trace_events = PerfMetricsExport._generate_perfetto_trace(
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
    )
    PerfMetricsExport._write_perfetto_trace(
        export_dir, global_step_group, trace_events
    )

  @staticmethod
  def _write_perfetto_trace(
      export_dir: str | None,
      global_step_group: SpanGroup,
      trace_events: str,
  ) -> None:
    """Writes Perfetto trace events to a file."""
    export_dir = export_dir or PerfMetricsExport._DEFAULT_EXPORT_DIR
    try:
      os.makedirs(export_dir, exist_ok=True)
      trace_filename = f"perf_trace_{int(global_step_group.begin * 1e6)}.json"
      with open(os.path.join(export_dir, trace_filename), "w") as f:
        f.write(trace_events)
    except Exception as e:  # pylint: disable=broad-except
      logging.error(
          "Failed to write perfetto trace. Skipping the trace dump: %s", e
      )

  @staticmethod
  def _generate_perfetto_trace(
      global_step_group: SpanGroup,
      rollout_spans: list[Span],
      refer_inference_spans: list[Span],
      actor_train_groups: list[SpanGroup],
  ) -> str:
    events = []

    # Metadata for process names
    events.extend([
        {
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {"name": "Main"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": "Rollout"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": 2,
            "tid": 0,
            "args": {"name": "Reference"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": 3,
            "tid": 0,
            "args": {"name": "Actor"},
        },
    ])

    def add_span(s: Span | SpanGroup, pid: int, tid: int):
      events.append({
          "name": s.name,
          "cat": "PERF",
          "ph": "X",
          "ts": int(s.begin * 1_000_000),
          "dur": int(s.duration * 1_000_000),
          "pid": pid,
          "tid": tid,
      })

    def add_group(g: SpanGroup, pid: int, tid: int):
      add_span(g, pid, tid)
      for inner in g.inner:
        if isinstance(inner, SpanGroup):
          add_group(inner, pid, tid)
        elif isinstance(inner, Span):
          add_span(inner, pid, tid)

    # Main
    if global_step_group:
      add_group(global_step_group, 0, 0)

    # Rollout
    for s in rollout_spans:
      add_span(s, 1, 0)

    # Reference
    for s in refer_inference_spans:
      add_span(s, 2, 0)

    # Actor
    for g in actor_train_groups:
      add_group(g, 3, 0)

    return json.dumps(events)

  @staticmethod
  def _grpo_extract_spans_and_groups(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> tuple[
      bool, SpanGroup, list[Span], list[Span], list[SpanGroup], list[Span]
  ]:
    """Extracts spans and span groups of the last global step for GRPO workflow."""

    global_steps: list[SpanGroup] = (
        query().main().last_group("global_step").get()
    )
    if not global_steps:
      logging.warning("global_step is None")
      return (False, SpanGroup(""), [], [], [], [])

    global_step_group: SpanGroup = global_steps[0]

    micro_batch: PerfSpanQuery = (
        query()
        .last_group("global_step")
        .all_groups("mini_batch_step")
        .all_groups("micro_batch_steps")
    )
    main_groups = micro_batch.main().get()
    rollout_groups = micro_batch.timeline(role_to_devices["rollout"][0]).get()
    refer_groups = micro_batch.timeline(role_to_devices["refer"][0]).get()
    actor_groups = micro_batch.timeline(role_to_devices["actor"][0]).get()

    if not rollout_groups or not refer_groups or not actor_groups:
      logging.warning("rollout_group or refer_group or actor_group is None")
      return (False, SpanGroup(""), [], [], [], [])

    rollout_span: list[Span] = []
    refer_inference_span: list[Span] = []
    actor_train_groups: list[SpanGroup] = []
    actor_train_step_span: list[Span] = []

    for group in rollout_groups:
      rollout_span.extend(group.find_all_inner_spans("rollout"))
    for group in refer_groups:
      refer_inference_span.extend(group.find_all_inner_spans("refer_inference"))
    for group in actor_groups:
      actor_train_groups.extend(group.find_all_inner_groups("actor_training"))
    # TODO(yangmu) rewrite this after peft_train_step is attached to device
    # timeline. Note that peft_train_step records the correct device timespan.
    for group in main_groups:
      for actor_train_group in group.find_all_inner_groups("actor_training"):
        actor_train_step_span.extend(
            actor_train_group.find_all_inner_spans("peft_train_step")
        )

    return (
        True,
        global_step_group,
        rollout_span,
        refer_inference_span,
        actor_train_groups,
        actor_train_step_span,
    )
