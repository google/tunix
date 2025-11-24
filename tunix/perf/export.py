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
import logging
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

  @staticmethod
  def from_role_to_devices(
      role_to_devices: dict[str, list[str]],
  ) -> MetricsExportFn:
    """Creates a metrics export function based on the role to devices mapping."""
    r2d = role_to_devices
    if r2d["rollout"] == r2d["actor"] == r2d["refer"]:
      return partial(PerfMetricsExport._grpo_metrics_colocated, r2d)
    elif r2d["rollout"] != r2d["actor"] == r2d["refer"]:
      return partial(
          PerfMetricsExport._grpo_metrics_rollout_1_actor_2_reference_2, r2d
      )
    elif r2d["rollout"] != r2d["actor"] != r2d["refer"]:
      return partial(PerfMetricsExport._grpo_metrics_fully_disaggregated, r2d)
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
        }
    )

  # TODO(yangmu): DEPRECATED: remove after all users use the new API.
  @staticmethod
  def create_metrics_export_fn(
      cluster_config: ClusterConfig,
  ) -> MetricsExportFn:
    return PerfMetricsExport.from_cluster_config(cluster_config)

  @staticmethod
  def _grpo_metrics_colocated(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> MetricsT:
    """GRPO workflow: rollout, actor and reference are colocated on the same mesh."""
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_training_spans,
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

    # training includes gradient update
    actor_train_grad_time: list[float] = [
        span.duration for span in actor_training_spans
    ]

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/actor_train_grad_time": (np.sum(actor_train_grad_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/actor_train_grad_time": (np.mean(actor_train_grad_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_metrics_rollout_1_actor_2_reference_2(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> MetricsT:
    """GRPO workflow: actor and reference are on the same mesh,rollout is on a different mesh."""
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_training_spans,
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

    # training includes gradient update
    actor_train_grad_time: list[float] = [
        span.duration for span in actor_training_spans
    ]

    first_micro_batch_rollout_time: float = (
        rollout_spans[0].end - global_step_group.begin
    )

    # append [0.0] to make size equal to micro batch
    between_micro_batch_gap_time: list[float] = [
        b.begin - a.end
        for a, b in zip(actor_training_spans[:-1], refer_inference_spans[1:])
    ] + [0.0]

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/first_micro_batch_rollout_time": (first_micro_batch_rollout_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/actor_train_grad_time": (np.sum(actor_train_grad_time), None),
        "perf/sum/between_micro_batch_gap_time": (np.sum(between_micro_batch_gap_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/actor_train_grad_time": (np.mean(actor_train_grad_time), None),
        "perf/mean/between_micro_batch_gap_time": (np.mean(between_micro_batch_gap_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_metrics_fully_disaggregated(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> MetricsT:
    """GRPO workflow: rollout, actor and reference are all on different meshes."""
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_training_spans,
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

    # training includes gradient update
    actor_train_grad_time: list[float] = [
        span.duration for span in actor_training_spans
    ]
    # append [0.0] to make size equal to micro batch
    actor_gap_time: list[float] = [
        b.end - a.begin
        for a, b in zip(actor_training_spans[:-1], actor_training_spans[1:])
    ] + [0.0]

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/refer_gap_time": (np.sum(refer_gap_time), None),
        "perf/sum/actor_train_grad_time": (np.sum(actor_train_grad_time), None),
        "perf/sum/actor_gap_time": (np.sum(actor_gap_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/refer_gap_time": (np.mean(refer_gap_time), None),
        "perf/mean/actor_train_grad_time": (np.mean(actor_train_grad_time), None),
        "perf/mean/actor_gap_time": (np.mean(actor_gap_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_extract_spans_and_groups(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> tuple[bool, SpanGroup, list[Span], list[Span], list[Span]]:
    """Extracts spans and span groups of the last global step for GRPO workflow."""

    global_steps: list[SpanGroup] = (
        query().main().last_group("global_step").get()
    )
    if not global_steps:
      logging.warning("global_step is None")
      return (False, SpanGroup(""), [], [], [])

    global_step_group: SpanGroup = global_steps[0]

    micro_batch: PerfSpanQuery = (
        query()
        .last_group("global_step")
        .all_groups("mini_batch_step")
        .all_groups("micro_batch_steps")
    )
    rollout_groups = micro_batch.timeline(role_to_devices["rollout"][0]).get()
    refer_groups = micro_batch.timeline(role_to_devices["refer"][0]).get()
    actor_groups = micro_batch.timeline(role_to_devices["actor"][0]).get()

    if not rollout_groups or not refer_groups or not actor_groups:
      logging.warning("rollout_group or refer_group or actor_group is None")
      return (False, SpanGroup(""), [], [], [])

    rollout_span: list[Span] = []
    refer_inference_span: list[Span] = []
    actor_training_span: list[Span] = []

    for group in rollout_groups:
      rollout_span.extend(group.find_all_inner_spans("rollout"))
    for group in refer_groups:
      refer_inference_span.extend(group.find_all_inner_spans("refer_inference"))
    for group in actor_groups:
      actor_training_span.extend(group.find_all_inner_spans("actor_training"))

    return (
        True,
        global_step_group,
        rollout_span,
        refer_inference_span,
        actor_training_span,
    )
