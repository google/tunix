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
    """GRPO colocated case: rollout, actor and reference are colocated on the same mesh."""
    global_step: SpanGroup | None = query().main().group("global_step").get()

    if global_step is None:
      logging.warning("global_step is None")
      return {}

    weight_sync: Span | None = global_step.find_last_inner_span("weight_sync")
    if weight_sync is None:
      weight_sync = Span("weight_sync", global_step.end)
      weight_sync.end = global_step.end

    micro_batch_query: PerfSpanQuery = (
        query()
        .group("global_step")
        .group("mini_batch_step")
        .group("micro_batch_steps")
    )
    rollout_group = micro_batch_query.timeline(
        role_to_devices["rollout"][0]
    ).get()
    refer_group = micro_batch_query.timeline(role_to_devices["refer"][0]).get()
    actor_group = micro_batch_query.timeline(role_to_devices["actor"][0]).get()

    if rollout_group is None or refer_group is None or actor_group is None:
      logging.warning("rollout_group or refer_group or actor_group is None")
      return {}

    rollout: list[Span] = rollout_group.find_all_inner_spans("rollout")
    refer_inference: list[Span] = refer_group.find_all_inner_spans(
        "refer_inference"
    )
    actor_training: list[Span] = actor_group.find_all_inner_spans(
        "actor_training"
    )

    global_step_time: float = global_step.duration
    weight_sync_time: float = weight_sync.duration

    rollout_time: list[float] = [span.duration for span in rollout]

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference
    ]

    # training includes gradient update
    actor_train_grad_time: list[float] = [
        span.duration for span in actor_training
    ]

    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_time": (np.sum(rollout_time), None),
        "perf/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/actor_train_grad_time": (np.sum(actor_train_grad_time), None),
        "perf/micro_batch/rollout_time": (np.mean(rollout_time), None),
        "perf/micro_batch/refer_inference_time": (
            np.mean(refer_inference_time),
            None,
        ),
        "perf/micro_batch/actor_train_grad_time": (
            np.mean(actor_train_grad_time),
            None,
        ),
    }

  @staticmethod
  def _grpo_metrics_rollout_1_actor_2_reference_2(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> MetricsT:
    """GRPO disaggregated case 1: actor and reference are on the same mesh,rollout is on a different mesh."""
    global_step: SpanGroup | None = query().main().group("global_step").get()

    if global_step is None:
      logging.warning("global_step is None")
      return {}

    weight_sync: Span | None = global_step.find_last_inner_span("weight_sync")
    if weight_sync is None:
      weight_sync = Span("weight_sync", global_step.end)
      weight_sync.end = global_step.end

    micro_batch_query: PerfSpanQuery = (
        query()
        .group("global_step")
        .group("mini_batch_step")
        .group("micro_batch_steps")
    )
    rollout_group = micro_batch_query.timeline(
        role_to_devices["rollout"][0]
    ).get()
    refer_group = micro_batch_query.timeline(role_to_devices["refer"][0]).get()
    actor_group = micro_batch_query.timeline(role_to_devices["actor"][0]).get()

    if rollout_group is None or refer_group is None or actor_group is None:
      logging.warning("rollout_group or refer_group or actor_group is None")
      return {}

    rollout: list[Span] = rollout_group.find_all_inner_spans("rollout")
    refer_inference: list[Span] = refer_group.find_all_inner_spans(
        "refer_inference"
    )
    actor_training: list[Span] = actor_group.find_all_inner_spans(
        "actor_training"
    )

    global_step_time: float = global_step.duration
    weight_sync_time: float = weight_sync.duration

    rollout_time: list[float] = [span.duration for span in rollout]
    rollout_idle_time: float = weight_sync.begin - rollout[-1].end

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference
    ]

    # training includes gradient update
    actor_train_grad_time: list[float] = [
        span.duration for span in actor_training
    ]

    first_micro_batch_rollout_time: float = rollout[0].begin - global_step.begin

    between_micro_batch_gap_time: list[float] = [
        b.begin - a.end
        for a, b in zip(actor_training[:-1], refer_inference[1:])
    ]

    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/rollout_time": (np.sum(rollout_time), None),
        "perf/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/actor_train_grad_time": (np.sum(actor_train_grad_time), None),
        "perf/first_micro_batch_rollout_time": (
            first_micro_batch_rollout_time,
            None,
        ),
        "perf/between_micro_batch_gap_time": (
            np.sum(between_micro_batch_gap_time),
            None,
        ),
        "perf/micro_batch/rollout_time": (np.mean(rollout_time), None),
        "perf/micro_batch/refer_inference_time": (
            np.mean(refer_inference_time),
            None,
        ),
        "perf/micro_batch/actor_train_grad_time": (
            np.mean(actor_train_grad_time),
            None,
        ),
        "perf/micro_batch/between_micro_batch_gap_time": (
            np.mean(between_micro_batch_gap_time),
            None,
        ),
    }

  @staticmethod
  def _grpo_metrics_fully_disaggregated(
      role_to_devices: dict[str, list[str]], query: PerfSpanQuery
  ) -> MetricsT:
    """GRPO disaggregated case 2: rollout, actor and reference are all on different meshes."""
    global_step: SpanGroup | None = query().main().group("global_step").get()

    if global_step is None:
      logging.warning("global_step is None")
      return {}

    weight_sync: Span | None = global_step.find_last_inner_span("weight_sync")

    if weight_sync is None:
      logging.warning("weight_sync is None")
      return {}

    micro_batch_query: PerfSpanQuery = (
        query()
        .group("global_step")
        .group("mini_batch_step")
        .group("micro_batch_steps")
    )
    rollout_group = micro_batch_query.timeline(
        role_to_devices["rollout"][0]
    ).get()
    refer_group = micro_batch_query.timeline(role_to_devices["refer"][0]).get()
    actor_group = micro_batch_query.timeline(role_to_devices["actor"][0]).get()

    if rollout_group is None or refer_group is None or actor_group is None:
      logging.warning("rollout_group or refer_group or actor_group is None")
      return {}

    rollout: list[Span] = rollout_group.find_all_inner_spans("rollout")
    refer_inference: list[Span] = refer_group.find_all_inner_spans(
        "refer_inference"
    )
    actor_training: list[Span] = actor_group.find_all_inner_spans(
        "actor_training"
    )

    global_step_time: float = global_step.duration
    weight_sync_time: float = weight_sync.duration

    rollout_time: list[float] = [span.duration for span in rollout]
    rollout_idle_time: float = weight_sync.begin - rollout[-1].end

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference
    ]
    # append [0.0] to make size equal to micro batch
    refer_gap_time: list[float] = [
        b.end - a.begin
        for a, b in zip(refer_inference[:-1], refer_inference[1:])
    ] + [0.0]

    # training includes gradient update
    actor_train_grad_time: list[float] = [
        span.duration for span in actor_training
    ]
    # append [0.0] to make size equal to micro batch
    actor_gap_time: list[float] = [
        b.end - a.begin for a, b in zip(actor_training[:-1], actor_training[1:])
    ] + [0.0]

    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/rollout_time": (np.sum(rollout_time), None),
        "perf/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/refer_gap_time": (np.sum(refer_gap_time), None),
        "perf/actor_train_grad_time": (np.sum(actor_train_grad_time), None),
        "perf/actor_gap_time": (np.sum(actor_gap_time), None),
        "perf/micro_batch/rollout_time": (np.mean(rollout_time), None),
        "perf/micro_batch/refer_inference_time": (
            np.mean(refer_inference_time),
            None,
        ),
        "perf/micro_batch/refer_gap_time": (np.mean(refer_gap_time), None),
        "perf/micro_batch/actor_train_grad_time": (
            np.mean(actor_train_grad_time),
            None,
        ),
        "perf/micro_batch/actor_gap_time": (np.mean(actor_gap_time), None),
    }
