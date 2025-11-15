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

from typing import Callable

import numpy as np
from tunix.perf import metrics
from tunix.perf import trace
from tunix.rl import rl_cluster


ClusterConfig = rl_cluster.ClusterConfig
PerfMetricsQuery = metrics.PerfMetricsQuery
PerfMetricsContext = metrics.PerfMetricsContext
MetricsT = metrics.MetricsT
MetricsExportFn = Callable[[PerfMetricsQuery, PerfMetricsContext], MetricsT]


class PerfMetricsExport:
  """Provides helper functions to create metrics export functions."""

  @staticmethod
  def create_metrics_export_fn(
      cluster_config: ClusterConfig,
  ) -> MetricsExportFn:
    """Creates a metrics export function based on the mesh topology in cluster config."""

    rollo_mesh = cluster_config.role_to_mesh[rl_cluster.Role.ROLLOUT]
    actor_mesh = cluster_config.role_to_mesh[rl_cluster.Role.ACTOR]
    refer_mesh = cluster_config.role_to_mesh[rl_cluster.Role.REFERENCE]

    rollo_devices = rollo_mesh.devices.flatten().tolist()
    actor_devices = actor_mesh.devices.flatten().tolist()
    refer_devices = refer_mesh.devices.flatten().tolist()

    rollo_tids = sorted(
        [trace.create_timeline_id(device) for device in rollo_devices]
    )
    actor_tids = sorted(
        [trace.create_timeline_id(device) for device in actor_devices]
    )
    refer_tids = sorted(
        [trace.create_timeline_id(device) for device in refer_devices]
    )

    # Colocated case: rollout, actor and reference are colocated on the same
    # mesh.
    def metrics_export_colocated(
        query: PerfMetricsQuery, context: PerfMetricsContext
    ) -> MetricsT:
      glob_step = query.main().busy().sum() + query.main().idle().sum()

      all_gap = [query.timeline(device).idle().sum() for device in rollo_tids]

      return {
          "perf/global_step_time": (glob_step, None),
          "perf/gap_time": (np.mean(all_gap), None),
      }

    # Disaggregated case 1: actor and reference are on the same mesh, rollout is
    # on a different mesh.
    def metrics_export_disagg_1(
        query: PerfMetricsQuery, context: PerfMetricsContext
    ) -> MetricsT:
      glob_step = query.main().busy().sum() + query.main().idle().sum()

      roll_idle = [query.timeline(device).idle().sum() for device in rollo_tids]
      infer_and_train_gap = [
          query.timeline(device).idle().sum() for device in refer_tids
      ]

      return {
          "perf/global_step_time": (glob_step, None),
          "perf/rollout_idle_time": (np.mean(roll_idle), None),
          "perf/inference_and_train_gap_time": (
              np.mean(infer_and_train_gap),
              None,
          ),
      }

    # Disaggregated case 2: rollout, actor and reference are all on different
    # meshes.
    def metrics_export_disagg_2(
        query: PerfMetricsQuery, context: PerfMetricsContext
    ) -> MetricsT:
      glob_step = query.main().busy().sum() + query.main().idle().sum()

      roll_idle = [query.timeline(device).idle().sum() for device in rollo_tids]
      infer_gap = [query.timeline(device).idle().sum() for device in refer_tids]
      train_gap = [query.timeline(device).idle().sum() for device in actor_tids]

      return {
          "perf/global_step_time": (glob_step, None),
          "perf/rollout_idle_time": (np.mean(roll_idle), None),
          "perf/inference_gap_time": (np.mean(infer_gap), None),
          "perf/train_gap_time": (np.mean(train_gap), None),
      }

    if rollo_tids == refer_tids == actor_tids:
      return metrics_export_colocated
    elif rollo_tids != refer_tids == actor_tids:
      return metrics_export_disagg_1
    elif rollo_tids != refer_tids != actor_tids:
      return metrics_export_disagg_2
    else:
      raise ValueError("Unsupported mesh configuration.")
