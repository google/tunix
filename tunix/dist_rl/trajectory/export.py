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

"""DataFrame export helpers for TrajectoryReader."""

from typing import Any
import pandas as pd
from tunix.dist_rl.trajectory import base_store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.rl.agentic.agents import agent_types


def export_trajectories_df(
    reader: base_store.TrajectoryReader[Any],
    trajectory_ids: list[str] | None = None,
) -> pd.DataFrame:
  """Exports left-joined trajectory metadata and metrics as a flat DataFrame.

  Performs an in-memory left join between `reader.get_trajectories_metadata()`
  and `reader.get_trajectories_metrics(None)` keyed on `trajectory_id`.
  Trajectories that do not have recorded metrics (e.g. in-flight or failed
  rollouts dropped before training) still appear in the exported DataFrame with
  null/None values for metric fields.

  Args:
    reader: The TrajectoryReader instance to export from.
    trajectory_ids: Optional list of trajectory IDs to export. If None, exports
      all trajectories in the run.

  Returns:
    A flat pandas DataFrame containing combined metadata and metrics fields,
    indexed by row order with `trajectory_id` as a column. If no trajectories
    are found, returns an empty DataFrame.
  """
  metadata_list = reader.get_trajectories_metadata(trajectory_ids)
  if not metadata_list:
    return pd.DataFrame()

  metrics_by_id = reader.get_trajectories_metrics(None)

  rows: list[dict[str, Any]] = []
  for raw_meta in metadata_list:
    row: dict[str, Any] = {"trajectory_id": raw_meta.trajectory_id}

    if raw_meta.session_id is not None:
      row["session_id"] = raw_meta.session_id

    if raw_meta.agent is not None:
      row["agent_name"] = raw_meta.agent.name
      if raw_meta.agent.version is not None:
        row["agent_version"] = raw_meta.agent.version

    if raw_meta.final_metrics is not None:
      row["total_prompt_tokens"] = raw_meta.final_metrics.total_prompt_tokens
      row["total_completion_tokens"] = (
          raw_meta.final_metrics.total_completion_tokens
      )
      row["total_steps"] = raw_meta.final_metrics.total_steps
      if raw_meta.final_metrics.total_cached_tokens is not None:
        row["total_cached_tokens"] = raw_meta.final_metrics.total_cached_tokens
      if raw_meta.final_metrics.total_cost_usd is not None:
        row["total_cost_usd"] = raw_meta.final_metrics.total_cost_usd
      if raw_meta.final_metrics.extra:
        for k, v in raw_meta.final_metrics.extra.items():
          row[k] = v

    meta: trajectory_lib.TrajectoryMetadata = raw_meta
    if not isinstance(raw_meta, trajectory_lib.TunixTrajectoryMetadata):
      if (
          raw_meta.extra
          and trajectory_lib.TUNIX_EXTENSIONS_KEY in raw_meta.extra
      ):
        meta = trajectory_lib.TunixTrajectoryMetadata.from_atif_metadata(
            raw_meta
        )

    if isinstance(meta, trajectory_lib.TunixTrajectoryMetadata):
      row["prompt_id"] = meta.prompt_id
      row["group_index"] = meta.group_index
      row["target_policy_versions"] = meta.target_policy_versions
      row["total_reward"] = meta.total_reward
      if meta.status is not None:
        if isinstance(meta.status, agent_types.TrajectoryStatus):
          row["status"] = meta.status.name
        else:
          row["status"] = str(meta.status)
      else:
        row["status"] = None
      row["masked_out"] = meta.masked_out
      row["hyperparams"] = meta.hyperparams

    if meta.extra:
      for k, v in meta.extra.items():
        if k != trajectory_lib.TUNIX_EXTENSIONS_KEY:
          row[k] = v

    # Left-join TrajectoryMetrics
    if meta.trajectory_id is not None:
      metrics = metrics_by_id.get(meta.trajectory_id)
      if metrics is not None and metrics.extra:
        for k, v in metrics.extra.items():
          row[k] = v

    rows.append(row)

  return pd.DataFrame(rows)


export_trajectories_metadata_df = export_trajectories_df
