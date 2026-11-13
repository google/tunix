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

"""Zero-dependency OpenTelemetry instrument catalog for dist_rl.

Every streaming instrument is declared once here as an `InstrumentSpec` (with
`deps = []` under the namespace package `tunix.dist_rl.common`, so OSS and
Blaze launchers never pull in JAX or compiled worker extensions). Producer
modules build their OpenTelemetry meters from these specs, and
`otel_dashboard.py` reads the same registry and groups charts by `role` with
no edits to dashboard code when new specs are added.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Literal

Kind = Literal["counter", "histogram", "gauge"]
Aligner = Literal["ALIGN_RATE", "ALIGN_DELTA", "ALIGN_MEAN", "ALIGN_MAX"]
Reducer = Literal[
    "REDUCE_SUM",
    "REDUCE_MEAN",
    "REDUCE_MAX",
    "REDUCE_PERCENTILE_50",
    "REDUCE_PERCENTILE_90",
    "REDUCE_PERCENTILE_99",
]
PlotType = Literal["LINE", "STACKED_AREA", "STACKED_BAR"]

VALID_AGGREGATIONS: Mapping[Kind, frozenset[tuple[Aligner, Reducer]]] = {
    "counter": frozenset({("ALIGN_RATE", "REDUCE_SUM")}),
    "histogram": frozenset({
        ("ALIGN_DELTA", "REDUCE_PERCENTILE_50"),
        ("ALIGN_DELTA", "REDUCE_PERCENTILE_90"),
        ("ALIGN_DELTA", "REDUCE_PERCENTILE_99"),
    }),
    "gauge": frozenset({
        ("ALIGN_MEAN", "REDUCE_MEAN"),
        ("ALIGN_MEAN", "REDUCE_SUM"),
        ("ALIGN_MAX", "REDUCE_MAX"),
    }),
}

DEFAULT_AGGREGATIONS: Mapping[Kind, tuple[tuple[Aligner, Reducer], ...]] = {
    "counter": (("ALIGN_RATE", "REDUCE_SUM"),),
    "histogram": (("ALIGN_DELTA", "REDUCE_PERCENTILE_50"),),
    "gauge": (("ALIGN_MEAN", "REDUCE_MEAN"),),
}

ROLE_TITLES: Mapping[str, str] = {
    "rollout": "Rollout workers (tunix.worker.*, all completed rollouts)",
    "trainer": "Trainer worker (tunix.trainer.*)",
    "orchestrator": "Orchestrator (tunix.orchestrator.*, streaming)",
    "fault_tolerance": "Fault tolerance & fleet (tunix.ft.*)",
    "inference": "Reference inference worker (tunix.inference.*)",
    "env": "Environment & sandbox (tunix.env.*)",
    "trajectory_store": "TrajectoryStore I/O (tunix.traj_store.*)",
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class InstrumentSpec:
  """Canonical definition of one streaming dist_rl OpenTelemetry instrument.

  Attributes:
    name: OpenTelemetry instrument name (for example
      `"tunix.worker.execution_sec"`).
    kind: Instrument kind (`"counter"`, `"histogram"`, or `"gauge"`).
    role: Logical role section for automatic dashboard grouping (for example
      `"rollout"`, `"trainer"`, `"orchestrator"`, `"fault_tolerance"`).
    unit: OpenTelemetry unit string (for example `"s"`, `"{token}"`, `"1"`).
    description: Instrument description passed to the OpenTelemetry meter.
    title: Optional chart title prefix; defaults to `name`.
    group_by: Bounded attribute names to group by on the dashboard (without the
      `"metric.label."` prefix). High-cardinality `"worker_id"` is forbidden.
    y_label: Optional chart y-axis label; defaults to `unit`.
    plot_type: Cloud Monitoring plot type (`"LINE"`, `"STACKED_AREA"`, or
      `"STACKED_BAR"`).
    aggregations: Optional sequence of `(aligner, reducer)` pairs to render as
      charts; defaults to `DEFAULT_AGGREGATIONS[kind]`.
  """

  name: str
  kind: Kind
  role: str
  unit: str
  description: str
  title: str = ""
  group_by: tuple[str, ...] = ()
  y_label: str = ""
  plot_type: PlotType = "LINE"
  aggregations: tuple[tuple[Aligner, Reducer], ...] | None = None

  def __post_init__(self) -> None:
    if not self.name.strip():
      raise ValueError("InstrumentSpec.name must be non-empty.")
    if not self.role.strip():
      raise ValueError(f"{self.name}: role must be non-empty.")
    if not self.unit.strip():
      raise ValueError(f"{self.name}: unit must be non-empty.")
    if not self.description.strip():
      raise ValueError(f"{self.name}: description must be non-empty.")
    for label in self.group_by:
      clean = label.removeprefix("metric.label.")
      if clean in ("worker_id", "tunix_worker_id"):
        raise ValueError(
            f"{self.name}: group_by must not contain high-cardinality"
            f" {label!r}."
        )
    if self.aggregations is not None and not self.aggregations:
      raise ValueError(f"{self.name}: aggregations must be non-empty when set.")
    valid = VALID_AGGREGATIONS[self.kind]
    for agg in self.effective_aggregations:
      if agg not in valid:
        raise ValueError(
            f"{self.name}: aggregation {agg} is invalid for {self.kind!r};"
            f" expected one of {sorted(valid)}."
        )

  @property
  def effective_title(self) -> str:
    """Returns the chart title, falling back to `name`."""
    return self.title or self.name

  @property
  def effective_y_label(self) -> str:
    """Returns the Y-axis label, falling back to `unit`."""
    return self.y_label or self.unit

  @property
  def effective_aggregations(self) -> tuple[tuple[Aligner, Reducer], ...]:
    """Returns explicit aggregations or kind-based defaults."""
    return self.aggregations or DEFAULT_AGGREGATIONS[self.kind]


_REGISTRY: dict[str, InstrumentSpec] = {}


def register(spec: InstrumentSpec) -> InstrumentSpec:
  """Registers `spec` globally and returns it.

  Re-registering an equal `InstrumentSpec` is a no-op; registering a different
  spec under the same `name` raises `ValueError`.

  Args:
    spec: Instrument specification to register.

  Returns:
    The registered `InstrumentSpec`.
  """
  existing = _REGISTRY.get(spec.name)
  if existing is not None:
    if existing == spec:
      return existing
    raise ValueError(
        f"Conflicting InstrumentSpec registration for {spec.name!r}:"
        f" existing={existing!r}, new={spec!r}."
    )
  _REGISTRY[spec.name] = spec
  return spec


def registered_specs() -> tuple[InstrumentSpec, ...]:
  """Returns all registered `InstrumentSpec` entries in registration order."""
  return tuple(_REGISTRY.values())


def role_title(role: str) -> str:
  """Returns the display header for `role`."""
  return ROLE_TITLES.get(role, role.replace("_", " ").title())


def specs_by_role(
    specs: Sequence[InstrumentSpec] | None = None,
) -> dict[str, tuple[InstrumentSpec, ...]]:
  """Groups `specs` (defaulting to the global registry) by `role`."""
  items = tuple(_REGISTRY.values()) if specs is None else tuple(specs)
  grouped: dict[str, list[InstrumentSpec]] = {}
  for spec in items:
    grouped.setdefault(spec.role, []).append(spec)
  ordered_roles = [role for role in ROLE_TITLES if role in grouped]
  ordered_roles.extend(role for role in grouped if role not in ROLE_TITLES)
  return {role: tuple(grouped[role]) for role in ordered_roles}


# Registered streaming worker instruments (single source of truth for both
# OpenTelemetry meter creation and automatic Cloud Monitoring dashboards).
# Histograms keep the default `plot_type="LINE"` because stacking per-label
# percentiles sums medians and double-counts overlapping intervals (such as
# `total_paused` with `pre`/`h2d`/`post`, or `trainer_idle` with
# `data_fetch_delay`).
TRAINER_STEP_DURATION_SPEC = register(
    InstrumentSpec(
        name="tunix.trainer.step.duration",
        kind="histogram",
        role="trainer",
        unit="s",
        description=(
            "Host-side execution or dispatch duration of trainer worker"
            " operations by phase."
        ),
        title="Trainer step duration p50 by phase",
        group_by=("phase",),
        y_label="seconds",
    )
)
TRAINER_TOKENS_PROCESSED_SPEC = register(
    InstrumentSpec(
        name="tunix.trainer.tokens.processed",
        kind="counter",
        role="trainer",
        unit="{token}",
        description=(
            "Total valid and padded tokens processed by the trainer worker"
            " across BatchAssembler-stamped RLTrainerPayload microbatches."
        ),
        title="Trainer tokens / s (valid vs. padded)",
        group_by=("token_type",),
        y_label="tokens/s",
        plot_type="STACKED_AREA",
    )
)
TRAINER_WEIGHT_SYNC_DURATION_SPEC = register(
    InstrumentSpec(
        name="tunix.trainer.weight_sync.duration",
        kind="histogram",
        role="trainer",
        unit="s",
        description=(
            "Duration of trainer weight sync staging and release (staging"
            " includes waiting on queued device work)."
        ),
        title="Trainer weight sync p50 by direction",
        group_by=("direction",),
        y_label="seconds",
    )
)
TRAINER_STARVATION_DURATION_SPEC = register(
    InstrumentSpec(
        name="tunix.trainer.starvation.duration",
        kind="histogram",
        role="trainer",
        unit="s",
        description=(
            "Host-side trainer idle time (now - last_free_time_s, anchored at"
            " host RPC completion) and data-fetch delay (max(0, now -"
            " max(last_free_time_s, mb_ready_time_s)), including post-assembly"
            " prep and RPC transfer across host wall clocks) before a"
            " microbatch."
        ),
        title="Trainer starvation p50 by kind",
        group_by=("kind",),
        y_label="seconds",
    )
)

ROLLOUT_TRAJ_COMPLETED_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.trajectories.completed",
        kind="counter",
        role="rollout",
        unit="{trajectory}",
        description="Count of completed rollout trajectories.",
        title="Completed trajectories / s by status",
        group_by=("status",),
        y_label="trajectories/s",
        plot_type="STACKED_AREA",
    )
)
ROLLOUT_TURNS_TOTAL_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.turns.total",
        kind="counter",
        role="rollout",
        unit="{turn}",
        description="Total assistant turns generated across trajectories.",
        title="Assistant turns / s by task",
        group_by=("task_name",),
        y_label="turns/s",
    )
)
ROLLOUT_PROMPT_TOKENS_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.prompt_tokens",
        kind="histogram",
        role="rollout",
        unit="{token}",
        description="Prompt token count per rollout trajectory.",
        title="Prompt tokens per trajectory p50",
        group_by=("task_name",),
        y_label="tokens",
    )
)
ROLLOUT_COMPLETION_TOKENS_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.completion_tokens",
        kind="histogram",
        role="rollout",
        unit="{token}",
        description="Completion token count per rollout trajectory.",
        title="Completion tokens per trajectory p50",
        group_by=("task_name",),
        y_label="tokens",
    )
)
ROLLOUT_EXECUTION_SEC_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.execution_sec",
        kind="histogram",
        role="rollout",
        unit="s",
        description="End-to-end episode execution time on the rollout worker.",
        title="Rollout execution duration",
        group_by=("status",),
        y_label="seconds",
        aggregations=(
            ("ALIGN_DELTA", "REDUCE_PERCENTILE_50"),
            ("ALIGN_DELTA", "REDUCE_PERCENTILE_99"),
        ),
    )
)
ROLLOUT_SAMPLER_TIME_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.sampler_time_s",
        kind="histogram",
        role="rollout",
        unit="s",
        description=(
            "Cumulative model sampler inference latency per trajectory."
        ),
        title="Rollout sampler inference time p50",
        group_by=("task_name",),
        y_label="seconds",
    )
)
ROLLOUT_ENV_TIME_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.env_time_sec",
        kind="histogram",
        role="rollout",
        unit="s",
        description="Environment interaction latency by operation.",
        title="Environment time p50 by op",
        group_by=("op",),
        y_label="seconds",
    )
)
ROLLOUT_REWARD_TIME_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.reward_time_sec",
        kind="histogram",
        role="rollout",
        unit="s",
        description="Final reward computation latency per trajectory.",
        title="Reward computation time p50",
        group_by=("task_name",),
        y_label="seconds",
    )
)
ROLLOUT_WEIGHT_SYNC_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.weight_sync.duration",
        kind="histogram",
        role="rollout",
        unit="s",
        description=(
            "Rollout worker weight sync phase and total paused duration."
        ),
        title="Rollout weight sync p50 by phase",
        group_by=("phase",),
        y_label="seconds",
    )
)
ROLLOUT_REQUESTS_INFLIGHT_SPEC = register(
    InstrumentSpec(
        name="tunix.worker.requests_inflight",
        kind="gauge",
        role="rollout",
        unit="{request}",
        description=(
            "Number of active rollout episodes in flight on this worker."
        ),
        title="Rollout requests in flight (fleet total)",
        y_label="requests",
        aggregations=(("ALIGN_MEAN", "REDUCE_SUM"),),
    )
)
