"""Trajectory implementation using Agent Trajectory Interchange Format (ATIF).

For more details on the ATIF specification, see:
https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import enum
import functools
from typing import Annotated, Any, ClassVar, Final, Generic, Literal, Self, Sequence, TypeVar, get_args

import numpy as np
import pydantic

# ==============================================================================
# --- Pure ATIF Base Classes ---
# ==============================================================================


class Source(enum.StrEnum):
  """Source of the step message."""

  SYSTEM = enum.auto()
  USER = enum.auto()
  AGENT = enum.auto()


def _serialize_array(value: list[Any] | np.ndarray | None) -> list[Any] | None:
  """Serializes a NumPy array or list into a plain Python list."""
  if value is None:
    return None
  if isinstance(value, np.ndarray):
    return value.tolist()
  return list(value)


# Sequences whose elements all have one of these exact types are copied without
# per-element recursion: one C-level scan of `map(type, ...)` vets the whole
# sequence. Any other element type, including subclasses such as `StrEnum` and
# NumPy scalars, takes the recursive path, which also returns it unchanged.
_JSON_SCALAR_TYPES: Final[frozenset[type[object]]] = frozenset(
    {bool, float, int, str, type(None)}
)


def _to_json_compatible(value: Any) -> Any:
  """Recursively converts NumPy arrays and tuples nested in `value` to lists."""
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, dict):
    return {k: _to_json_compatible(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    # `to_atif_step()` packs long token, mask, and logprob lists into `extra`;
    # copy scalar-only sequences without a Python call per element.
    if _JSON_SCALAR_TYPES.issuperset(map(type, value)):
      return list(value)
    return [_to_json_compatible(v) for v in value]
  return value


def _serialize_dict(value: dict[str, Any] | None) -> dict[str, Any] | None:
  """Recursively converts any nested NumPy arrays within a dictionary to lists."""
  if value is None:
    return None
  return _to_json_compatible(value)


# Marks a key missing from a `dict.get` lookup.
_MISSING: Final[object] = object()


def _deep_copy_into_memo(
    value: Any, memo: dict[int, Any], originals: dict[int, object]
) -> Any:
  """Returns a deep copy of `value`, sharing `memo` with `copy.deepcopy`.

  `copy.deepcopy` makes a Python call per element of a list or dict, which
  dominates the cost of copying long token ID and logprob lists and nested
  JSON-like dicts such as tool definitions. This function copies exact `list`s
  and `dict`s itself: it reuses elements of `_JSON_SCALAR_TYPES`, which are
  immutable, without a call, and copies a list holding only such elements with
  one C-level `list.copy()`. Like `copy.deepcopy`, it records each copy in
  `memo` before copying the values nested in it, so shared and cyclic
  references are copied once. Any other value goes to `copy.deepcopy` with the
  same `memo`. For a pydantic model, it copies `__dict__` this way and then
  calls the model's `__deepcopy__`, which finds that copy in `memo`.

  Args:
    value: The value to copy.
    memo: The `copy.deepcopy` memo, mapping the `id()` of each copied original
      to its copy.
    originals: The lists, dicts, and models copied so far, keyed by `id()`.
      Holding them keeps their `id()`s in `memo` from being reused by new
      objects until the copy is complete, as `copy.deepcopy` does for the
      originals it copies.

  Returns:
    A deep copy of `value`.
  """
  value_type = type(value)
  if value_type in _JSON_SCALAR_TYPES:
    return value
  memoized_copy = memo.get(id(value), _MISSING)
  if memoized_copy is not _MISSING:
    return memoized_copy
  # Subclasses of `list` and `dict` are left to `copy.deepcopy`, which
  # preserves their type.
  if value_type is list:
    originals[id(value)] = value
    if _JSON_SCALAR_TYPES.issuperset(map(type, value)):
      scalar_list_copy = value.copy()
      memo[id(value)] = scalar_list_copy
      return scalar_list_copy
    list_copy: list[Any] = []
    memo[id(value)] = list_copy
    for element in value:
      if type(element) not in _JSON_SCALAR_TYPES:
        element = _deep_copy_into_memo(element, memo, originals)
      list_copy.append(element)
    return list_copy
  if value_type is dict:
    originals[id(value)] = value
    dict_copy: dict[Any, Any] = {}
    memo[id(value)] = dict_copy
    for key, element in value.items():
      # Copies the value before the key, in the order `copy.deepcopy` does.
      if type(element) not in _JSON_SCALAR_TYPES:
        element = _deep_copy_into_memo(element, memo, originals)
      if type(key) not in _JSON_SCALAR_TYPES:
        key = _deep_copy_into_memo(key, memo, originals)
      dict_copy[key] = element
    return dict_copy
  if isinstance(value, pydantic.BaseModel):
    originals[id(value)] = value
    # Copied here first, so that `__deepcopy__` finds this copy in `memo`.
    _deep_copy_into_memo(value.__dict__, memo, originals)
    # Calls `__deepcopy__` itself, as `copy.deepcopy(value, memo)` does on a
    # memo miss. If a field refers back to `value`, copying `__dict__` has
    # already put an inner copy of `value` in `memo`, which `copy.deepcopy`
    # would return instead of making the outer copy.
    copied_model = value.__deepcopy__(memo)
    memo[id(value)] = copied_model
    return copied_model
  return copy.deepcopy(value, memo)


_ModelT = TypeVar("_ModelT", bound=pydantic.BaseModel)


def deep_copy(model: _ModelT) -> _ModelT:
  """Returns a deep copy of `model`, copying plain lists and dicts faster.

  The copy matches `copy.deepcopy(model)` in types, values, and the sharing of
  nested objects, so for models without extra or private attributes, such as
  all models in this module, it also matches `model.model_copy(deep=True)`.
  See `_deep_copy_into_memo` for how the copy is made faster.

  Args:
    model: The model to copy.

  Returns:
    A copy of `model` that shares no mutable state with it.
  """
  return _deep_copy_into_memo(model, memo={}, originals={})


TUNIX_EXTENSIONS_KEY: Final[str] = "_tunix_extensions"
_EXTRA_FIELD: Final[str] = "extra"


@functools.lru_cache(maxsize=None)
def _get_field_names(model_cls: type[pydantic.BaseModel]) -> set[str]:
  """Returns the cached set of field names for `model_cls`."""
  return set(model_cls.model_fields)


def _get_non_none_fields(
    model: pydantic.BaseModel,
    field_names: set[str],
) -> dict[str, Any]:
  """Returns non-None field values read directly from `model`."""
  values_by_field = {}
  for field in field_names:
    value = getattr(model, field)
    if value is not None:
      values_by_field[field] = value
  return values_by_field


def _pack_subclass_values_into_extra(
    source_model: Step | TrajectoryMetadata,
    target_cls: type[pydantic.BaseModel],
    exclude_field_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
  """Packs subclass-specific fields into the `extra['tunix_extensions']` dict."""
  source_field_names = _get_field_names(type(source_model))
  target_field_names = _get_field_names(target_cls)
  if source_field_names == target_field_names:
    return None

  subclass_field_names = (
      source_field_names - target_field_names - exclude_field_names
  )
  target_values_by_field = _get_non_none_fields(
      source_model, target_field_names - {_EXTRA_FIELD}
  )

  # Serialize only subclass extension fields.
  subclass_values_by_field = {}
  if subclass_field_names:
    subclass_values_by_field = source_model.model_dump(
        include=subclass_field_names,
        exclude_none=True,
    )

  extra = dict(source_model.extra or {})
  if subclass_values_by_field:
    tunix_ext = extra.get(TUNIX_EXTENSIONS_KEY) or {}
    extra[TUNIX_EXTENSIONS_KEY] = tunix_ext | subclass_values_by_field
  if extra:
    target_values_by_field[_EXTRA_FIELD] = extra
  return target_values_by_field


def _unpack_subclass_values_from_extra(
    source_model: Step | TrajectoryMetadata,
    target_cls: type[pydantic.BaseModel],
) -> dict[str, Any]:
  """Extracts `source_model.extra[TUNIX_EXTENSIONS_KEY]` into top-level values."""
  extra = dict(source_model.extra or {})
  tunix_ext = dict(extra.pop(TUNIX_EXTENSIONS_KEY, None) or {})

  target_field_names = _get_field_names(target_cls)
  source_field_names = _get_field_names(type(source_model))
  subclass_field_names = target_field_names - source_field_names

  target_values_by_field = _get_non_none_fields(
      source_model, (source_field_names & target_field_names) - {_EXTRA_FIELD}
  )

  # Promote only target subclass fields from `extra[TUNIX_EXTENSIONS_KEY]`,
  # leaving any non-subclass keys in place so `extra="forbid"` is not tripped.
  for field in tunix_ext.keys() & subclass_field_names:
    target_values_by_field[field] = tunix_ext.pop(field)
  if tunix_ext:
    extra[TUNIX_EXTENSIONS_KEY] = tunix_ext
  if extra:
    target_values_by_field[_EXTRA_FIELD] = extra
  return target_values_by_field


_StepT = TypeVar("_StepT", "TunixAgentStep", "TunixEnvStep")


def _unpack_step_from_atif(
    step: Step,
    target_cls: type[_StepT],
    step_id_offset: int = 1,
) -> _StepT:
  """Rehydrates a 1-indexed ATIF Step into a 0-indexed Tunix step subclass."""
  if isinstance(step, target_cls):
    return step
  if step.step_id < step_id_offset:
    raise ValueError(
        f"Expected a {step_id_offset}-indexed ATIF step_id, got"
        f" {step.step_id}; this step may already use the 0-indexed Tunix"
        " convention."
    )
  target_values_by_field = _unpack_subclass_values_from_extra(step, target_cls)
  target_values_by_field["step_id"] = step.step_id - step_id_offset
  return target_cls.model_validate(target_values_by_field)


# `np.ndarray` comes first in these unions: smart-mode union validation returns
# the first exact match, so an array input passes one `isinstance` check. With
# `list[...]` first, the lax list validator would iterate and convert every
# array element before the array matched exactly and the result was discarded.
IntArray = Annotated[
    np.ndarray | list[int] | None,
    pydantic.PlainSerializer(
        _serialize_array, return_type=list[int] | None, when_used="always"
    ),
]

FloatArray = Annotated[
    np.ndarray | list[float] | None,
    pydantic.PlainSerializer(
        _serialize_array, return_type=list[float] | None, when_used="always"
    ),
]

MetadataDict = Annotated[
    dict[str, Any] | None,
    pydantic.PlainSerializer(
        _serialize_dict, return_type=dict[str, Any] | None, when_used="always"
    ),
]

ArgumentsDict = Annotated[
    dict[str, Any],
    pydantic.PlainSerializer(
        _serialize_dict, return_type=dict[str, Any], when_used="always"
    ),
]


class SubagentTrajectoryRef(pydantic.BaseModel):
  """Reference to a delegated subagent trajectory."""

  trajectory_id: str | None = pydantic.Field(
      default=None,
      description="ID of the subagent trajectory in parent.",
  )
  session_id: str | None = pydantic.Field(
      default=None,
      description="Run identity of the subagent, for debugging/correlation.",
  )
  trajectory_path: str | None = pydantic.Field(
      default=None,
      description="Path/URL of external subagent trajectory file.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom metadata about the subagent execution.",
  )

  @pydantic.model_validator(mode="after")
  def validate_is_resolvable(self) -> SubagentTrajectoryRef:
    if self.trajectory_id is None and self.trajectory_path is None:
      raise ValueError(
          "SubagentTrajectoryRef must set either trajectory_id or"
          " trajectory_path."
      )
    return self


class ObservationResult(pydantic.BaseModel):
  """A single result within an observation."""

  source_call_id: str | None = pydantic.Field(
      default=None,
      description="The corresponding tool_call_id from the step's tool_calls.",
  )
  content: str | None = pydantic.Field(
      default=None,
      description="Output or result from the action/tool execution.",
  )
  subagent_trajectory_ref: list[SubagentTrajectoryRef] | None = pydantic.Field(
      default=None,
      description="References to delegated subagent trajectories.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom observation result metadata.",
  )


class Observation(pydantic.BaseModel):
  """Environment feedback/result after actions or system events."""

  results: list[ObservationResult] = pydantic.Field(
      default_factory=list,
      description="Array of result objects from actions or tool calls.",
  )


class Metrics(pydantic.BaseModel):
  """LLM operational and confidence data for a step."""

  prompt_tokens: int | None = pydantic.Field(
      default=None,
      description="Total input tokens, including cached and non-cached.",
  )
  completion_tokens: int | None = pydantic.Field(
      default=None,
      description="Total generated output tokens.",
  )
  cached_tokens: int | None = pydantic.Field(
      default=None,
      description="Number of prompt tokens that hit the cache.",
  )
  cost_usd: float | None = pydantic.Field(
      default=None,
      description="Monetary cost of the model call in USD.",
  )
  prompt_token_ids: list[int] | None = pydantic.Field(
      default=None,
      description="Sequence of token IDs sent to the LLM.",
  )
  completion_token_ids: list[int] | None = pydantic.Field(
      default=None,
      description="Sequence of token IDs generated by the LLM.",
  )
  logprobs: list[float] | None = pydantic.Field(
      default=None,
      description="Log probability assigned to each generated token.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Other operational metrics.",
  )


class FinalMetrics(pydantic.BaseModel):
  """Aggregate statistics for the entire trajectory."""

  total_prompt_tokens: int | None = pydantic.Field(
      default=None,
      description="Sum of all prompt tokens across all steps.",
  )
  total_completion_tokens: int | None = pydantic.Field(
      default=None,
      description="Sum of all completion tokens across all steps.",
  )
  total_cached_tokens: int | None = pydantic.Field(
      default=None,
      description="Sum of all cached tokens across all steps.",
  )
  total_cost_usd: float | None = pydantic.Field(
      default=None,
      description="Total monetary cost of the trajectory in USD.",
  )
  total_steps: int | None = pydantic.Field(
      default=None,
      description="Total step count in the trajectory.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom aggregate metrics.",
  )


class ToolCall(pydantic.BaseModel):
  """A tool call within a step."""

  tool_call_id: str = pydantic.Field(
      description="Unique identifier for the tool invocation.",
  )
  function_name: str = pydantic.Field(
      description="Name of the function or tool being called.",
  )
  arguments: ArgumentsDict = pydantic.Field(
      default_factory=dict,
      description="JSON-serializable arguments passed to the function.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom tool-call-level metadata.",
  )


_AgentOnlyField = Literal[
    "model_name",
    "reasoning_effort",
    "reasoning_content",
    "tool_calls",
    "metrics",
]
_AGENT_ONLY_FIELDS: Final[tuple[_AgentOnlyField, ...]] = get_args(
    _AgentOnlyField
)

_LlmOnlyField = Literal[
    "metrics",
    "reasoning_content",
    "model_name",
    "reasoning_effort",
]
_LLM_ONLY_FIELDS: Final[tuple[_LlmOnlyField, ...]] = get_args(_LlmOnlyField)


class Step(pydantic.BaseModel):
  """A single turn/interaction step."""

  model_config = pydantic.ConfigDict(extra="forbid")

  step_id: int = pydantic.Field(
      description="Ordinal index of the turn (starting from 1).",
  )
  timestamp: datetime.datetime | None = pydantic.Field(
      default=None,
      description="UTC timestamp indicating when the step occurred.",
  )
  source: Source = pydantic.Field(
      description="Originator of the step (system, user, or agent).",
  )
  model_name: str | None = pydantic.Field(
      default=None,
      description="The specific LLM model used for this turn.",
  )
  reasoning_effort: str | float | None = pydantic.Field(
      default=None,
      description="Qualitative or quantitative measure of effort.",
  )
  message: str = pydantic.Field(
      description="Dialogue message content.",
  )
  reasoning_content: str | None = pydantic.Field(
      default=None,
      description="Agent's explicit internal reasoning or thoughts.",
  )
  tool_calls: list[ToolCall] | None = pydantic.Field(
      default=None,
      description="Structured actions or tools invoked by the agent.",
  )
  observation: Observation | None = pydantic.Field(
      default=None,
      description="Environment feedback resulting from the step's actions.",
  )
  metrics: Metrics | None = pydantic.Field(
      default=None,
      description="LLM operational and confidence metrics for this step.",
  )
  is_copied_context: bool | None = pydantic.Field(
      default=None,
      description="True if step was copied from a previous run.",
  )
  llm_call_count: int | None = pydantic.Field(
      default=None,
      ge=0,
      description="Number of LLM inferences this step represents.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom step-level metadata.",
  )

  @pydantic.model_validator(mode="after")
  def validate_agent_only_fields(self) -> Step:
    """Validate that certain fields are only present for agent steps."""
    if self.source == Source.AGENT:
      return self
    for field in _AGENT_ONLY_FIELDS:
      if getattr(self, field) is not None:
        raise ValueError(
            f"Field '{field}' is only applicable when source is 'agent', "
            f"but source is '{self.source}'"
        )
    return self

  @pydantic.model_validator(mode="after")
  def validate_llm_call_count_zero_fields(self) -> Step:
    """Enforce ATIF v1.7 no-LLM orchestration rule."""
    if self.llm_call_count != 0 or self.source != Source.AGENT:
      return self
    for field in _LLM_ONLY_FIELDS:
      if getattr(self, field) is not None:
        raise ValueError(
            f"Field '{field}' must be absent when llm_call_count is 0 "
            "(deterministic dispatch on a 'source: agent' step)"
        )
    return self

  def to_atif_step(self, step_id_offset: int = 0) -> Step:
    """Converts this step to a base ATIF Step, storing subclass fields in extra."""
    target_values_by_field = _pack_subclass_values_into_extra(self, Step)
    if target_values_by_field is None:
      return self
    target_values_by_field["step_id"] = self.step_id + step_id_offset
    return Step.model_validate(target_values_by_field)


class Agent(pydantic.BaseModel):
  """Basic agent metadata."""

  name: str = pydantic.Field(
      description="Name of the agent system.",
  )
  version: str = pydantic.Field(
      description="Version of the agent system.",
  )
  model_name: str | None = pydantic.Field(
      default=None,
      description="Default LLM model used for this trajectory.",
  )
  tool_definitions: list[dict[str, Any]] | None = pydantic.Field(
      default=None,
      description="Array of tool definitions available to the agent.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom agent configuration details.",
  )


class TrajectoryMetadata(pydantic.BaseModel):
  """Metadata for a trajectory (excluding steps and subagents)."""

  METADATA_TYPE: ClassVar[str] = "base"
  _REGISTRY: ClassVar[dict[str, type[TrajectoryMetadata]]] = {}

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    meta_type = cls.__dict__.get("METADATA_TYPE")
    if meta_type:
      if meta_type in cls._REGISTRY:
        raise ValueError(
            f"METADATA_TYPE {meta_type} is already registered to"
            f" {cls._REGISTRY[meta_type].__qualname__}; cannot register"
            f" {cls.__qualname__}."
        )
      cls._REGISTRY[meta_type] = cls

  model_config = pydantic.ConfigDict(extra="forbid")

  schema_version: str = pydantic.Field(
      default="ATIF-v1.7",
      description="Compatibility version of the ATIF schema.",
  )
  session_id: str | None = pydantic.Field(
      default=None,
      description="Run-scoped identity sharing among segment files.",
  )
  trajectory_id: str | None = pydantic.Field(
      default=None,
      description="Canonical unique identifier for this trajectory document.",
  )
  agent: Agent = pydantic.Field(
      description="Metadata describing the execution agent.",
  )
  notes: str | None = pydantic.Field(
      default=None,
      description="Custom information, design notes, or explanations.",
  )
  final_metrics: FinalMetrics | None = pydantic.Field(
      default=None,
      description="Aggregate metrics for the entire run.",
  )
  continued_trajectory_ref: str | None = pydantic.Field(
      default=None,
      description="Reference to the continuation trajectory file.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom root-level metadata.",
  )

  def to_atif_metadata(self) -> TrajectoryMetadata:
    """Converts this metadata to base ATIF TrajectoryMetadata, storing subclass fields in extra."""
    target_values_by_field = _pack_subclass_values_into_extra(
        self,
        TrajectoryMetadata,
        exclude_field_names={"steps", "subagent_trajectories"},
    )
    if target_values_by_field is None:
      return self
    return TrajectoryMetadata.model_validate(target_values_by_field)

  def get_extensions(self) -> dict[str, Any]:
    """Returns the packed subclass extensions dictionary from `extra`."""
    if not self.extra:
      return {}
    return self.extra.get(TUNIX_EXTENSIONS_KEY) or {}

  @classmethod
  def from_atif_metadata(cls: type[Self], metadata: TrajectoryMetadata) -> Self:
    """Rehydrates base ATIF TrajectoryMetadata into this metadata subclass."""
    if isinstance(metadata, cls):
      return metadata
    target_values_by_field = _unpack_subclass_values_from_extra(metadata, cls)
    return cls.model_validate(target_values_by_field)

  def _create_paired_trajectory(
      self,
      trajectory_cls: type[TrajectoryT],
      steps: Sequence[Any] | None,
      subagent_trajectories: Sequence[Any] | None,
  ) -> TrajectoryT:
    """Constructs `trajectory_cls` from this metadata, steps, and subagents.

    Verifies that `trajectory_cls` is a subclass of `type(self)` before
    instantiation so a `TrajectoryMetadata` subclass that does not override
    `create_trajectory()` raises `TypeError`.

    Args:
      trajectory_cls: Concrete Trajectory class to build.
      steps: Sequence of step instances, or None.
      subagent_trajectories: Sequence of subagent trajectory instances, or None.

    Returns:
      The constructed paired trajectory instance.

    Raises:
      TypeError: If `trajectory_cls` is not a subclass of `type(self)`.
    """
    # Completing a Trajectory is degenerate, and pydantic's parametrized
    # classes would fail this spuriously; exclude both.
    if not isinstance(self, Trajectory) and not issubclass(
        trajectory_cls, type(self)
    ):
      raise TypeError(
          f"Expected trajectory_cls to be a subclass of {type(self).__name__}"
      )
    metadata_fields = _get_field_names(type(self)) - {
        "steps",
        "subagent_trajectories",
    }
    data = _get_non_none_fields(self, metadata_fields)
    if steps is not None:
      data["steps"] = list(steps)
    if subagent_trajectories is not None:
      data["subagent_trajectories"] = list(subagent_trajectories)
    return trajectory_cls(**data)

  def create_trajectory(
      self,
      steps: Sequence[Any] | None = None,
      subagent_trajectories: Sequence[Any] | None = None,
  ) -> Trajectory[Any]:
    """Creates a full Trajectory from this metadata, steps, and subagents.

    Subclasses must override this method to construct their corresponding
    `Trajectory` subclass via `_create_paired_trajectory`.

    Args:
      steps: Sequence of step instances, or None.
      subagent_trajectories: Sequence of subagent trajectory instances, or None.

    Returns:
      The constructed trajectory instance.
    """
    return self._create_paired_trajectory(
        Trajectory, steps, subagent_trajectories
    )


# TrajectoryMetadata is the base class, so __init_subclass__ does not run on it.
TrajectoryMetadata._REGISTRY[TrajectoryMetadata.METADATA_TYPE] = (  # pylint: disable=protected-access
    TrajectoryMetadata
)


StepT = TypeVar("StepT", bound=Step)


class Trajectory(TrajectoryMetadata, Generic[StepT]):
  """Root trajectory object containing the interaction history."""

  # First step_id in a well-formed trajectory; step_ids must be sequential
  # from here.
  _STEP_ID_START: ClassVar[int] = 1

  steps: list[StepT] = pydantic.Field(
      default_factory=list,
      description="Sequential step history.",
  )
  subagent_trajectories: list[Trajectory[StepT]] | None = pydantic.Field(
      default=None,
      description="Array of embedded subagent trajectories.",
  )

  @pydantic.field_validator("steps")
  @classmethod
  def validate_step_ids(cls, steps: list[StepT]) -> list[StepT]:
    """Validate that step_ids are sequential from `_STEP_ID_START`."""
    steps.sort(key=lambda step: step.step_id)
    for expected_step_id, step in enumerate(steps, start=cls._STEP_ID_START):
      if step.step_id != expected_step_id:
        raise ValueError(
            f"Expected step_id {expected_step_id} (sequential from"
            f" {cls._STEP_ID_START}), got {step.step_id}"
        )
    return steps

  @pydantic.field_validator("subagent_trajectories")
  @classmethod
  def validate_embedded_subagent_trajectory_ids(
      cls, subagent_trajectories: list[Trajectory[StepT]] | None
  ) -> list[Trajectory[StepT]] | None:
    """Every embedded subagent must carry a unique, non-null trajectory_id."""
    if not subagent_trajectories:
      return subagent_trajectories
    seen: set[str] = set()
    for i, traj in enumerate(subagent_trajectories):
      if traj.trajectory_id is None:
        raise ValueError(
            f"subagent_trajectories[{i}].trajectory_id is required "
            "for embedded subagents."
        )
      if traj.trajectory_id in seen:
        raise ValueError(
            f"subagent_trajectories[{i}].trajectory_id: duplicate ID "
            f"'{traj.trajectory_id}'"
        )
      seen.add(traj.trajectory_id)
    return subagent_trajectories

  def add_step(
      self: "Trajectory[Step]",
      source: Source,
      message: str,
      *,
      timestamp: datetime.datetime | None = None,
      reasoning_content: str | None = None,
      tool_calls: list[ToolCall] | None = None,
      observation: Observation | None = None,
      metrics: Metrics | None = None,
      model_name: str | None = None,
      reasoning_effort: str | float | None = None,
      is_copied_context: bool | None = None,
      llm_call_count: int | None = None,
      extra: dict[str, Any] | None = None,
  ) -> Step:
    """Helper to create and append a step, automatically assigning step_id.

    Only builds a plain `Step`, so `self` is bound to `Trajectory[Step]`: a
    subclass that narrows `StepT` must override this method to build its own
    step type, and gets a type error at the call site if it does not.

    Args:
      source: Originator of the step (system, user, or agent).
      message: Dialogue message content.
      timestamp: When the step occurred; defaults to now, in UTC.
      reasoning_content: Agent's explicit internal reasoning or thoughts.
      tool_calls: Structured actions or tools invoked by the agent.
      observation: Environment feedback resulting from the step's actions.
      metrics: LLM operational and confidence metrics for this step.
      model_name: The specific LLM model used for this turn.
      reasoning_effort: Qualitative or quantitative measure of effort.
      is_copied_context: True if step was copied from a previous run.
      llm_call_count: Number of LLM inferences this step represents.
      extra: Custom step-level metadata.

    Returns:
      The newly created and appended step.
    """
    step_id = len(self.steps) + self._STEP_ID_START
    new_step = Step(
        step_id=step_id,
        timestamp=timestamp or datetime.datetime.now(datetime.timezone.utc),
        source=source,
        message=message,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        observation=observation,
        metrics=metrics,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        is_copied_context=is_copied_context,
        llm_call_count=llm_call_count,
        extra=extra,
    )
    self.steps.append(new_step)
    return new_step

  def get_metadata(self) -> TrajectoryMetadata:
    """Returns trajectory metadata (excluding steps and sub-trajectories)."""
    data = self.model_dump(exclude={"steps", "subagent_trajectories"})
    return TrajectoryMetadata(**data)

  def to_json_dict(self) -> dict[str, Any]:
    """Serializes the model to a dictionary suitable for JSON, excluding Nones."""
    return self.model_dump(exclude_none=True, mode="json")

  @classmethod
  def from_json_dict(cls, data: dict[str, Any]) -> Self:
    """Deserializes a dictionary into a Trajectory object."""
    return cls.model_validate(data)


TrajectoryT = TypeVar("TrajectoryT", bound=Trajectory[Any])


@dataclasses.dataclass
class TrajectoryError:
  """Structured error payload returned over streams or futures when generation fails."""

  trajectory_id: str
  prompt_id: str
  error_message: str
  error_type: str = "RuntimeError"
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


# ==============================================================================
# --- Tunix RL Extensions ---
# ==============================================================================


class TunixAgentStep(Step):
  """A single turn/interaction agent step with Tunix RL extensions."""

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  mc_return: float | None = pydantic.Field(
      default=None,
      description="Monte Carlo return from this step to episode end.",
  )
  assistant_tokens: IntArray = pydantic.Field(
      default=None,
      description="Token IDs generated by the assistant for this step.",
  )
  assistant_masks: IntArray = pydantic.Field(
      default=None,
      description="Masks for assistant tokens.",
  )
  logprobs: FloatArray = pydantic.Field(
      default=None,
      description="Log probabilities for assistant tokens.",
  )
  policy_version: int | None = pydantic.Field(
      default=None,
      description="Policy/weight version used to generate this step.",
  )

  @pydantic.model_validator(mode="after")
  def validate_agent_source(self) -> TunixAgentStep:
    """Validate that source is 'agent' for TunixAgentStep."""
    if self.source != Source.AGENT:
      raise ValueError(
          "TunixAgentStep is only applicable when source is 'agent', but source"
          f" is '{self.source}'"
      )
    return self

  def to_atif_step(self, step_id_offset: int = 1) -> Step:
    """Converts this 0-indexed Tunix step to a 1-indexed base ATIF Step."""
    return super().to_atif_step(step_id_offset=step_id_offset)

  @classmethod
  def from_atif_step(
      cls, step: Step, step_id_offset: int = 1
  ) -> TunixAgentStep:
    """Rehydrates a 1-indexed ATIF Step into a 0-indexed TunixAgentStep."""
    return _unpack_step_from_atif(step, cls, step_id_offset=step_id_offset)


class TunixEnvStep(Step):
  """A single turn/interaction environment step with Tunix RL extensions."""

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  reward: float | None = pydantic.Field(
      default=None,
      description="Immediate reward signal from the environment.",
  )
  done: bool | None = pydantic.Field(
      default=None,
      description="Terminal state flag indicating if the episode ended.",
  )
  env_tokens: IntArray = pydantic.Field(
      default=None,
      description="Token IDs generated by the environment for this step.",
  )
  env_masks: IntArray = pydantic.Field(
      default=None,
      description="Masks for environment tokens.",
  )

  @pydantic.model_validator(mode="after")
  def validate_env_source(self) -> TunixEnvStep:
    """Validate that source is 'system' or 'user' for TunixEnvStep."""
    if self.source not in (Source.SYSTEM, Source.USER):
      raise ValueError(
          "TunixEnvStep is only applicable when source is 'system' or 'user',"
          f" but source is '{self.source}'"
      )
    return self

  def to_atif_step(self, step_id_offset: int = 1) -> Step:
    """Converts this 0-indexed Tunix step to a 1-indexed base ATIF Step."""
    return super().to_atif_step(step_id_offset=step_id_offset)

  @classmethod
  def from_atif_step(cls, step: Step, step_id_offset: int = 1) -> TunixEnvStep:
    """Rehydrates a 1-indexed ATIF Step into a 0-indexed TunixEnvStep."""
    return _unpack_step_from_atif(step, cls, step_id_offset=step_id_offset)


def _upcast_atif_step(step: Step) -> TunixAgentStep | TunixEnvStep:
  """Rehydrates a 1-indexed ATIF Step into the Tunix subclass for its source.

  Args:
    step: A Tunix step (returned as-is) or a 1-indexed base ATIF step.

  Returns:
    The step as a `TunixAgentStep` or `TunixEnvStep`.

  Raises:
    ValueError: If `step.source` has no matching Tunix step subclass.
  """
  if isinstance(step, (TunixAgentStep, TunixEnvStep)):
    return step
  match step.source:
    case Source.AGENT:
      return TunixAgentStep.from_atif_step(step)
    case Source.USER | Source.SYSTEM:
      return TunixEnvStep.from_atif_step(step)
    case _:
      raise ValueError(f"Unsupported step source: {step.source}")


class TunixTrajectoryMetadata(TrajectoryMetadata):
  """Tunix-specific trajectory metadata extending base ATIF TrajectoryMetadata."""

  METADATA_TYPE: ClassVar[str] = "tunix"

  prompt_id: str | None = pydantic.Field(
      default=None,
      description="Identifier for the initial prompt/task.",
  )
  group_index: int = pydantic.Field(
      default=0,
      description="Sample index within group for rollouts.",
  )
  target_policy_versions: list[int] | None = pydantic.Field(
      default=None,
      description="List of policy versions for each step in the trajectory.",
  )
  status: str | None = pydantic.Field(
      default=None,
      description=(
          "Trajectory status as an `agent_types.TrajectoryStatus` name (e.g."
          ' "RUNNING", "SUCCEEDED", "FAILED"), or None if not stated.'
      ),
  )
  total_reward: float | None = pydantic.Field(
      default=None,
      description="Total cumulative reward.",
  )
  hyperparams: MetadataDict = pydantic.Field(
      default=None,
      description="Hyperparameters / generation kwargs.",
  )
  env_time: MetadataDict = pydantic.Field(
      default=None,
      description="Timing information for environment operations.",
  )
  reward_time: MetadataDict = pydantic.Field(
      default=None,
      description="Timing information for reward operations.",
  )

  def create_trajectory(
      self,
      steps: Sequence[Step] | None = None,
      subagent_trajectories: Sequence[Trajectory] | None = None,
  ) -> TunixTrajectory:
    """Creates a TunixTrajectory, rehydrating 1-indexed ATIF inputs.

    Args:
      steps: Tunix steps (kept as-is) or 1-indexed ATIF steps (rehydrated into
        the Tunix subclass for their source), or None.
      subagent_trajectories: Embedded Tunix subagent trajectories (kept as-is)
        or 1-indexed ATIF trajectories (rehydrated recursively), or None.

    Returns:
      The constructed TunixTrajectory.
    """
    if steps is not None:
      steps = [_upcast_atif_step(step) for step in steps]
    if subagent_trajectories is not None:
      subagent_trajectories = [
          TunixTrajectory.from_atif_trajectory(subagent_trajectory)
          for subagent_trajectory in subagent_trajectories
      ]
    return self._create_paired_trajectory(
        TunixTrajectory, steps, subagent_trajectories
    )


class TunixTrajectory(
    TunixTrajectoryMetadata,
    Trajectory[TunixAgentStep | TunixEnvStep],
):
  """Tunix-specific trajectory object containing the interaction history."""

  _STEP_ID_START: ClassVar[int] = 0

  subagent_trajectories: list[TunixTrajectory] | None = pydantic.Field(
      default=None,
      description="Array of embedded subagent trajectories.",
  )

  def add_step(
      self,
      source: Source,
      message: str,
      *,
      timestamp: datetime.datetime | None = None,
      reasoning_content: str | None = None,
      tool_calls: list[ToolCall] | None = None,
      observation: Observation | None = None,
      metrics: Metrics | None = None,
      model_name: str | None = None,
      reasoning_effort: str | float | None = None,
      is_copied_context: bool | None = None,
      llm_call_count: int | None = None,
      reward: float | None = None,
      done: bool | None = None,
      mc_return: float | None = None,
      assistant_tokens: list[int] | np.ndarray | None = None,
      assistant_masks: list[int] | np.ndarray | None = None,
      env_tokens: list[int] | np.ndarray | None = None,
      env_masks: list[int] | np.ndarray | None = None,
      logprobs: list[float] | np.ndarray | None = None,
      policy_version: int | None = None,
      extra: dict[str, Any] | None = None,
  ) -> TunixAgentStep | TunixEnvStep:
    """Helper to create and append a step, automatically assigning step_id."""
    step_id = len(self.steps) + self._STEP_ID_START
    ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
    if source == Source.AGENT:
      new_step = TunixAgentStep(
          step_id=step_id,
          timestamp=ts,
          source=source,
          message=message,
          reasoning_content=reasoning_content,
          tool_calls=tool_calls,
          observation=observation,
          metrics=metrics,
          model_name=model_name,
          reasoning_effort=reasoning_effort,
          is_copied_context=is_copied_context,
          llm_call_count=llm_call_count,
          mc_return=mc_return,
          assistant_tokens=assistant_tokens,
          assistant_masks=assistant_masks,
          logprobs=logprobs,
          policy_version=policy_version,
          extra=extra,
      )
    else:
      new_step = TunixEnvStep(
          step_id=step_id,
          timestamp=ts,
          source=source,
          message=message,
          observation=observation,
          reward=reward,
          done=done,
          env_tokens=env_tokens,
          env_masks=env_masks,
          extra=extra,
      )
    self.steps.append(new_step)
    return new_step

  def get_metadata(self) -> TunixTrajectoryMetadata:
    """Returns trajectory metadata (excluding steps and sub-trajectories)."""
    data = self.model_dump(exclude={"steps", "subagent_trajectories"})
    return TunixTrajectoryMetadata(**data)

  @classmethod
  def from_atif_trajectory(cls, atif_trajectory: Trajectory) -> TunixTrajectory:
    """Rehydrates a 1-indexed ATIF Trajectory to a 0-indexed TunixTrajectory."""
    if isinstance(atif_trajectory, cls):
      return atif_trajectory

    return TunixTrajectoryMetadata.from_atif_metadata(
        atif_trajectory
    ).create_trajectory(
        steps=atif_trajectory.steps,
        subagent_trajectories=atif_trajectory.subagent_trajectories,
    )
