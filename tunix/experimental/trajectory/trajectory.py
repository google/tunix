"""Trajectory implementation using Agent Trajectory Interchange Format (ATIF).

For more details on the ATIF specification, see:
https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import copy
import dataclasses
import datetime
import enum
import functools
import math
from typing import Annotated, Any, ClassVar, Final, Generic, Literal, Self, TypeVar, TypedDict, get_args
import weakref

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


def _deep_copy_into_memo(value: Any, memo: dict[int, Any]) -> Any:
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
    if _JSON_SCALAR_TYPES.issuperset(map(type, value)):
      scalar_list_copy = value.copy()
      memo[id(value)] = scalar_list_copy
      return scalar_list_copy
    list_copy: list[Any] = []
    memo[id(value)] = list_copy
    for element in value:
      if type(element) not in _JSON_SCALAR_TYPES:
        element = _deep_copy_into_memo(element, memo)
      list_copy.append(element)
    return list_copy
  if value_type is dict:
    dict_copy: dict[Any, Any] = {}
    memo[id(value)] = dict_copy
    for key, element in value.items():
      # Copies the value before the key, in the order `copy.deepcopy` does.
      if type(element) not in _JSON_SCALAR_TYPES:
        element = _deep_copy_into_memo(element, memo)
      if type(key) not in _JSON_SCALAR_TYPES:
        key = _deep_copy_into_memo(key, memo)
      dict_copy[key] = element
    return dict_copy
  if isinstance(value, pydantic.BaseModel):
    # Copied here first, so that `__deepcopy__` finds this copy in `memo`.
    _deep_copy_into_memo(value.__dict__, memo)
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
  return _deep_copy_into_memo(model, memo={})


_EXTRA_FIELD: Final[str] = "extra"


@functools.lru_cache(maxsize=None)
def _get_field_names(model_cls: type[pydantic.BaseModel]) -> set[str]:
  """Returns the cached set of field names for `model_cls`."""
  return set(model_cls.model_fields)


@functools.lru_cache(maxsize=None)
def _get_field_serializers(
    model_cls: type[pydantic.BaseModel],
) -> Mapping[str, Callable[..., Any]]:
  """Returns cached custom PlainSerializer functions keyed by field name."""
  serializers: dict[str, Callable[..., Any]] = {}
  for name, field_info in model_cls.model_fields.items():
    for meta in field_info.metadata:
      if isinstance(meta, pydantic.PlainSerializer):
        serializers[name] = meta.func
  return serializers


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
    subclass_model: Step | TrajectoryMetadata,
    base_atif_cls: type[pydantic.BaseModel],
    exclude_field_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
  """Packs subclass-specific fields into `extra[subclass_model.EXTENSIONS_KEY]`.

  Args:
    subclass_model: The `Step` or `TrajectoryMetadata` instance being projected
      down to its base ATIF model.
    base_atif_cls: The target base ATIF model class (`Step` or
      `TrajectoryMetadata`).
    exclude_field_names: Subclass fields to omit rather than pack into `extra`
      (e.g. `steps` and `subagent_trajectories` when projecting a `Trajectory`
      to `TrajectoryMetadata`).

  Returns:
    A field-value dict ready for `base_atif_cls.model_validate(...)`, or `None`
    if `subclass_model` defines no fields beyond `base_atif_cls`.
  """
  source_field_names = _get_field_names(type(subclass_model))
  base_field_names = _get_field_names(base_atif_cls)
  if source_field_names == base_field_names:
    return None

  subclass_field_names = (
      source_field_names - base_field_names - exclude_field_names
  )
  base_values_by_field = _get_non_none_fields(
      subclass_model, base_field_names - {_EXTRA_FIELD}
  )

  # Convert subclass extension fields directly without invoking `model_dump`,
  # which would re-traverse serialized array lists to validate `return_type`.
  field_serializers = _get_field_serializers(type(subclass_model))
  subclass_fields = {
      field: getattr(subclass_model, field) for field in subclass_field_names
  }
  subclass_values_by_field = {
      field: (
          field_serializers[field](value)
          if field in field_serializers
          else _to_json_compatible(value)
      )
      for field, value in subclass_fields.items()
  }

  extra = dict(subclass_model.extra or {})
  if subclass_values_by_field:
    extensions_key = subclass_model.EXTENSIONS_KEY
    existing_extensions = extra.get(extensions_key) or {}
    extra[extensions_key] = existing_extensions | subclass_values_by_field
  if extra:
    base_values_by_field[_EXTRA_FIELD] = extra
  return base_values_by_field


def _unpack_subclass_values_from_extra(
    atif_model: Step | TrajectoryMetadata,
    subclass_cls: type[Step | TrajectoryMetadata],
) -> dict[str, Any]:
  """Extracts `atif_model.extra[subclass_cls.EXTENSIONS_KEY]` into top-level fields.

  Args:
    atif_model: The base ATIF `Step` or `TrajectoryMetadata` instance holding
      packed extension fields in `extra`.
    subclass_cls: The concrete `Step` or `TrajectoryMetadata` subclass to
      rehydrate into.

  Returns:
    A field-value dict ready for `subclass_cls.model_validate(...)`.
  """
  extensions_key = getattr(subclass_cls, "EXTENSIONS_KEY", None)
  extra = dict(atif_model.extra or {})
  packed_extensions = (
      dict(extra.pop(extensions_key, None) or {}) if extensions_key else {}
  )

  subclass_all_fields = _get_field_names(subclass_cls)
  base_field_names = _get_field_names(type(atif_model))
  extension_field_names = subclass_all_fields - base_field_names

  values_by_field = _get_non_none_fields(
      atif_model, (base_field_names & subclass_all_fields) - {_EXTRA_FIELD}
  )

  # Promote only fields declared on `subclass_cls` from `extra[extensions_key]`,
  # leaving any unrecognized keys in place so `extra="forbid"` is not tripped.
  for field in packed_extensions.keys() & extension_field_names:
    values_by_field[field] = packed_extensions.pop(field)
  if packed_extensions and extensions_key:
    extra[extensions_key] = packed_extensions
  if extra:
    values_by_field[_EXTRA_FIELD] = extra
  return values_by_field


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

_INT3D_DTYPE: Final[np.dtype[np.int16]] = np.dtype("<i2")
_INT3D_DTYPE_NAME: Final[Literal["int16"]] = "int16"
_INT3D_BLOB_KEYS: Final[frozenset[str]] = frozenset({"shape", "dtype", "data"})


class SerializedInt3DArray(TypedDict):
  """Compact JSON-serializable base64 binary representation of a 3D int16 array."""

  shape: list[int]
  dtype: Literal["int16"]
  data: str


def _validate_int_3d_array(
    value: np.ndarray | list[list[list[int]]] | Mapping[str, Any] | None,
) -> np.ndarray | None:
  """Validates and normalizes a 3D int16 array or decodes its base64 blob."""
  if value is None:
    return None
  if isinstance(value, Mapping):
    if set(value.keys()) != _INT3D_BLOB_KEYS:
      raise ValueError(
          f"Invalid serialized Int3DArray keys {sorted(value.keys())};"
          f" expected {sorted(_INT3D_BLOB_KEYS)}."
      )
    dtype = value["dtype"]
    if dtype != _INT3D_DTYPE_NAME:
      raise ValueError(
          f"Unsupported Int3DArray dtype {dtype!r}; expected"
          f" {_INT3D_DTYPE_NAME!r}."
      )
    raw_shape = value["shape"]
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 3
        or not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
            for dim in raw_shape
        )
    ):
      raise ValueError(
          f"Invalid Int3DArray shape {raw_shape!r}; expected 3 non-negative"
          " integers."
      )
    shape = (raw_shape[0], raw_shape[1], raw_shape[2])
    encoded = value["data"]
    if not isinstance(encoded, str):
      raise ValueError(
          "Expected base64 string for Int3DArray data, got"
          f" {type(encoded).__name__}."
      )
    try:
      raw_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
      raise ValueError(f"Failed to decode Int3DArray blob: {exc}") from exc
    expected_bytes = math.prod(shape) * _INT3D_DTYPE.itemsize
    if len(raw_bytes) != expected_bytes:
      raise ValueError(
          f"Decoded Int3DArray byte length {len(raw_bytes)} does not match"
          f" shape {shape} * {_INT3D_DTYPE.itemsize} ({expected_bytes} bytes)."
      )
    return (
        np.frombuffer(raw_bytes, dtype=_INT3D_DTYPE)
        .reshape(shape)
        .astype(np.int16, copy=True)
    )
  if isinstance(value, (np.ndarray, list)):
    arr = np.asarray(value)
    if arr.ndim != 3:
      raise ValueError(
          f"Expected a 3D array for Int3DArray, got shape {arr.shape}."
      )
    if arr.size == 0:
      return np.zeros(arr.shape, dtype=np.int16)
    if not np.issubdtype(arr.dtype, np.integer) or np.issubdtype(
        arr.dtype, np.bool_
    ):
      raise ValueError(
          f"Expected an integer array for Int3DArray, got dtype {arr.dtype}."
      )
    if arr.dtype != np.int16:
      info = np.iinfo(np.int16)
      min_val, max_val = int(arr.min()), int(arr.max())
      if min_val < info.min or max_val > info.max:
        raise ValueError(
            f"Int3DArray values [{min_val}, {max_val}] exceed int16 range"
            f" [{info.min}, {info.max}]."
        )
    return np.ascontiguousarray(arr, dtype=np.int16)
  raise ValueError(f"Unsupported type for Int3DArray: {type(value).__name__}.")


def _serialize_int_3d_array(
    value: np.ndarray | list[list[list[int]]] | Mapping[str, Any] | None,
) -> SerializedInt3DArray | None:
  """Serializes a 3D int16 array into a base64 binary blob dict."""
  if value is None:
    return None
  if (
      isinstance(value, np.ndarray)
      and value.ndim == 3
      and value.dtype == _INT3D_DTYPE
      and value.flags.c_contiguous
  ):
    arr = value
  else:
    validated = _validate_int_3d_array(value)
    assert validated is not None
    arr = np.ascontiguousarray(validated, dtype=_INT3D_DTYPE)
  return {
      "shape": list(arr.shape),
      "dtype": _INT3D_DTYPE_NAME,
      "data": base64.b64encode(arr.tobytes()).decode("ascii"),
  }


Int3DArray = Annotated[
    np.ndarray | list[list[list[int]]] | None,
    pydantic.BeforeValidator(_validate_int_3d_array),
    pydantic.PlainSerializer(
        _serialize_int_3d_array,
        return_type=SerializedInt3DArray | None,
        when_used="always",
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


def _validate_subclass_extensions_key(cls: type[pydantic.BaseModel]) -> None:
  """Raises TypeError if `cls` does not define a non-empty EXTENSIONS_KEY."""
  extensions_key = getattr(cls, "EXTENSIONS_KEY", None)
  if not isinstance(extensions_key, str) or not extensions_key:
    raise TypeError(
        f"{cls.__name__} must define a non-empty string 'EXTENSIONS_KEY'"
        " ClassVar."
    )


class Step(pydantic.BaseModel):
  """A single turn/interaction step."""

  model_config = pydantic.ConfigDict(extra="forbid")

  # Key in `extra` under which subclass-specific fields are packed when
  # projecting to base ATIF. Subclasses must define this ClassVar.
  EXTENSIONS_KEY: ClassVar[str]

  @classmethod
  def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    """Validates that Step subclasses define a non-empty EXTENSIONS_KEY."""
    super().__pydantic_init_subclass__(**kwargs)
    _validate_subclass_extensions_key(cls)

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

  # TODO(tunix-dev): Add automatic Step subclass registration and resolution
  # (mirroring TrajectoryMetadata._SUBCLASS_FIELDS and resolve_subclass) so
  # persistent stores rehydrate custom Step subclasses automatically.
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

  model_config = pydantic.ConfigDict(extra="forbid")

  # Key in `extra` under which subclass-specific fields are packed when
  # projecting to base ATIF. Subclasses must define this ClassVar.
  EXTENSIONS_KEY: ClassVar[str]

  # Maps each registered TrajectoryMetadata subclass to the set of extension
  # field names it defines beyond the base ATIF TrajectoryMetadata schema.
  # WeakKeyDictionary avoids keeping dynamically defined/test classes alive.
  _SUBCLASS_FIELDS: ClassVar[
      MutableMapping[type[TrajectoryMetadata], frozenset[str]]
  ] = weakref.WeakKeyDictionary()

  @classmethod
  def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    """Validates `EXTENSIONS_KEY` and registers extension fields for subclasses.

    Unlike Python's standard `__init_subclass__` (which runs before Pydantic's
    metaclass builds `cls.model_fields`), `__pydantic_init_subclass__` is
    invoked by Pydantic after `cls.model_fields` is populated. We verify that
    `cls` defines a non-empty `EXTENSIONS_KEY` and record the set of fields that
    `cls` adds on top of base `TrajectoryMetadata` so `resolve_subclass` can
    match packed `extra[cls.EXTENSIONS_KEY]` keys back to the subclass that
    defined them.
    """
    super().__pydantic_init_subclass__(**kwargs)
    # `Trajectory` (defined later in this file) and its subclasses inherit from
    # `TrajectoryMetadata` to share root-level ATIF fields, but represent full
    # trajectories rather than standalone metadata classes.
    trajectory_base = globals().get("Trajectory")
    if cls.__name__ == "Trajectory" or (
        trajectory_base is not None and issubclass(cls, trajectory_base)
    ):
      return
    _validate_subclass_extensions_key(cls)
    extension_fields = frozenset(
        cls.model_fields.keys() - TrajectoryMetadata.model_fields.keys()
    )
    if extension_fields:
      cls._SUBCLASS_FIELDS[cls] = extension_fields

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
    exclude_field_names = (
        {"steps", "subagent_trajectories"}
        if isinstance(self, Trajectory)
        else frozenset()
    )
    target_values_by_field = _pack_subclass_values_into_extra(
        self,
        TrajectoryMetadata,
        exclude_field_names=exclude_field_names,
    )
    if target_values_by_field is None:
      return self
    return TrajectoryMetadata.model_validate(target_values_by_field)

  def get_extensions(self) -> dict[str, Any]:
    """Returns the packed subclass extensions dictionary from `extra`."""
    if not self.extra:
      return {}
    extensions_key = getattr(
        self.resolve_subclass(self), "EXTENSIONS_KEY", None
    )
    if not extensions_key:
      return {}
    packed_extensions = self.extra.get(extensions_key)
    return packed_extensions if isinstance(packed_extensions, dict) else {}

  @classmethod
  def resolve_subclass(
      cls, metadata: TrajectoryMetadata
  ) -> type[TrajectoryMetadata]:
    """Resolves the concrete TrajectoryMetadata class for `metadata`.

    When `metadata` is deserialized from storage as a base `TrajectoryMetadata`,
    subclass-specific fields for a registered subclass `subclass` are stored in
    `extra[subclass.EXTENSIONS_KEY]`. This method resolves the registered
    subclass in two tiers:
      1. Exact or subset match (`extension_keys <= subclass_fields`): covers
         normal persistence (`extension_keys == subclass_fields`) and forward
         schema evolution where new optional fields were added to the subclass.
         Picks the narrowest covering subclass, breaking ties in favor of `cls`
         when `resolve_subclass` is called on a specific subclass.
      2. Superset match (`subclass_fields <= extension_keys`): covers backward
         schema evolution where a deprecated field was removed from the subclass
         (or a caller placed custom keys in `extra[subclass.EXTENSIONS_KEY]`).
         Picks the widest subclass whose full field set is present, breaking
         ties in favor of `cls`; remaining unrecognized keys stay preserved in
         `extra[subclass.EXTENSIONS_KEY]`.
    Falls back to base `TrajectoryMetadata` when no extension keys are present
    or no registered subclass matches.

    Args:
      metadata: A `TrajectoryMetadata` instance (either already a concrete
        subclass or a base `TrajectoryMetadata` unpacked from ATIF).

    Returns:
      The resolved `TrajectoryMetadata` class.
    """
    metadata_cls = type(metadata)
    if not isinstance(metadata, Trajectory):
      if metadata_cls is not TrajectoryMetadata:
        return metadata_cls
    else:
      # `Trajectory` subclasses multiply inherit from `Trajectory` and their
      # paired `TrajectoryMetadata` subclass (e.g. `TunixTrajectory` inherits
      # from `Trajectory` and `TunixTrajectoryMetadata`). Walk the Method
      # Resolution Order (`__mro__`) to find the paired metadata subclass.
      for base in metadata_cls.__mro__:
        if base in cls._SUBCLASS_FIELDS:
          return base
    if not metadata.extra:
      return TrajectoryMetadata
    subset_matches: list[type[TrajectoryMetadata]] = []
    superset_matches: list[type[TrajectoryMetadata]] = []
    for subclass, subclass_fields in cls._SUBCLASS_FIELDS.items():
      packed_extensions = metadata.extra.get(subclass.EXTENSIONS_KEY)
      if not isinstance(packed_extensions, Mapping) or not packed_extensions:
        continue
      extension_keys = packed_extensions.keys()
      if extension_keys <= subclass_fields:
        subset_matches.append(subclass)
      elif subclass_fields <= extension_keys:
        superset_matches.append(subclass)
    # In Python, `False < True` (0 < 1). To break equal-length ties in favor of
    # `cls`, `min()` uses `candidate is not cls` (False=0 for `cls`) and `max()`
    # uses `candidate is cls` (True=1 for `cls`).
    if subset_matches:
      return min(
          subset_matches,
          key=lambda candidate: (
              len(cls._SUBCLASS_FIELDS[candidate]),
              candidate is not cls,
          ),
      )
    if superset_matches:
      return max(
          superset_matches,
          key=lambda candidate: (
              len(cls._SUBCLASS_FIELDS[candidate]),
              candidate is cls,
          ),
      )
    return TrajectoryMetadata

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
    data = {field: getattr(self, field) for field in metadata_fields}
    if steps is not None:
      data["steps"] = list(steps)
    if subagent_trajectories is not None:
      data["subagent_trajectories"] = list(subagent_trajectories)
    return trajectory_cls(**data)

  # TODO(tunix-dev): Auto-pair or synthesize a Trajectory subclass when a
  # TrajectoryMetadata subclass does not require custom step upcasting, removing
  # the need to manually define a Trajectory subclass and override
  # create_trajectory.
  def create_trajectory(
      self,
      steps: Sequence[Any] | None = None,
      subagent_trajectories: Sequence[Any] | None = None,
  ) -> Trajectory[Any]:
    """Creates a full Trajectory from this metadata, steps, and subagents.

    Subclasses must override this method to construct their corresponding
    `Trajectory` subclass via `_create_paired_trajectory`. Persistent stores
    (`FileTrajectoryStore`, `SqlTrajectoryStore`) pass deserialized base ATIF
    `Step` instances into `steps`; subclasses that also define custom `Step`
    subclasses should rehydrate them here (see
    `TunixTrajectoryMetadata.create_trajectory`).

    Args:
      steps: Sequence of step instances, or None.
      subagent_trajectories: Sequence of subagent trajectory instances, or None.

    Returns:
      The constructed trajectory instance.
    """
    return self._create_paired_trajectory(
        Trajectory, steps, subagent_trajectories
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

TUNIX_EXTENSIONS_KEY: Final[str] = "_tunix_extensions"


class TunixAgentStep(Step):
  """A single turn/interaction agent step with Tunix RL extensions."""

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  EXTENSIONS_KEY: ClassVar[str] = TUNIX_EXTENSIONS_KEY

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
  prefill_routed_experts: Int3DArray = pydantic.Field(
      default=None,
      description=(
          "MoE routed expert IDs returned by this step's model call, shape"
          " (T, L, K), covering sequence positions [prefill_start,"
          " prefill_start + T). The last sampled token of a turn (plus any"
          " chat-parser suffix tokens appended to assistant_tokens) is only"
          " routed by the next turn's prefill, so rows are: turn 0 = prompt +"
          " this turn's routed assistant prefix; turn t > 0 = previous turn's"
          " unrouted assistant tail + previous env tokens + this turn's routed"
          " assistant prefix. Written once and never rewritten."
      ),
  )
  prefill_start: int | None = pydantic.Field(
      default=None,
      description=(
          "Position of the first `prefill_routed_experts` row in the unpadded"
          " prompt + conversation token sequence."
      ),
  )
  prefill_num_context: int | None = pydantic.Field(
      default=None,
      description=(
          "Number of leading `prefill_routed_experts` rows that cover tokens"
          " before this step's generation (turn 0: the prompt; turn t > 0:"
          " the previous assistant tail + previous env tokens)."
      ),
  )

  @pydantic.model_validator(mode="after")
  def validate_agent_source(self) -> TunixAgentStep:
    """Validate source and prefill routing invariants for TunixAgentStep."""
    if self.source != Source.AGENT:
      raise ValueError(
          "TunixAgentStep is only applicable when source is 'agent', but source"
          f" is '{self.source}'"
      )
    prefill_fields = (
        self.prefill_routed_experts,
        self.prefill_start,
        self.prefill_num_context,
    )
    if any(x is not None for x in prefill_fields) and not all(
        x is not None for x in prefill_fields
    ):
      raise ValueError(
          "prefill_routed_experts, prefill_start, and prefill_num_context must"
          " be either all None or all set"
      )
    if self.prefill_routed_experts is not None:
      assert self.prefill_start is not None
      assert self.prefill_num_context is not None
      if self.prefill_start < 0:
        raise ValueError(
            f"prefill_start must be >= 0, got {self.prefill_start}"
        )
      if not (
          0 <= self.prefill_num_context <= len(self.prefill_routed_experts)
      ):
        raise ValueError(
            "prefill_num_context must be in"
            f" [0, {len(self.prefill_routed_experts)}], got"
            f" {self.prefill_num_context}"
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

  EXTENSIONS_KEY: ClassVar[str] = TUNIX_EXTENSIONS_KEY

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

  EXTENSIONS_KEY: ClassVar[str] = TUNIX_EXTENSIONS_KEY

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
  masked_out: bool | None = pydantic.Field(
      default=None,
      description=(
          "True if the overlong filter zeroed this trajectory's training"
          " masks; None until the episode is post-processed."
      ),
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
      prefill_routed_experts: list[list[list[int]]] | np.ndarray | None = None,
      prefill_start: int | None = None,
      prefill_num_context: int | None = None,
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
          prefill_routed_experts=prefill_routed_experts,
          prefill_start=prefill_start,
          prefill_num_context=prefill_num_context,
          extra=extra,
      )
    else:
      if (
          prefill_routed_experts is not None
          or prefill_start is not None
          or prefill_num_context is not None
      ):
        raise ValueError(
            "prefill_routed_experts, prefill_start, and prefill_num_context"
            f" are only valid when source is 'agent', got '{source}'"
        )
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
