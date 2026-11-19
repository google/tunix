"""Trajectory implementation using Agent Trajectory Interchange Format (ATIF).

For more details on the ATIF specification, see:
https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence, Set
import copy
import dataclasses
import datetime
import enum
import functools
import logging
import math
import sys
import types
from typing import Annotated, Any, ClassVar, Final, Generic, Literal, Self, TypeGuard, TypeVar, TypedDict, Union, cast, get_args, get_origin
import weakref

import numpy as np
import pydantic
from tunix.rl import packing

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


class _JsonCompatibleDict(dict[str, Any]):
  """A dict whose values `_pack_subclass_values_into_extra` already converted.

  Projecting a subclass model to base ATIF stores its serialized extension
  fields under `extra[EXTENSIONS_KEY]` in this type when every value is
  JSON-compatible and every `list` stored directly in it is a flat list of JSON
  scalars (token IDs, masks, logprobs, policy versions). `_to_json_compatible`
  and `_deep_copy_into_memo` therefore copy those lists with one C-level
  `list.copy()` instead of scanning every element again on each `model_dump()`
  or `deep_copy()`.

  Only `_pack_subclass_values_into_extra` creates instances; if any packed list
  holds nested containers, it stores a plain `dict` instead. A plain `dict` with
  the same keys that a caller builds by hand also takes the regular conversion
  path, so unconverted arrays or tuples in it are still serialized correctly.
  """

  __slots__ = ()


def _to_json_compatible(value: Any) -> Any:
  """Recursively converts NumPy arrays and tuples nested in `value` to lists.

  Every `list` and `dict` in the result is a new object, so the output of
  `model_dump()` never shares a container with the model's `extra`.

  Args:
    value: The value to convert.

  Returns:
    The JSON-compatible equivalent of `value`.
  """
  value_type = type(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if value_type is _JsonCompatibleDict:
    # Already converted when packed; copy flat scalar lists in one C call.
    out: dict[str, Any] = {}
    for key, element in value.items():
      element_type = type(element)
      if element_type is list:
        out[key] = element.copy()
      elif element_type in _JSON_SCALAR_TYPES:
        out[key] = element
      else:
        out[key] = _to_json_compatible(element)
    return out
  if isinstance(value, dict):
    return {k: _to_json_compatible(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    # Caller-supplied token or ID lists in `extra` are common; vet scalar-only
    # sequences with one C-level scan and copy them without a Python call per
    # element.
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


def _copy_scalar_list(value: list[Any], memo: dict[int, Any]) -> list[Any]:
  """Copies a list known to hold only JSON scalars, recording it in `memo`."""
  memoized_copy = memo.get(id(value), _MISSING)
  if memoized_copy is not _MISSING:
    return memoized_copy
  list_copy = value.copy()
  memo[id(value)] = list_copy
  return list_copy


def _is_json_scalar_annotation(annotation: Any) -> bool:
  """Returns whether `annotation` only admits `_JSON_SCALAR_TYPES` values."""
  if get_origin(annotation) is Annotated:
    annotation = get_args(annotation)[0]
  if annotation in _JSON_SCALAR_TYPES:
    return True
  if get_origin(annotation) in (types.UnionType, Union):
    return all(
        _is_json_scalar_annotation(member) for member in get_args(annotation)
    )
  return False


def _is_scalar_list_annotation(annotation: Any) -> bool:
  """Returns whether `annotation` only admits `list` values of JSON scalars.

  Matches `list[T]` (and unions of `list[T]` with non-list types such as `None`
  or `np.ndarray`, like `IntArray` and `FloatArray`) where `T` only admits
  `_JSON_SCALAR_TYPES` elements. Nested lists such as `Int3DArray`
  (`list[list[list[int]]]`) and model or dict lists return `False`.

  Args:
    annotation: A resolved field annotation from `model_fields`.

  Returns:
    `True` if any `list` value validated by the field is a flat list of JSON
    scalars.
  """
  if get_origin(annotation) is Annotated:
    annotation = get_args(annotation)[0]
  origin = get_origin(annotation)
  members = (
      get_args(annotation)
      if origin in (types.UnionType, Union)
      else (annotation,)
  )
  has_scalar_list = False
  for member in members:
    if get_origin(member) is Annotated:
      member = get_args(member)[0]
    if get_origin(member) is list:
      args = get_args(member)
      if len(args) != 1 or not _is_json_scalar_annotation(args[0]):
        return False
      has_scalar_list = True
    elif member is not np.ndarray and not _is_json_scalar_annotation(member):
      return False
  return has_scalar_list


def _scalar_list_fields(
    model_cls: type[pydantic.BaseModel],
) -> frozenset[str]:
  """Returns the names of fields on `model_cls` annotated as flat scalar lists."""
  return frozenset(
      name
      for name, field in model_cls.model_fields.items()
      if _is_scalar_list_annotation(field.annotation)
  )


def _deep_copy_model_dict(
    model: pydantic.BaseModel, memo: dict[int, Any]
) -> dict[str, Any]:
  """Deep-copies `model.__dict__` into `memo`.

  Fields that the model class lists in `SCALAR_LIST_FIELDS` hold flat lists of
  JSON scalars once validated (token IDs, masks, logprobs, policy versions), so
  a `list` in such a field is copied with one C-level `list.copy()` instead of
  a per-element type scan. Every other value takes `_deep_copy_into_memo`.

  Args:
    model: The model whose `__dict__` to copy.
    memo: The `copy.deepcopy` memo, mapping the `id()` of each copied original
      to its copy.

  Returns:
    A deep copy of `model.__dict__`.
  """
  model_dict = model.__dict__
  memoized_copy = memo.get(id(model_dict), _MISSING)
  if memoized_copy is not _MISSING:
    return memoized_copy
  scalar_list_fields = (
      type(model).SCALAR_LIST_FIELDS
      if isinstance(model, (Metrics, Step, TrajectoryMetadata))
      else frozenset()
  )
  dict_copy: dict[str, Any] = {}
  memo[id(model_dict)] = dict_copy
  for key, element in model_dict.items():
    element_type = type(element)
    if element_type is list and key in scalar_list_fields:
      element = _copy_scalar_list(element, memo)
    elif element_type not in _JSON_SCALAR_TYPES:
      element = _deep_copy_into_memo(element, memo)
    dict_copy[key] = element
  return dict_copy


def _deep_copy_into_memo(value: Any, memo: dict[int, Any]) -> Any:
  """Returns a deep copy of `value`, sharing `memo` with `copy.deepcopy`.

  `copy.deepcopy` makes a Python call per element of a list or dict, which
  dominates the cost of copying long token ID and logprob lists and nested
  JSON-like dicts such as tool definitions. This function copies exact `list`s,
  `dict`s, `_JsonCompatibleDict`s, and NumPy arrays without object elements
  itself: it reuses elements of `_JSON_SCALAR_TYPES`, which are immutable,
  without a call, copies a list holding only such elements with one C-level
  `list.copy()`, and copies an array with one `ndarray.copy()`. Like
  `copy.deepcopy`, it records each copy in `memo` before copying the values
  nested in it, so shared and cyclic references are copied once. Any other
  value goes to `copy.deepcopy` with the same `memo`. For a pydantic model, it
  copies `__dict__` via `_deep_copy_model_dict` and then calls the model's
  `__deepcopy__`, which finds that copy in `memo`.

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
  if value_type is np.ndarray and not value.dtype.hasobject:
    # Same buffer copy `ndarray.__deepcopy__` makes for these dtypes, without
    # the `copy.deepcopy` dispatch. Arrays holding Python objects, including
    # structured dtypes with an object field, must deep-copy their elements.
    array_copy = value.copy()
    memo[id(value)] = array_copy
    return array_copy
  if value_type is _JsonCompatibleDict:
    packed_copy = _JsonCompatibleDict()
    memo[id(value)] = packed_copy
    for key, element in value.items():
      element_type = type(element)
      if element_type is list:
        # Flat scalar lists by construction; see `_JsonCompatibleDict`.
        element = _copy_scalar_list(element, memo)
      elif element_type not in _JSON_SCALAR_TYPES:
        element = _deep_copy_into_memo(element, memo)
      packed_copy[key] = element
    return packed_copy
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
    _deep_copy_model_dict(value, memo)
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

# Subclass fields promoted onto TrajectoryMetadata whose legacy packed entries
# inside `extra[EXTENSIONS_KEY]` remain valid for backward compatibility when
# the top-level canonical field is unset.
_PROMOTED_METADATA_EXTENSION_FIELDS: Final[frozenset[str]] = frozenset(
    {"status", "total_reward"}
)


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


def _is_flat_scalar_source(
    raw: Any, field: str, scalar_list_fields: frozenset[str]
) -> bool:
  """Returns whether packing `raw` yields a flat list of JSON scalars."""
  if isinstance(raw, np.ndarray):
    return raw.ndim == 1 and raw.dtype.kind in "biuf"
  return isinstance(raw, list) and field in scalar_list_fields


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
    # Caller-supplied extensions are converted here so that everything stored
    # in the packed dict is JSON-compatible; `_JsonCompatibleDict` is used only
    # when every top-level list is a flat list of JSON scalars.
    existing_extensions = _to_json_compatible(extra.get(extensions_key) or {})
    packed = existing_extensions | subclass_values_by_field
    scalar_list_fields = type(subclass_model).SCALAR_LIST_FIELDS
    is_flat = all(
        not isinstance(value, list)
        or _is_flat_scalar_source(
            subclass_fields.get(key), key, scalar_list_fields
        )
        or _JSON_SCALAR_TYPES.issuperset(map(type, value))
        for key, value in packed.items()
    )
    extra[extensions_key] = _JsonCompatibleDict(packed) if is_flat else packed
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

  # Promote fields declared on `subclass_cls` (plus legacy promoted metadata
  # fields when unset at the top level), leaving unrecognized keys in `extra`.
  promoted_fallbacks = (
      _PROMOTED_METADATA_EXTENSION_FIELDS - values_by_field.keys()
      if isinstance(atif_model, TrajectoryMetadata)
      else frozenset()
  )
  for field in packed_extensions.keys() & (
      extension_field_names | promoted_fallbacks
  ):
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
_UINT8_DTYPE: Final[np.dtype[np.uint8]] = np.dtype("uint8")
_INT3D_DTYPE_NAME: Final[Literal["int16"]] = "int16"
_UINT8_DTYPE_NAME: Final[Literal["uint8"]] = "uint8"
_INT3D_BLOB_KEYS: Final[frozenset[str]] = frozenset({"shape", "dtype", "data"})


class SerializedInt3DArray(TypedDict):
  """JSON-serializable base64 binary representation of MoE routed experts.

  Supported layouts:
  - 3D `int16` `(T, L, K)`: raw expert IDs (`0 <= id < E`), `-1` for unset. Raw
    unshifted expert IDs must be passed as `int16`.
  - 3D `uint8` `(T, L, K)`: `+1`-shifted expert IDs (`id + 1`), `0` for unset
    (any `uint8` array is interpreted as already `+1`-shifted).
  - 4D `uint8` `(T, L, K // 4, 5)` (or `(T, L, 1, ceil(10 * K / 8))` when
    `K % 4 != 0`): `uint10` packing `+1`-shifted 10-bit values (`id + 1`, `0`
    for unset) into little-endian bytes.

  Note: Offline readers or older binaries consuming a `TrajectoryStore` written
  with `dtype="uint8"` blobs must include this schema version (otherwise older
  readers fail with `Unsupported Int3DArray dtype 'uint8'`).
  """

  shape: list[int]
  dtype: Literal["int16", "uint8"]
  data: str


def _is_valid_uint8_routed_shape(shape: Sequence[int]) -> bool:
  """Returns True if `shape` is a valid 3D uint8 or 4D uint10-packed shape."""
  return len(shape) == 3 or (
      len(shape) == 4
      and packing.uint10_logical_k(shape[2], shape[3]) is not None
  )


def _decode_int_3d_array_blob(value: Mapping[str, Any]) -> np.ndarray:
  """Decodes and validates a base64-serialized routed-experts blob dict."""
  if set(value.keys()) != _INT3D_BLOB_KEYS:
    raise ValueError(
        f"Invalid serialized Int3DArray keys {sorted(value.keys())};"
        f" expected {sorted(_INT3D_BLOB_KEYS)}."
    )
  dtype = value["dtype"]
  if dtype == _INT3D_DTYPE_NAME:
    np_dtype, out_dtype = _INT3D_DTYPE, np.int16
  elif dtype == _UINT8_DTYPE_NAME:
    np_dtype, out_dtype = _UINT8_DTYPE, np.uint8
  else:
    raise ValueError(
        f"Unsupported Int3DArray dtype {dtype!r}; expected"
        f" {_INT3D_DTYPE_NAME!r} or {_UINT8_DTYPE_NAME!r}."
    )
  raw_shape = value["shape"]
  valid_dims = isinstance(raw_shape, (list, tuple)) and all(
      isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
      for dim in raw_shape
  )
  valid_rank = valid_dims and (
      (dtype == _INT3D_DTYPE_NAME and len(raw_shape) == 3)
      or (
          dtype == _UINT8_DTYPE_NAME and _is_valid_uint8_routed_shape(raw_shape)
      )
  )
  if not valid_rank:
    raise ValueError(
        f"Invalid Int3DArray shape {raw_shape!r} for dtype {dtype!r}; expected"
        " 3 non-negative integers (or 4D uint10 shape for uint8)."
    )
  shape = tuple(int(dim) for dim in raw_shape)
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
  expected_bytes = math.prod(shape) * np_dtype.itemsize
  if len(raw_bytes) != expected_bytes:
    raise ValueError(
        f"Decoded Int3DArray byte length {len(raw_bytes)} does not match"
        f" shape {shape} * {np_dtype.itemsize} ({expected_bytes} bytes)."
    )
  return (
      np.frombuffer(raw_bytes, dtype=np_dtype)
      .reshape(shape)
      .astype(out_dtype, copy=True)
  )


def _is_valid_contiguous_int_3d_array(value: Any) -> TypeGuard[np.ndarray]:
  """Returns True if `value` is already a C-contiguous valid Int3DArray ndarray."""
  return (
      isinstance(value, np.ndarray)
      and value.flags.c_contiguous
      and (
          (value.dtype == _INT3D_DTYPE and value.ndim == 3)
          or (
              value.dtype == _UINT8_DTYPE
              and _is_valid_uint8_routed_shape(value.shape)
          )
      )
  )


def _validate_int_3d_array(
    value: np.ndarray | list[list[list[int]]] | Mapping[str, Any] | None,
) -> np.ndarray | None:
  """Validates a 3D int16/uint8 or 4D uint10-packed uint8 array or blob."""
  if value is None:
    return None
  if _is_valid_contiguous_int_3d_array(value):
    return value
  if isinstance(value, Mapping):
    return _decode_int_3d_array_blob(value)
  if not isinstance(value, (np.ndarray, list)):
    raise ValueError(
        f"Unsupported type for Int3DArray: {type(value).__name__}."
    )
  arr = np.asarray(value)
  if isinstance(value, np.ndarray) and arr.dtype == np.uint8:
    if _is_valid_uint8_routed_shape(arr.shape):
      return np.ascontiguousarray(arr, dtype=np.uint8)
    raise ValueError(
        "Expected a 3D uint8 array or 4D uint10-packed uint8 array for"
        f" Int3DArray, got shape {arr.shape}."
    )
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


def _serialize_int_3d_array(
    value: np.ndarray | list[list[list[int]]] | Mapping[str, Any] | None,
) -> SerializedInt3DArray | None:
  """Serializes a 3D int16/uint8 or 4D uint10-packed uint8 array into base64."""
  if value is None:
    return None
  if _is_valid_contiguous_int_3d_array(value):
    contig = value
  else:
    arr = _validate_int_3d_array(value)
    if arr is None:
      return None
    contig = np.ascontiguousarray(
        arr,
        dtype=_UINT8_DTYPE if arr.dtype == np.uint8 else _INT3D_DTYPE,
    )
  dtype_name: Literal["int16", "uint8"] = (
      _UINT8_DTYPE_NAME if contig.dtype == np.uint8 else _INT3D_DTYPE_NAME
  )
  return {
      "shape": list(contig.shape),
      "dtype": dtype_name,
      "data": (
          base64.b64encode(
              memoryview(contig)  # pyrefly: ignore[bad-argument-type]
          ).decode("ascii")
      ),
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

  # Fields whose `list` values hold only JSON scalars once validated, so
  # `deep_copy` copies them with one `list.copy()` instead of a per-element
  # scan. Populated from `model_fields` for `Metrics` and its subclasses.
  SCALAR_LIST_FIELDS: ClassVar[frozenset[str]] = frozenset()

  @classmethod
  def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    super().__pydantic_init_subclass__(**kwargs)
    cls.SCALAR_LIST_FIELDS = _scalar_list_fields(cls)

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


Metrics.SCALAR_LIST_FIELDS = _scalar_list_fields(Metrics)


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


class TrajectoryMetrics(pydantic.BaseModel):
  """Flexible metric capture expanding beyond `FinalMetrics`.

  Captures lightweight trajectory-level summaries, execution timings, RL
  training statistics, and other environment or computation metrics. Heavy
  per-token arrays belong on `Step`.
  """

  model_config = pydantic.ConfigDict(extra="forbid")

  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom metrics",
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

  # Fields whose `list` values hold only JSON scalars once validated, so
  # `deep_copy` copies them with one `list.copy()` instead of a per-element
  # scan. Populated in `__pydantic_init_subclass__` from `model_fields`.
  SCALAR_LIST_FIELDS: ClassVar[frozenset[str]] = frozenset()

  # Heavy array fields that stores may be configured to drop before writing
  # (via `exclude_step_arrays`), e.g. for observability-only runs. Subclasses
  # declare their own fields; base ATIF steps define none.
  EXCLUDABLE_ARRAY_FIELDS: ClassVar[frozenset[str]] = frozenset()

  @classmethod
  def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    """Validates `EXTENSIONS_KEY` and records scalar-list fields."""
    super().__pydantic_init_subclass__(**kwargs)
    _validate_subclass_extensions_key(cls)
    cls.SCALAR_LIST_FIELDS = _scalar_list_fields(cls)

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

  def without_excluded_fields(self, exclude: Set[str]) -> Self:
    """Returns this step with the `exclude` fields set to None.

    Uses a shallow `model_copy`, so this step is never mutated and the dropped
    values are never copied. Names this step's class does not define, or whose
    values are already None, are ignored. Callers are expected to pass names
    validated against `EXCLUDABLE_ARRAY_FIELDS`.

    Args:
      exclude: Names of fields to drop.

    Returns:
      A shallow copy with the fields dropped, or `self` if nothing was dropped.
    """
    if not exclude:
      return self
    fields = type(self).model_fields
    update = {
        field: None
        for field in exclude
        if field in fields and getattr(self, field) is not None
    }
    return self.model_copy(update=update) if update else self

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


def validate_step_array_exclusions(
    fields: Collection[str],
    allowed_fields: Set[str] | None = None,
) -> frozenset[str]:
  """Validates and normalizes a collection of step array fields to exclude.

  The check is schema-agnostic: callers that know the step schema pass its
  excludable fields (e.g. `TrajectoryMetadata.excludable_step_array_fields()`)
  as `allowed_fields`.

  Args:
    fields: Names of step array fields to drop before queueing writes.
    allowed_fields: If given, every name in `fields` must be in this set.

  Returns:
    The validated field names as a frozenset.

  Raises:
    TypeError: If `fields` is not a collection, is a string, bytes, or mapping
      (e.g. `{"logprobs": False}`), or contains non-string entries.
    ValueError: If `allowed_fields` is given and `fields` contains names
      outside it.
  """
  if isinstance(fields, (str, bytes, Mapping)) or not isinstance(
      fields, Collection
  ):
    raise TypeError(
        "exclude_step_arrays must be a collection of field names (e.g. a"
        f" list), not a string, bytes, or mapping; got {fields!r}."
    )
  excluded = frozenset(fields)
  non_str = [field for field in excluded if not isinstance(field, str)]
  if non_str:
    raise TypeError(
        "exclude_step_arrays must contain only field name strings; got"
        f" {non_str!r}."
    )
  if allowed_fields is not None:
    unknown = excluded - allowed_fields
    if unknown:
      raise ValueError(
          f"Unsupported exclude_step_arrays fields {sorted(unknown)}; supported"
          f" fields are {sorted(allowed_fields)}."
      )
  return excluded


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


def _rehydrate_sequence(
    items: Sequence[Any],
    rehydrator: Callable[[Any], Any] | None,
) -> list[Any]:
  """Copies `items` into a list, applying `rehydrator` when provided."""
  if rehydrator is None:
    return list(items)
  return [rehydrator(item) for item in items]


class TrajectoryMetadata(pydantic.BaseModel):
  """Metadata for a trajectory (excluding steps and subagents)."""

  model_config = pydantic.ConfigDict(extra="forbid")

  # Key in `extra` under which subclass-specific fields are packed when
  # projecting to base ATIF. Subclasses must define this ClassVar.
  EXTENSIONS_KEY: ClassVar[str]

  # Fields whose `list` values hold only JSON scalars once validated, so
  # `deep_copy` copies them with one `list.copy()` instead of a per-element
  # scan. Populated in `__pydantic_init_subclass__` from `model_fields`.
  SCALAR_LIST_FIELDS: ClassVar[frozenset[str]] = frozenset()

  # Step classes written to trajectories with this metadata type. Used to
  # validate a store's `exclude_step_arrays` against the bound step schema.
  STEP_CLASSES: ClassVar[tuple[type[Step], ...]] = (Step,)

  # Maps each registered TrajectoryMetadata subclass to the set of extension
  # field names it defines beyond the base ATIF TrajectoryMetadata schema.
  # WeakKeyDictionary avoids keeping dynamically defined/test classes alive.
  _SUBCLASS_FIELDS: ClassVar[
      MutableMapping[type[TrajectoryMetadata], frozenset[str]]
  ] = weakref.WeakKeyDictionary()

  # Maps each registered TrajectoryMetadata subclass to its paired Trajectory
  # subclass. WeakKeyDictionary avoids keeping dynamically defined/test classes alive.
  _PAIRED_TRAJECTORY_CLASSES: ClassVar[
      MutableMapping[type[TrajectoryMetadata], type[Trajectory[Any]]]
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
    if cls.__name__ == "Trajectory":
      return
    if trajectory_base is not None and issubclass(cls, trajectory_base):
      for base in cls.__bases__:
        if (
            isinstance(base, type)
            and issubclass(base, TrajectoryMetadata)
            and base is not TrajectoryMetadata
            and not issubclass(base, trajectory_base)
        ):
          TrajectoryMetadata._PAIRED_TRAJECTORY_CLASSES[base] = cls
      return
    _validate_subclass_extensions_key(cls)
    cls.SCALAR_LIST_FIELDS = _scalar_list_fields(cls)
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
  attempt: int = pydantic.Field(
      default=1,
      ge=1,
      description="1-based retry attempt counter for this trajectory.",
  )
  status: str | None = pydantic.Field(
      default=None,
      description=(
          "Execution outcome or lifecycle status of the trajectory, or None"
          " if not stated."
      ),
  )
  total_reward: float | None = pydantic.Field(
      default=None,
      description="Total cumulative reward, or None if unreported.",
  )
  extra: MetadataDict = pydantic.Field(
      default=None,
      description="Custom root-level metadata.",
  )

  @classmethod
  def excludable_step_array_fields(cls) -> frozenset[str]:
    """Returns the step array fields a store bound to `cls` may exclude."""
    return frozenset().union(
        *(step_cls.EXCLUDABLE_ARRAY_FIELDS for step_cls in cls.STEP_CLASSES)
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
      unpromoted_keys = (
          packed_extensions.keys() - _PROMOTED_METADATA_EXTENSION_FIELDS
      )
      if not unpromoted_keys and all(
          getattr(metadata, key) is not None for key in packed_extensions
      ):
        continue
      if unpromoted_keys <= subclass_fields:
        subset_matches.append(subclass)
      elif subclass_fields <= unpromoted_keys:
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
      *,
      step_rehydrator: Callable[[Any], Any] | None = None,
      subagent_rehydrator: Callable[..., Any] | None = None,
      strict: bool = True,
      is_incomplete: bool = False,
  ) -> TrajectoryT:
    """Constructs `trajectory_cls` from this metadata, steps, and subagents.

    Verifies that `trajectory_cls` is a subclass of `type(self)` before
    instantiation so passing an incompatible trajectory class raises
    `TypeError`.

    Args:
      trajectory_cls: Concrete Trajectory class to build.
      steps: Sequence of step instances, or None.
      subagent_trajectories: Sequence of subagent trajectory instances, or None.
      step_rehydrator: Optional callable applied to each step before validation.
      subagent_rehydrator: Optional callable applied to each subagent trajectory
        before validation.
      strict: Whether to raise on validation or step-sequence errors. When
        False, skips corrupt items, retains the longest contiguous step prefix
        starting from `trajectory_cls.STEP_ID_START`, and resets `status` and
        `total_reward` to `None` when incomplete.
      is_incomplete: Whether any step was already skipped as corrupted before
        calling this method.

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
    if strict:
      if steps is not None:
        data["steps"] = _rehydrate_sequence(steps, step_rehydrator)
      if subagent_trajectories is not None:
        data["subagent_trajectories"] = _rehydrate_sequence(
            subagent_trajectories,
            subagent_rehydrator,
        )
      return trajectory_cls(**data)

    rehydrated_steps: list[Any] = []
    if steps:
      for step in steps:
        try:
          rehydrated_step = (
              step_rehydrator(step) if step_rehydrator is not None else step
          )
          if not isinstance(rehydrated_step, Step):
            raise TypeError(
                f"Expected Step instance, got {type(rehydrated_step).__name__}"
            )
          rehydrated_steps.append(rehydrated_step)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.warning(
              "Skipping corrupted step %s in trajectory %r.",
              getattr(step, "step_id", None),
              self.trajectory_id,
              exc_info=True,
          )
    sorted_steps = sorted(rehydrated_steps, key=lambda s: s.step_id)
    valid_steps: list[Any] = []
    expected_step_id = trajectory_cls.STEP_ID_START
    for step in sorted_steps:
      if step.step_id != expected_step_id:
        break
      valid_steps.append(step)
      expected_step_id += 1
    if steps is not None:
      data["steps"] = valid_steps
    if subagent_trajectories is not None:
      valid_subagents: list[Any] = []
      seen_subagent_ids: set[str] = set()
      for subagent_trajectory in subagent_trajectories:
        try:
          rehydrated_subagent = (
              subagent_rehydrator(subagent_trajectory, strict=False)
              if subagent_rehydrator is not None
              else subagent_trajectory
          )
          if not isinstance(rehydrated_subagent, Trajectory):
            raise TypeError(
                "Expected Trajectory instance, got"
                f" {type(rehydrated_subagent).__name__}"
            )
          subagent_id = rehydrated_subagent.trajectory_id
          if subagent_id is None or subagent_id in seen_subagent_ids:
            raise ValueError(
                f"Invalid or duplicate subagent trajectory_id: {subagent_id!r}"
            )
          seen_subagent_ids.add(subagent_id)
          valid_subagents.append(rehydrated_subagent)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.warning(
              "Skipping corrupted subagent trajectory %r in trajectory %r.",
              getattr(subagent_trajectory, "trajectory_id", None),
              self.trajectory_id,
              exc_info=True,
          )
      data["subagent_trajectories"] = valid_subagents
    trajectory = trajectory_cls(**data)
    if not strict:
      self._reset_incomplete_outcome(
          trajectory,
          num_input_steps=len(steps) if steps is not None else 0,
          is_incomplete=is_incomplete,
      )
    return trajectory

  def _reset_incomplete_outcome(
      self,
      trajectory: Trajectory[Any],
      *,
      num_input_steps: int,
      is_incomplete: bool,
  ) -> None:
    """Clears completion fields on `trajectory` when steps are incomplete."""
    num_valid_steps = len(trajectory.steps)
    expected_total = (
        self.final_metrics.total_steps
        if self.final_metrics is not None
        else None
    )
    tail_mismatch = self.status in ("COMPLETED", "SUCCEEDED") and (
        (expected_total is not None and num_valid_steps < expected_total)
        or (expected_total is None and num_valid_steps == 0)
    )
    if is_incomplete or num_valid_steps < num_input_steps or tail_mismatch:
      trajectory.status = None
      trajectory.total_reward = None

  @classmethod
  def _get_or_create_paired_trajectory_cls(cls) -> type[Trajectory[Any]]:
    """Resolves or dynamically synthesizes a paired Trajectory subclass for cls."""
    trajectory_base = globals().get("Trajectory")
    if trajectory_base is None:
      raise RuntimeError("Trajectory class is not yet defined.")
    if cls is TrajectoryMetadata:
      return trajectory_base
    if issubclass(cls, trajectory_base):
      return cls
    paired_cls = TrajectoryMetadata._PAIRED_TRAJECTORY_CLASSES.get(cls)
    if paired_cls is not None:
      return paired_cls
    for subcls in cls.__subclasses__():
      if issubclass(subcls, trajectory_base):
        TrajectoryMetadata._PAIRED_TRAJECTORY_CLASSES[cls] = subcls
        return subcls
    paired_cls = cast(
        type[Trajectory[Any]],
        type(
            f"{cls.__name__}Trajectory",
            (cls, trajectory_base),
            {
                "__module__": cls.__module__,
                "__doc__": (
                    f"Auto-synthesized Trajectory paired with {cls.__name__}."
                ),
            },
        ),
    )
    paired_cls.model_rebuild()
    parent_module = sys.modules.get(cls.__module__)
    if parent_module is not None:
      setattr(parent_module, paired_cls.__name__, paired_cls)
    TrajectoryMetadata._PAIRED_TRAJECTORY_CLASSES[cls] = paired_cls
    return paired_cls

  def create_trajectory(
      self,
      steps: Sequence[Any] | None = None,
      subagent_trajectories: Sequence[Any] | None = None,
      *,
      strict: bool = True,
      is_incomplete: bool = False,
  ) -> Trajectory[Any]:
    """Creates a full Trajectory from this metadata, steps, and subagents.

    When called on a `TrajectoryMetadata` subclass that does not override
    this method, automatically resolves or synthesizes the paired `Trajectory`
    subclass. Subclasses with custom step upcasting should override this
    method (see `TunixTrajectoryMetadata.create_trajectory`).

    Args:
      steps: Sequence of step instances, or None.
      subagent_trajectories: Sequence of subagent trajectory instances, or None.
      strict: Whether to enforce strict sequential step IDs and raise on step
        validation errors. When False, skips corrupted steps, retains the
        longest contiguous prefix starting from `STEP_ID_START`, and resets
        `status` and `total_reward` to `None` when incomplete.
      is_incomplete: Whether any step was already skipped as corrupted before
        calling this method.

    Returns:
      The constructed trajectory instance.
    """
    trajectory_cls = self._get_or_create_paired_trajectory_cls()
    return self._create_paired_trajectory(
        trajectory_cls,
        steps,
        subagent_trajectories,
        strict=strict,
        is_incomplete=is_incomplete,
    )


StepT = TypeVar("StepT", bound=Step)


class Trajectory(TrajectoryMetadata, Generic[StepT]):
  """Root trajectory object containing the interaction history."""

  # First step_id in a well-formed trajectory; step_ids must be sequential
  # from here.
  STEP_ID_START: ClassVar[int] = 1

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
    """Validate that step_ids are sequential from `STEP_ID_START`."""
    for expected_step_id, step in enumerate(steps, start=cls.STEP_ID_START):
      if step.step_id != expected_step_id:
        steps.sort(key=lambda s: s.step_id)
        for sorted_expected_id, sorted_step in enumerate(
            steps, start=cls.STEP_ID_START
        ):
          if sorted_step.step_id != sorted_expected_id:
            raise ValueError(
                f"Expected step_id {sorted_expected_id} (sequential from"
                f" {cls.STEP_ID_START}), got {sorted_step.step_id}"
            )
        break
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
    step_id = len(self.steps) + self.STEP_ID_START
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

  EXCLUDABLE_ARRAY_FIELDS: ClassVar[frozenset[str]] = frozenset({
      "assistant_tokens",
      "assistant_masks",
      "logprobs",
      "prefill_routed_experts",
  })

  # Scalars that must be all set or all None with `prefill_routed_experts` (see
  # `validate_agent_source`), so they are dropped along with it.
  _PREFILL_COMPANION_FIELDS: ClassVar[frozenset[str]] = frozenset({
      "prefill_start",
      "prefill_num_context",
  })

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
          "MoE routed expert IDs returned by this step's model call, covering"
          " sequence positions [prefill_start, prefill_start + T). Stored in"
          " one of three layouts (decode with"
          " tunix.rl.packing.decode_routed_experts): (1) 3D int16 (T, L, K)"
          " holding raw expert IDs with -1 for unset (raw IDs must be written"
          " as int16); (2) 3D uint8 (T, L, K) holding +1-shifted expert IDs"
          " (id + 1) with 0 for unset; or (3) 4D uint8 (T, L, K // 4, 5) (or"
          " (T, L, 1, ceil(10 * K / 8)) when K % 4 != 0) packing +1-shifted"
          " 10-bit values (id + 1, 0 for unset) into bytes. The last sampled"
          " token of a turn (plus any chat-parser"
          " suffix tokens appended to assistant_tokens) is only routed by the"
          " next turn's prefill, so rows are: turn 0 = prompt + this turn's"
          " routed assistant prefix; turn t > 0 = previous turn's unrouted"
          " assistant tail + previous env tokens + this turn's routed"
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

  def without_excluded_fields(self, exclude: Set[str]) -> Self:
    """Drops `exclude` fields, clearing prefill scalars with routed experts.

    Args:
      exclude: Names of fields to drop.

    Returns:
      A shallow copy with the fields dropped, or `self` if nothing was dropped.
    """
    if "prefill_routed_experts" in exclude:
      exclude = exclude | self._PREFILL_COMPANION_FIELDS
    return super().without_excluded_fields(exclude)

  @pydantic.model_validator(mode="after")
  def validate_assistant_tokens_match_logprobs(self) -> TunixAgentStep:
    """Validate that `assistant_tokens` and `logprobs` have equal lengths."""
    if (
        self.assistant_tokens is not None
        and self.logprobs is not None
        and len(self.assistant_tokens) != len(self.logprobs)
    ):
      raise ValueError(
          f"Length of assistant_tokens ({len(self.assistant_tokens)}) must"
          f" match length of logprobs ({len(self.logprobs)})."
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

  # Note: `converter.to_tunix_trajectory` uses the previous step's
  # `env_tokens` length to split `prefill_num_context` when reconstructing
  # multi-turn MoE routing, so keep `env_tokens` when `prefill_routed_experts`
  # is retained and that reconstruction is needed.
  EXCLUDABLE_ARRAY_FIELDS: ClassVar[frozenset[str]] = frozenset({
      "env_tokens",
      "env_masks",
  })

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

  STEP_CLASSES: ClassVar[tuple[type[Step], ...]] = (
      TunixAgentStep,
      TunixEnvStep,
  )

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
  gamma: float | None = pydantic.Field(
      default=None,
      description="Discount factor for Monte Carlo return calculation.",
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
      subagent_trajectories: Sequence[Trajectory[Any]] | None = None,
      *,
      strict: bool = True,
      is_incomplete: bool = False,
  ) -> TunixTrajectory:
    """Creates a TunixTrajectory, rehydrating 1-indexed ATIF inputs.

    Args:
      steps: Tunix steps (kept as-is) or 1-indexed ATIF steps (rehydrated into
        the Tunix subclass for their source), or None.
      subagent_trajectories: Embedded Tunix subagent trajectories (kept as-is)
        or 1-indexed ATIF trajectories (rehydrated recursively), or None.
      strict: Whether to enforce sequential step IDs and raise on step or
        subagent rehydration errors. When False, skips corrupted items, retains
        the longest contiguous prefix from `STEP_ID_START`, and resets `status`
        and `total_reward` to `None` when incomplete.
      is_incomplete: Whether any step was already skipped as corrupted before
        calling this method.

    Returns:
      The constructed TunixTrajectory.
    """
    return self._create_paired_trajectory(
        TunixTrajectory,
        steps,
        subagent_trajectories,
        step_rehydrator=_upcast_atif_step,
        subagent_rehydrator=TunixTrajectory.from_atif_trajectory,
        strict=strict,
        is_incomplete=is_incomplete,
    )


class TunixTrajectory(
    TunixTrajectoryMetadata,
    Trajectory[TunixAgentStep | TunixEnvStep],
):
  """Tunix-specific trajectory object containing the interaction history."""

  STEP_ID_START: ClassVar[int] = 0

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
    step_id = len(self.steps) + self.STEP_ID_START
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
  def from_atif_trajectory(
      cls,
      atif_trajectory: Trajectory[Any],
      *,
      strict: bool = True,
  ) -> TunixTrajectory:
    """Rehydrates a 1-indexed ATIF Trajectory to a 0-indexed TunixTrajectory."""
    if isinstance(atif_trajectory, cls):
      return atif_trajectory

    return TunixTrajectoryMetadata.from_atif_metadata(
        atif_trajectory
    ).create_trajectory(
        steps=atif_trajectory.steps,
        subagent_trajectories=atif_trajectory.subagent_trajectories,
        strict=strict,
    )
