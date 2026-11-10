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

"""Schema diagnostics: statistics specific to one trajectory schema.

A `SchemaDiagnostics` subclass declares the columns, run summary and output
lines that one `TrajectoryMetadata` subclass contributes to the explorer:

  class TunixDiagnostics(SchemaDiagnostics[TunixTrajectoryMetadata]):
    COLUMN_DTYPES = {"masked_out": "boolean"}

Diagnostics are not registered; callers pass instances explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
import types
from typing import Any, ClassVar, Generic, TypeVar, final, get_args, get_origin

import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import formatting

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

# The supported column dtypes, each with the types of the non-null values that
# a column of the dtype holds.
_NULLABLE_DTYPES: Mapping[str, tuple[type[Any], ...]] = types.MappingProxyType({
    "string": (str,),
    "Int64": (int,),
    "Float64": (int, float),
    "boolean": (bool,),
})

# An `EXTENSIONS_KEY` of the form `_<namespace>_extensions`, capturing the
# namespace: lowercase snake case without leading, trailing or double
# underscores.
_EXTENSIONS_KEY_PATTERN = re.compile(
    r"_([a-z][a-z0-9]*(?:_[a-z0-9]+)*)_extensions"
)


class SchemaDiagnostics(Generic[MetadataT]):
  """Diagnostics for one schema, a `TrajectoryMetadata` or `Trajectory` subclass.

  A subclass passes its schema as the generic parameter, declares its columns in
  `COLUMN_DTYPES`, and may override `compute_columns`, `compute_run_summary` and
  the `format_*` hooks, which all have defaults. `extract` and `summarize` are
  final. Instances hold no state.

  `compute_columns` and `format_show_header` are called only for trajectories
  that are instances of the schema. `compute_run_summary` and
  `format_summary_columns` receive this class's columns, under their own names
  (e.g. `df["masked_out"]`), with one row per trajectory of the run; the rows of
  trajectories of other schemas are null.

  Attributes:
    SCHEMA: The generic parameter of the class's `SchemaDiagnostics[X]` base.
    NAMESPACE: The key this class's values are reported under, derived from
      `SCHEMA.EXTENSIONS_KEY` ("_tunix_extensions" -> "tunix").
    COLUMN_DTYPES: The nullable pandas dtype of each column: "string", "Int64",
      "Float64" or "boolean".
  """

  SCHEMA: ClassVar[type[trajectory_lib.TrajectoryMetadata]]
  NAMESPACE: ClassVar[str]
  COLUMN_DTYPES: ClassVar[Mapping[str, str]]

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Derives `SCHEMA` and `NAMESPACE`, and validates the subclass.

    Args:
      **kwargs: Forwarded to `object.__init_subclass__`.

    Raises:
      TypeError: If the class does not subclass `SchemaDiagnostics[X]` directly
        for a `TrajectoryMetadata` subclass `X`, if `X.EXTENSIONS_KEY` does not
        have the form `_<namespace>_extensions`, if the class overrides
        `extract` or `summarize`, or if a column name is not an identifier or
        its dtype is not nullable.
      AttributeError: If the class declares no `COLUMN_DTYPES`, or `X` has no
        `EXTENSIONS_KEY`.
    """
    super().__init_subclass__(**kwargs)
    cls.SCHEMA = _schema_of(cls)
    cls.NAMESPACE = _namespace_of(cls.SCHEMA)
    if "extract" in vars(cls):
      raise TypeError(
          f"{cls.__name__} overrides extract(); override compute_columns()"
          " instead."
      )
    if "summarize" in vars(cls):
      raise TypeError(
          f"{cls.__name__} overrides summarize(); override"
          " compute_run_summary() instead."
      )
    for column, dtype in cls.COLUMN_DTYPES.items():
      if not column.isidentifier():
        raise TypeError(
            f"{cls.__name__}: column {column!r} is not an identifier."
        )
      if dtype not in _NULLABLE_DTYPES:
        raise TypeError(
            f"{cls.__name__}: column {column!r} has dtype {dtype!r}; expected"
            f" a nullable dtype, one of {list(_NULLABLE_DTYPES)}."
        )

  @final
  def extract(self, record: MetadataT) -> Mapping[str, JsonScalar]:
    """Returns this class's value of each column for one record.

    The values are those of `compute_columns(record)`, in `COLUMN_DTYPES`
    order, with non-finite floats (NaN, inf) replaced by None.

    Args:
      record: A trajectory, or trajectory metadata, of this class's schema.

    Raises:
      ValueError: If `compute_columns` does not return exactly the columns of
        `COLUMN_DTYPES`.
      TypeError: If a value does not fit its column's dtype.
    """
    computed = self.compute_columns(record)
    if computed.keys() != self.COLUMN_DTYPES.keys():
      raise ValueError(
          f"{type(self).__name__}.compute_columns() returned columns"
          f" {list(computed)}; expected {list(self.COLUMN_DTYPES)}."
      )
    values: dict[str, JsonScalar] = {}
    for column, dtype in self.COLUMN_DTYPES.items():
      value = computed[column]
      if isinstance(value, float) and not math.isfinite(value):
        value = None
      if not _fits_dtype(value, dtype):
        raise TypeError(
            f"{type(self).__name__}.compute_columns() returned"
            f" {column}={value!r}, which does not fit dtype {dtype!r}."
        )
      values[column] = value
    return values

  def compute_columns(self, record: MetadataT) -> Mapping[str, JsonScalar]:
    """Returns the value of each of this class's columns for one record.

    `extract` checks and returns the result. The default reads the record's
    attribute named after each column.

    Args:
      record: A trajectory, or trajectory metadata, of this class's schema.

    Raises:
      AttributeError: If the record has no attribute named after a column.
    """
    return {column: getattr(record, column) for column in self.COLUMN_DTYPES}

  @final
  def summarize(self, df: pd.DataFrame) -> Mapping[str, JsonValue]:
    """Returns this class's run summary, computed from its columns.

    The summary is that of `compute_run_summary(df)`, checked to be standard
    JSON serializable.

    Args:
      df: This class's columns, one row per trajectory of the run.

    Raises:
      TypeError: If `compute_run_summary` returns a value that is not JSON
        serializable, such as a numpy integer.
      ValueError: If `compute_run_summary` returns a non-finite float (NaN,
        inf), which standard JSON cannot represent.
    """
    run_summary = self.compute_run_summary(df)
    try:
      json.dumps(dict(run_summary), allow_nan=False)
    except TypeError as e:
      raise TypeError(
          f"{type(self).__name__}.compute_run_summary() returned a value that"
          f" is not JSON serializable: {e}"
      ) from e
    except ValueError as e:
      raise ValueError(
          f"{type(self).__name__}.compute_run_summary() returned a value that"
          f" is not standard JSON: {e}"
      ) from e
    return run_summary

  def compute_run_summary(self, df: pd.DataFrame) -> Mapping[str, JsonValue]:
    """Returns this class's run summary, computed from its columns.

    `summarize` checks and returns the result. The default returns an empty
    mapping.

    Args:
      df: This class's columns, one row per trajectory of the run.
    """
    del df  # Unused.
    return {}

  # Presentation hooks, used only by the explorer's CLI commands.

  def format_show_header(
      self, values: Mapping[str, JsonScalar]
  ) -> Sequence[str]:
    """Returns the lines this class adds to a trajectory's header in `show`.

    The default returns one `column: value` line per column, with `-` for None.

    Args:
      values: This class's `extract()` result for the trajectory.
    """
    return [
        f"{column}: {formatting.format_optional(values[column])}"
        for column in self.COLUMN_DTYPES
    ]

  def format_summary_columns(
      self, df: pd.DataFrame
  ) -> Mapping[str, pd.Series]:
    """Returns the columns this class adds to the `summary` table, by header.

    The default returns each column under its own name.

    Args:
      df: This class's columns, one row per trajectory of the run.
    """
    return {column: df[column] for column in self.COLUMN_DTYPES}

  def format_summary_totals(
      self, run_summary: Mapping[str, JsonValue], num_trajectories: int
  ) -> Sequence[str]:
    """Returns the lines this class adds to the run totals in `summary`.

    The default returns no lines.

    Args:
      run_summary: This class's `summarize()` result.
      num_trajectories: The number of trajectories in the run.
    """
    del run_summary, num_trajectories  # Unused.
    return ()


def _schema_of(
    cls: type[SchemaDiagnostics[Any]],
) -> type[trajectory_lib.TrajectoryMetadata]:
  """Returns the schema `X` of `cls`'s `SchemaDiagnostics[X]` base."""
  for base in types.get_original_bases(cls):
    if get_origin(base) is SchemaDiagnostics:
      (schema,) = get_args(base)
      if isinstance(schema, type) and issubclass(
          schema, trajectory_lib.TrajectoryMetadata
      ):
        return schema
  raise TypeError(
      f"{cls.__name__} must subclass SchemaDiagnostics[X] directly, where X is"
      " a TrajectoryMetadata subclass."
  )


def _namespace_of(schema: type[trajectory_lib.TrajectoryMetadata]) -> str:
  """Returns the namespace in `schema.EXTENSIONS_KEY` ("_tunix_extensions")."""
  match = _EXTENSIONS_KEY_PATTERN.fullmatch(schema.EXTENSIONS_KEY)
  if match is None:
    raise TypeError(
        f"{schema.__name__}.EXTENSIONS_KEY is {schema.EXTENSIONS_KEY!r};"
        " schema diagnostics need the form '_<namespace>_extensions', with a"
        " lowercase snake_case namespace."
    )
  return match.group(1)


def _fits_dtype(value: object, dtype: str) -> bool:
  """Returns whether a column of the nullable pandas `dtype` holds `value`.

  None fits every dtype, and a bool fits only "boolean".
  """
  if value is None:
    return True
  if isinstance(value, bool):
    return dtype == "boolean"
  return isinstance(value, _NULLABLE_DTYPES[dtype])
