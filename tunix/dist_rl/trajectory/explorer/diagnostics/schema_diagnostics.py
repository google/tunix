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

"""Schema diagnostics: statistics that only one trajectory schema reports.

`stats.py` derives only what every ATIF trajectory reports (steps, tokens and
tool calls). A schema that stores more, i.e. a `TrajectoryMetadata` subclass
such as `TunixTrajectoryMetadata`, gets its own per-trajectory values, run
totals and output lines from a `SchemaDiagnostics` subclass:

  class TunixDiagnostics(SchemaDiagnostics[TunixTrajectoryMetadata]):
    COLUMN_DTYPES = {"masked_out": "boolean"}

Nothing is registered: callers pass diagnostics instances explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
import types
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import formatting

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

# The Python types each supported pandas dtype holds, besides None. Only
# nullable dtypes are supported, because a record may not report a value:
# "int64" fails on a null and "bool" silently turns it into True or False.
_NULLABLE_DTYPES: Mapping[str, tuple[type[Any], ...]] = types.MappingProxyType({
    "string": (str,),
    "Int64": (int,),
    "Float64": (int, float),
    "float64": (int, float),
    "boolean": (bool,),
})

# An `EXTENSIONS_KEY` such as "_tunix_extensions", capturing its namespace
# ("tunix"): lowercase snake case without leading, trailing or double
# underscores.
_EXTENSIONS_KEY_PATTERN = re.compile(
    r"_([a-z][a-z0-9]*(?:_[a-z0-9]+)*)_extensions"
)


class SchemaDiagnostics(Generic[MetadataT]):
  """Diagnostics for one schema, a `TrajectoryMetadata` or `Trajectory` subclass.

  A subclass names its schema as the generic parameter and declares its columns
  in `COLUMN_DTYPES`; every method has a default. The diagnostics apply to the
  records that are instances of the schema. Instances hold no state.

  Hooks that take a DataFrame receive one row per trajectory and only this
  class's own columns, under their own names (e.g. `df["masked_out"]`).

  Attributes:
    SCHEMA: The schema, derived from the class's `SchemaDiagnostics[X]` base.
    NAMESPACE: The name this class's values are reported under, derived from
      `SCHEMA.EXTENSIONS_KEY` ("_tunix_extensions" -> "tunix").
    COLUMN_DTYPES: The nullable pandas dtype of each column: "string", "Int64",
      "Float64", "float64" or "boolean".
  """

  SCHEMA: ClassVar[type[trajectory_lib.TrajectoryMetadata]]
  NAMESPACE: ClassVar[str]
  COLUMN_DTYPES: ClassVar[Mapping[str, str]]

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Derives `SCHEMA` and `NAMESPACE`, and validates `COLUMN_DTYPES`.

    Args:
      **kwargs: Forwarded to `object.__init_subclass__`.

    Raises:
      TypeError: If the class does not subclass `SchemaDiagnostics[X]` directly
        for a `TrajectoryMetadata` subclass `X`, if `X.EXTENSIONS_KEY` does not
        have the form `_<namespace>_extensions`, or if a column name is not an
        identifier or its dtype is not nullable.
      AttributeError: If the class declares no `COLUMN_DTYPES`, or `X` has no
        `EXTENSIONS_KEY`.
    """
    super().__init_subclass__(**kwargs)
    cls.SCHEMA = _schema_of(cls)
    cls.NAMESPACE = _namespace_of(cls.SCHEMA)
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

  def extract(self, record: MetadataT) -> Mapping[str, JsonScalar]:
    """Returns this class's value of each column for one record.

    The default reads the record's attribute named after each column, and
    reports a non-finite float (NaN, inf) as None.

    Args:
      record: A trajectory, or trajectory metadata, of this class's schema.

    Raises:
      AttributeError: If the record has no attribute named after a column.
      TypeError: If a value does not fit its column's dtype.
    """
    values: dict[str, JsonScalar] = {}
    for column, dtype in self.COLUMN_DTYPES.items():
      value = getattr(record, column)
      if isinstance(value, float) and not math.isfinite(value):
        value = None
      if not _fits_dtype(value, dtype):
        raise TypeError(
            f"{type(self).__name__}: {column}={value!r} does not fit dtype"
            f" {dtype!r}."
        )
      values[column] = value
    return values

  def summarize_run(self, df: pd.DataFrame) -> dict[str, JsonValue]:
    """Returns run-level figures computed from this class's columns.

    The default returns none.

    Args:
      df: This class's columns, one row per trajectory.
    """
    del df  # Unused by the default.
    return {}

  # Presentation hooks, used only to render the explorer's CLI output.

  def format_show_lines(
      self, values: Mapping[str, JsonScalar]
  ) -> Sequence[str]:
    """Returns the lines to add to a trajectory's header in `show`.

    The default prints one `column: value` line per column, with `-` for an
    unreported value.

    Args:
      values: This class's `extract()` result for the trajectory.
    """
    return [
        f"{column}: {formatting.format_optional(values[column])}"
        for column in self.COLUMN_DTYPES
    ]

  def summary_columns(self, df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Returns the columns to add to the `summary` table, by header.

    The default adds every column as is, under its own name.

    Args:
      df: This class's columns, one row per trajectory.
    """
    return {column: df[column] for column in self.COLUMN_DTYPES}

  def format_totals(
      self, summary: Mapping[str, JsonValue], num_trajectories: int
  ) -> Sequence[str]:
    """Returns the lines to add to the run totals in `summary`.

    The default adds none.

    Args:
      summary: This class's `summarize_run()` result.
      num_trajectories: The number of trajectories in the run.
    """
    del summary, num_trajectories  # Unused by the default.
    return ()


def _schema_of(
    cls: type[SchemaDiagnostics[Any]],
) -> type[trajectory_lib.TrajectoryMetadata]:
  """Returns the schema `X` of `cls`'s `SchemaDiagnostics[X]` base."""
  # Python erases `[X]` from `cls.__bases__`, but keeps the parameterized base
  # in the original bases (PEP 560).
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
  """Returns whether a column of the nullable pandas `dtype` holds `value`."""
  if value is None:
    return True
  # `bool` is a subclass of `int`, but only "boolean" columns hold booleans.
  if isinstance(value, bool):
    return dtype == "boolean"
  return isinstance(value, _NULLABLE_DTYPES[dtype])
