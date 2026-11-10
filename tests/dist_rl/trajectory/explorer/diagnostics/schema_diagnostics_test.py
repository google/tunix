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

"""Unit tests for the schema diagnostics contract."""

from collections.abc import Mapping
import math
from typing import Any, ClassVar

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from tunix.dist_rl.trajectory import trajectory as trajectory_lib
from tunix.dist_rl.trajectory.explorer.diagnostics import schema_diagnostics

_AGENT = trajectory_lib.Agent(name="a", version="1.0")


class _EvalMetadata(trajectory_lib.TrajectoryMetadata):
  """A hypothetical evaluation schema with a field for every supported dtype."""

  EXTENSIONS_KEY: ClassVar[str] = "_acme_eval_extensions"

  verdict: str | None = None
  num_retries: int | None = None
  latency_ms: float | None = None
  passed: bool | None = None
  tags: list[str] | None = None


class _EvalDiagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):

  COLUMN_DTYPES = {
      "verdict": "string",
      "num_retries": "Int64",
      "latency_ms": "Float64",
      "passed": "boolean",
  }


def _metadata(**fields: Any) -> _EvalMetadata:
  """Builds evaluation metadata with the given fields."""
  return _EvalMetadata(agent=_AGENT, **fields)


def _overriding(
    values: Mapping[str, schema_diagnostics.JsonScalar],
) -> schema_diagnostics.SchemaDiagnostics[_EvalMetadata]:
  """Returns diagnostics whose `compute_columns` returns `values`."""

  class Overriding(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):
    COLUMN_DTYPES = {"verdict": "string", "num_retries": "Int64"}

    def compute_columns(
        self, record: _EvalMetadata
    ) -> Mapping[str, schema_diagnostics.JsonScalar]:
      del record  # Unused.
      return values

  return Overriding()


def _summarizing(
    run_summary: Mapping[str, schema_diagnostics.JsonValue],
) -> schema_diagnostics.SchemaDiagnostics[_EvalMetadata]:
  """Returns diagnostics whose `compute_run_summary` returns `run_summary`."""

  class Summarizing(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):
    COLUMN_DTYPES = {"verdict": "string"}

    def compute_run_summary(
        self, df: pd.DataFrame
    ) -> Mapping[str, schema_diagnostics.JsonValue]:
      del df  # Unused.
      return run_summary

  return Summarizing()


class DefinitionTest(parameterized.TestCase):

  def test_schema_and_namespace_are_derived(self) -> None:
    self.assertIs(_EvalDiagnostics.SCHEMA, _EvalMetadata)
    self.assertEqual(_EvalDiagnostics.NAMESPACE, "acme_eval")

  def test_a_trajectory_class_can_be_the_schema(self) -> None:
    class TunixTrajectoryDiagnostics(
        schema_diagnostics.SchemaDiagnostics[trajectory_lib.TunixTrajectory]
    ):
      COLUMN_DTYPES = {"prompt_id": "string"}

    self.assertIs(
        TunixTrajectoryDiagnostics.SCHEMA, trajectory_lib.TunixTrajectory
    )
    self.assertEqual(TunixTrajectoryDiagnostics.NAMESPACE, "tunix")

  def test_the_schema_must_be_the_generic_parameter(self) -> None:
    with self.assertRaisesRegex(TypeError, "must subclass SchemaDiagnostics"):

      class Unparameterized(schema_diagnostics.SchemaDiagnostics):  # pylint: disable=unused-variable
        COLUMN_DTYPES = {}

    with self.assertRaisesRegex(TypeError, "must subclass SchemaDiagnostics"):

      class SubclassOfDiagnostics(_EvalDiagnostics):  # pylint: disable=unused-variable
        pass

  @parameterized.named_parameters(
      ("no_extensions_suffix", "_acme_eval"),
      ("no_leading_underscore", "acme_eval_extensions"),
      ("upper_case", "_Acme_extensions"),
      ("double_underscore", "_acme__eval_extensions"),
  )
  def test_the_extensions_key_must_name_a_namespace(self, key: str) -> None:
    class Schema(trajectory_lib.TrajectoryMetadata):
      EXTENSIONS_KEY: ClassVar[str] = key
      verdict: str | None = None

    with self.assertRaisesRegex(TypeError, "EXTENSIONS_KEY"):

      class Diagnostics(schema_diagnostics.SchemaDiagnostics[Schema]):  # pylint: disable=unused-variable
        COLUMN_DTYPES = {"verdict": "string"}

  def test_a_schema_without_an_extensions_key_is_rejected(self) -> None:
    with self.assertRaises(AttributeError):

      class Diagnostics(  # pylint: disable=unused-variable
          schema_diagnostics.SchemaDiagnostics[
              trajectory_lib.TrajectoryMetadata
          ]
      ):
        COLUMN_DTYPES = {"notes": "string"}

  @parameterized.named_parameters(
      ("non_nullable_integer", {"num_retries": "int64"}),
      ("non_nullable_boolean", {"passed": "bool"}),
      ("float_with_nan_nulls", {"latency_ms": "float64"}),
      ("unknown_dtype", {"verdict": "str"}),
      ("column_that_is_not_an_identifier", {"not a name": "string"}),
  )
  def test_invalid_columns_are_rejected(
      self, column_dtypes: dict[str, str]
  ) -> None:
    with self.assertRaises(TypeError):

      class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):  # pylint: disable=unused-variable
        COLUMN_DTYPES = column_dtypes

  def test_columns_must_be_declared(self) -> None:
    with self.assertRaises(AttributeError):

      class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):  # pylint: disable=unused-variable
        pass

  def test_extract_cannot_be_overridden(self) -> None:
    with self.assertRaisesRegex(TypeError, "override compute_columns"):

      class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):  # pylint: disable=unused-variable
        COLUMN_DTYPES = {"verdict": "string"}

        # pyrefly: ignore[bad-override]
        def extract(  # pylint: disable=overridden-final-method
            self, record: _EvalMetadata
        ) -> Mapping[str, schema_diagnostics.JsonScalar]:
          return {"verdict": record.verdict}

  def test_summarize_cannot_be_overridden(self) -> None:
    with self.assertRaisesRegex(TypeError, "override compute_run_summary"):

      class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):  # pylint: disable=unused-variable
        COLUMN_DTYPES = {"verdict": "string"}

        # pyrefly: ignore[bad-override]
        def summarize(  # pylint: disable=overridden-final-method
            self, df: pd.DataFrame
        ) -> Mapping[str, schema_diagnostics.JsonValue]:
          return {"num_trajectories": len(df)}


class ComputeColumnsTest(absltest.TestCase):

  def test_reads_the_attribute_named_after_each_column(self) -> None:
    record = _metadata(
        verdict="pass", num_retries=2, latency_ms=12.5, passed=True
    )

    self.assertEqual(
        _EvalDiagnostics().compute_columns(record),
        {
            "verdict": "pass",
            "num_retries": 2,
            "latency_ms": 12.5,
            "passed": True,
        },
    )

  def test_unset_values_are_unreported(self) -> None:
    self.assertEqual(
        _EvalDiagnostics().compute_columns(_metadata()),
        {
            "verdict": None,
            "num_retries": None,
            "latency_ms": None,
            "passed": None,
        },
    )

  def test_a_column_the_record_lacks_raises(self) -> None:
    class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):
      COLUMN_DTYPES = {"not_a_field": "string"}

    with self.assertRaises(AttributeError):
      Diagnostics().compute_columns(_metadata())


class ExtractTest(parameterized.TestCase):

  def test_returns_the_computed_columns_in_column_order(self) -> None:
    values = _overriding({"num_retries": None, "verdict": "pass"}).extract(
        _metadata()
    )

    self.assertEqual(values, {"verdict": "pass", "num_retries": None})
    self.assertEqual(list(values), ["verdict", "num_retries"])

  @parameterized.named_parameters(
      ("nan", math.nan),
      ("inf", math.inf),
      ("negative_inf", -math.inf),
  )
  def test_non_finite_floats_are_unreported(self, value: float) -> None:
    values = _EvalDiagnostics().extract(_metadata(latency_ms=value))

    self.assertIsNone(values["latency_ms"])

  @parameterized.named_parameters(
      ("list_in_a_string_column", {"tags": "string"}, {"tags": ["a"]}),
      ("bool_in_an_integer_column", {"passed": "Int64"}, {"passed": True}),
      (
          "string_in_a_boolean_column",
          {"verdict": "boolean"},
          {"verdict": "yes"},
      ),
  )
  def test_a_value_that_does_not_fit_its_dtype_is_rejected(
      self, column_dtypes: dict[str, str], fields: dict[str, Any]
  ) -> None:
    class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):
      COLUMN_DTYPES = column_dtypes

    with self.assertRaisesRegex(TypeError, "does not fit dtype"):
      Diagnostics().extract(_metadata(**fields))

  @parameterized.named_parameters(
      ("missing_column", {"verdict": "pass"}, ValueError),
      (
          "extra_column",
          {"verdict": "pass", "num_retries": 1, "typo": 1},
          ValueError,
      ),
      ("int_in_a_string_column", {"verdict": 5, "num_retries": 1}, TypeError),
      (
          "float_in_an_integer_column",
          {"verdict": "pass", "num_retries": 1.5},
          TypeError,
      ),
      (
          "numpy_integer_in_an_integer_column",
          {"verdict": "pass", "num_retries": np.int64(1)},
          TypeError,
      ),
  )
  def test_computed_columns_are_checked(
      self,
      values: Mapping[str, schema_diagnostics.JsonScalar],
      error: type[Exception],
  ) -> None:
    with self.assertRaisesRegex(error, r"^Overriding\.compute_columns\(\)"):
      _overriding(values).extract(_metadata())


class SummarizeTest(parameterized.TestCase):

  def test_returns_the_computed_run_summary(self) -> None:
    run_summary: dict[str, schema_diagnostics.JsonValue] = {
        "num_passed": 1,
        "pass_rate": 0.5,
        "tags": {"a": [None]},
    }

    self.assertEqual(
        _summarizing(run_summary).summarize(pd.DataFrame()), run_summary
    )

  @parameterized.named_parameters(
      ("numpy_integer", {"num_passed": np.int64(1)}),
      ("numpy_boolean", {"all_passed": np.bool_(True)}),
      ("pandas_null", {"max_retries": pd.NA}),
  )
  def test_a_run_summary_that_is_not_json_is_rejected(
      self, run_summary: Mapping[str, schema_diagnostics.JsonValue]
  ) -> None:
    with self.assertRaisesRegex(
        TypeError, r"^Summarizing\.compute_run_summary\(\)"
    ):
      _summarizing(run_summary).summarize(pd.DataFrame())

  @parameterized.named_parameters(
      ("nan", {"pass_rate": float("nan")}),
      ("inf", {"pass_rate": float("inf")}),
      ("nested_negative_inf", {"rates": {"a": [float("-inf")]}}),
  )
  def test_a_run_summary_with_a_non_finite_float_is_rejected(
      self, run_summary: Mapping[str, schema_diagnostics.JsonValue]
  ) -> None:
    with self.assertRaisesRegex(
        ValueError, r"^Summarizing\.compute_run_summary\(\)"
    ):
      _summarizing(run_summary).summarize(pd.DataFrame())


class DefaultHooksTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.diagnostics = _EvalDiagnostics()

  def test_show_header_names_every_column(self) -> None:
    lines = self.diagnostics.format_show_header({
        "verdict": "pass",
        "num_retries": None,
        "latency_ms": 12.5,
        "passed": False,
    })

    self.assertEqual(
        list(lines),
        [
            "verdict: pass",
            "num_retries: -",
            "latency_ms: 12.5",
            "passed: False",
        ],
    )

  def test_summary_columns_are_the_columns_under_their_own_names(self) -> None:
    df = pd.DataFrame({
        "verdict": ["pass"],
        "num_retries": [1],
        "latency_ms": [2.5],
        "passed": [True],
    })

    columns = self.diagnostics.format_summary_columns(df)

    self.assertEqual(list(columns), list(df.columns))
    for name, column in columns.items():
      pd.testing.assert_series_equal(column, df[name])

  def test_run_summary_and_summary_totals_are_empty(self) -> None:
    self.assertEqual(self.diagnostics.summarize(pd.DataFrame()), {})
    self.assertEmpty(
        self.diagnostics.format_summary_totals({}, num_trajectories=2)
    )


if __name__ == "__main__":
  absltest.main()
