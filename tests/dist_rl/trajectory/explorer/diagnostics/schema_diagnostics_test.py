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

import math
from typing import Any, ClassVar

from absl.testing import absltest
from absl.testing import parameterized
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


class ExtractTest(parameterized.TestCase):

  def test_reads_the_attribute_named_after_each_column(self) -> None:
    record = _metadata(
        verdict="pass", num_retries=2, latency_ms=12.5, passed=True
    )

    self.assertEqual(
        _EvalDiagnostics().extract(record),
        {
            "verdict": "pass",
            "num_retries": 2,
            "latency_ms": 12.5,
            "passed": True,
        },
    )

  def test_unset_values_are_unreported(self) -> None:
    self.assertEqual(
        _EvalDiagnostics().extract(_metadata()),
        {
            "verdict": None,
            "num_retries": None,
            "latency_ms": None,
            "passed": None,
        },
    )

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

  def test_a_column_the_record_lacks_raises(self) -> None:
    class Diagnostics(schema_diagnostics.SchemaDiagnostics[_EvalMetadata]):
      COLUMN_DTYPES = {"not_a_field": "string"}

    with self.assertRaises(AttributeError):
      Diagnostics().extract(_metadata())


class DefaultHooksTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.diagnostics = _EvalDiagnostics()

  def test_show_lines_name_every_column(self) -> None:
    lines = self.diagnostics.format_show_lines({
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

    columns = self.diagnostics.summary_columns(df)

    self.assertEqual(list(columns), list(df.columns))
    for name, column in columns.items():
      pd.testing.assert_series_equal(column, df[name])

  def test_run_summary_and_totals_are_empty(self) -> None:
    self.assertEqual(self.diagnostics.summarize_run(pd.DataFrame()), {})
    self.assertEmpty(self.diagnostics.format_totals({}, num_trajectories=2))


if __name__ == "__main__":
  absltest.main()
