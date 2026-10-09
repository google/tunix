import collections
from collections.abc import Iterator
import copy
import json
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pydantic
from tunix.experimental.trajectory import converter
from tunix.experimental.trajectory import trajectory
from tunix.experimental.trajectory import trajectory_testing
from tunix.rl.agentic.agents import agent_types

_SAMPLE_ATIF_PATH = os.path.join(
    os.path.dirname(trajectory.__file__), "testdata", "sample_atif_v1_7.json"
)


class SubagentTrajectoryRefTest(parameterized.TestCase):

  def test_subagent_trajectory_ref_valid_trajectory_id(self):
    # Valid: only trajectory_id
    ref1 = trajectory.SubagentTrajectoryRef(trajectory_id="sub-1")
    self.assertEqual(ref1.trajectory_id, "sub-1")

  def test_subagent_trajectory_ref_valid_trajectory_path(self):
    # Valid: only trajectory_path
    ref2 = trajectory.SubagentTrajectoryRef(trajectory_path="path/to/sub.json")
    self.assertEqual(ref2.trajectory_path, "path/to/sub.json")

  def test_subagent_trajectory_ref_invalid_no_id_or_path(self):
    # Invalid: neither trajectory_id nor trajectory_path
    with self.assertRaises(ValueError):
      trajectory.SubagentTrajectoryRef(session_id="session-1")


class StepTest(trajectory_testing.TrajectoryTestCase):

  @parameterized.named_parameters(
      ("metrics", "metrics", trajectory.Metrics(prompt_tokens=10)),
      (
          "tool_calls",
          "tool_calls",
          [trajectory.ToolCall(tool_call_id="c1", function_name="f1")],
      ),
      ("reasoning_effort", "reasoning_effort", 1.0),
      ("model_name", "model_name", "dummy_value"),
      ("reasoning_content", "reasoning_content", "dummy_value"),
  )
  def test_validate_agent_only_fields(self, field_name, value):
    # Invalid: non-agent step containing agent-only field
    kwargs = {
        "step_id": 1,
        "source": trajectory.Source.USER,
        "message": "Hello",
        field_name: value,
    }
    with self.assertRaises(ValueError):
      trajectory.Step(**kwargs)

  @parameterized.named_parameters(
      ("metrics", "metrics", trajectory.Metrics(prompt_tokens=10)),
      ("reasoning_effort", "reasoning_effort", 1.0),
      ("reasoning_content", "reasoning_content", "dummy_value"),
      ("model_name", "model_name", "dummy_value"),
  )
  def test_validate_llm_call_count_zero_prohibits_llm_fields(
      self, field_name, value
  ):
    # Invalid: agent step with llm_call_count=0 containing LLM fields
    kwargs = {
        "step_id": 1,
        "source": trajectory.Source.AGENT,
        "message": "Deterministic action",
        "llm_call_count": 0,
        field_name: value,
    }
    with self.assertRaises(ValueError):
      trajectory.Step(**kwargs)

  def test_validate_llm_call_count_zero_allows_non_llm_fields(self):
    # Valid: agent step with llm_call_count=0 without LLM-specific fields.
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Deterministic action",
        llm_call_count=0,
    )
    self.assertEqual(step.llm_call_count, 0)

  def test_step_policy_version_agent_step_success(self):
    step = trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Agent turn",
        policy_version=42,
    )
    self.assertEqual(step.policy_version, 42)
    step_dict = step.model_dump()
    self.assertEqual(step_dict["policy_version"], 42)
    restored_step = trajectory.TunixAgentStep(**step_dict)
    self.assertEqual(restored_step, step)


class MetadataDictSerializationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="primitive_list",
          extra={"tokens": [1, 2, 3]},
          expected={"tokens": [1, 2, 3]},
      ),
      dict(
          testcase_name="mixed_primitive_list",
          extra={"values": [True, 1, 2.5, "a", None]},
          expected={"values": [True, 1, 2.5, "a", None]},
      ),
      dict(
          testcase_name="empty_list",
          extra={"values": []},
          expected={"values": []},
      ),
      dict(
          testcase_name="primitive_tuple",
          extra={"pair": (1, 2)},
          expected={"pair": [1, 2]},
      ),
      dict(
          testcase_name="ndarray",
          extra={"arr": np.array([1, 2])},
          expected={"arr": [1, 2]},
      ),
      dict(
          testcase_name="ndarray_in_list",
          extra={"values": [1, np.array([2, 3])]},
          expected={"values": [1, [2, 3]]},
      ),
      dict(
          testcase_name="nested_sequences",
          extra={"rows": [[1, 2], (3, 4)]},
          expected={"rows": [[1, 2], [3, 4]]},
      ),
      dict(
          testcase_name="dict_in_list",
          extra={"items": [{"arr": np.array([1])}]},
          expected={"items": [{"arr": [1]}]},
      ),
      dict(
          testcase_name="non_primitive_scalars",
          extra={"values": [trajectory.Source.AGENT, np.int64(7)]},
          expected={"values": [trajectory.Source.AGENT, np.int64(7)]},
      ),
  )
  def test_model_dump_converts_nested_arrays_and_tuples_to_lists(
      self, extra, expected
  ):
    step = trajectory.Step(
        step_id=1, source=trajectory.Source.AGENT, message="msg", extra=extra
    )

    self.assertEqual(step.model_dump()["extra"], expected)

  def test_model_dump_does_not_recurse_into_primitive_list_elements(self):
    tokens = list(range(1000))
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"tokens": tokens},
    )

    with mock.patch.object(
        trajectory,
        "_to_json_compatible",
        wraps=trajectory._to_json_compatible,
    ) as mock_to_json_compatible:
      dumped_extra = step.model_dump()["extra"]

    self.assertEqual(dumped_extra, {"tokens": tokens})
    # One call for the `extra` dict and one for the list, none per element.
    self.assertEqual(mock_to_json_compatible.call_count, 2)


class _IterationCountingArray(np.ndarray):
  """A NumPy array that counts the calls to its `__iter__`."""

  iteration_count: int

  def __array_finalize__(self, obj: np.ndarray | None) -> None:
    """Starts the iteration count of each new array at zero."""
    del obj  # Unused.
    self.iteration_count = 0

  def __iter__(self) -> Iterator[Any]:
    """Counts the call, then iterates over the array's elements."""
    self.iteration_count += 1
    return super().__iter__()


class ArrayFieldValidationTest(parameterized.TestCase):
  """Tests validation of the NumPy array fields of Tunix steps."""

  @parameterized.named_parameters(
      dict(
          testcase_name="assistant_tokens",
          step_cls=trajectory.TunixAgentStep,
          source=trajectory.Source.AGENT,
          field_name="assistant_tokens",
          dtype=np.int32,
      ),
      dict(
          testcase_name="assistant_masks",
          step_cls=trajectory.TunixAgentStep,
          source=trajectory.Source.AGENT,
          field_name="assistant_masks",
          dtype=np.int32,
      ),
      dict(
          testcase_name="logprobs",
          step_cls=trajectory.TunixAgentStep,
          source=trajectory.Source.AGENT,
          field_name="logprobs",
          dtype=np.float32,
      ),
      dict(
          testcase_name="env_tokens",
          step_cls=trajectory.TunixEnvStep,
          source=trajectory.Source.USER,
          field_name="env_tokens",
          dtype=np.int32,
      ),
      dict(
          testcase_name="env_masks",
          step_cls=trajectory.TunixEnvStep,
          source=trajectory.Source.USER,
          field_name="env_masks",
          dtype=np.int32,
      ),
  )
  def test_array_is_stored_without_iterating_its_elements(
      self,
      step_cls: type[trajectory.TunixAgentStep | trajectory.TunixEnvStep],
      source: trajectory.Source,
      field_name: str,
      dtype: type[np.generic],
  ) -> None:
    """Verifies arrays are kept as they are, without iterating over them."""
    array = np.arange(1000, dtype=dtype).view(_IterationCountingArray)

    step = step_cls(
        step_id=0, source=source, message="msg", **{field_name: array}
    )

    self.assertIs(getattr(step, field_name), array)
    # Guards the speed of validating arrays, which relies on `np.ndarray`
    # preceding `list[...]` in the field's union: see `trajectory.IntArray`.
    self.assertEqual(array.iteration_count, 0)

  @parameterized.named_parameters(
      dict(testcase_name="list", tokens=[1, 2, 3]),
      dict(testcase_name="tuple", tokens=(1, 2, 3)),
  )
  def test_sequence_is_validated_into_int_list(
      self, tokens: list[int] | tuple[int, ...]
  ) -> None:
    """Verifies lists and tuples of ints are validated into lists of ints."""
    step = trajectory.TunixAgentStep(
        step_id=0,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=tokens,
    )

    self.assertIsInstance(step.assistant_tokens, list)
    self.assertEqual(step.assistant_tokens, [1, 2, 3])

  @parameterized.named_parameters(
      dict(testcase_name="string", tokens="abc"),
      dict(testcase_name="non_int_element", tokens=[1, "x"]),
      dict(testcase_name="numpy_scalar", tokens=np.int32(1)),
  )
  def test_invalid_tokens_raise_validation_error(self, tokens: Any) -> None:
    """Verifies tokens other than arrays and sequences of ints are rejected."""
    with self.assertRaises(pydantic.ValidationError):
      trajectory.TunixAgentStep(
          step_id=0,
          source=trajectory.Source.AGENT,
          message="msg",
          assistant_tokens=tokens,
      )


class _TokenList(list[int]):
  """A list subclass, which `copy.deepcopy` preserves."""


class DeepCopyTest(trajectory_testing.TrajectoryTestCase):
  """Tests `trajectory.deep_copy`."""

  def _assert_identical(self, actual: Any, expected: Any) -> None:
    """Asserts that `actual` and `expected` have equal types and values.

    Unlike `assertEqual`, this also compares the types of nested values, such as
    `bool`, `int`, and `float`, or `list` and `tuple`, and the order of dict
    keys. Models are compared by their set fields and `__dict__`, and NumPy
    arrays by their dtypes and elements.

    Args:
      actual: The value to check.
      expected: The value that `actual` must match.
    """
    self.assertIs(type(actual), type(expected))
    if isinstance(expected, pydantic.BaseModel):
      self.assertEqual(actual.model_fields_set, expected.model_fields_set)
      self._assert_identical(actual.__dict__, expected.__dict__)
    elif isinstance(expected, dict):
      self.assertEqual(list(actual), list(expected))
      for key, value in expected.items():
        self._assert_identical(actual[key], value)
    elif isinstance(expected, (list, tuple)):
      self.assertLen(actual, len(expected))
      for actual_element, expected_element in zip(actual, expected):
        self._assert_identical(actual_element, expected_element)
    elif isinstance(expected, np.ndarray):
      self.assertEqual(actual.dtype, expected.dtype)
      np.testing.assert_array_equal(actual, expected)
    else:
      self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name="step", model=trajectory_testing.STEP_1_1),
      dict(
          testcase_name="tunix_env_step",
          model=trajectory_testing.TUNIX_ENV_STEP_0,
      ),
      dict(
          testcase_name="tunix_agent_step",
          model=trajectory_testing.TUNIX_AGENT_STEP_1,
      ),
      dict(
          testcase_name="tunix_metadata",
          model=trajectory_testing.TUNIX_METADATA_1,
      ),
      dict(
          testcase_name="tunix_trajectory",
          model=trajectory_testing.TUNIX_TRAJECTORY_1,
      ),
      dict(
          testcase_name="atif_trajectory",
          model=trajectory_testing.PAIRED_ATIF_TRAJECTORY,
      ),
      dict(
          testcase_name="mixed_extra",
          model=trajectory.TunixAgentStep(
              step_id=0,
              source=trajectory.Source.AGENT,
              message="msg",
              assistant_tokens=np.array([10, 20], dtype=np.int32),
              logprobs=[-0.5, -1.0],
              extra={
                  "scalars": [1, True, 2.0, None, "s"],
                  "rows": [[1, 2], (3, [4.0]), {"ids": [5]}],
                  "empty": {"list": [], "dict": {}, "tuple": ()},
                  "array": np.arange(3, dtype=np.float32),
                  "source": trajectory.Source.USER,
              },
          ),
      ),
  )
  def test_deep_copy_matches_model_copy_deep(
      self, model: pydantic.BaseModel
  ) -> None:
    """Verifies the copy has the types and values of `model_copy(deep=True)`."""
    copied = trajectory.deep_copy(model)

    self.assertIsNot(copied, model)
    self._assert_identical(copied, model.model_copy(deep=True))

  def test_deep_copy_shares_no_mutable_state(self) -> None:
    """Verifies changes to the copy leave the original unchanged."""
    step = trajectory.TunixAgentStep(
        step_id=0,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([10, 20]),
        metrics=trajectory.Metrics(
            prompt_token_ids=[1, 2], logprobs=[-0.1, -0.2]
        ),
        extra={"ids": [3, 4], "rows": [[5], {"ids": [6]}]},
    )

    copied = trajectory.deep_copy(step)
    copied.assistant_tokens[0] = 0
    copied.metrics.prompt_token_ids.append(3)
    copied.metrics.logprobs[0] = 0.0
    copied.extra["ids"].append(5)
    copied.extra["rows"][0].append(7)
    copied.extra["rows"][1]["ids"].append(8)

    np.testing.assert_array_equal(step.assistant_tokens, [10, 20])
    self.assertEqual(step.metrics.prompt_token_ids, [1, 2])
    self.assertEqual(step.metrics.logprobs, [-0.1, -0.2])
    self.assertEqual(step.extra, {"ids": [3, 4], "rows": [[5], {"ids": [6]}]})

  def test_deep_copy_preserves_shared_references(self) -> None:
    """Verifies a list referred to several times is copied once."""
    shared_ids = [1, 2, 3]
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"a": shared_ids, "b": shared_ids, "rows": [shared_ids]},
    )

    copied = trajectory.deep_copy(step)

    self.assertIsNot(copied.extra["a"], shared_ids)
    self.assertIs(copied.extra["b"], copied.extra["a"])
    self.assertIs(copied.extra["rows"][0], copied.extra["a"])

  @parameterized.named_parameters(
      dict(testcase_name="tuple_first", keys=("pair", "ids")),
      dict(testcase_name="list_first", keys=("ids", "pair")),
  )
  def test_deep_copy_shares_list_reached_through_tuple(
      self, keys: tuple[str, str]
  ) -> None:
    """Verifies a list reached directly and through a tuple is copied once."""
    shared_ids = [1, 2, 3]
    values = {"pair": (shared_ids, "x"), "ids": shared_ids}
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={key: values[key] for key in keys},
    )

    copied = trajectory.deep_copy(step)

    self.assertIsNot(copied.extra["ids"], shared_ids)
    self.assertEqual(copied.extra["ids"], [1, 2, 3])
    self.assertIs(copied.extra["pair"][0], copied.extra["ids"])
    self.assertEqual(copied.extra["pair"][1], "x")

  def test_deep_copy_copies_cyclic_list(self) -> None:
    """Verifies a list that contains itself is copied with the cycle kept."""
    cyclic = [1]
    cyclic.append(cyclic)
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"cyclic": cyclic},
    )

    copied = trajectory.deep_copy(step)

    copied_cyclic = copied.extra["cyclic"]
    self.assertIsNot(copied_cyclic, cyclic)
    self.assertEqual(copied_cyclic[0], 1)
    self.assertIs(copied_cyclic[1], copied_cyclic)

  def test_deep_copy_preserves_shared_models(self) -> None:
    """Verifies a model referred to twice is copied once."""
    shared_metrics = trajectory.Metrics(prompt_token_ids=[1, 2])
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        metrics=shared_metrics,
        extra={"metrics": shared_metrics},
    )

    copied = trajectory.deep_copy(step)

    self.assertIsNot(copied.metrics, shared_metrics)
    self.assertIs(copied.extra["metrics"], copied.metrics)

  def test_deep_copy_copies_model_its_fields_refer_back_to_like_model_copy(
      self,
  ) -> None:
    """Verifies a model in a reference cycle is copied as `model_copy` does."""
    step = trajectory.Step(
        step_id=1, source=trajectory.Source.AGENT, message="msg", extra={}
    )
    step.extra["step"] = step

    copied = trajectory.deep_copy(step)

    # Matches `model_copy(deep=True)`: the back reference becomes a second
    # model object that shares the outer copy's `__dict__`.
    self.assertIsNot(copied, step)
    self.assertIsNot(copied.extra["step"], copied)
    self.assertIs(copied.extra["step"].__dict__, copied.__dict__)

  def test_deep_copy_preserves_list_subclass(self) -> None:
    """Verifies a list subclass keeps its type in the copy."""
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"ids": _TokenList([1, 2])},
    )

    copied = trajectory.deep_copy(step)

    self.assertIs(type(copied.extra["ids"]), _TokenList)
    self.assertEqual(copied.extra["ids"], [1, 2])
    self.assertIsNot(copied.extra["ids"], step.extra["ids"])

  def test_deep_copy_preserves_nested_dict_subclass(self) -> None:
    """Verifies a nested `defaultdict` keeps its type and default factory."""
    ids_by_name = collections.defaultdict(list, {"a": [1, 2]})
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"ids_by_name": ids_by_name},
    )

    copied = trajectory.deep_copy(step)

    copied_ids_by_name = copied.extra["ids_by_name"]
    self.assertIs(type(copied_ids_by_name), collections.defaultdict)
    self.assertIs(copied_ids_by_name.default_factory, list)
    self.assertEqual(copied_ids_by_name, {"a": [1, 2]})
    self.assertIsNot(copied_ids_by_name["a"], ids_by_name["a"])

  def test_deep_copy_copies_dict_with_tuple_keys(self) -> None:
    """Verifies a dict with tuple keys is copied with its keys and values."""
    ids_by_pair = {(1, 2): [3], (4, 5): [6]}
    step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"ids_by_pair": ids_by_pair},
    )

    copied = trajectory.deep_copy(step)

    copied_ids_by_pair = copied.extra["ids_by_pair"]
    self.assertEqual(copied_ids_by_pair, {(1, 2): [3], (4, 5): [6]})
    self.assertIsNot(copied_ids_by_pair[(1, 2)], ids_by_pair[(1, 2)])


class TrajectoryTest(trajectory_testing.TrajectoryTestCase):
  sample_atif_trajectory: trajectory.Trajectory

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with open(_SAMPLE_ATIF_PATH, "r", encoding="utf-8") as f:
      cls.sample_atif_trajectory = trajectory.Trajectory.from_json_dict(
          json.load(f)
      )

  def test_basic_serialization_and_deserialization(self):
    traj = self.sample_atif_trajectory
    self.assertEqual(traj.schema_version, "ATIF-v1.7")
    self.assertEqual(traj.session_id, "session-123")
    self.assertEqual(traj.trajectory_id, "traj-456")
    self.assertLen(traj.steps, 2)
    self.assertEqual(traj.steps[0].message, "List directory contents")

    serialized = traj.to_json_dict()
    reloaded = trajectory.Trajectory.from_json_dict(serialized)
    self.assertTrajectoryEqual(reloaded, traj)

  def test_dynamic_step_logging(self):
    traj = trajectory.TunixTrajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    self.assertEmpty(traj.steps)

    step1 = traj.add_step(source=trajectory.Source.USER, message="Start task")
    self.assertEqual(step1.step_id, 0)
    self.assertLen(traj.steps, 1)
    self.assertEqual(traj.steps[0].message, "Start task")

    step2 = traj.add_step(
        source=trajectory.Source.AGENT,
        message="Working",
        reasoning_content="Logic here",
        policy_version=5,
    )
    self.assertEqual(step2.step_id, 1)
    self.assertLen(traj.steps, 2)
    self.assertEqual(traj.steps[1].reasoning_content, "Logic here")
    self.assertEqual(traj.steps[1].policy_version, 5)

  def test_trajectory_metadata_target_policy_versions(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="traj_pv_test",
        agent=trajectory.Agent(name="test_agent", version="1.0"),
        target_policy_versions=[1, 2, 3],
    )
    self.assertEqual(meta.target_policy_versions, [1, 2, 3])
    dumped = meta.model_dump()
    self.assertEqual(dumped["target_policy_versions"], [1, 2, 3])
    restored = trajectory.TunixTrajectoryMetadata.model_validate(dumped)
    self.assertEqual(restored.target_policy_versions, [1, 2, 3])

  def test_observation_and_metrics_serialization(self):
    data = {
        "schema_version": "ATIF-v1.7",
        "session_id": "test-session",
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{
            "step_id": 1,
            "source": "agent",
            "message": "Call tool",
            "tool_calls": [{
                "tool_call_id": "call-1",
                "function_name": "calculator",
                "arguments": {"expr": "2+2"},
            }],
            "observation": {
                "results": [{
                    "source_call_id": "call-1",
                    "content": "4",
                }]
            },
            "metrics": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "cost_usd": 0.0001,
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5],
                "logprobs": [-0.1, -0.2],
                "extra": {"latency": 0.5},
            },
        }],
        "final_metrics": {
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0001,
            "total_steps": 1,
            "extra": {"overall_latency": 0.5},
        },
    }

    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertLen(traj.steps, 1)
    step = traj.steps[0]
    self.assertEqual(step.message, "Call tool")
    self.assertEqual(step.tool_calls[0].tool_call_id, "call-1")
    self.assertEqual(step.observation.results[0].content, "4")
    self.assertEqual(step.metrics.prompt_tokens, 10)
    self.assertEqual(step.metrics.extra["latency"], 0.5)
    self.assertEqual(traj.final_metrics.total_steps, 1)

    serialized = traj.to_json_dict()
    self.assertEqual(serialized["final_metrics"]["total_steps"], 1)
    self.assertEqual(
        serialized["steps"][0]["observation"]["results"][0]["content"], "4"
    )
    self.assertEqual(serialized["steps"][0]["metrics"]["prompt_tokens"], 10)

  def test_add_step_with_observation_and_metrics(self):
    traj = trajectory.Trajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    obs = trajectory.Observation(
        results=[
            trajectory.ObservationResult(
                source_call_id="call-1", content="result"
            )
        ]
    )
    metrics = trajectory.Metrics(prompt_tokens=20, completion_tokens=10)

    step = traj.add_step(
        source="agent",
        message="Running...",
        observation=obs,
        metrics=metrics,
    )

    self.assertEqual(step.step_id, 1)
    self.assertEqual(step.observation.results[0].content, "result")
    self.assertEqual(step.metrics.prompt_tokens, 20)

  def test_add_step_with_all_optional_fields(self):
    traj = trajectory.Trajectory(
        agent=trajectory.Agent(name="test-agent", version="1.0")
    )
    step = traj.add_step(
        source=trajectory.Source.AGENT,
        message="Running...",
        model_name="gpt-4",
        reasoning_effort=1.5,
        is_copied_context=True,
        llm_call_count=2,
        extra={"key": "val"},
    )
    expected_data = {
        "step_id": 1,
        "source": "agent",
        "message": "Running...",
        "model_name": "gpt-4",
        "reasoning_effort": 1.5,
        "is_copied_context": True,
        "llm_call_count": 2,
        "extra": {"key": "val"},
    }
    self.assertIsNotNone(step.timestamp)
    actual_data = step.model_dump(
        exclude={"timestamp"}, exclude_none=True, mode="json"
    )
    self.assertDictEqual(actual_data, expected_data)

  def test_validate_step_ids(self):
    # Invalid: non-sequential step IDs
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "First"},
            {"step_id": 3, "source": "agent", "message": "Third"},
        ],
    }
    with self.assertRaises(ValueError):
      trajectory.Trajectory.from_json_dict(data)

  def test_unordered_steps_are_sorted(self):
    # Valid: out-of-order steps are sorted by step_id
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [
            {"step_id": 2, "source": "agent", "message": "Second"},
            {"step_id": 1, "source": "user", "message": "First"},
        ],
    }
    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertEqual(traj.steps[0].step_id, 1)
    self.assertEqual(traj.steps[1].step_id, 2)

  def test_validate_embedded_subagent_missing_trajectory_id(self):
    # Invalid: missing trajectory_id on embedded subagent
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "First"}],
        "subagent_trajectories": [{
            "agent": {"name": "sub-agent", "version": "1.0"},
            "steps": [{"step_id": 1, "source": "agent", "message": "Sub"}],
        }],
    }
    with self.assertRaises(ValueError):
      trajectory.Trajectory.from_json_dict(data)

  def test_validate_embedded_subagent_duplicate_trajectory_id(self):
    # Invalid: duplicate trajectory_id on embedded subagents
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "First"}],
        "subagent_trajectories": [
            {
                "trajectory_id": "dup-id",
                "agent": {"name": "sub-1", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 1"}
                ],
            },
            {
                "trajectory_id": "dup-id",
                "agent": {"name": "sub-2", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 2"}
                ],
            },
        ],
    }
    with self.assertRaises(ValueError):
      trajectory.Trajectory.from_json_dict(data)

  def test_validate_embedded_subagent_unique_trajectory_id(self):
    # Valid: unique trajectory_id on embedded subagents
    data = {
        "agent": {"name": "test-agent", "version": "1.0"},
        "steps": [{"step_id": 1, "source": "user", "message": "First"}],
        "subagent_trajectories": [
            {
                "trajectory_id": "sub-1",
                "agent": {"name": "sub-1", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 1"}
                ],
            },
            {
                "trajectory_id": "sub-2",
                "agent": {"name": "sub-2", "version": "1.0"},
                "steps": [
                    {"step_id": 1, "source": "agent", "message": "Sub 2"}
                ],
            },
        ],
    }
    traj = trajectory.Trajectory.from_json_dict(data)
    self.assertLen(traj.subagent_trajectories, 2)

  def test_get_metadata(self):
    traj = self.sample_atif_trajectory
    traj.add_step(source=trajectory.Source.USER, message="Hello")

    meta = traj.get_metadata()
    self.assertIsInstance(meta, trajectory.TrajectoryMetadata)
    self.assertNotIsInstance(meta, trajectory.Trajectory)
    self.assertEqual(meta.trajectory_id, "traj-456")
    self.assertFalse(hasattr(meta, "steps"))
    self.assertFalse(hasattr(meta, "subagent_trajectories"))

  def test_create_trajectory_with_steps_and_subagent_trajectories(self):
    meta = trajectory.TrajectoryMetadata(
        trajectory_id="traj_parent",
        session_id="session_1",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
        notes="parent notes",
        extra={"custom": "field"},
    )
    steps = [
        trajectory.Step(
            step_id=1, source=trajectory.Source.USER, message="Start"
        ),
        trajectory.Step(
            step_id=2, source=trajectory.Source.AGENT, message="Respond"
        ),
    ]
    subagents = [
        trajectory.Trajectory(
            trajectory_id="sub_1",
            agent=trajectory.Agent(name="sub_agent", version="1.0"),
            steps=[
                trajectory.Step(
                    step_id=1, source=trajectory.Source.AGENT, message="Sub"
                )
            ],
        )
    ]

    traj = meta.create_trajectory(steps=steps, subagent_trajectories=subagents)

    self.assertIsInstance(traj, trajectory.Trajectory)
    self.assertEqual(traj.trajectory_id, "traj_parent")
    self.assertEqual(traj.session_id, "session_1")
    self.assertEqual(traj.notes, "parent notes")
    self.assertEqual(traj.extra, {"custom": "field"})
    self.assertLen(traj.steps, 2)
    self.assertLen(traj.subagent_trajectories, 1)
    self.assertEqual(traj.subagent_trajectories[0].trajectory_id, "sub_1")
    self.assertTrajectoryEqual(traj.subagent_trajectories[0], subagents[0])
    # The input sequences are copied, not aliased.
    self.assertIsNot(traj.steps, steps)
    self.assertIsNot(traj.subagent_trajectories, subagents)

  def test_create_trajectory_without_steps_or_subagent_trajectories(self):
    meta = trajectory.TrajectoryMetadata(
        trajectory_id="traj_empty",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )

    traj = meta.create_trajectory()

    self.assertIsInstance(traj, trajectory.Trajectory)
    self.assertEqual(traj.trajectory_id, "traj_empty")
    self.assertEmpty(traj.steps)
    self.assertIsNone(traj.subagent_trajectories)

  def test_tunix_create_trajectory_with_subagent_trajectories(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_parent",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
        prompt_id="prompt_abc",
        group_index=3,
        status="COMPLETED",
        total_reward=1.5,
        hyperparams={"temperature": 0.7},
    )
    steps = [
        trajectory.TunixAgentStep(
            step_id=0, source=trajectory.Source.AGENT, message="Act"
        )
    ]
    subagents = [
        trajectory.TunixTrajectory(
            trajectory_id="tunix_sub_1",
            agent=trajectory.Agent(name="sub_agent", version="1.0"),
        )
    ]

    traj = meta.create_trajectory(steps=steps, subagent_trajectories=subagents)

    self.assertIsInstance(traj, trajectory.TunixTrajectory)
    self.assertEqual(traj.trajectory_id, "tunix_parent")
    self.assertEqual(traj.prompt_id, "prompt_abc")
    self.assertEqual(traj.group_index, 3)
    self.assertEqual(traj.status, "COMPLETED")
    self.assertEqual(traj.total_reward, 1.5)
    self.assertEqual(traj.hyperparams, {"temperature": 0.7})
    self.assertLen(traj.steps, 1)
    self.assertLen(traj.subagent_trajectories, 1)
    self.assertEqual(traj.subagent_trajectories[0].trajectory_id, "tunix_sub_1")
    self.assertIsNot(traj.subagent_trajectories, subagents)

  def test_tunix_create_trajectory_rehydrates_atif_subagent_trajectories(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_parent",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )
    atif_subagent = trajectory.Trajectory(
        trajectory_id="atif_sub",
        agent=trajectory.Agent(name="sub_agent", version="1.0"),
        steps=[
            trajectory.Step(
                step_id=1, source=trajectory.Source.USER, message="Sub prompt"
            ),
            trajectory.Step(
                step_id=2, source=trajectory.Source.AGENT, message="Sub reply"
            ),
        ],
    )

    traj = meta.create_trajectory(subagent_trajectories=[atif_subagent])

    self.assertLen(traj.subagent_trajectories, 1)
    subagent = traj.subagent_trajectories[0]
    self.assertIsInstance(subagent, trajectory.TunixTrajectory)
    self.assertEqual(subagent.trajectory_id, "atif_sub")
    self.assertLen(subagent.steps, 2)
    self.assertIsInstance(subagent.steps[0], trajectory.TunixEnvStep)
    self.assertEqual(subagent.steps[0].step_id, 0)
    self.assertEqual(subagent.steps[0].message, "Sub prompt")
    self.assertIsInstance(subagent.steps[1], trajectory.TunixAgentStep)
    self.assertEqual(subagent.steps[1].step_id, 1)
    self.assertEqual(subagent.steps[1].message, "Sub reply")

  def test_tunix_create_trajectory_without_subagent_trajectories(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_empty",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )

    traj = meta.create_trajectory()

    self.assertIsInstance(traj, trajectory.TunixTrajectory)
    self.assertEmpty(traj.steps)
    self.assertIsNone(traj.subagent_trajectories)

  def test_tunix_create_trajectory_promotes_plain_atif_steps(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_promote",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )
    plain_steps = [
        trajectory.Step(
            step_id=1, source=trajectory.Source.USER, message="User prompt"
        ),
        trajectory.Step(
            step_id=2, source=trajectory.Source.AGENT, message="Agent response"
        ),
        trajectory.Step.model_validate(
            {"step_id": 3, "source": "agent", "message": "Dict agent step"}
        ),
        trajectory.Step.model_validate(
            {"step_id": 4, "source": "user", "message": "Dict user step"}
        ),
    ]

    traj = meta.create_trajectory(steps=plain_steps)

    self.assertIsInstance(traj, trajectory.TunixTrajectory)
    self.assertLen(traj.steps, 4)
    self.assertIsInstance(traj.steps[0], trajectory.TunixEnvStep)
    self.assertEqual(traj.steps[0].step_id, 0)
    self.assertEqual(traj.steps[0].message, "User prompt")
    self.assertIsInstance(traj.steps[1], trajectory.TunixAgentStep)
    self.assertEqual(traj.steps[1].step_id, 1)
    self.assertEqual(traj.steps[1].message, "Agent response")
    self.assertIsInstance(traj.steps[2], trajectory.TunixAgentStep)
    self.assertEqual(traj.steps[2].step_id, 2)
    self.assertEqual(traj.steps[2].message, "Dict agent step")
    self.assertIsInstance(traj.steps[3], trajectory.TunixEnvStep)
    self.assertEqual(traj.steps[3].step_id, 3)
    self.assertEqual(traj.steps[3].message, "Dict user step")

  def test_tunix_create_trajectory_rejects_non_step_inputs(self):
    # Only typed `Step` instances are accepted; raw dicts are not silently
    # reinterpreted as either ATIF or Tunix steps.
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_promote_err",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )
    tunix_shaped_dict = {
        "step_id": 1,
        "source": "user",
        "message": "m",
        "reward": 1.0,
    }
    for non_step in ({"invalid": "data"}, tunix_shaped_dict, "not_a_step"):
      with self.subTest(non_step=non_step):
        with self.assertRaises(AttributeError):
          meta.create_trajectory(steps=[non_step])  # pytype: disable=wrong-arg-types

  def test_tunix_create_trajectory_propagates_step_rehydration_errors(self):
    # Rehydration failures surface as-is instead of falling back to treating
    # the input as an already-Tunix step.
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_promote_err",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )
    bad_ext_step = trajectory.Step(
        step_id=1,
        source=trajectory.Source.USER,
        message="m",
        extra={trajectory.TUNIX_EXTENSIONS_KEY: {"reward": "not_a_float"}},
    )

    with self.assertRaisesRegex(pydantic.ValidationError, "reward"):
      meta.create_trajectory(steps=[bad_ext_step])

  def test_tunix_create_trajectory_rejects_unknown_source(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="tunix_unknown_source",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
    )
    unknown_source_step = trajectory.Step.model_construct(
        step_id=1, source="bogus", message="m"
    )

    with self.assertRaisesRegex(ValueError, "Unsupported step source"):
      meta.create_trajectory(steps=[unknown_source_step])

  def test_create_trajectory_rejects_unpaired_subclass_without_fields(self):
    """Without the check this would silently yield a plain Trajectory."""

    class _NoNewFields(trajectory.TrajectoryMetadata):
      """Adds behavior but no fields, and forgets to override."""

      EXTENSIONS_KEY = "_test_extensions"

      def is_interesting(self) -> bool:
        return self.notes is not None

    meta = _NoNewFields(agent=trajectory.Agent(name="a", version="1.0"))

    with self.assertRaisesRegex(
        TypeError,
        "Expected trajectory_cls to be a subclass of _NoNewFields"
    ):
      meta.create_trajectory()

  def test_create_trajectory_rejects_unpaired_subclass_with_fields(self):
    """A subclass adding fields reports the pairing, not the rejected field."""

    class _NewField(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      custom_tag: str = ""

    try:
      meta = _NewField(
          agent=trajectory.Agent(name="a", version="1.0"),
          custom_tag="experiment_42",
      )

      with self.assertRaisesRegex(
          TypeError, "Expected trajectory_cls to be a subclass of _NewField"
      ):
        meta.create_trajectory()
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(_NewField, None)

  def test_step_initialization_with_rl_fields(self):
    step = trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Running bash command",
        reasoning_content="Thought process",
        assistant_tokens=np.array([10, 20]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.1, -0.2]),
        mc_return=1.5,
        extra={"custom_key": "custom_val"},
    )
    self.assertEqual(step.step_id, 1)
    self.assertEqual(step.source, trajectory.Source.AGENT)
    self.assertEqual(step.message, "Running bash command")
    self.assertEqual(step.reasoning_content, "Thought process")
    np.testing.assert_array_equal(step.assistant_tokens, np.array([10, 20]))
    np.testing.assert_array_equal(step.assistant_masks, np.array([1, 1]))
    np.testing.assert_array_equal(step.logprobs, np.array([-0.1, -0.2]))
    self.assertEqual(step.mc_return, 1.5)
    self.assertEqual(step.extra, {"custom_key": "custom_val"})

  def test_step_env_fields(self):
    step = trajectory.TunixEnvStep(
        step_id=2,
        source=trajectory.Source.SYSTEM,
        message="Observation result",
        reward=1.0,
        done=True,
        env_tokens=np.array([100]),
        env_masks=np.array([1]),
    )
    self.assertEqual(step.reward, 1.0)
    self.assertTrue(step.done)
    np.testing.assert_array_equal(step.env_tokens, np.array([100]))
    np.testing.assert_array_equal(step.env_masks, np.array([1]))

  def test_step_json_serialization_and_deserialization(self):
    step = trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="Thinking",
        assistant_tokens=np.array([10, 20]),
        logprobs=np.array([-0.5, -0.3]),
        mc_return=2.0,
    )
    json_str = step.model_dump_json(exclude_none=True)
    loaded_dict = json.loads(json_str)
    self.assertEqual(loaded_dict["assistant_tokens"], [10, 20])
    self.assertEqual(loaded_dict["logprobs"], [-0.5, -0.3])
    self.assertEqual(loaded_dict["mc_return"], 2.0)

    reloaded_step = trajectory.TunixAgentStep.model_validate_json(json_str)
    self.assertEqual(reloaded_step.assistant_tokens, [10, 20])
    self.assertEqual(reloaded_step.logprobs, [-0.5, -0.3])
    self.assertEqual(reloaded_step.mc_return, 2.0)

  def test_step_equality(self):
    step1 = trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([1, 2]),
    )
    step2 = trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([1, 2]),
    )
    step3 = trajectory.TunixAgentStep(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        assistant_tokens=np.array([1, 3]),
    )
    self.assertStepEqual(step1, step2)
    self.assertNotEqual(step1.model_dump(), step3.model_dump())

  def test_step_equality_with_extra_numpy_arrays(self):
    step1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={
            "arr": np.array([1, 2]),
            "nested": {"val": np.array([3, 4])},
        },
    )
    step2 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={
            "arr": np.array([1, 2]),
            "nested": {"val": np.array([3, 4])},
        },
    )
    step3 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={
            "arr": np.array([1, 2]),
            "nested": {"val": np.array([3, 5])},
        },
    )
    step4 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra={"arr": np.array([1, 2])},
    )
    step5 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        extra=None,
    )
    self.assertStepEqual(step1, step2)
    self.assertNotEqual(step1.model_dump(), step3.model_dump())
    self.assertNotEqual(step1.model_dump(), step4.model_dump())
    self.assertNotEqual(step1.model_dump(), step5.model_dump())

  def test_step_equality_with_nested_tool_calls_and_observations(self):
    step1 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call-1",
                function_name="fn",
                arguments={"arr": np.array([1, 2])},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call-1",
                    content="output",
                    extra={"res_arr": np.array([10, 20])},
                )
            ]
        ),
    )
    step2 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call-1",
                function_name="fn",
                arguments={"arr": np.array([1, 2])},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call-1",
                    content="output",
                    extra={"res_arr": np.array([10, 20])},
                )
            ]
        ),
    )
    step3 = trajectory.Step(
        step_id=1,
        source=trajectory.Source.AGENT,
        message="msg",
        tool_calls=[
            trajectory.ToolCall(
                tool_call_id="call-1",
                function_name="fn",
                arguments={"arr": np.array([1, 99])},
            )
        ],
        observation=trajectory.Observation(
            results=[
                trajectory.ObservationResult(
                    source_call_id="call-1",
                    content="output",
                    extra={"res_arr": np.array([10, 20])},
                )
            ]
        ),
    )
    self.assertStepEqual(step1, step2)
    self.assertNotEqual(step1.model_dump(), step3.model_dump())

  def test_trajectory_metadata_first_class_fields(self):
    meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="traj_1",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
        prompt_id="prompt_abc",
        group_index=2,
        status="COMPLETED",
        total_reward=4.2,
        hyperparams={"temperature": 0.7},
        env_time={"init": 0.1, "step": 0.5},
        reward_time={"eval": 0.2},
        extra={"unstructured_key": "val"},
    )
    self.assertIsInstance(meta, trajectory.TunixTrajectoryMetadata)
    self.assertIsInstance(meta, trajectory.TrajectoryMetadata)
    self.assertEqual(meta.trajectory_id, "traj_1")
    self.assertEqual(meta.prompt_id, "prompt_abc")
    self.assertEqual(meta.group_index, 2)
    self.assertEqual(meta.status, "COMPLETED")
    self.assertEqual(meta.total_reward, 4.2)
    self.assertEqual(meta.hyperparams, {"temperature": 0.7})
    self.assertEqual(meta.env_time, {"init": 0.1, "step": 0.5})
    self.assertEqual(meta.reward_time, {"eval": 0.2})
    self.assertEqual(meta.extra, {"unstructured_key": "val"})

  def test_trajectory_get_metadata_preserves_first_class_fields(self):
    traj = trajectory.TunixTrajectory(
        trajectory_id="traj_1",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
        prompt_id="prompt_abc",
        group_index=2,
        status="COMPLETED",
        total_reward=4.2,
        hyperparams={"top_p": 0.9},
        env_time={"init": 0.1},
        reward_time={"eval": 0.2},
        extra={"custom": "field"},
        steps=[
            trajectory.TunixAgentStep(
                step_id=0,
                source=trajectory.Source.AGENT,
                message="hello",
            )
        ],
    )
    meta = traj.get_metadata()
    self.assertIsInstance(meta, trajectory.TunixTrajectoryMetadata)
    self.assertIsInstance(meta, trajectory.TrajectoryMetadata)
    self.assertEqual(meta.trajectory_id, "traj_1")
    self.assertEqual(meta.prompt_id, "prompt_abc")
    self.assertEqual(meta.group_index, 2)
    self.assertEqual(meta.status, "COMPLETED")
    self.assertEqual(meta.total_reward, 4.2)
    self.assertEqual(meta.hyperparams, {"top_p": 0.9})
    self.assertEqual(meta.env_time, {"init": 0.1})
    self.assertEqual(meta.reward_time, {"eval": 0.2})
    self.assertEqual(meta.extra, {"custom": "field"})

  def test_trajectory_serialization_deserialization_with_first_class_metadata(
      self,
  ):
    traj = trajectory.TunixTrajectory(
        trajectory_id="traj_roundtrip",
        agent=trajectory.Agent(name="agent_test", version="1.0"),
        prompt_id="p1",
        group_index=1,
        status="RUNNING",
        total_reward=1.0,
        hyperparams={"temperature": 0.5},
        env_time={"step": 0.05},
        reward_time={"reward": 0.01},
        extra={"meta": "data"},
        steps=[
            trajectory.TunixAgentStep(
                step_id=0,
                source=trajectory.Source.AGENT,
                message="action message",
            )
        ],
    )
    data = traj.to_json_dict()
    restored = trajectory.TunixTrajectory.from_json_dict(data)
    self.assertEqual(restored.trajectory_id, traj.trajectory_id)
    self.assertEqual(restored.prompt_id, "p1")
    self.assertEqual(restored.group_index, 1)
    self.assertEqual(restored.status, "RUNNING")
    self.assertEqual(restored.total_reward, 1.0)
    self.assertEqual(restored.hyperparams, {"temperature": 0.5})
    self.assertEqual(restored.env_time, {"step": 0.05})
    self.assertEqual(restored.reward_time, {"reward": 0.01})
    self.assertEqual(restored.extra, {"meta": "data"})
    self.assertLen(restored.steps, 1)

  def test_to_tunix_trajectory_multi_turn_conversion(self):
    # 1. Create a mock trajectory metadata via create_trajectory_metadata()
    class MockRolloutRequest:
      prompt_id = "prompt_123"
      group_index = 0
      generation_kwargs = {"temperature": 0.7, "top_p": 0.9}
      metadata = {"req_key": "req_val"}

    class MockAgent:
      name = "multi_turn_agent"
      version = "2.0"
      trajectory = agent_types.Trajectory(
          reward=4.5,
          env_time={"init": 0.1, "step": 0.6},
          reward_time={"eval": 0.25},
      )

    meta = converter.create_trajectory_metadata(
        traj_id="traj_multi_turn_test",
        request=MockRolloutRequest(),
        agent=MockAgent(),
        target_policy_versions=[1, 2, 3],
        status="SUCCEEDED",
        extra={"custom_extra": "custom_val"},
    )

    # 2. Add the first step to it via calling create_task_step (step_id = 0)
    step_0 = converter.create_task_step("Solve task step by step")
    self.assertIsNotNone(step_0)
    self.assertEqual(step_0.step_id, 0)
    self.assertEqual(step_0.source, trajectory.Source.USER)

    # 3. Add Tunix turn 0 (tunix_step_id=0) agent step -> converted step_id = 1
    mock_agent_step_1 = agent_types.Step(
        model_response="Action 1 response",
        thought="Thought 1",
        action=agent_types.Action(
            action={
                "id": "call_1",
                "name": "tool_1",
                "arguments": {"arg1": "val1"},
            }
        ),
        assistant_tokens=np.array([101, 102]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.1, -0.2]),
        mc_return=1.0,
        info={"trace_1": "abc"},
    )
    step_1 = converter.create_agent_step(mock_agent_step_1, tunix_step_id=0)
    self.assertIsNotNone(step_1)
    self.assertEqual(step_1.step_id, 1)

    # 4. Add Tunix turn 0 (tunix_step_id=0) env step -> converted step_id = 2
    mock_env_step_1 = agent_types.Step(
        observation="Observation 1 result",
        reward=0.5,
        done=False,
        env_tokens=np.array([201]),
        env_masks=np.array([1]),
        info={"env_meta_1": "val1"},
    )
    step_2 = converter.create_env_step(mock_env_step_1, tunix_step_id=0)
    self.assertIsNotNone(step_2)
    self.assertEqual(step_2.step_id, 2)

    # 5. Add Tunix turn 1 (tunix_step_id=1) agent step -> converted step_id = 3
    mock_agent_step_2 = agent_types.Step(
        model_response="Action 2 response",
        thought="Thought 2",
        action=agent_types.Action(
            action={
                "id": "call_2",
                "name": "tool_2",
                "arguments": {"arg2": "val2"},
            }
        ),
        assistant_tokens=np.array([103, 104]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.3, -0.4]),
        mc_return=2.0,
        info={"trace_2": "def"},
    )
    step_3 = converter.create_agent_step(mock_agent_step_2, tunix_step_id=1)
    self.assertIsNotNone(step_3)
    self.assertEqual(step_3.step_id, 3)

    # 6. Add Tunix turn 1 (tunix_step_id=1) env step -> converted step_id = 4
    mock_env_step_2 = agent_types.Step(
        observation="Observation 2 result",
        reward=1.0,
        done=False,
        env_tokens=np.array([202]),
        env_masks=np.array([1]),
        info={"env_meta_2": "val2"},
    )
    step_4 = converter.create_env_step(mock_env_step_2, tunix_step_id=1)
    self.assertIsNotNone(step_4)
    self.assertEqual(step_4.step_id, 4)

    # 7. Add Tunix turn 2 (tunix_step_id=2) agent step -> converted step_id = 5
    mock_agent_step_3 = agent_types.Step(
        model_response="Action 3 response",
        thought="Thought 3",
        action=None,
        assistant_tokens=np.array([105, 106]),
        assistant_masks=np.array([1, 1]),
        logprobs=np.array([-0.5, -0.6]),
        mc_return=3.0,
        info={"trace_3": "ghi"},
    )
    step_5 = converter.create_agent_step(mock_agent_step_3, tunix_step_id=2)
    self.assertIsNotNone(step_5)
    self.assertEqual(step_5.step_id, 5)

    # 8. Add Tunix turn 2 (tunix_step_id=2) env step -> converted step_id = 6
    mock_env_step_3 = agent_types.Step(
        observation="Observation 3 result",
        reward=3.0,
        done=True,
        env_tokens=np.array([203]),
        env_masks=np.array([1]),
        info={"env_meta_3": "val3"},
    )
    step_6 = converter.create_env_step(mock_env_step_3, tunix_step_id=2)
    self.assertIsNotNone(step_6)
    self.assertEqual(step_6.step_id, 6)

    traj = trajectory.TunixTrajectory(
        **meta.model_dump(),
        steps=[step_0, step_1, step_2, step_3, step_4, step_5, step_6],
    )

    # 9. Call to_tunix_trajectory to convert the trajectory
    converted_traj = converter.to_tunix_trajectory(traj)

    # 10. Verify received trajectory
    # a) Trajectory task should have the same prompt as Step #0
    self.assertEqual(
        converted_traj.task, {"prompts": ["Solve task step by step"]}
    )
    self.assertEqual(converted_traj.task["prompts"][0], traj.steps[0].message)

    # b) Step #0 contains (3) and (4)
    self.assertLen(converted_traj.steps, 3)
    expected_step_0 = converter.to_tunix_step(
        agent_step=step_1, env_step=step_2
    )
    self.assertStepEqual(converted_traj.steps[0], expected_step_0)
    self.assertEqual(
        converted_traj.steps[0].model_response, "Action 1 response"
    )
    self.assertEqual(converted_traj.steps[0].thought, "Thought 1")
    self.assertEqual(
        converted_traj.steps[0].action,
        agent_types.Action(
            action={
                "id": "call_1",
                "name": "tool_1",
                "arguments": {"arg1": "val1"},
            }
        ),
    )
    self.assertEqual(
        converted_traj.steps[0].observation, "Observation 1 result"
    )
    self.assertEqual(converted_traj.steps[0].reward, 0.5)
    self.assertFalse(converted_traj.steps[0].done)
    self.assertEqual(converted_traj.steps[0].mc_return, 1.0)
    np.testing.assert_array_equal(
        converted_traj.steps[0].assistant_tokens, np.array([101, 102])
    )
    np.testing.assert_array_equal(
        converted_traj.steps[0].assistant_masks, np.array([1, 1])
    )
    np.testing.assert_array_equal(
        converted_traj.steps[0].logprobs, np.array([-0.1, -0.2])
    )
    np.testing.assert_array_equal(
        converted_traj.steps[0].env_tokens, np.array([201])
    )
    np.testing.assert_array_equal(
        converted_traj.steps[0].env_masks, np.array([1])
    )
    self.assertEqual(
        converted_traj.steps[0].info,
        {"trace_1": "abc", "env_meta_1": "val1"},
    )

    # c) Step #1 contains (5) and (6)
    expected_step_1 = converter.to_tunix_step(
        agent_step=step_3, env_step=step_4
    )
    self.assertStepEqual(converted_traj.steps[1], expected_step_1)
    self.assertEqual(
        converted_traj.steps[1].model_response, "Action 2 response"
    )
    self.assertEqual(converted_traj.steps[1].thought, "Thought 2")
    self.assertEqual(
        converted_traj.steps[1].action,
        agent_types.Action(
            action={
                "id": "call_2",
                "name": "tool_2",
                "arguments": {"arg2": "val2"},
            }
        ),
    )


class AtifProjectionTest(trajectory_testing.TrajectoryTestCase):
  """Covers the `to_atif_*()` projection of Tunix models into base ATIF."""

  def test_to_atif_step_on_tunix_agent_step_packs_subclass_fields_into_extra(
      self,
  ):
    agent_step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={
            "extra": {
                "user_key": "val",
            }
        }
    )

    atif_agent_step = agent_step.to_atif_step()

    self.assertIs(type(atif_agent_step), trajectory.Step)
    self.assertEqual(atif_agent_step.step_id, agent_step.step_id + 1)
    self.assertEqual(
        atif_agent_step.extra,
        {
            "user_key": "val",
            trajectory.TUNIX_EXTENSIONS_KEY: {
                "mc_return": 2.5,
                "assistant_tokens": [10, 20],
                "assistant_masks": [1, 1],
                "logprobs": [-0.5, -0.2],
                "policy_version": 3,
                "prefill_routed_experts": trajectory._serialize_int_3d_array(
                    [[[1, 2]], [[3, 4]], [[5, 6]]]
                ),
                "prefill_start": 0,
                "prefill_num_context": 1,
            },
        },
    )

  def test_to_atif_step_on_tunix_env_step_packs_subclass_fields_into_extra(
      self,
  ):
    env_step = trajectory_testing.TUNIX_ENV_STEP_0

    atif_env_step = env_step.to_atif_step()

    self.assertIs(type(atif_env_step), trajectory.Step)
    self.assertEqual(atif_env_step.step_id, env_step.step_id + 1)
    self.assertEqual(
        atif_env_step.extra,
        {
            "env_extra_key": "env_extra_val",
            trajectory.TUNIX_EXTENSIONS_KEY: {
                "reward": 1.0,
                "done": False,
                "env_tokens": [1, 2],
                "env_masks": [1, 1],
            },
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="agent_step",
          step_cls=trajectory.TunixAgentStep,
          tunix_step=trajectory_testing.TUNIX_AGENT_STEP_1,
      ),
      dict(
          testcase_name="env_step",
          step_cls=trajectory.TunixEnvStep,
          tunix_step=trajectory_testing.TUNIX_ENV_STEP_0,
      ),
  )
  def test_to_atif_step_packs_every_tunix_only_field_into_extra(
      self, step_cls, tunix_step
  ):
    tunix_only_field_names = set(step_cls.model_fields) - set(
        trajectory.Step.model_fields
    )

    atif_step = tunix_step.to_atif_step()

    self.assertContainsSubset(
        tunix_only_field_names, atif_step.extra[trajectory.TUNIX_EXTENSIONS_KEY]
    )

  def test_to_atif_metadata_packs_every_tunix_only_field_into_extra(self):
    tunix_only_field_names = set(
        trajectory.TunixTrajectoryMetadata.model_fields
    ) - set(trajectory.TrajectoryMetadata.model_fields)

    atif_metadata = trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()

    self.assertContainsSubset(
        tunix_only_field_names,
        atif_metadata.extra[trajectory.TUNIX_EXTENSIONS_KEY],
    )

  def test_to_atif_metadata_on_tunix_metadata_packs_subclass_fields_into_extra(
      self,
  ):
    atif_metadata = trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()

    self.assertIs(type(atif_metadata), trajectory.TrajectoryMetadata)
    self.assertEqual(atif_metadata.trajectory_id, "t_atif")
    self.assertEqual(atif_metadata.session_id, "sess_01")
    self.assertEqual(atif_metadata.notes, "Metadata projection test")
    self.assertEqual(
        atif_metadata.extra,
        {
            "user_meta": "val",
            trajectory.TUNIX_EXTENSIONS_KEY: {
                "prompt_id": "p_1",
                "group_index": 2,
                "target_policy_versions": [2, 3],
                "status": "SUCCEEDED",
                "total_reward": 3.5,
                "masked_out": False,
                "hyperparams": {"temperature": 0.7},
                "env_time": {"step_0": 0.05},
                "reward_time": {"step_1": 0.02},
            },
        },
    )

  def test_to_atif_step_retains_none_valued_subclass_fields_in_extra(self):
    sparse_agent_step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={
            "mc_return": None,
            "assistant_tokens": None,
            "assistant_masks": None,
            "logprobs": None,
            "prefill_routed_experts": None,
            "prefill_start": None,
            "prefill_num_context": None,
            "extra": None,
        }
    )

    atif_step = sparse_agent_step.to_atif_step()

    self.assertEqual(
        atif_step.extra,
        {
            trajectory.TUNIX_EXTENSIONS_KEY: {
                "mc_return": None,
                "assistant_tokens": None,
                "assistant_masks": None,
                "logprobs": None,
                "policy_version": 3,
                "prefill_routed_experts": None,
                "prefill_start": None,
                "prefill_num_context": None,
            }
        },
    )

  def test_to_atif_metadata_retains_none_valued_subclass_fields_in_extra(self):
    sparse_tunix_meta = trajectory.TunixTrajectoryMetadata(
        trajectory_id="t_sparse",
        agent=trajectory.Agent(name="agent", version="1.0"),
    )

    atif_metadata = sparse_tunix_meta.to_atif_metadata()
    reloaded = trajectory.TrajectoryMetadata.model_validate_json(
        atif_metadata.model_dump_json(exclude_none=True)
    )

    expected_keys = set(trajectory.TunixTrajectoryMetadata.model_fields) - set(
        trajectory.TrajectoryMetadata.model_fields
    )
    self.assertEqual(set(atif_metadata.get_extensions().keys()), expected_keys)
    self.assertEqual(set(reloaded.get_extensions().keys()), expected_keys)
    self.assertIsNone(reloaded.get_extensions()["prompt_id"])
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(reloaded),
        trajectory.TunixTrajectoryMetadata,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="base_step",
          atif_model=trajectory_testing.STEP_1_1,
          project=lambda model: model.to_atif_step(),
      ),
      dict(
          testcase_name="base_metadata",
          atif_model=trajectory_testing.METADATA_1,
          project=lambda model: model.to_atif_metadata(),
      ),
  )
  def test_to_atif_projection_on_base_atif_model_is_noop(
      self, atif_model, project
  ):
    projected_model = project(atif_model)

    self.assertIs(projected_model, atif_model)

  def test_to_atif_metadata_on_tunix_trajectory_strips_steps_and_packs_extra(
      self,
  ):
    atif_metadata = trajectory_testing.TUNIX_TRAJECTORY_1.to_atif_metadata()

    self.assertIs(type(atif_metadata), trajectory.TrajectoryMetadata)
    self.assertEqual(atif_metadata.trajectory_id, "t_atif")
    self.assertEqual(
        atif_metadata.extra,
        {
            "user_meta": "val",
            trajectory.TUNIX_EXTENSIONS_KEY: {
                "prompt_id": "p_1",
                "group_index": 2,
                "target_policy_versions": [2, 3],
                "status": "SUCCEEDED",
                "total_reward": 3.5,
                "masked_out": False,
                "hyperparams": {"temperature": 0.7},
                "env_time": {"step_0": 0.05},
                "reward_time": {"step_1": 0.02},
            },
        },
    )

  def test_to_atif_metadata_on_base_trajectory_strips_steps(self):
    base_atif_metadata = trajectory_testing.TRAJECTORY_1.to_atif_metadata()

    self.assertIs(type(base_atif_metadata), trajectory.TrajectoryMetadata)
    self.assertEqual(base_atif_metadata, trajectory_testing.METADATA_1)

  def test_get_extensions_on_projected_metadata_returns_packed_extensions(self):
    atif_metadata = trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()

    self.assertEqual(
        atif_metadata.get_extensions(),
        {
            "prompt_id": "p_1",
            "group_index": 2,
            "target_policy_versions": [2, 3],
            "status": "SUCCEEDED",
            "total_reward": 3.5,
            "masked_out": False,
            "hyperparams": {"temperature": 0.7},
            "env_time": {"step_0": 0.05},
            "reward_time": {"step_1": 0.02},
        },
    )

  @parameterized.named_parameters(
      dict(testcase_name="none_extra", extra=None),
      dict(testcase_name="user_extra_without_extensions", extra={"k": "v"}),
  )
  def test_get_extensions_without_packed_extensions_returns_empty_dict(
      self, extra
  ):
    metadata = trajectory_testing.METADATA_1.model_copy(update={"extra": extra})

    self.assertEqual(metadata.get_extensions(), {})


class AtifRehydrationTest(trajectory_testing.TrajectoryTestCase):
  """Covers the `from_atif_*()` rehydration of base ATIF into Tunix models."""

  @parameterized.named_parameters(
      dict(
          testcase_name="agent_step",
          step_cls=trajectory.TunixAgentStep,
          tunix_step=trajectory_testing.TUNIX_AGENT_STEP_1,
          expected_extra={"user_key": "val"},
      ),
      dict(
          testcase_name="env_step",
          step_cls=trajectory.TunixEnvStep,
          tunix_step=trajectory_testing.TUNIX_ENV_STEP_0,
          expected_extra={"env_extra_key": "env_extra_val"},
      ),
  )
  def test_from_atif_step_round_trip_restores_tunix_step(
      self, step_cls, tunix_step, expected_extra
  ):
    atif_step = tunix_step.to_atif_step()

    rehydrated_step = step_cls.from_atif_step(atif_step)

    self.assertStepEqual(rehydrated_step, tunix_step)
    self.assertEqual(rehydrated_step.extra, expected_extra)

  @parameterized.named_parameters(
      dict(
          testcase_name="agent_step",
          step_cls=trajectory.TunixAgentStep,
          tunix_step=trajectory_testing.TUNIX_AGENT_STEP_1,
          array_field_names=(
              "assistant_tokens",
              "assistant_masks",
              "logprobs",
              "prefill_routed_experts",
          ),
      ),
      dict(
          testcase_name="env_step",
          step_cls=trajectory.TunixEnvStep,
          tunix_step=trajectory_testing.TUNIX_ENV_STEP_0,
          array_field_names=("env_tokens", "env_masks"),
      ),
  )
  def test_from_atif_step_round_trip_preserves_array_values(
      self, step_cls, tunix_step, array_field_names
  ):
    # `assertStepEqual` compares serialized dumps, where both ndarrays and
    # lists render as lists, so array payloads are checked element-wise here.
    # Only the values survive the round trip; the ndarray container does not.
    rehydrated_step = step_cls.from_atif_step(tunix_step.to_atif_step())

    for field_name in array_field_names:
      np.testing.assert_array_equal(
          getattr(rehydrated_step, field_name),
          getattr(tunix_step, field_name),
          err_msg=f"Field '{field_name}' mismatch",
      )

  def test_from_atif_step_round_trip_restores_unset_tunix_fields(self):
    sparse_agent_step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={
            "mc_return": None,
            "assistant_tokens": None,
            "assistant_masks": None,
            "logprobs": None,
            "prefill_routed_experts": None,
            "prefill_start": None,
            "prefill_num_context": None,
            "extra": None,
        }
    )

    rehydrated_step = trajectory.TunixAgentStep.from_atif_step(
        sparse_agent_step.to_atif_step()
    )

    self.assertStepEqual(rehydrated_step, sparse_agent_step)

  def test_from_atif_trajectory_round_trip_restores_tunix_trajectory(self):
    tunix_trajectory = trajectory_testing.TUNIX_TRAJECTORY_1
    atif_trajectory = trajectory_testing.to_atif_trajectory(tunix_trajectory)

    rehydrated_trajectory = trajectory.TunixTrajectory.from_atif_trajectory(
        atif_trajectory
    )

    self.assertIs(type(rehydrated_trajectory), trajectory.TunixTrajectory)
    # Stores omit subagent trajectories, so the projection carries none and
    # the round trip restores every other field.
    self.assertTrajectoryEqual(
        rehydrated_trajectory,
        tunix_trajectory.model_copy(update={"subagent_trajectories": None}),
    )

  def test_from_atif_trajectory_renumbers_steps_to_zero_indexed(self):
    atif_trajectory = trajectory_testing.to_atif_trajectory(
        trajectory_testing.TUNIX_TRAJECTORY_1
    )
    self.assertEqual([step.step_id for step in atif_trajectory.steps], [1, 2])

    rehydrated_trajectory = trajectory.TunixTrajectory.from_atif_trajectory(
        atif_trajectory
    )

    self.assertEqual(
        [step.step_id for step in rehydrated_trajectory.steps], [0, 1]
    )

  def test_from_atif_trajectory_upcasts_steps_by_source(self):
    atif_trajectory = trajectory_testing.to_atif_trajectory(
        trajectory_testing.TUNIX_TRAJECTORY_1
    )

    rehydrated_trajectory = trajectory.TunixTrajectory.from_atif_trajectory(
        atif_trajectory
    )

    env_step, agent_step = rehydrated_trajectory.steps
    self.assertIs(type(env_step), trajectory.TunixEnvStep)
    self.assertIs(type(agent_step), trajectory.TunixAgentStep)

  def test_from_atif_trajectory_rehydrates_nested_subagent_trajectories(self):
    atif_subagent_trajectory = trajectory_testing.to_atif_trajectory(
        trajectory_testing.TUNIX_SUBAGENT_TRAJECTORY_1
    )
    atif_trajectory = trajectory_testing.to_atif_trajectory(
        trajectory_testing.TUNIX_TRAJECTORY_1
    ).model_copy(update={"subagent_trajectories": [atif_subagent_trajectory]})

    rehydrated_trajectory = trajectory.TunixTrajectory.from_atif_trajectory(
        atif_trajectory
    )

    (rehydrated_subagent,) = rehydrated_trajectory.subagent_trajectories
    self.assertIs(type(rehydrated_subagent), trajectory.TunixTrajectory)
    self.assertEqual(rehydrated_subagent.trajectory_id, "sub_traj_1")
    self.assertEqual(
        [step.step_id for step in rehydrated_subagent.steps], [0, 1]
    )

  def test_from_atif_metadata_round_trip_restores_tunix_metadata(self):
    tunix_metadata = trajectory_testing.TUNIX_METADATA_1
    atif_metadata = tunix_metadata.to_atif_metadata()

    rehydrated_metadata = trajectory.TunixTrajectoryMetadata.from_atif_metadata(
        atif_metadata
    )

    self.assertIs(type(rehydrated_metadata), trajectory.TunixTrajectoryMetadata)
    self.assertEqual(rehydrated_metadata, tunix_metadata)
    self.assertEqual(rehydrated_metadata.extra, {"user_meta": "val"})

  def test_from_atif_metadata_on_base_trajectory_strips_steps_and_rehydrates(
      self,
  ):
    tunix_trajectory = trajectory_testing.TUNIX_TRAJECTORY_1
    atif_trajectory = trajectory_testing.to_atif_trajectory(tunix_trajectory)

    rehydrated_metadata = trajectory.TunixTrajectoryMetadata.from_atif_metadata(
        atif_trajectory
    )

    self.assertIs(type(rehydrated_metadata), trajectory.TunixTrajectoryMetadata)
    self.assertEqual(rehydrated_metadata, tunix_trajectory.get_metadata())

  def test_from_atif_step_without_tunix_extensions_rehydrates_with_defaults(
      self,
  ):
    base_atif_step = trajectory_testing.STEP_1_1

    rehydrated_step = trajectory.TunixAgentStep.from_atif_step(base_atif_step)

    self.assertIsInstance(rehydrated_step, trajectory.TunixAgentStep)
    self.assertEqual(rehydrated_step.step_id, base_atif_step.step_id - 1)
    self.assertIsNone(rehydrated_step.mc_return)
    self.assertIsNone(rehydrated_step.extra)

  def test_from_atif_metadata_without_tunix_extensions_rehydrates_with_defaults(
      self,
  ):
    base_atif_metadata = trajectory_testing.METADATA_1

    rehydrated_metadata = trajectory.TunixTrajectoryMetadata.from_atif_metadata(
        base_atif_metadata
    )

    self.assertIsInstance(
        rehydrated_metadata, trajectory.TunixTrajectoryMetadata
    )
    self.assertEqual(
        rehydrated_metadata.trajectory_id, base_atif_metadata.trajectory_id
    )
    self.assertIsNone(rehydrated_metadata.prompt_id)
    self.assertIsNone(rehydrated_metadata.extra)

  def test_base_trajectory_metadata_from_atif_metadata(self):
    base_meta = trajectory_testing.METADATA_1
    self.assertIs(
        trajectory.TrajectoryMetadata.from_atif_metadata(base_meta), base_meta
    )

    class CustomMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      tag: str = "default_tag"

    try:
      rehydrated_subclass = CustomMetadata.from_atif_metadata(base_meta)
      self.assertIsInstance(rehydrated_subclass, CustomMetadata)
      self.assertEqual(rehydrated_subclass.tag, "default_tag")
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(CustomMetadata, None)

  def test_pydantic_init_subclass_registers_metadata_subclasses_only(self):
    self.assertIn(
        trajectory.TunixTrajectoryMetadata,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )
    self.assertEqual(
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS[
            trajectory.TunixTrajectoryMetadata
        ],
        frozenset({
            "prompt_id",
            "group_index",
            "target_policy_versions",
            "status",
            "total_reward",
            "masked_out",
            "hyperparams",
            "env_time",
            "reward_time",
        }),
    )
    self.assertNotIn(
        trajectory.TrajectoryMetadata,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )
    self.assertNotIn(
        trajectory.Trajectory,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )
    self.assertNotIn(
        trajectory.TunixTrajectory,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )

    class _EmptyMetadataSubclass(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"

    class _CustomTrajectorySubclass(trajectory.Trajectory):
      custom_traj_field: str = "ignored"

    self.assertNotIn(
        _EmptyMetadataSubclass,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )
    self.assertNotIn(
        _CustomTrajectorySubclass,
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS,
    )

  def test_resolve_subclass_with_overlapping_fields(self):
    base_meta = trajectory_testing.METADATA_1
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(base_meta),
        trajectory.TrajectoryMetadata,
    )
    tunix_atif_meta = trajectory_testing.TUNIX_METADATA_1.to_atif_metadata()
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(tunix_atif_meta),
        trajectory.TunixTrajectoryMetadata,
    )

    class SubclassA(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      shared_tag: str = "a"
      only_a: int = 1

    class SubclassB(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      shared_tag: str = "b"
      only_b: int = 2

    try:
      meta_a = SubclassA(
          trajectory_id="a",
          agent=trajectory.Agent(name="agent", version="1.0"),
      ).to_atif_metadata()
      meta_b = SubclassB(
          trajectory_id="b",
          agent=trajectory.Agent(name="agent", version="1.0"),
      ).to_atif_metadata()
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(meta_a),
          SubclassA,
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(meta_b),
          SubclassB,
      )
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(SubclassA, None)
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(SubclassB, None)

  def test_resolve_subclass_hierarchical_inheritance_and_definition_order(self):
    # Define a wider subclass BEFORE a narrower subclass, and a parent/child
    # subclass hierarchy, to verify that `resolve_subclass` always picks the
    # most specific matching subclass regardless of definition order.
    class WideMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      hier_field_1: str = "w1"
      hier_field_2: int | None = None
      hier_field_3: bool = True

    class ParentMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      hier_field_1: str = "p1"

    class ChildMetadata(ParentMetadata):
      hier_field_2: int = 42

    try:
      parent_orig = ParentMetadata(
          trajectory_id="parent",
          agent=trajectory.Agent(name="agent", version="1.0"),
          hier_field_1="parent_val",
      )
      child_orig = ChildMetadata(
          trajectory_id="child",
          agent=trajectory.Agent(name="agent", version="1.0"),
          hier_field_1="child_val",
          hier_field_2=99,
      )
      wide_orig = WideMetadata(
          trajectory_id="wide",
          agent=trajectory.Agent(name="agent", version="1.0"),
          hier_field_1="wide_val",
          hier_field_2=7,
          hier_field_3=False,
      )

      parent_atif = parent_orig.to_atif_metadata()
      child_atif = child_orig.to_atif_metadata()
      wide_atif = wide_orig.to_atif_metadata()

      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(parent_atif),
          ParentMetadata,
      )
      self.assertEqual(
          ParentMetadata.from_atif_metadata(parent_atif), parent_orig
      )

      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(child_atif),
          ChildMetadata,
      )
      self.assertEqual(ChildMetadata.from_atif_metadata(child_atif), child_orig)

      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(wide_atif),
          WideMetadata,
      )
      self.assertEqual(WideMetadata.from_atif_metadata(wide_atif), wide_orig)
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(WideMetadata, None)
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(ParentMetadata, None)
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(ChildMetadata, None)

  def test_resolve_subclass_on_already_typed_instances_and_trajectories(self):
    class BehaviorOnlyMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"

    behav_meta = BehaviorOnlyMetadata(
        trajectory_id="b1",
        agent=trajectory.Agent(name="agent", version="1.0"),
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(behav_meta),
        BehaviorOnlyMetadata,
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(
            trajectory_testing.TUNIX_METADATA_1
        ),
        trajectory.TunixTrajectoryMetadata,
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(
            trajectory_testing.TUNIX_TRAJECTORY_1
        ),
        trajectory.TunixTrajectoryMetadata,
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(
            trajectory_testing.TRAJECTORY_1
        ),
        trajectory.TrajectoryMetadata,
    )
    unknown_ext_meta = trajectory_testing.METADATA_1.model_copy(
        update={
            "extra": {
                trajectory.TUNIX_EXTENSIONS_KEY: {
                    "unregistered_ext_key": "value"
                }
            }
        }
    )
    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(unknown_ext_meta),
        trajectory.TrajectoryMetadata,
    )

  def test_resolve_subclass_with_optional_only_and_overlapping_none_fields(
      self,
  ):
    class OptionalOnlyMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      opt_field: str | None = None

    class SiblingA(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      shared_tag: str = "shared"
      opt_a: str | None = None

    class SiblingB(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      shared_tag: str = "shared"
      opt_b: str | None = None

    class ParentOptMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      parent_tag: str = "p"

    class ChildOptMetadata(ParentOptMetadata):
      child_opt: int | None = None

    try:
      # 1. Optional-only subclass logged when all extension fields are None.
      opt_orig = OptionalOnlyMetadata(
          trajectory_id="opt_1",
          agent=trajectory.Agent(name="agent", version="1.0"),
      )
      opt_atif = trajectory.TrajectoryMetadata.model_validate_json(
          opt_orig.to_atif_metadata().model_dump_json(exclude_none=True)
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(opt_atif),
          OptionalOnlyMetadata,
      )
      self.assertEqual(
          OptionalOnlyMetadata.from_atif_metadata(opt_atif), opt_orig
      )

      # 2. Sibling subclasses with overlapping `shared_tag` and unset optional
      # distinguishing fields (`opt_a=None` vs `opt_b=None`).
      sib_a_orig = SiblingA(
          trajectory_id="sib_a",
          agent=trajectory.Agent(name="agent", version="1.0"),
          shared_tag="same",
      )
      sib_b_orig = SiblingB(
          trajectory_id="sib_b",
          agent=trajectory.Agent(name="agent", version="1.0"),
          shared_tag="same",
      )
      sib_a_atif = trajectory.TrajectoryMetadata.model_validate_json(
          sib_a_orig.to_atif_metadata().model_dump_json(exclude_none=True)
      )
      sib_b_atif = trajectory.TrajectoryMetadata.model_validate_json(
          sib_b_orig.to_atif_metadata().model_dump_json(exclude_none=True)
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(sib_a_atif), SiblingA
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(sib_b_atif), SiblingB
      )

      # 3. Child subclass whose added optional field is None does not downgrade
      # to parent subclass.
      child_orig = ChildOptMetadata(
          trajectory_id="child_none",
          agent=trajectory.Agent(name="agent", version="1.0"),
          parent_tag="p_val",
          child_opt=None,
      )
      child_atif = trajectory.TrajectoryMetadata.model_validate_json(
          child_orig.to_atif_metadata().model_dump_json(exclude_none=True)
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(child_atif),
          ChildOptMetadata,
      )
      self.assertEqual(
          ChildOptMetadata.from_atif_metadata(child_atif), child_orig
      )
    finally:
      for cls_to_remove in (
          OptionalOnlyMetadata,
          SiblingA,
          SiblingB,
          ParentOptMetadata,
          ChildOptMetadata,
      ):
        trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(cls_to_remove, None)

  def test_resolve_subclass_and_from_atif_metadata_schema_evolution(self):
    ext_key = "_evolved_extensions"

    class EvolvedV2Metadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = ext_key
      evolved_field_1: str = "v1"
      evolved_field_2: int = 0
      newly_added_v2_field: str | None = "v2_default"

    try:
      # 1. Forward schema evolution (field added in v2): persisted v1 payload
      # only contains {evolved_field_1, evolved_field_2}.
      v1_persisted_missing_new_field = trajectory.TrajectoryMetadata(
          trajectory_id="schema_add",
          agent=trajectory.Agent(name="agent", version="1.0"),
          extra={
              ext_key: {
                  "evolved_field_1": "from_v1",
                  "evolved_field_2": 7,
              }
          },
      )
      resolved_cls = trajectory.TrajectoryMetadata.resolve_subclass(
          v1_persisted_missing_new_field
      )
      self.assertIs(resolved_cls, EvolvedV2Metadata)
      rehydrated_added = resolved_cls.from_atif_metadata(
          v1_persisted_missing_new_field
      )
      self.assertIsInstance(rehydrated_added, EvolvedV2Metadata)
      self.assertEqual(rehydrated_added.evolved_field_1, "from_v1")
      self.assertEqual(rehydrated_added.evolved_field_2, 7)
      self.assertEqual(rehydrated_added.newly_added_v2_field, "v2_default")
      self.assertIsNone(rehydrated_added.extra)

      # 2. Backward schema evolution (field removed in v2): persisted v1 payload
      # contains all v2 fields plus a removed `deprecated_v1_field`.
      v1_persisted_with_removed_field = trajectory.TrajectoryMetadata(
          trajectory_id="schema_remove",
          agent=trajectory.Agent(name="agent", version="1.0"),
          extra={
              ext_key: {
                  "evolved_field_1": "from_v1",
                  "evolved_field_2": 9,
                  "newly_added_v2_field": "explicit_v2",
                  "deprecated_v1_field": "legacy_val",
              }
          },
      )
      resolved_removed_cls = trajectory.TrajectoryMetadata.resolve_subclass(
          v1_persisted_with_removed_field
      )
      self.assertIs(resolved_removed_cls, EvolvedV2Metadata)
      rehydrated_removed = resolved_removed_cls.from_atif_metadata(
          v1_persisted_with_removed_field
      )
      self.assertIsInstance(rehydrated_removed, EvolvedV2Metadata)
      self.assertEqual(rehydrated_removed.evolved_field_1, "from_v1")
      self.assertEqual(rehydrated_removed.evolved_field_2, 9)
      self.assertEqual(rehydrated_removed.newly_added_v2_field, "explicit_v2")
      self.assertEqual(
          rehydrated_removed.extra,
          {ext_key: {"deprecated_v1_field": "legacy_val"}},
      )
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(
          EvolvedV2Metadata, None
      )

  def test_resolve_subclass_and_round_trip_with_custom_client_extensions_key(
      self,
  ):
    class ClientAMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_client_a_extensions"
      shared_field: str = "a"
      status: str | None = None

    class ClientBMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_client_b_extensions"
      shared_field: str = "b"
      status: str | None = None

    try:
      orig_a = ClientAMetadata(
          trajectory_id="client_a_1",
          agent=trajectory.Agent(name="agent", version="1.0"),
          shared_field="val_a",
          status="RUNNING",
          extra={"user_meta": "keep_me"},
      )
      orig_b = ClientBMetadata(
          trajectory_id="client_b_1",
          agent=trajectory.Agent(name="agent", version="1.0"),
          shared_field="val_b",
          status="SUCCEEDED",
      )

      atif_a = orig_a.to_atif_metadata()
      atif_b = orig_b.to_atif_metadata()

      self.assertEqual(
          atif_a.extra,
          {
              "user_meta": "keep_me",
              "_client_a_extensions": {
                  "shared_field": "val_a",
                  "status": "RUNNING",
              },
          },
      )
      self.assertEqual(
          atif_a.get_extensions(),
          {"shared_field": "val_a", "status": "RUNNING"},
      )
      self.assertEqual(
          atif_b.get_extensions(),
          {"shared_field": "val_b", "status": "SUCCEEDED"},
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(atif_a),
          ClientAMetadata,
      )
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(atif_b),
          ClientBMetadata,
      )
      self.assertEqual(ClientAMetadata.from_atif_metadata(atif_a), orig_a)
      self.assertEqual(ClientBMetadata.from_atif_metadata(atif_b), orig_b)
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(ClientAMetadata, None)
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(ClientBMetadata, None)

  def test_subclasses_must_define_extensions_key(self):
    self.assertIsNone(getattr(trajectory.Step, "EXTENSIONS_KEY", None))
    self.assertIsNone(
        getattr(trajectory.TrajectoryMetadata, "EXTENSIONS_KEY", None)
    )
    self.assertIsNone(getattr(trajectory.Trajectory, "EXTENSIONS_KEY", None))
    self.assertEqual(
        trajectory.TunixAgentStep.EXTENSIONS_KEY,
        trajectory.TUNIX_EXTENSIONS_KEY,
    )
    self.assertEqual(
        trajectory.TunixEnvStep.EXTENSIONS_KEY,
        trajectory.TUNIX_EXTENSIONS_KEY,
    )
    self.assertEqual(
        trajectory.TunixTrajectoryMetadata.EXTENSIONS_KEY,
        trajectory.TUNIX_EXTENSIONS_KEY,
    )
    self.assertEqual(
        trajectory.TunixTrajectory.EXTENSIONS_KEY,
        trajectory.TUNIX_EXTENSIONS_KEY,
    )

    with self.assertRaisesRegex(
        TypeError,
        "MissingKeyMetadata must define a non-empty string 'EXTENSIONS_KEY'",
    ):

      class MissingKeyMetadata(trajectory.TrajectoryMetadata):  # pylint: disable=unused-variable
        custom_field: str = "x"

    with self.assertRaisesRegex(
        TypeError,
        "EmptyKeyMetadata must define a non-empty string 'EXTENSIONS_KEY'",
    ):

      class EmptyKeyMetadata(trajectory.TrajectoryMetadata):  # pylint: disable=unused-variable
        EXTENSIONS_KEY = ""
        custom_field: str = "x"

    with self.assertRaisesRegex(
        TypeError,
        "MissingKeyStep must define a non-empty string 'EXTENSIONS_KEY'",
    ):

      class MissingKeyStep(trajectory.Step):  # pylint: disable=unused-variable
        custom_step_field: int = 1

  def test_metadata_subclass_can_define_steps_field(self):
    class StepCountMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      steps: int = 0

    try:
      self.assertIn(
          StepCountMetadata, trajectory.TrajectoryMetadata._SUBCLASS_FIELDS
      )
      orig = StepCountMetadata(
          trajectory_id="step_count_1",
          agent=trajectory.Agent(name="agent", version="1.0"),
          steps=42,
      )
      atif = orig.to_atif_metadata()
      self.assertIs(
          trajectory.TrajectoryMetadata.resolve_subclass(atif),
          StepCountMetadata,
      )
      self.assertEqual(StepCountMetadata.from_atif_metadata(atif), orig)
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(
          StepCountMetadata, None
      )

  def test_create_paired_trajectory_preserves_explicit_none_over_non_none_default(
      self,
  ):
    class DefaultedTagMetadata(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      tag: str | None = "default_tag"

      def create_trajectory(
          self,
          steps: list[trajectory.Step] | None = None,
          subagent_trajectories: list["DefaultedTagTrajectory"] | None = None,
      ) -> "DefaultedTagTrajectory":
        return self._create_paired_trajectory(
            DefaultedTagTrajectory, steps, subagent_trajectories
        )

    class DefaultedTagTrajectory(trajectory.Trajectory, DefaultedTagMetadata):
      pass

    try:
      meta = DefaultedTagMetadata(
          trajectory_id="explicit_none_tag",
          agent=trajectory.Agent(name="agent", version="1.0"),
          tag=None,
      )
      traj = meta.create_trajectory([trajectory_testing.STEP_1_1])
      self.assertIsInstance(traj, DefaultedTagTrajectory)
      self.assertIsNone(traj.tag)
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(
          DefaultedTagMetadata, None
      )

  def test_resolve_subclass_breaks_ties_in_favor_of_target_cls(self):
    class DuplicateFieldsA(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      shared_metric: int = 1

    class DuplicateFieldsB(trajectory.TrajectoryMetadata):
      EXTENSIONS_KEY = "_test_extensions"
      shared_metric: int = 2

    try:
      atif = DuplicateFieldsB(
          trajectory_id="tie_break_1",
          agent=trajectory.Agent(name="agent", version="1.0"),
          shared_metric=99,
      ).to_atif_metadata()
      self.assertIs(DuplicateFieldsA.resolve_subclass(atif), DuplicateFieldsA)
      self.assertIs(DuplicateFieldsB.resolve_subclass(atif), DuplicateFieldsB)
    finally:
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(DuplicateFieldsA, None)
      trajectory.TrajectoryMetadata._SUBCLASS_FIELDS.pop(DuplicateFieldsB, None)

  @parameterized.named_parameters(
      dict(
          testcase_name="tunix_agent_step",
          tunix_model=trajectory_testing.TUNIX_AGENT_STEP_1,
          rehydrate=trajectory.TunixAgentStep.from_atif_step,
      ),
      dict(
          testcase_name="tunix_env_step",
          tunix_model=trajectory_testing.TUNIX_ENV_STEP_0,
          rehydrate=trajectory.TunixEnvStep.from_atif_step,
      ),
      dict(
          testcase_name="tunix_metadata",
          tunix_model=trajectory_testing.TUNIX_METADATA_1,
          rehydrate=trajectory.TunixTrajectoryMetadata.from_atif_metadata,
      ),
      dict(
          testcase_name="tunix_trajectory",
          tunix_model=trajectory_testing.TUNIX_TRAJECTORY_1,
          rehydrate=trajectory.TunixTrajectoryMetadata.from_atif_metadata,
      ),
      dict(
          testcase_name="tunix_trajectory_from_trajectory",
          tunix_model=trajectory_testing.TUNIX_TRAJECTORY_1,
          rehydrate=trajectory.TunixTrajectory.from_atif_trajectory,
      ),
  )
  def test_from_atif_on_already_tunix_model_is_noop(
      self, tunix_model, rehydrate
  ):
    self.assertIs(rehydrate(tunix_model), tunix_model)

  def test_from_atif_step_with_custom_step_id_offset(self):
    tunix_step = trajectory_testing.TUNIX_AGENT_STEP_1
    atif_step = tunix_step.to_atif_step(step_id_offset=0)

    rehydrated_step = trajectory.TunixAgentStep.from_atif_step(
        atif_step, step_id_offset=0
    )

    self.assertEqual(rehydrated_step.step_id, tunix_step.step_id)

  def test_from_atif_step_on_zero_indexed_step_raises_value_error(self):
    # `from_atif_step` assumes the 1-indexed ATIF numbering. An already
    # 0-indexed step is rejected instead of silently underflowing to -1.
    zero_indexed_atif_step = trajectory_testing.STEP_1_1.model_copy(
        update={"step_id": 0}
    )

    with self.assertRaisesRegex(
        ValueError, "Expected a 1-indexed ATIF step_id, got 0"
    ):
      trajectory.TunixAgentStep.from_atif_step(zero_indexed_atif_step)

  def test_from_atif_step_with_foreign_tunix_extensions_key_keeps_it_nested(
      self,
  ):
    # `TUNIX_EXTENSIONS_KEY` is reserved by convention alone. Only fields the
    # Tunix subclass actually declares are promoted, so a key a caller stashed
    # there stays nested instead of tripping `extra="forbid"`.
    atif_step = trajectory_testing.STEP_1_1.model_copy(
        update={
            "extra": {
                "user_key": "user_val",
                trajectory.TUNIX_EXTENSIONS_KEY: {"caller_key": "caller_val"},
            }
        }
    )

    rehydrated_step = trajectory.TunixAgentStep.from_atif_step(atif_step)

    self.assertEqual(
        rehydrated_step.extra,
        {
            "user_key": "user_val",
            trajectory.TUNIX_EXTENSIONS_KEY: {"caller_key": "caller_val"},
        },
    )

  def test_from_atif_step_does_not_mutate_source_step_extra(self):
    # Unpacking pops promoted fields out of the nested extension dict, so it
    # must work on a copy rather than the dict the source step holds.
    atif_step = trajectory_testing.TUNIX_AGENT_STEP_1.to_atif_step()
    extra_before = copy.deepcopy(atif_step.extra)

    trajectory.TunixAgentStep.from_atif_step(atif_step)

    self.assertEqual(atif_step.extra, extra_before)

  def test_from_atif_step_with_foreign_tunix_extensions_key_does_not_shadow_field(
      self,
  ):
    # A foreign key naming a real field must not be promoted over the value the
    # step already carries.
    atif_step = trajectory_testing.STEP_1_1.model_copy(
        update={
            "extra": {trajectory.TUNIX_EXTENSIONS_KEY: {"message": "spoofed"}}
        }
    )

    rehydrated_step = trajectory.TunixAgentStep.from_atif_step(atif_step)

    self.assertEqual(
        rehydrated_step.message, trajectory_testing.STEP_1_1.message
    )
    self.assertEqual(
        rehydrated_step.extra,
        {trajectory.TUNIX_EXTENSIONS_KEY: {"message": "spoofed"}},
    )

  def test_atif_step_round_trip_with_foreign_tunix_extensions_key_is_lossless(
      self,
  ):
    # Packing merges foreign keys into `TUNIX_EXTENSIONS_KEY` rather than
    # overwriting, so unpacking must leave them behind for the projection to be
    # an inverse.
    tunix_step = trajectory_testing.TUNIX_AGENT_STEP_1.model_copy(
        update={
            "extra": {
                "user_key": "user_val",
                trajectory.TUNIX_EXTENSIONS_KEY: {"caller_key": "caller_val"},
            }
        }
    )

    round_tripped_step = trajectory.TunixAgentStep.from_atif_step(
        tunix_step.to_atif_step()
    )

    self.assertStepEqual(round_tripped_step, tunix_step)

  def test_atif_metadata_round_trip_with_foreign_tunix_extensions_key_is_lossless(
      self,
  ):
    tunix_metadata = trajectory_testing.TUNIX_METADATA_1.model_copy(
        update={
            "extra": {
                "user_meta": "val",
                trajectory.TUNIX_EXTENSIONS_KEY: {"caller_key": "caller_val"},
            }
        }
    )
    atif_metadata = tunix_metadata.to_atif_metadata()

    self.assertIs(
        trajectory.TrajectoryMetadata.resolve_subclass(atif_metadata),
        trajectory.TunixTrajectoryMetadata,
    )
    round_tripped_metadata = (
        trajectory.TunixTrajectoryMetadata.from_atif_metadata(atif_metadata)
    )

    self.assertEqual(
        round_tripped_metadata.model_dump(), tunix_metadata.model_dump()
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="agent_class_rejects_user_source",
          step_cls=trajectory.TunixAgentStep,
          source=trajectory.Source.USER,
          expected_error=(
              "TunixAgentStep is only applicable when source is 'agent'"
          ),
      ),
      dict(
          testcase_name="env_class_rejects_agent_source",
          step_cls=trajectory.TunixEnvStep,
          source=trajectory.Source.AGENT,
          expected_error=(
              "TunixEnvStep is only applicable when source is 'system' or"
              " 'user'"
          ),
      ),
  )
  def test_from_atif_step_with_mismatched_source_raises_value_error(
      self, step_cls, source, expected_error
  ):
    atif_step = trajectory_testing.STEP_1_1.model_copy(
        update={"source": source}
    )

    with self.assertRaisesRegex(ValueError, expected_error):
      step_cls.from_atif_step(atif_step)


class Int3DArraySerializationTest(parameterized.TestCase):
  """Tests compact base64 binary serialization and validation for Int3DArray."""

  def test_int_3d_array_serializes_to_base64_blob_and_round_trips(self):
    arr = np.arange(24, dtype=np.int16).reshape(3, 4, 2)
    step = trajectory.TunixAgentStep(
        step_id=0,
        source=trajectory.Source.AGENT,
        message="routed",
        prefill_routed_experts=arr,
        prefill_start=0,
        prefill_num_context=1,
    )

    dumped = step.model_dump(mode="json")
    blob = dumped["prefill_routed_experts"]
    self.assertEqual(blob["shape"], [3, 4, 2])
    self.assertEqual(blob["dtype"], "int16")
    self.assertIsInstance(blob["data"], str)

    restored = trajectory.TunixAgentStep.model_validate(dumped)
    self.assertIsInstance(restored.prefill_routed_experts, np.ndarray)
    self.assertEqual(restored.prefill_routed_experts.dtype, np.int16)
    np.testing.assert_array_equal(restored.prefill_routed_experts, arr)

  def test_int_3d_array_empty_shape_round_trips(self):
    arr = np.zeros((0, 4, 2), dtype=np.int16)
    step = trajectory.TunixAgentStep(
        step_id=0,
        source=trajectory.Source.AGENT,
        message="empty_routed",
        prefill_routed_experts=arr,
        prefill_start=0,
        prefill_num_context=0,
    )
    dumped = step.model_dump(mode="json")
    self.assertEqual(
        dumped["prefill_routed_experts"],
        {"shape": [0, 4, 2], "dtype": "int16", "data": ""},
    )
    restored = trajectory.TunixAgentStep.model_validate(dumped)
    np.testing.assert_array_equal(restored.prefill_routed_experts, arr)

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_ndim",
          bad_value=np.zeros((3, 4), dtype=np.int16),
          expected_error="Expected a 3D array",
      ),
      dict(
          testcase_name="float_array",
          bad_value=np.zeros((1, 2, 2), dtype=np.float32),
          expected_error="Expected an integer array",
      ),
      dict(
          testcase_name="out_of_int16_range",
          bad_value=[[[40000, 1]]],
          expected_error="exceed int16 range",
      ),
      dict(
          testcase_name="bad_blob_keys",
          bad_value={"shape": [1, 1, 1], "dtype": "int16"},
          expected_error="Invalid serialized Int3DArray keys",
      ),
      dict(
          testcase_name="bad_blob_dtype",
          bad_value={"shape": [1, 1, 1], "dtype": "int32", "data": "AAAA"},
          expected_error="Unsupported Int3DArray dtype",
      ),
      dict(
          testcase_name="bad_blob_shape",
          bad_value={"shape": [1, -1, 1], "dtype": "int16", "data": ""},
          expected_error="Invalid Int3DArray shape",
      ),
      dict(
          testcase_name="bad_base64",
          bad_value={"shape": [1, 1, 1], "dtype": "int16", "data": "%%%"},
          expected_error="Failed to decode Int3DArray blob",
      ),
      dict(
          testcase_name="byte_length_mismatch",
          bad_value={"shape": [2, 1, 1], "dtype": "int16", "data": "AA=="},
          expected_error="does not match shape",
      ),
  )
  def test_int_3d_array_rejects_invalid_inputs(self, bad_value, expected_error):
    with self.assertRaisesRegex(ValueError, expected_error):
      trajectory.TunixAgentStep(
          step_id=0,
          source=trajectory.Source.AGENT,
          message="invalid",
          prefill_routed_experts=bad_value,
          prefill_start=0,
          prefill_num_context=0,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_start_and_num_context",
          kwargs=dict(
              prefill_routed_experts=np.zeros((2, 2, 1), dtype=np.int16)
          ),
          expected_error="either all None or all set",
      ),
      dict(
          testcase_name="missing_routed_experts",
          kwargs=dict(prefill_start=0, prefill_num_context=1),
          expected_error="either all None or all set",
      ),
      dict(
          testcase_name="negative_prefill_start",
          kwargs=dict(
              prefill_routed_experts=np.zeros((2, 2, 1), dtype=np.int16),
              prefill_start=-1,
              prefill_num_context=1,
          ),
          expected_error="prefill_start must be >= 0",
      ),
      dict(
          testcase_name="num_context_exceeds_rows",
          kwargs=dict(
              prefill_routed_experts=np.zeros((2, 2, 1), dtype=np.int16),
              prefill_start=0,
              prefill_num_context=3,
          ),
          expected_error="prefill_num_context must be in",
      ),
  )
  def test_agent_step_rejects_invalid_prefill_metadata(
      self, kwargs, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      trajectory.TunixAgentStep(
          step_id=0,
          source=trajectory.Source.AGENT,
          message="bad_prefill",
          **kwargs,
      )

  def test_add_step_rejects_prefill_fields_for_non_agent_source(self):
    traj = trajectory.TunixTrajectory(
        session_id="s1",
        agent=trajectory.Agent(name="a", version="1"),
    )
    with self.assertRaisesRegex(
        ValueError, "only valid when source is 'agent'"
    ):
      traj.add_step(
          trajectory.Source.USER,
          "env",
          prefill_routed_experts=np.zeros((1, 2, 1), dtype=np.int16),
          prefill_start=0,
          prefill_num_context=0,
      )


if __name__ == "__main__":
  absltest.main()
