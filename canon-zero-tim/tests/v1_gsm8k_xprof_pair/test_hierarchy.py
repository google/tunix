#!/usr/bin/env python3
"""Host and synthetic gates for the P60-2 XProf hierarchy."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


HIERARCHY = _load(
    "v1_gsm8k_xprof_hierarchy",
    TASK / "scripts/census_gsm8k_xprof_hierarchy.py",
)
TRACE_CENSUS = _load(
    "v1_gsm8k_xprof_trace",
    TASK / "scripts/census_gsm8k_xprof_trace.py",
)
GSM8K_XPROF = _load("p60_gsm8k_xprof", ROOT / "tunix/rl/gsm8k_xprof.py")


def _span(name, start, duration, *, line_name="python3", **stats):
  return HIERARCHY.Span(name, start, duration, line_name, stats)


def _fixture(groups: int = 16):
  """One synthetic well-formed update at the requested group count.

  Every offset derives from `groups` so the same builder exercises the
  16-group DP4xTP1 layout and the 32-group DP2xTP2 layout: the forward
  parent closes before loss_pullback, which closes before the first train,
  and the optimizer commit is owned by the last train.
  """
  forward_end = 30 + groups * 10
  loss_start = forward_end + 10
  train_base = loss_start + 30
  optimizer_start = train_base + groups * 35 + 5
  update_end = optimizer_start + 70
  spans = [
      _span("zero_tim_update", 0, update_end, update_step=2),
      _span("forward_groups", 10, forward_end - 10),
      _span("loss_pullback", loss_start, 10),
      _span("optimizer_commit", optimizer_start, 50, update_step=2),
  ]
  for index in range(groups):
    spans.append(_span("forward_group", 25 + index * 10, 5, group_index=index))
    train_start = train_base + index * 35
    train_duration = (
        optimizer_start + 55 - train_start if index == groups - 1 else 32
    )
    spans.append(_span(
        "train",
        train_start,
        train_duration,
        _r="1",
        step_num=2 * groups + index,
    ))
    start = train_start + 1
    spans.extend((
        _span("reverse_group", start, 30, group_index=index),
        _span("replay_forward", start + 1, 4, group_index=index),
        _span("model_backward", start + 6, 8, group_index=index),
        _span("report_adjoint", start + 16, 4, group_index=index),
        _span("fixed_dp_reduce", start + 21, 4, group_index=index),
        _span(
            "gradient_accumulate",
            start + 26,
            4,
            group_index=index,
            micro_step=index,
            is_last_accumulate=int(index == groups - 1),
        ),
    ))
  device_steps = {f"/device:TPU:{index}": 100 for index in range(8)}
  compiler_counts = {name: 0 for name in HIERARCHY.COMPILER_EVENTS}
  return spans, device_steps, compiler_counts


class HierarchyTest(unittest.TestCase):

  def test_parent_annotations_are_constructed_after_trace_start(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    end = learner.index("segmented_result = self._run_p28_g6_update(")
    seam = learner[learner.rfind("marker_prefix", 0, end):end]
    start = seam.index("_canon_xprof_update_entry()")
    self.assertLess(start, seam.index("xprof_train_schedule ="))
    self.assertLess(start, seam.index("train_step_annotation ="))
    self.assertLess(start, seam.index("update_annotation ="))

  def test_microstep_and_optimizer_metadata_are_wired_to_runtime(self):
    def annotation_calls(path):
      tree = ast.parse(path.read_text())
      calls = {}
      for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "trace_annotation"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
          calls.setdefault(node.args[0].value, []).append(node)
      return calls

    adapter_calls = annotation_calls(
        ROOT / "tunix/rl/canonical_qwen3_adapter.py"
    )
    accumulator = adapter_calls["gradient_accumulate"][0]
    accumulator_metadata = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in accumulator.keywords
    }
    self.assertEqual(accumulator_metadata, {
        "group_index": "index",
        "micro_step": "index",
        "is_last_accumulate": "int(index == len(specs) - 1)",
    })
    learner_calls = annotation_calls(
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    )
    optimizer = learner_calls["optimizer_commit"][0]
    optimizer_metadata = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in optimizer.keywords
    }
    self.assertEqual(
        optimizer_metadata,
        {"update_step": "self.rl_cluster.global_steps"},
    )
    adapter = (
        ROOT / "tunix/rl/canonical_qwen3_adapter.py"
    ).read_text()
    self.assertIn("xprof_train_schedule.transaction(index)", adapter)
    self.assertNotIn(
        'with gsm8k_xprof.trace_annotation("reverse_groups"):',
        adapter[adapter.index("  def segmented_dp_grpo_value_and_grad("):],
    )
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    self.assertIn("xprof_train_schedule.optimizer_commit()", learner)

  def test_label_flag_noop_positive_and_fail_closed(self):
    for value in (None, "", "0"):
      environment = {} if value is None else {"CANON_XPROF_LABELS": value}
      with self.subTest(value=value), mock.patch.dict(
          os.environ, environment, clear=True
      ):
        self.assertFalse(GSM8K_XPROF.labels_enabled())
        self.assertIsInstance(
            GSM8K_XPROF.trace_annotation("forward_group", group_index=0),
            contextlib.nullcontext,
        )
        self.assertIsInstance(
            GSM8K_XPROF.train_step_annotation(step_num=1),
            contextlib.nullcontext,
        )

    calls = []
    fake_jax = types.SimpleNamespace(
        profiler=types.SimpleNamespace(
            TraceAnnotation=lambda name, **stats: calls.append(
                ("trace", name, stats)
            ) or contextlib.nullcontext(),
            StepTraceAnnotation=lambda name, **stats: calls.append(
                ("step", name, stats)
            ) or contextlib.nullcontext(),
        )
    )
    with mock.patch.dict(
        os.environ, {"CANON_XPROF_LABELS": "1"}, clear=True
    ), mock.patch.dict(sys.modules, {"jax": fake_jax}):
      GSM8K_XPROF.trace_annotation("reverse_group", group_index=3)
      GSM8K_XPROF.trace_annotation(
          "gradient_accumulate",
          group_index=15,
          micro_step=15,
          is_last_accumulate=1,
      )
      GSM8K_XPROF.trace_annotation("optimizer_commit", update_step=2)
      GSM8K_XPROF.train_step_annotation(step_num=32)
    self.assertEqual(calls, [
        ("trace", "reverse_group", {"group_index": 3}),
        (
            "trace",
            "gradient_accumulate",
            {
                "group_index": 15,
                "micro_step": 15,
                "is_last_accumulate": 1,
            },
        ),
        ("trace", "optimizer_commit", {"update_step": 2}),
        ("step", "train", {"step_num": 32}),
    ])

    with mock.patch.dict(
        os.environ, {"CANON_XPROF_LABELS": "invalid"}, clear=True
    ):
      with self.assertRaisesRegex(ValueError, "must be unset/0/1"):
        GSM8K_XPROF.trace_annotation("forward_groups")
    with mock.patch.dict(
        os.environ, {"CANON_XPROF_LABELS": "1"}, clear=True
    ):
      with self.assertRaisesRegex(ValueError, "invalid XProf annotation name"):
        GSM8K_XPROF.trace_annotation("group/123")
      with self.assertRaisesRegex(ValueError, "integer-valued"):
        GSM8K_XPROF.trace_annotation("forward_group", group_index="3")

  def test_pure_hierarchy_validator_positive(self):
    spans, device_steps, compiler_counts = _fixture()
    self.assertEqual(
        HIERARCHY.validate_hierarchy(
            spans,
            device_step_counts=device_steps,
            compiler_counts=compiler_counts,
            expected_update_step=2,
        ),
        [],
    )
    self.assertEqual(
        TRACE_CENSUS.validate_trace(
            spans, compiler_counts=compiler_counts, expected_update_step=2
        ),
        [],
    )

  def test_pure_hierarchy_validator_covers_the_dp2_geometry(self):
    """The same interval semantics hold at the registered 32-group count."""
    spans, device_steps, compiler_counts = _fixture(groups=32)
    self.assertEqual(
        HIERARCHY.validate_hierarchy(
            spans,
            device_step_counts=device_steps,
            compiler_counts=compiler_counts,
            expected_update_step=2,
            expected_groups=32,
        ),
        [],
    )
    self.assertEqual(
        TRACE_CENSUS.validate_trace(
            spans,
            compiler_counts=compiler_counts,
            expected_update_step=2,
            expected_groups=32,
        ),
        [],
    )
    # A dp2 capture judged with the dp4 expectation (and the reverse)
    # rings instead of passing: count mismatches on every grouped name
    # plus the train step-number window.
    cross = HIERARCHY.validate_hierarchy(
        spans,
        device_step_counts=device_steps,
        compiler_counts=compiler_counts,
        expected_update_step=2,
        expected_groups=16,
    )
    self.assertIn("train:count=32 expected=16", cross)
    dp4_spans, dp4_steps, dp4_compilers = _fixture(groups=16)
    cross = HIERARCHY.validate_hierarchy(
        dp4_spans,
        device_step_counts=dp4_steps,
        compiler_counts=dp4_compilers,
        expected_update_step=2,
        expected_groups=32,
    )
    self.assertIn("train:count=16 expected=32", cross)

  def test_pure_hierarchy_validator_strong_negatives(self):
    spans, device_steps, compiler_counts = _fixture()
    cases = {}
    cases["missing_parent"] = [
        span for span in spans if span.name != "zero_tim_update"
    ]
    cases["duplicate_group"] = spans + [
        _span("forward_group", 25, 5, group_index=0)
    ]
    cases["orphan_optimizer"] = [
        _span("optimizer_commit", 970, 1, update_step=2)
        if span.name == "optimizer_commit" else span
        for span in spans
    ]
    cases["orphan_child"] = [
        _span("model_backward", 880, 2, group_index=3)
        if span.name == "model_backward" and span.stats.get("group_index") == 3
        else span
        for span in spans
    ]
    cases["wrong_count"] = [
        span
        for span in spans
        if not (
            span.name == "reverse_group"
            and span.stats.get("group_index") == 15
        )
    ]
    cases["wrong_micro_step"] = [
        _span(
            "gradient_accumulate",
            span.start_ns,
            span.duration_ns,
            group_index=3,
            micro_step=4,
            is_last_accumulate=0,
        )
        if span.name == "gradient_accumulate"
        and span.stats.get("group_index") == 3
        else span
        for span in spans
    ]
    cases["wrong_last_accumulate"] = [
        _span(
            "gradient_accumulate",
            span.start_ns,
            span.duration_ns,
            group_index=3,
            micro_step=3,
            is_last_accumulate=1,
        )
        if span.name == "gradient_accumulate"
        and span.stats.get("group_index") == 3
        else span
        for span in spans
    ]
    cases["wrong_optimizer_update"] = [
        _span("optimizer_commit", 900, 50, update_step=3)
        if span.name == "optimizer_commit" else span
        for span in spans
    ]
    cases["wrong_host_track"] = [
        _span(
            "model_backward",
            span.start_ns,
            span.duration_ns,
            line_name="worker",
            group_index=3,
        )
        if span.name == "model_backward"
        and span.stats.get("group_index") == 3
        else span
        for span in spans
    ]
    for name, changed in cases.items():
      with self.subTest(name=name):
        self.assertTrue(HIERARCHY.validate_hierarchy(
            changed,
            device_step_counts=device_steps,
            compiler_counts=compiler_counts,
            expected_update_step=2,
        ))

    missing_steps = dict(device_steps)
    missing_steps["/device:TPU:7"] = 0
    reasons = HIERARCHY.validate_hierarchy(
        spans,
        device_step_counts=missing_steps,
        compiler_counts=compiler_counts,
        expected_update_step=2,
    )
    self.assertIn("device_steps:/device:TPU:7=empty", reasons)

    wrong_track = cases["wrong_host_track"]
    reasons = HIERARCHY.validate_hierarchy(
        wrong_track,
        device_step_counts=device_steps,
        compiler_counts=compiler_counts,
        expected_update_step=2,
    )
    self.assertIn(
        "model_backward[3]:host_line=worker expected=python3",
        reasons,
    )

    compile_reasons = HIERARCHY.validate_hierarchy(
        spans,
        device_step_counts=device_steps,
        compiler_counts={
            **compiler_counts,
            "PJRT_Client_Compile": 1,
        },
        expected_update_step=2,
    )
    self.assertIn(
        "captured_compile:PJRT_Client_Compile=1 expected=0",
        compile_reasons,
    )

  def test_train_microstep_schedule_keeps_last_open_through_optimizer(self):
    calls = []

    class RecordingContext:

      def __init__(self, kind, name, stats):
        self.record = (kind, name, stats)

      def __enter__(self):
        calls.append(("enter",) + self.record)
        return self

      def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        calls.append(("exit",) + self.record)
        return False

    fake_jax = types.SimpleNamespace(
        profiler=types.SimpleNamespace(
            TraceAnnotation=lambda name, **stats: RecordingContext(
                "trace", name, stats
            ),
            StepTraceAnnotation=lambda name, **stats: RecordingContext(
                "step", name, stats
            ),
        )
    )
    with mock.patch.dict(
        os.environ, {"CANON_XPROF_LABELS": "1"}, clear=True
    ), mock.patch.dict(sys.modules, {"jax": fake_jax}):
      schedule = GSM8K_XPROF.ZeroHpTrainMicrostepSchedule(update_step=2)
      with schedule:
        for index in range(16):
          with schedule.transaction(index):
            calls.append(("body", index))
        with schedule.optimizer_commit():
          calls.append(("optimizer_body",))
    entered_steps = [
        item[3]["step_num"]
        for item in calls
        if item[:3] == ("enter", "step", "train")
    ]
    self.assertEqual(entered_steps, list(range(32, 48)))
    last_exit = calls.index(
        ("exit", "step", "train", {"step_num": 47})
    )
    optimizer_exit = calls.index(
        ("exit", "trace", "optimizer_commit", {"update_step": 2})
    )
    self.assertGreater(last_exit, optimizer_exit)
    self.assertFalse(any(
        item[:3] == ("enter", "step", "train")
        and item[3]["step_num"] == 48
        for item in calls
    ))

    with mock.patch.dict(
        os.environ, {"CANON_XPROF_LABELS": "1"}, clear=True
    ), mock.patch.dict(sys.modules, {"jax": fake_jax}):
      with self.assertRaisesRegex(RuntimeError, "before all transactions"):
        with GSM8K_XPROF.ZeroHpTrainMicrostepSchedule(update_step=2):
          pass
      schedule = GSM8K_XPROF.ZeroHpTrainMicrostepSchedule(update_step=2)
      with self.assertRaisesRegex(RuntimeError, "16 completed"):
        with schedule:
          with schedule.optimizer_commit():
            pass


if __name__ == "__main__":
  unittest.main()
