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
GSM8K_XPROF = _load("p60_gsm8k_xprof", ROOT / "tunix/rl/gsm8k_xprof.py")


def _span(name, start, duration, *, line_name="python3", **stats):
  return HIERARCHY.Span(name, start, duration, line_name, stats)


def _fixture():
  spans = [
      _span("train", 0, 1000, _r="1", step_num="1"),
      _span("zero_tim_update", 10, 980),
      _span("forward_groups", 20, 180),
      _span("loss_pullback", 210, 10),
      _span("reverse_groups", 230, 620),
      _span("optimizer_commit", 900, 50, update_step=1),
  ]
  for index in range(16):
    spans.append(_span("forward_group", 25 + index * 10, 5, group_index=index))
    start = 240 + index * 35
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
            is_last_accumulate=int(index == 15),
        ),
    ))
  device_steps = {f"/device:TPU:{index}": 100 for index in range(8)}
  return spans, device_steps


class HierarchyTest(unittest.TestCase):

  def test_parent_annotations_are_constructed_after_trace_start(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    end = learner.index(
        "segmented_result = self._run_p28_g6_update(",
        learner.index("train_step_annotation ="),
    )
    seam = learner[learner.rfind("marker_prefix", 0, end):end]
    start = seam.index("_canon_xprof_update_entry()")
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
      GSM8K_XPROF.trace_annotation("optimizer_commit", update_step=1)
      GSM8K_XPROF.train_step_annotation(step_num=1)
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
        ("trace", "optimizer_commit", {"update_step": 1}),
        ("step", "train", {"step_num": 1}),
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
    spans, device_steps = _fixture()
    self.assertEqual(
        HIERARCHY.validate_hierarchy(
            spans, device_step_counts=device_steps, expected_step=1
        ),
        [],
    )

  def test_pure_hierarchy_validator_strong_negatives(self):
    spans, device_steps = _fixture()
    cases = {}
    cases["missing_parent"] = [
        span for span in spans if span.name != "zero_tim_update"
    ]
    cases["duplicate_group"] = spans + [
        _span("forward_group", 25, 5, group_index=0)
    ]
    cases["orphan_optimizer"] = [
        _span("optimizer_commit", 995, 1)
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
        _span("optimizer_commit", 900, 50, update_step=2)
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
            changed, device_step_counts=device_steps, expected_step=1
        ))

    missing_steps = dict(device_steps)
    missing_steps["/device:TPU:7"] = 0
    reasons = HIERARCHY.validate_hierarchy(
        spans, device_step_counts=missing_steps, expected_step=1
    )
    self.assertIn("device_steps:/device:TPU:7=empty", reasons)

    wrong_track = cases["wrong_host_track"]
    reasons = HIERARCHY.validate_hierarchy(
        wrong_track, device_step_counts=device_steps, expected_step=1
    )
    self.assertIn(
        "model_backward[3]:host_line=worker expected=python3",
        reasons,
    )


if __name__ == "__main__":
  unittest.main()
