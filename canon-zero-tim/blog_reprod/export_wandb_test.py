#!/usr/bin/env python3
"""Tests for export_wandb.py, with a fake W&B Api injected through make_api.

    python3 -m unittest export_wandb_test        # run from this directory
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import export_wandb
import yaml


class FakeRun:

  def __init__(self, rows, config, summary):
    self._rows = rows
    self.config = config
    self.summary = summary

  def scan_history(self):
    return iter(self._rows)


class FakeApi:
  """Records the run path it was asked for and serves canned runs."""

  def __init__(self, runs):
    self._runs = runs
    self.requested = []

  def run(self, path):
    self.requested.append(path)
    return self._runs[path]


ALPHA = FakeRun(
    # Out of order on purpose, and the second row introduces a new column.
    rows=[
        {"_step": 1, "_runtime": 2.5, "loss": 0.5, "flag": True},
        {"_step": 0, "_runtime": 0.0, "loss": 1.0, "note": "start"},
    ],
    config={"seed": 42, "model": "Qwen/Qwen3-8B", "_wandb": {"cli": "0.1"}},
    summary={"b": 2, "a": 1},
)
BETA = FakeRun(
    rows=[{"_step": 0, "solve": 0.25}],
    config={"seed": 7},
    summary={"solve": 0.25},
)
RUNS = {
    "zero-tim-p57-frozenlake-tim/alpha": ALPHA,
    "team/zero-tim-p57-frozenlake-tim/beta": BETA,
}


class ExportWandbTest(unittest.TestCase):

  def setUp(self):
    self._api = FakeApi(RUNS)
    self._saved_make_api = export_wandb.make_api
    export_wandb.make_api = lambda *a, **k: self._api
    self._saved_env = {
        key: os.environ.get(key) for key in ("WANDB_API_KEY", "WANDB_ENTITY")
    }
    os.environ["WANDB_API_KEY"] = "not-a-real-key"
    os.environ.pop("WANDB_ENTITY", None)
    self._tmp = tempfile.TemporaryDirectory()
    self.out = Path(self._tmp.name)
    self.addCleanup(self._restore)

  def _restore(self):
    export_wandb.make_api = self._saved_make_api
    for key, value in self._saved_env.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value
    self._tmp.cleanup()

  def export(self, *extra, run="alpha", out=None):
    out = out or (self.out / run)
    return export_wandb.main([
        "--project", "zero-tim-p57-frozenlake-tim",
        "--run", run,
        "--out", str(out),
        *extra,
    ])

  def test_history_columns_are_first_seen_order_and_steps_ascend(self):
    self.assertEqual(self.export(), 0)
    text = (self.out / "alpha" / "history.csv").read_text()
    self.assertEqual(
        text.splitlines()[0], "_step,_runtime,loss,flag,note"
    )
    self.assertEqual(
        text.splitlines()[1:], ["0,0.0,1.0,,start", "1,2.5,0.5,true,"]
    )
    self.assertTrue(text.endswith("\n"))

  def test_config_drops_wandb_block_and_sorts(self):
    self.assertEqual(self.export(), 0)
    path = self.out / "alpha" / "config.yaml"
    self.assertEqual(
        path.read_text(), "model: Qwen/Qwen3-8B\nseed: 42\n"
    )
    self.assertEqual(yaml.safe_load(path.read_text()), {"model": "Qwen/Qwen3-8B", "seed": 42})

  def test_summary_is_sorted_json(self):
    self.assertEqual(self.export(), 0)
    path = self.out / "alpha" / "summary.json"
    self.assertEqual(path.read_text(), '{\n  "a": 1,\n  "b": 2\n}')
    self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})

  def test_entity_flag_and_env_build_the_run_path(self):
    self.assertEqual(self.export("--entity", "team", run="beta"), 0)
    os.environ["WANDB_ENTITY"] = "team"
    self.assertEqual(self.export(run="beta", out=self.out / "beta2"), 0)
    self.assertEqual(
        self._api.requested,
        ["team/zero-tim-p57-frozenlake-tim/beta"] * 2,
    )

  def test_missing_api_key_exits_two_with_a_message(self):
    os.environ.pop("WANDB_API_KEY")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
      self.assertEqual(self.export(), 2)
    self.assertIn("WANDB_API_KEY", stderr.getvalue())
    self.assertFalse((self.out / "alpha").exists())
    self.assertEqual(self._api.requested, [])

  def test_cell_rendering(self):
    self.assertEqual(export_wandb.cell(None), "")
    self.assertEqual(export_wandb.cell("text"), "text")
    self.assertEqual(export_wandb.cell(True), "true")
    self.assertEqual(export_wandb.cell(3), "3")
    self.assertEqual(export_wandb.cell(0.1), "0.1")


if __name__ == "__main__":
  unittest.main()
