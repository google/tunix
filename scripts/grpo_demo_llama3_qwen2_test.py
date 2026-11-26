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
"""Regression tests for spawn-safe imports of grpo_demo_llama3_qwen2."""

import argparse
import contextlib
import io
import os
import pathlib
import runpy
import sys
import types
from unittest import mock

from absl.testing import absltest
import jax
import transformers
from tunix.examples.data import math_dataset
from tunix.rl import rl_cluster
from tunix.rl.grpo import grpo_learner
from tunix.sft import utils as sft_utils
from tunix.tests import test_common

_SCRIPT_PATH = pathlib.Path(__file__).parent / "grpo_demo_llama3_qwen2.py"


class _FailingPathwaysModule(types.ModuleType):

  def initialize(self) -> None:
    raise AssertionError("pathwaysutils.initialize must not run on import.")


class GrpoDemoLlama3Qwen2Test(absltest.TestCase):

  def test_mp_main_import_has_no_runtime_side_effects(self):
    # Simulates multiprocessing "spawn" worker re-execution via runpy.run_path
    # with run_name="__mp_main__" and inherited CLI arguments.
    self.enter_context(
        mock.patch.object(
            sys,
            "argv",
            [
                "grpo_demo_llama3_qwen2.py",
                "--root-dir=/tmp/grpo_test",
                "--rollout-engine=vllm",
            ],
        )
    )
    self.enter_context(
        mock.patch.dict(os.environ, {"JAX_PLATFORMS": "proxy"}, clear=False)
    )
    os.environ.pop("SKIP_JAX_PRECOMPILE", None)
    if "pathwaysutils" in sys.modules:
      orig_pathways = sys.modules["pathwaysutils"]
      self.addCleanup(sys.modules.__setitem__, "pathwaysutils", orig_pathways)
    else:
      self.addCleanup(sys.modules.pop, "pathwaysutils", None)
    sys.modules["pathwaysutils"] = _FailingPathwaysModule("pathwaysutils")

    fail_on_import = AssertionError("Must not execute during __mp_main__.")
    runtime_guards = [
        self.enter_context(
            mock.patch.object(
                argparse.ArgumentParser,
                "parse_args",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                jax,
                "device_count",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                jax,
                "devices",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                sft_utils,
                "show_hbm_usage",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                test_common,
                "delete_directory",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                test_common,
                "clear_jax_arrays",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                test_common,
                "download_from_huggingface",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                transformers.AutoTokenizer,
                "from_pretrained",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                math_dataset,
                "create_dataset",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                rl_cluster,
                "RLEngine",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
        self.enter_context(
            mock.patch.object(
                grpo_learner,
                "GRPOLearner",
                autospec=True,
                side_effect=fail_on_import,
            )
        ),
    ]

    module_globals = runpy.run_path(str(_SCRIPT_PATH), run_name="__mp_main__")

    self.assertNotIn("SKIP_JAX_PRECOMPILE", os.environ)
    for guard in runtime_guards:
      guard.assert_not_called()

    self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", module_globals["MODEL_CONFIG"])
    self.assertEqual(
        module_globals["extract_hash_answer"]("step 1 #### 42"), "42"
    )
    self.assertIsNone(module_globals["extract_hash_answer"]("no hash"))
    self.assertEqual(
        module_globals["match_format_exactly"](
            [""], ["<reasoning>ok</reasoning><answer>42</answer>"]
        ),
        [3.0],
    )
    self.assertEqual(
        module_globals["match_format_approximately"](
            [""], ["<reasoning>ok</reasoning><answer>42</answer>"]
        ),
        [2.0],
    )
    self.assertEqual(
        module_globals["check_answer"]([""], ["<answer>42</answer>"], ["42"]),
        [3.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
      self.assertEqual(
          module_globals["check_numbers"](
              [""], ["<answer>42</answer>"], ["42"], question=["q"]
          ),
          [1.5],
      )

  def test_mp_main_import_without_cli_args_does_not_parse_args(self):
    self.enter_context(
        mock.patch.object(sys, "argv", ["grpo_demo_llama3_qwen2.py"])
    )
    self.enter_context(
        mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}, clear=False)
    )
    mock_device_count = self.enter_context(
        mock.patch.object(
            jax,
            "device_count",
            autospec=True,
            side_effect=AssertionError("Must not initialize JAX on import."),
        )
    )

    module_globals = runpy.run_path(str(_SCRIPT_PATH), run_name="__mp_main__")

    self.assertIn("MODEL_CONFIG", module_globals)
    mock_device_count.assert_not_called()

  def test_main_help_parses_cli_args_without_initializing_jax(self):
    self.enter_context(
        mock.patch.object(sys, "argv", ["grpo_demo_llama3_qwen2.py", "--help"])
    )
    self.enter_context(
        mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}, clear=False)
    )
    mock_device_count = self.enter_context(
        mock.patch.object(
            jax,
            "device_count",
            autospec=True,
            side_effect=AssertionError("Must not initialize JAX for --help."),
        )
    )

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
      with self.assertRaises(SystemExit) as cm:
        runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")

    self.assertEqual(cm.exception.code, 0)
    self.assertIn("--root-dir", stdout_buf.getvalue())
    self.assertIn("--rollout-engine", stdout_buf.getvalue())
    mock_device_count.assert_not_called()


if __name__ == "__main__":
  absltest.main()
