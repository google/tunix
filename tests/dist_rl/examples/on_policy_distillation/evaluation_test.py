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

"""CPU tests for held-out metrics and actual distributed generation requests."""

import asyncio
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.examples.on_policy_distillation import evaluation
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution


def _examples():
  return [
      evaluation.EvaluationExample(
          example_id="heldout-0",
          question="Count the apples.",
          gold_answer="123456789",
      ),
      evaluation.EvaluationExample(
          example_id="heldout-1", question="Count the pears.", gold_answer="4"
      ),
  ]


def _item(prompt_id, answer, *, version=2, status="SUCCEEDED"):
  return datatypes.TrajectoryItem(
      prompt_id=prompt_id,
      group_index=0,
      traj={
          "status": status,
          "policy_version": version,
          "policy_versions": [version],
          "conversation_text": [
              {"role": "user", "content": "Never score this \\boxed{999}."},
              {"role": "assistant", "content": answer},
          ],
          "conversation_tokens": np.array([3, 4, 0], dtype=np.int32),
          "conversation_masks": np.ones(3, dtype=np.float32),
      },
  )


class EvaluationConfigTest(unittest.TestCase):

  def test_config_rejects_training_data_and_unknown_or_invalid_fields(self):
    for settings in (
        {"split": "train"},
        {"split": "test+train"},
        {"split": "test[:100]+train[:100]"},
        {"dataset_name": "gsm8k"},
        {"dataset_name": "gsm8k:9.0.0"},
        {"batch_size": 0},
        {"batch_size": True},
        {"max_examples": -1},
        {"max_response_length": 0},
        {"seed": -1},
        {"timeout_seconds": float("nan")},
        {"temperature": 1.0},
    ):
      with self.subTest(settings=settings):
        with self.assertRaises(ValueError):
          evaluation.EvaluationConfig.from_dict(settings)
    self.assertEqual(evaluation.EvaluationConfig().split, "test")
    self.assertEqual(
        evaluation.EvaluationConfig.from_dict({"split": "test[:128]"}).split,
        "test[:128]",
    )

  def test_numeric_answer_canonicalization_is_exact_and_not_a_last_number_guess(
      self,
  ):
    for answer, expected in (
        ("4.000", "4"),
        ("1,000", "1000"),
        ("1,000 / 2", "500"),
        (" -1,234.5 ", "-2469/2"),
        ("0.5", "1/2"),
        ("1/2", "1/2"),
        (r"\frac{1}{2}", "1/2"),
        (r"\dfrac{1}{2}", "1/2"),
        (r"\frac { 1,000 } { 2 }", "500"),
        ("-0.0", "0"),
        ("1e2", "100"),
    ):
      with self.subTest(answer=answer):
        self.assertEqual(evaluation.canonical_number(answer), expected)
    for answer in (
        None,
        "",
        "NaN",
        "inf",
        "4 dollars",
        "4+1",
        "answer 4",
        "1/0",
        "1e99999",
        "1,2",
        "1,2,3",
        "1, 2",
        "1 2",
        "4 .0",
        "1,23/2",
        r"\frac{1 2}{2}",
    ):
      with self.subTest(answer=answer):
        self.assertIsNone(evaluation.canonical_number(answer))

  def test_loader_pins_test_version_order_and_rejects_incomplete_full_split(
      self,
  ):
    from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k

    rows = [
        {"question": "first", "answer": "1"},
        {"question": "second", "answer": "2"},
    ]
    config = evaluation.EvaluationConfig(max_examples=1)
    with mock.patch.object(
        gsm8k, "load_gsm8k_dataset", return_value=rows
    ) as load:
      examples = evaluation.load_examples("dataset-cache", config)
    self.assertEqual([row.question for row in examples], ["first"])
    load.assert_called_once_with(
        split="test",
        data_dir="dataset-cache",
        shuffle=False,
        seed=42,
        dataset_name="gsm8k:1.0.0",
    )
    with mock.patch.object(gsm8k, "load_gsm8k_dataset", return_value=rows):
      with self.assertRaisesRegex(ValueError, "1319"):
        evaluation.load_examples("dataset-cache", evaluation.EvaluationConfig())


class EvaluationEngineTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.output = Path(self.directory.name) / "evaluation" / "final.json"
    self.config = evaluation.EvaluationConfig(
        batch_size=2, max_response_length=64
    )
    self.identity = {
        "repo_id": "Qwen/Qwen3-8B",
        "revision": "a" * 40,
        "evaluation_source": "live_synced_rollout",
    }

  async def _evaluate(self, engine, **overrides):
    kwargs = dict(
        config=self.config,
        output_path=self.output,
        label="final",
        model_identity=self.identity,
        policy_version=2,
    )
    kwargs.update(overrides)
    return await evaluation.evaluate_engine(engine, _examples(), **kwargs)

  async def test_live_engine_request_has_no_labels_and_report_grades_assistant_only(
      self,
  ):
    calls = []

    async def respond(method_name, *, requests):
      self.assertEqual(method_name, "generate")
      responses = []
      for req in requests:
        calls.append(req)
        question = req.metadata["env_config"]["question"]
        answer = (
            r"done</reasoning><answer>\boxed{123456789.0}</answer>"
            if "apples" in question
            else "The last number is 4, without a box."
        )
        responses.append(
            datatypes.RolloutResponse(
                request_id=req.request_id,
                status="COMPLETED",
                payload=_item(req.prompt_id, answer),
            )
        )
      return responses

    handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    handle.asubmit = mock.AsyncMock(side_effect=respond)
    registry = worker_registry.WorkerRegistry()
    registry.register_handle("rollout-eval", [datatypes.Role.ROLLOUT], handle)
    engine = distributed_rl_engine.DistributedRLEngine(registry)
    engine._policy_version = 2
    report = await self._evaluate(engine)
    self.assertEqual(report["metrics"]["answer_accuracy"], 0.5)
    self.assertEqual(report["metrics"]["strict_boxed_accuracy"], 0.0)
    self.assertEqual(report["metrics"]["format_accuracy"], 0.5)
    self.assertEqual(report["metrics"]["num_examples"], 2)
    self.assertEqual(report["results"][0]["canonical_answer"], "123456789")
    self.assertIsNone(report["results"][1]["extracted_answer"])
    self.assertEqual(report["dataset"]["name"], "gsm8k:1.0.0")
    self.assertEqual(json.loads(self.output.read_text()), report)
    self.assertEqual(len(calls), 2)
    for request in calls:
      self.assertEqual(request.target_policy_version, 2)
      self.assertEqual(request.max_turns, 1)
      self.assertEqual(request.max_response_length, 64)
      self.assertEqual(
          {
              key: value
              for key, value in request.generation_kwargs.items()
              if key != "seed"
          },
          {
              "temperature": 0.0,
              "top_p": 1.0,
              "top_k": 0,
              "max_generation_steps": 64,
              "return_logprobs": False,
          },
      )
      self.assertIsInstance(request.generation_kwargs["seed"], int)
      self.assertGreaterEqual(request.generation_kwargs["seed"], 0)
      self.assertNotIn("answer", request.metadata["env_config"])
      self.assertNotIn("gold_answer", request.metadata["env_config"])
      self.assertNotIn("123456789", str(request.metadata))
      self.assertNotIn("123456789", request.prompt)

  async def test_response_order_does_not_change_question_alignment(self):
    async def generate(prompts, generation_args):
      del generation_args
      return [
          _item(prompts[1]["prompt_id"], r"\boxed{4}"),
          _item(prompts[0]["prompt_id"], r"\boxed{0}"),
      ]

    report = await self._evaluate(
        mock.Mock(generate=mock.AsyncMock(side_effect=generate))
    )
    self.assertEqual(
        [row["example_id"] for row in report["results"]],
        ["heldout-0", "heldout-1"],
    )
    self.assertEqual(
        [row["answer_correct"] for row in report["results"]], [False, True]
    )

  async def test_missing_bad_or_stale_responses_fail_without_publishing_report(
      self,
  ):
    for problem in (
        "count",
        "duplicate",
        "unknown",
        "stale",
        "failed",
        "mask",
        "tokens",
        "text",
        "null_text",
        "missing_assistant",
        "error",
    ):
      with self.subTest(problem=problem):

        async def generate(prompts, generation_args):
          del generation_args
          items = [_item(p["prompt_id"], r"\boxed{4}") for p in prompts]
          if problem == "count":
            return items[:1]
          if problem == "duplicate":
            return [items[0], items[0]]
          if problem == "unknown":
            items[0].prompt_id = "unknown"
          elif problem == "stale":
            items[0].traj["policy_version"] = 1
          elif problem == "failed":
            items[0].traj["status"] = "FAILED"
          elif problem == "mask":
            items[0].traj["conversation_masks"] = np.array([1])
          elif problem == "tokens":
            items[0].traj["conversation_tokens"] = []
          elif problem == "text":
            del items[0].traj["conversation_text"]
          elif problem == "null_text":
            items[0].traj["conversation_text"] = None
          elif problem == "missing_assistant":
            items[0].traj["conversation_text"] = [
                {"role": "user", "content": r"\boxed{4}"}
            ]
          elif problem == "error":
            items[0].metadata["error"] = "backend failure"
          return items

        with self.assertRaises(ValueError):
          await self._evaluate(
              mock.Mock(generate=mock.AsyncMock(side_effect=generate))
          )
        self.assertFalse(self.output.exists())

  async def test_timeout_cancels_generation_and_does_not_publish_report(self):
    cancelled = asyncio.Event()

    async def generate(*args, **kwargs):
      del args, kwargs
      try:
        await asyncio.sleep(30)
      finally:
        cancelled.set()

    engine = mock.Mock(generate=mock.AsyncMock(side_effect=generate))
    with self.assertRaises(TimeoutError):
      await self._evaluate(
          engine, config=dataclasses.replace(self.config, timeout_seconds=0.01)
      )
    self.assertTrue(cancelled.is_set())
    self.assertFalse(self.output.exists())

  async def test_existing_report_is_preserved_before_generation(self):
    self.output.parent.mkdir()
    self.output.write_text("existing evidence")
    engine = mock.Mock(generate=mock.AsyncMock())
    with self.assertRaises(FileExistsError):
      await self._evaluate(engine)
    engine.generate.assert_not_awaited()
    self.assertEqual(self.output.read_text(), "existing evidence")

  async def test_truncated_responses_stay_in_accuracy_denominator(self):
    async def generate(prompts, generation_args):
      del generation_args
      return [
          _item(
              p["prompt_id"], "incomplete", status="MAX_CONTEXT_LIMIT_REACHED"
          )
          for p in prompts
      ]

    report = await self._evaluate(
        mock.Mock(generate=mock.AsyncMock(side_effect=generate))
    )
    self.assertEqual(report["metrics"]["num_examples"], 2)
    self.assertEqual(report["metrics"]["answer_accuracy"], 0.0)
    self.assertEqual(report["metrics"]["truncated_count"], 2)


if __name__ == "__main__":
  unittest.main()
