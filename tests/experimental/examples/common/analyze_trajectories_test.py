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

"""Tests for analyze_trajectories."""

import csv
import os
import tempfile
from absl.testing import absltest
from tunix.experimental.examples.common import analyze_trajectories


class AnalyzeTrajectoriesTest(absltest.TestCase):

  def test_compute_group_advantages_zero_variance(self):
    advs, mean_r, std_r = analyze_trajectories.compute_group_advantages(
        [0.0, 0.0, 0.0, 0.0]
    )
    self.assertEqual(advs, [0.0, 0.0, 0.0, 0.0])
    self.assertEqual(mean_r, 0.0)
    self.assertEqual(std_r, 0.0)

  def test_compute_group_advantages_mixed(self):
    advs, mean_r, std_r = analyze_trajectories.compute_group_advantages(
        [1.0, 0.0]
    )
    self.assertAlmostEqual(mean_r, 0.5)
    self.assertGreater(std_r, 0.0)
    self.assertGreater(advs[0], 0.0)
    self.assertLess(advs[1], 0.0)
    self.assertAlmostEqual(advs[0] + advs[1], 0.0)

  def test_compute_group_advantages_empty_raises(self):
    with self.assertRaises(ValueError):
      analyze_trajectories.compute_group_advantages([])

  def test_analyze_all_zero_step(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "p1",
            "reward": "0.0",
            "completion_tokens": "100",
        },
        {
            "global_step": "0",
            "prompt_id": "p1",
            "reward": "0.0",
            "completion_tokens": "110",
        },
        {
            "global_step": "0",
            "prompt_id": "p2",
            "reward": "0.0",
            "completion_tokens": "95",
        },
        {
            "global_step": "0",
            "prompt_id": "p2",
            "reward": "0.0",
            "completion_tokens": "105",
        },
    ]
    analysis = analyze_trajectories.analyze_trajectories(rows)
    self.assertEqual(analysis["total_trajectories"], 4)
    self.assertEqual(analysis["steps"][0]["active_groups"], 0)
    self.assertIn("~0.0", analysis["steps"][0]["loss_verdict"])

  def test_analyze_active_step(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "p1",
            "reward": "1.0",
            "completion_tokens": "100",
        },
        {
            "global_step": "0",
            "prompt_id": "p1",
            "reward": "0.0",
            "completion_tokens": "110",
        },
    ]
    analysis = analyze_trajectories.analyze_trajectories(rows)
    self.assertEqual(analysis["steps"][0]["active_groups"], 1)
    self.assertIn("ACTIVE", analysis["steps"][0]["loss_verdict"])

  def test_response_snippet_truncated(self):
    long_response = "Step 1: calculate. " * 20
    rows = [
        {
            "global_step": "0",
            "prompt_id": "p1",
            "question": "What is 2+2?",
            "completion": long_response,
            "reward": "1.0",
            "completion_tokens": "50",
        },
        {
            "global_step": "0",
            "prompt_id": "p1",
            "question": "What is 2+2?",
            "completion": "Second rollout response",
            "reward": "0.0",
            "completion_tokens": "20",
        },
    ]
    analysis = analyze_trajectories.analyze_trajectories(rows)
    group = analysis["steps"][0]["groups"][0]
    self.assertTrue(group["response"].endswith("..."))
    self.assertLessEqual(len(group["response"]), 123)
    self.assertTrue(group["response"].startswith("Step 1: calculate."))

  def test_missing_prompt_id_raises_key_error(self):
    rows = [
        {
            "global_step": "0",
            "question": "What is 2+2?",  # No prompt_id!
            "reward": "1.0",
            "completion_tokens": "100",
        }
    ]
    with self.assertRaises(KeyError):
      analyze_trajectories.analyze_trajectories(rows)

  def test_missing_global_step_raises_key_error(self):
    rows = [
        {
            "prompt_id": "p1",
            "reward": "1.0",
            "completion_tokens": "100",
        }
    ]
    with self.assertRaises(KeyError):
      analyze_trajectories.analyze_trajectories(rows)

  def test_invalid_global_step_raises_value_error(self):
    rows = [
        {
            "global_step": "not_an_int",
            "prompt_id": "p1",
            "reward": "1.0",
            "completion_tokens": "100",
        }
    ]
    with self.assertRaises(ValueError):
      analyze_trajectories.analyze_trajectories(rows)

  def test_missing_reward_raises_key_error(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "p1",
            "completion_tokens": "100",
        }
    ]
    with self.assertRaises(KeyError):
      analyze_trajectories.analyze_trajectories(rows)

  def test_invalid_reward_raises_value_error(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "p1",
            "reward": "not_a_float",
            "completion_tokens": "100",
        }
    ]
    with self.assertRaises(ValueError):
      analyze_trajectories.analyze_trajectories(rows)

  def test_completion_tokens_none_is_allowed(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "prompt_0",
            "reward": "0.5",
            "completion_tokens": "None",
            "question": "What is 2+2?",
            "completion": "4",
        }
    ]
    analysis = analyze_trajectories.analyze_trajectories(rows)
    self.assertEqual(analysis["total_trajectories"], 1)
    self.assertIsNone(analysis["steps"][0]["avg_completion_tokens"])

  def test_completion_tokens_missing_is_allowed(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "prompt_0",
            "reward": "0.5",
        }
    ]
    analysis = analyze_trajectories.analyze_trajectories(rows)
    self.assertEqual(analysis["total_trajectories"], 1)
    self.assertIsNone(analysis["steps"][0]["avg_completion_tokens"])

  def test_invalid_completion_tokens_raises_value_error(self):
    rows = [
        {
            "global_step": "0",
            "prompt_id": "prompt_0",
            "reward": "0.5",
            "completion_tokens": "not_an_int",
        }
    ]
    with self.assertRaises(ValueError):
      analyze_trajectories.analyze_trajectories(rows)

  def test_chat_formatted_completion_unwrapped(self):
    chat_repr = (
        "[{'role': 'user', 'content': 'What is 2+2?'}, {'role': 'assistant',"
        " 'content': 'The answer is 4.'}]"
    )
    rows = [
        {
            "global_step": "0",
            "prompt_id": "prompt_0",
            "reward": "1.0",
            "question": "What is 2+2?",
            "completion": chat_repr,
            "completion_tokens": "None",
        }
    ]
    analysis = analyze_trajectories.analyze_trajectories(rows)
    group = analysis["steps"][0]["groups"][0]
    self.assertEqual(group["response"], "The answer is 4.")

  def test_empty_rows_raises_value_error(self):
    with self.assertRaises(ValueError):
      analyze_trajectories.analyze_trajectories([])

  def test_load_and_report_end_to_end(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      csv_path = os.path.join(tmp_dir, "trajectory_log.csv")
      with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_step",
                "prompt_id",
                "question",
                "reward",
                "completion_tokens",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "global_step": "0",
            "prompt_id": "p1",
            "question": "What is 2+2?",
            "reward": "1.0",
            "completion_tokens": "50",
        })
        writer.writerow({
            "global_step": "0",
            "prompt_id": "p1",
            "question": "What is 2+2?",
            "reward": "0.0",
            "completion_tokens": "60",
        })

      rows, selected_file, all_files = analyze_trajectories.load_trajectories(
          tmp_dir
      )
      self.assertLen(rows, 2)
      self.assertEqual(selected_file, csv_path)
      self.assertEqual(all_files, [csv_path])
      analysis = analyze_trajectories.analyze_trajectories(rows)
      self.assertEqual(analysis["total_steps"], 1)

  def test_csv_missing_required_column_header_raises_key_error(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      csv_path = os.path.join(tmp_dir, "trajectory_log.csv")
      with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["global_step", "reward"],  # missing prompt_id & completion_tokens
        )
        writer.writeheader()
        writer.writerow({"global_step": "0", "reward": "1.0"})

      with self.assertRaises(KeyError):
        analyze_trajectories.load_trajectories(csv_path)

  def test_empty_csv_file_raises_value_error(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      csv_path = os.path.join(tmp_dir, "trajectory_log.csv")
      with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(analyze_trajectories.REQUIRED_COLUMNS),
        )
        writer.writeheader()
        # No rows written!

      with self.assertRaises(ValueError):
        analyze_trajectories.load_trajectories(csv_path)

  def test_resolve_csv_file_non_csv_raises_value_error(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      txt_path = os.path.join(tmp_dir, "trajectory.txt")
      with open(txt_path, "w") as f:
        f.write("test")
      with self.assertRaises(ValueError):
        analyze_trajectories.resolve_csv_file(txt_path)

  def test_resolve_csv_file_picks_latest_modified(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      old_file = os.path.join(tmp_dir, "trajectory_log_1789429058.csv")
      new_file = os.path.join(tmp_dir, "trajectory_log_1789512708.csv")
      with open(old_file, "w") as f:
        f.write("dummy")
      with open(new_file, "w") as f:
        f.write("dummy")
      os.utime(old_file, (1000, 1000))
      os.utime(new_file, (2000, 2000))

      selected, all_files = analyze_trajectories.resolve_csv_file(tmp_dir)
      self.assertEqual(selected, new_file)
      self.assertEqual(all_files[0], new_file)
      self.assertEqual(all_files[1], old_file)

  def test_resolve_csv_file_direct_path(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      target_file = os.path.join(tmp_dir, "my_run.csv")
      with open(target_file, "w") as f:
        f.write("dummy")

      selected, all_files = analyze_trajectories.resolve_csv_file(target_file)
      self.assertEqual(selected, target_file)
      self.assertIn(target_file, all_files)

  def test_parse_orchestrator_log_nonexistent_raises(self):
    with self.assertRaises(FileNotFoundError):
      analyze_trajectories.parse_orchestrator_log("/nonexistent/orchestrator.log")


if __name__ == "__main__":
  absltest.main()
