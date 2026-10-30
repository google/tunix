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

"""Checks real shell argument construction without starting TPU workers."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

_EXAMPLES = Path(__file__).resolve().parents[4] / "tunix/dist_rl/examples"


def _env(**overrides):
  return {"PATH": os.environ["PATH"], **overrides}


def _orchestrator_command(**overrides):
  source = (_EXAMPLES / "math_gsm8k_dist/launcher.sh").read_text()
  # Evaluate the actual defaults and argv construction, omitting worker launch,
  # model preparation, port waits and cleanup. The emitted argv is NUL-delimited
  # so a tokenizer path containing spaces remains one argument.
  defaults = source[: source.index("TRAINER_TPU_CHIPS=")]
  start = source.index("  ORCHESTRATOR_CMD=(")
  end = source.index("  export JAX_PLATFORMS=cpu", start)
  script = (
      defaults
      + source[start:end]
      + '\nprintf "%s\\0" "${ORCHESTRATOR_CMD[@]}"\n'
  )
  result = subprocess.run(
      ["bash"],
      input=script.encode(),
      env=_env(**overrides),
      capture_output=True,
      check=True,
  )
  return result.stdout.decode().rstrip("\0").split("\0")


class Gsm8kLauncherTest(unittest.TestCase):

  def test_full_recipe_identity_evaluation_and_dataset_settings_reach_cli(self):
    command = _orchestrator_command(
        OPD_MODELS_MANIFEST="/prepared/models.json",
        OPD_EVAL_CONFIG="/run/evaluation.json",
        OPD_EVAL_OUTPUT_DIR="/run/evaluation reports",
        OPD_EVAL_ONLY="true",
        OPD_REQUIRE_RESUME="true",
        TFDS_DATASET="gsm8k:1.0.0",
        GENERATION_SEED="7",
    )
    for expected in (
        "--opd_models_manifest=/prepared/models.json",
        "--opd_eval_config=/run/evaluation.json",
        "--opd_eval_output_dir=/run/evaluation reports",
        "--opd_eval_only",
        "--opd_require_resume",
        "--tfds_dataset=gsm8k:1.0.0",
        "--generation_seed=7",
    ):
      self.assertIn(expected, command)

  def test_legacy_data_seed_does_not_enable_a_generation_seed(self):
    command = _orchestrator_command(SEED="7")
    self.assertIn("--seed=7", command)
    self.assertFalse(
        any(arg.startswith("--generation_seed") for arg in command)
    )

  def test_bfloat16_and_rematerialization_reach_worker_commands(self):
    source = (_EXAMPLES / "math_gsm8k_dist/launcher.sh").read_text()
    defaults = source[: source.index("TRAINER_TPU_CHIPS=")]
    for name, ending in (
        ("TRAINER_CMD", '  if [[ -n "$OPT_CHAIN_TYPE" ]]'),
        ("ROLLOUT_CMD", '  if [[ -n "$MAXTEXT_MODEL_NAME" ]]'),
    ):
      with self.subTest(worker=name):
        start = source.index(f"  {name}=(")
        end = source.index(ending, start)
        script = (
            defaults
            + source[start:end]
            + f'\nprintf "%s\\0" "${{{name}[@]}}"\n'
        )
        result = subprocess.run(
            ["bash"],
            input=script.encode(),
            env=_env(
                MODEL_DTYPE="bfloat16",
                MODEL_LOAD_DTYPE="bfloat16",
                REMAT_CONFIG="decoder",
            ),
            capture_output=True,
            check=True,
        )
        command = result.stdout.decode().rstrip("\0").split("\0")
        self.assertIn("--model_dtype=bfloat16", command)
        if name == "TRAINER_CMD":
          self.assertIn("--model_load_dtype=bfloat16", command)
          self.assertIn("--remat_config=decoder", command)

  def test_prepared_snapshot_symlinks_are_accepted_without_download(self):
    source = (_EXAMPLES / "math_gsm8k_dist/launcher.sh").read_text()
    functions = source[
        source.index("has_direct_safetensors() {") : source.index(
            "convert_maxtext_ckpt() {"
        )
    ]
    script = functions + """
print_safetensors_candidates() { :; }
download_model_dir() { echo UNEXPECTED_DOWNLOAD >&2; exit 9; }
ensure_model_dir
"""
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      snapshot = root / "snapshot"
      snapshot.mkdir()
      shard = root / "weights"
      shard.write_bytes(b"prepared")
      (snapshot / "model.safetensors").symlink_to(shard)
      result = subprocess.run(
          ["bash"],
          input=script,
          text=True,
          env=_env(MODEL_DIR=str(snapshot), ALLOW_MODEL_DOWNLOAD="false"),
          capture_output=True,
          check=False,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      (snapshot / "model.safetensors").unlink()
      result = subprocess.run(
          ["bash"],
          input=script,
          text=True,
          env=_env(MODEL_DIR=str(snapshot), ALLOW_MODEL_DOWNLOAD="false"),
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("ALLOW_MODEL_DOWNLOAD=false", result.stderr)
      self.assertNotIn("UNEXPECTED_DOWNLOAD", result.stderr)

  def test_opd_and_sampling_environment_reaches_the_cli(self):
    command = _orchestrator_command(
        OPD_TEACHER_URL="http://teacher:30001/generate",
        OPD_TEACHER_TOKENIZER_PATH="/snapshots/fixed teacher",
        OPD_COEFFICIENT="0.5",
        OPD_TASK_REWARD_COEFFICIENT="1",
        OPD_TEACHER_TIMEOUT_SECONDS="12.5",
        OPD_MAX_CONCURRENCY="3",
        TEMPERATURE="1.0",
        TOP_P="1.0",
        TOP_K="0",
        IN_FLIGHT_WEIGHT_UPDATES="false",
    )
    for expected in (
        "--opd_teacher_url=http://teacher:30001/generate",
        "--opd_teacher_tokenizer_path=/snapshots/fixed teacher",
        "--opd_coefficient=0.5",
        "--opd_task_reward_coefficient=1",
        "--opd_teacher_timeout_seconds=12.5",
        "--opd_max_concurrency=3",
        "--temperature=1.0",
        "--top_p=1.0",
        "--top_k=0",
        "--in_flight_weight_updates=false",
    ):
      self.assertIn(expected, command)

  def test_ordinary_grpo_does_not_enable_opd(self):
    command = _orchestrator_command()
    self.assertFalse(any(arg.startswith("--opd_") for arg in command))
    self.assertIn("--top_k=-1", command)

  def test_missing_teacher_tokenizer_fails_before_worker_start(self):
    result = subprocess.run(
        ["bash", str(_EXAMPLES / "math_gsm8k_dist/launcher.sh")],
        env=_env(OPD_TEACHER_URL="http://teacher:30001"),
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("OPD_TEACHER_TOKENIZER_PATH is required", result.stderr)

  def test_opd_recipe_inherits_native_smoke_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      recipes = root / "recipes"
      recipes.mkdir()
      for name in (
          "run_gsm8k_opd_qwen3_0p6b.sh",
          "ci_smoke_gsm8k_qwen3_0p6b.sh",
      ):
        shutil.copyfile(_EXAMPLES / "recipes" / name, recipes / name)
      launcher = root / "math_gsm8k_dist/launcher.sh"
      launcher.parent.mkdir()
      launcher.write_text(
          '#!/bin/bash\nprintf "%s\\0" "$TRAINER_BACKEND" "$SAMPLER" '
          '"$WEIGHT_SYNC_MODE" "$MAX_STEPS" "$NUM_GENERATIONS" '
          '"$BETA" "$MAX_STALENESS" "$IN_FLIGHT_WEIGHT_UPDATES" '
          '"$OPD_TASK_REWARD_COEFFICIENT" "$@"\n'
      )
      launcher.chmod(0o755)
      result = subprocess.run(
          [
              "bash",
              str(recipes / "run_gsm8k_opd_qwen3_0p6b.sh"),
              "arg with spaces",
          ],
          env=_env(
              OPD_TEACHER_URL="http://teacher:30001",
              OPD_TEACHER_TOKENIZER_PATH="/snapshots/teacher",
          ),
          capture_output=True,
          check=True,
      )
    self.assertEqual(
        result.stdout.decode().rstrip("\0").split("\0"),
        [
            "tunix",
            "inprocess_vllm",
            "raiden",
            "2",
            "2",
            "0",
            "0",
            "false",
            "0.0",
            "arg with spaces",
        ],
    )


if __name__ == "__main__":
  unittest.main()
