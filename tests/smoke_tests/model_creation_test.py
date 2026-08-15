# Copyright 2025 Google LLC
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

import os
import atexit
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tunix.cli.utils import model

atexit.register(lambda: os._exit(0))



class ModelIntegrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.download_dir = self.temp_dir.name

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  @parameterized.named_parameters(
      dict(
          testcase_name="qwen2_5_0_5b",
          model_name="qwen2.5-0.5b",
          model_source="huggingface",
          model_id="Qwen/Qwen2.5-0.5B",
          tokenizer_path="Qwen/Qwen2.5-0.5B",
          tokenizer_type="huggingface",
          expected_tokenizer_path="Qwen/Qwen2.5-0.5B",
      ),
      dict(
          testcase_name="gemma3_270m",
          model_name="gemma-3-270m",
          model_source="gcs",
          model_id="google/gemma-3-270m",
          model_path="gs://gemma-data/checkpoints/gemma3-270m-pt",
          tokenizer_path=model._DEFAULT_TOKENIZER_PATH,
          tokenizer_type="sentencepiece",
          expected_tokenizer_path=(
              "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
          ),
      ),
      dict(
          testcase_name="gemma-2-2b-it",
          model_name="gemma-2-2b-it",
          model_source="kaggle",
          model_id="google/gemma-2-2b-it",
          model_path="google/gemma-2/flax/gemma2-2b-it",
          tokenizer_path=model._DEFAULT_TOKENIZER_PATH,
          tokenizer_type="sentencepiece",
          expected_tokenizer_path=r"^/tmp/[^/]+/models/google/gemma-2/flax/gemma2-2b-it/\d+/tokenizer\.model$",
      ),
  )
  def test_create_model(
      self,
      model_name,
      model_source,
      model_id,
      tokenizer_path,
      tokenizer_type,
      expected_tokenizer_path,
      model_path=None,
  ):
    if model_source == "kaggle":
      try:
        import kagglesdk.kaggle_env  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
        if not hasattr(kagglesdk.kaggle_env, 'get_web_endpoint') and hasattr(kagglesdk.kaggle_env, 'get_endpoint'):
          kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      try:
        import kagglehub  # pylint: disable=g-import-not-at-top,unused-import
      except ImportError:
        self.skipTest(
            "Skipping Kaggle test because kagglehub is not available or has "
            "import mismatches."
        )
    model_config = {
        "model_name": model_name,
        "model_source": model_source,
        "model_id": model_id,
        "model_path": model_path,
        "model_download_path": self.download_dir,
        "intermediate_ckpt_dir": os.path.join(self.download_dir, "intermediate_ckpt"),
        "lora_config": None,
        "model_display": False,
    }

    tokenizer_config = {
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "add_bos": False,
        "add_eos": False,
    }

    devices = jax.devices()
    mesh = jax.sharding.Mesh(
        np.array(devices[:1]).reshape((1, 1)), ("tp", "fsdp")
    )

    model_obj, tokenizer_path = model.create_model(
        model_config, tokenizer_config, mesh
    )
    self.assertIsNotNone(model_obj)

    self.assertRegex(tokenizer_path, expected_tokenizer_path)


if __name__ == "__main__":
  absltest.main()
