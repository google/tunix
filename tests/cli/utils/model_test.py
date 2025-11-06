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

from absl.testing import absltest
from absl.testing import parameterized
from tunix.cli.utils import model


@parameterized.named_parameters(
    dict(
        testcase_name="gemma-2b",
        model_name="gemma-2b",
    ),
    dict(
        testcase_name="gemma2-2b",
        model_name="gemma2-2b",
    ),
    dict(
        testcase_name="gemma2-2b-it",
        model_name="gemma2-2b-it",
    ),
    dict(
        testcase_name="gemma2-9b",
        model_name="gemma2-9b",
    ),
    dict(
        testcase_name="gemma3-1b",
        model_name="gemma3-1b",
    ),
    dict(
        testcase_name="gemma3-270m",
        model_name="gemma3-270m",
    ),
    dict(
        testcase_name="llama3.2-1b",
        model_name="llama3.2-1b",
    ),
    dict(
        testcase_name="llama3.1-8b",
        model_name="llama3.1-8b",
    ),
    dict(
        testcase_name="qwen2.5-7b",
        model_name="qwen2.5-7b",
    ),
    dict(
        testcase_name="qwen3-14b",
        model_name="qwen3-14b",
    ),
)

class ModelTest(parameterized.TestCase):

  def test_obtain_model_params_valid(self, model_name: str):
    model.obtain_model_params(model_name)

  def test_create_model_dynamically_routing(self, model_name: str):
    model_module = model.get_model_module(model_name)
    if not model_name.startswith("gemma"):
      # TODO(b/444572467)
      getattr(model_module, "create_model_from_safe_tensors")


if __name__ == "__main__":
  absltest.main()
