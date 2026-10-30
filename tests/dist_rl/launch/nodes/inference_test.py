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

"""Unit tests for the distributed RL reference inference worker node."""

from unittest import mock

from absl.testing import absltest
from jax.sharding import Mesh
from tunix.dist_rl.launch.nodes import inference
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.worker import inference_worker as exp_inference_worker
from tunix.models import automodel
from tunix.rl.inference import inference_worker as rl_inference_worker


class InferenceMainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_context = mock.MagicMock()
    self.mock_mesh = mock.MagicMock(spec=Mesh)

    self.enter_context(
        mock.patch.object(
            node_utils, "ensure_local_safetensors", return_value="/models/ref"
        )
    )
    self.enter_context(
        mock.patch.object(
            node_utils,
            "load_hf_tokenizer",
            return_value=mock.Mock(pad_token_id=2, eos_token_id=9),
        )
    )
    self.enter_context(
        mock.patch.object(
            node_utils, "create_mesh", return_value=self.mock_mesh
        )
    )
    self.mock_from_pretrained = self.enter_context(
        mock.patch.object(
            automodel.AutoModel,
            "from_pretrained",
            return_value=(mock.sentinel.ref_model, "/models/ref"),
        )
    )
    self.mock_rl_worker_cls = self.enter_context(
        mock.patch.object(
            rl_inference_worker,
            "InferenceWorker",
            return_value=mock.sentinel.core_worker,
        )
    )
    self.mock_exp_worker_cls = self.enter_context(
        mock.patch.object(
            exp_inference_worker,
            "InferenceWorker",
            return_value=mock.sentinel.worker_service,
        )
    )
    self.mock_serve_worker = self.enter_context(
        mock.patch.object(
            node_utils, "serve_worker", new_callable=mock.AsyncMock
        )
    )

  def test_main_loads_model_with_separate_load_dtype_and_serves_worker(self):
    inference.main(
        [
            "reference_model_config.model_name=Qwen3-4B",
            "reference_model_config.dtype=bfloat16",
            "reference_model_config.load_dtype=float32",
            "rl_training_config.compute_logps_micro_batch_size=4",
            "cluster_config.ports.inference=20052",
        ],
        context=self.mock_context,
    )

    self.mock_context.jax.initialize.assert_called_once()
    self.mock_from_pretrained.assert_called_once_with(
        model_id="Qwen3-4B",
        mesh=self.mock_mesh,
        model_path="/models/ref",
        dtype="bfloat16",
        load_dtype="float32",
    )
    self.mock_rl_worker_cls.assert_called_once_with(
        {"reference": mock.sentinel.ref_model}
    )
    self.mock_exp_worker_cls.assert_called_once_with(
        mock.sentinel.core_worker,
        worker_id="reference-inference-0",
        pad_id=2,
        eos_id=9,
        chunk_size=4,
        execution_context=self.mock_mesh,
    )
    self.mock_serve_worker.assert_awaited_once_with(
        mock.sentinel.worker_service,
        process_context=self.mock_context,
        role="inference",
        worker_id_str="reference-inference-0",
        port=20052,
    )

  def test_main_raises_without_discovery_context(self):
    with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
      inference.main([], context=None)


if __name__ == "__main__":
  absltest.main()
