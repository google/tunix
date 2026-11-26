# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for `model_contracts` tile policies and `round_up` helper."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.zero_tim_kernel import model_contracts

_EXPECTED_TILES = {
    "qwen1p7b": (128, 256, 256),
    "qwen1p7b_tp1": (128, 256, 256),
    "qwen1p7b_tp2": (128, 256, 256),
    "qwen4b": (128, 128, 128),
    "qwen4b_tp4": (128, 128, 128),
    "qwen8b": (128, 256, 256),
    "qwen8b_tp1": (128, 256, 256),
    "qwen8b_tp2": (128, 256, 256),
    "qwen8b_tp8": (128, 128, 128),
    "qwen32b": (128, 128, 128),
}


class ModelContractsTest(parameterized.TestCase):

  def test_round_up(self):
    self.assertEqual(model_contracts.round_up(1, 128), 128)
    self.assertEqual(model_contracts.round_up(128, 128), 128)
    self.assertEqual(model_contracts.round_up(1216, 128), 1280)
    self.assertEqual(model_contracts.round_up(37984, 512), 38400)
    self.assertEqual(model_contracts.round_up(37984, 256), 38144)
    with self.assertRaises(ValueError):
      model_contracts.round_up(0, 128)
    with self.assertRaises(ValueError):
      model_contracts.round_up(128, 0)

  def test_registry_matches_expected_contracts(self):
    self.assertEqual(model_contracts.LM_HEAD_TILE, 256)
    self.assertEqual(model_contracts.SWIGLU_FEATURE_TILE, 256)
    self.assertCountEqual(model_contracts.CONTRACTS, _EXPECTED_TILES)

  @parameterized.parameters(*sorted(_EXPECTED_TILES))
  def test_contract_tiles(self, name):
    contract = model_contracts.get_contract(name)
    self.assertEqual(contract.tiles, _EXPECTED_TILES[name])

  def test_build_contract_with_custom_vocab(self):
    contract = model_contracts.build_contract(
        "tiny",
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        tp_size=4,
        tiles=(128, 256, 256),
        vocab_size=1000,
    )
    self.assertEqual(contract.vocab_size, 1000)
    self.assertEqual(contract.tiles, (128, 256, 256))

  def test_lookup(self):
    self.assertEqual(
        model_contracts.find_contract(hidden_size=4096, tp_size=4).name,
        "qwen8b",
    )
    self.assertEqual(
        model_contracts.find_contract(hidden_size=2560, tp_size=8).name,
        "qwen4b",
    )
    with self.assertRaises(KeyError):
      model_contracts.find_contract(hidden_size=1234, tp_size=4)
    with self.assertRaises(KeyError):
      model_contracts.get_contract("llama")

  def test_validation_fails_closed(self):
    self.assertEqual(model_contracts.ModelContract(tp_size=3).vocab_size, 0)
    with self.assertRaises(ValueError):
      model_contracts.ModelContract(block_m=0)
    with self.assertRaises(ValueError):
      model_contracts.ModelContract(tp_size=0)
    with self.assertRaises(ValueError):
      model_contracts.ModelContract(tp_size=8, vocab_size=1002)
    with self.assertRaises(ValueError):
      model_contracts.ModelContract(tp_size=4, num_attention_heads=15)
    with self.assertRaises(ValueError):
      model_contracts.build_contract(
          "bad",
          hidden_size=1024,
          intermediate_size=3072,
          num_attention_heads=15,
          num_kv_heads=8,
          head_dim=128,
          tp_size=4,
          tiles=(128, 256, 256),
      )


if __name__ == "__main__":
  absltest.main()
