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
"""Checks the per-model tile/padding contracts resolve to the expected sites."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from tunix.zero_tim_kernel import model_contracts

# Expected (k_local, n_local) of q/k/v/o/gate/up/down for each contract.
_EXPECTED_SITES = {
    "qwen1p7b": (
        (2048, 512),
        (2048, 256),
        (2048, 256),
        (512, 2048),
        (2048, 1536),
        (2048, 1536),
        (1536, 2048),
    ),
    "qwen1p7b_tp1": (
        (2048, 2048),
        (2048, 1024),
        (2048, 1024),
        (2048, 2048),
        (2048, 6144),
        (2048, 6144),
        (6144, 2048),
    ),
    "qwen1p7b_tp2": (
        (2048, 1024),
        (2048, 512),
        (2048, 512),
        (1024, 2048),
        (2048, 3072),
        (2048, 3072),
        (3072, 2048),
    ),
    "qwen4b": (
        (2560, 512),
        (2560, 128),
        (2560, 128),
        (512, 2560),
        (2560, 1216),
        (2560, 1216),
        (1216, 2560),
    ),
    "qwen4b_tp4": (
        (2560, 1024),
        (2560, 256),
        (2560, 256),
        (1024, 2560),
        (2560, 2432),
        (2560, 2432),
        (2432, 2560),
    ),
    "qwen8b": (
        (4096, 1024),
        (4096, 256),
        (4096, 256),
        (1024, 4096),
        (4096, 3072),
        (4096, 3072),
        (3072, 4096),
    ),
    "qwen8b_tp1": (
        (4096, 4096),
        (4096, 1024),
        (4096, 1024),
        (4096, 4096),
        (4096, 12288),
        (4096, 12288),
        (12288, 4096),
    ),
    "qwen8b_tp2": (
        (4096, 2048),
        (4096, 512),
        (4096, 512),
        (2048, 4096),
        (4096, 6144),
        (4096, 6144),
        (6144, 4096),
    ),
    "qwen8b_tp8": (
        (4096, 512),
        (4096, 128),
        (4096, 128),
        (512, 4096),
        (4096, 1536),
        (4096, 1536),
        (1536, 4096),
    ),
    "qwen32b": (
        (5120, 1024),
        (5120, 128),
        (5120, 128),
        (1024, 5120),
        (5120, 3200),
        (5120, 3200),
        (3200, 5120),
    ),
}

# Expected (tiles, matmul_k_padding, matmul_n_padding, swiglu_feature_padding).
_EXPECTED_PADDING = {
    "qwen1p7b": ((128, 256, 256), {}, {37984: 38144}, {}),
    "qwen1p7b_tp1": ((128, 256, 256), {}, {151936: 152064}, {}),
    "qwen1p7b_tp2": ((128, 256, 256), {}, {75968: 76032}, {}),
    "qwen4b": (
        (128, 128, 128),
        {1216: 1280},
        {1216: 1280, 18992: 19200},
        {1216: 1280},
    ),
    "qwen4b_tp4": (
        (128, 128, 128),
        {2432: 2560},
        {2432: 2560, 37984: 38144},
        {2432: 2560},
    ),
    "qwen8b": ((128, 256, 256), {}, {37984: 38144}, {}),
    "qwen8b_tp1": ((128, 256, 256), {}, {151936: 152064}, {}),
    "qwen8b_tp2": ((128, 256, 256), {}, {75968: 76032}, {}),
    "qwen8b_tp8": ((128, 128, 128), {}, {18992: 19200}, {}),
    "qwen32b": (
        (128, 128, 128),
        {},
        {18992: 19200},
        {3200: 3328},
    ),
}


class ModelContractsTest(parameterized.TestCase):

  def test_registry_matches_expected_contracts(self):
    self.assertCountEqual(model_contracts.CONTRACTS, _EXPECTED_SITES)

  @parameterized.parameters(*sorted(_EXPECTED_SITES))
  def test_contract_is_consistent(self, name):
    model_contracts.get_contract(name).validate()

  @parameterized.parameters(*sorted(_EXPECTED_SITES))
  def test_derived_sites_match_expected_tables(self, name):
    contract = model_contracts.get_contract(name)
    self.assertEqual(
        tuple((site.k_local, site.n_local) for site in contract.sites),
        _EXPECTED_SITES[name],
    )
    self.assertEqual(
        [site.family for site in contract.sites if site.contract_parallel],
        ["o_proj", "down_proj"],
    )
    self.assertEqual(
        (
            contract.tiles,
            contract.matmul_k_padding,
            contract.matmul_n_padding,
            contract.swiglu_feature_padding,
        ),
        _EXPECTED_PADDING[name],
    )

  def test_build_contract_with_custom_vocab_validates(self):
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
    contract.validate()
    self.assertEqual(contract.vocab_size, 1000)
    self.assertEqual(contract.local_vocab, 250)
    self.assertEqual(contract.matmul_n_padding, {250: 256})

  @parameterized.parameters(*sorted(_EXPECTED_SITES))
  def test_lm_head_width_is_admitted(self, name):
    contract = model_contracts.get_contract(name)
    local_vocab = contract.local_vocab
    padded = contract.matmul_n_padding.get(local_vocab, local_vocab)
    self.assertEqual(padded % contract.block_n, 0)
    self.assertGreaterEqual(padded, local_vocab)

  def test_model_env(self):
    self.assertEqual(
        model_contracts.get_contract("qwen8b").model_env(),
        {
            "CANON_QWEN3_HIDDEN_SIZE": "4096",
            "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
            "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
            "CANON_QWEN3_NUM_KV_HEADS": "8",
            "CANON_QWEN3_HEAD_DIM": "128",
            "CANON_QWEN3_TP_SIZE": "4",
        },
    )

  def test_match_site(self):
    contract = model_contracts.get_contract("qwen8b")
    site = contract.match_site("model.layers.3.mlp.down_proj", "mn,np->mp")
    assert site is not None
    self.assertEqual(site.family, "down_proj")
    self.assertTrue(site.contract_parallel)
    q_site = contract.match_site("layers.0.attn.q_proj", "TD,NDH->TNH")
    assert q_site is not None
    self.assertEqual(q_site.family, "q_proj")
    self.assertIsNone(contract.match_site("model.embedder", "mn,np->mp"))
    with self.assertRaises(RuntimeError):
      contract.match_site("layers.0.attn.o_proj", "mn,np->mp")

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

  def test_validate_fails_closed(self):
    qwen8b = model_contracts.get_contract("qwen8b")
    with self.assertRaisesRegex(ValueError, "lm_head"):
      dataclasses.replace(qwen8b, matmul_n_padding={}).validate()
    with self.assertRaisesRegex(ValueError, "must grow"):
      dataclasses.replace(
          qwen8b, matmul_n_padding={37984: 38144, 512: 256}
      ).validate()
    qwen32b = model_contracts.get_contract("qwen32b")
    with self.assertRaisesRegex(ValueError, "SwiGLU"):
      dataclasses.replace(qwen32b, swiglu_feature_padding={}).validate()


if __name__ == "__main__":
  absltest.main()
