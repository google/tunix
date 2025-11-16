# Copyright 2025 Google LLC
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

# BEGIN-GOOGLE-INTERNAL
# Tests for Llama3 model parameter loading from safetensors files.
# WARNING: This test is intended for external environments, such as GCE.
# It should not be run as part of a standard internal codebase or Blaze build.

# Setup:
# 1. Run `huggingface-cli login` to authenticate with Hugging Face
# 2. Ensure you have the corresponding model access.

# Usage:
# Script: python params_test.py
# Jupyter: %run params_test.py

# Each test validates model loading, device placement, and display
# functionality.
# Tests are skipped if model paths are not configured.
# END-GOOGLE-INTERNAL

import unittest

from absl.testing import parameterized
from flax import nnx
import jax
from tunix.models.gemma3 import params as gemma3_params_lib
import numpy as np
from flax.traverse_util import flatten_dict


class GemmaParamsTest(parameterized.TestCase):

  @parameterized.named_parameters(
    dict(
        testcase_name="gemma3",
        model_type="gemma3",
    ),
    dict(
        testcase_name="gemma2",
        model_type="gemma2",
    ),
  )
  def test_map_from_upstream_checkpoint(self, model_type):
    # Tiny shapes to demonstrate logic only
    embed         = np.arange(5*3, dtype=np.float32).reshape(5, 3)        # (vocab=5, dim=3)
    final_scale   = np.arange(3, dtype=np.float32)                        # (3,)
    gate_up       = np.arange(2*6*3, dtype=np.float32).reshape(2, 6, 3)   # -> two (3,6) after .T
    down          = np.arange(6*3, dtype=np.float32).reshape(6, 3)        # stays (6,3)
    q_w           = np.arange(4*3*2, dtype=np.float32).reshape(4, 3, 2)   # (4,3,2)
    kv_w          = np.arange(2*1*3*2, dtype=np.float32).reshape(2, 1, 3, 2)  # (2,1,3,2)
    o_w           = np.arange(4*2*3, dtype=np.float32).reshape(4, 2, 3)   # (4,2,3)
    pre_attn      = np.arange(3, dtype=np.float32)
    post_attn     = np.arange(3, dtype=np.float32)
    pre_ffw       = np.arange(3, dtype=np.float32)
    post_ffw      = np.arange(3, dtype=np.float32)
    siglip_dummy  = np.array([1.0], dtype=np.float32)
    mm_dummy      = np.array([2.0], dtype=np.float32)

    upstream = {
      "transformer/embedder": {"input_embedding": embed},
      "transformer/final_norm": {"scale": final_scale},

      # Should be skipped:
      "transformer/siglip_encoder": {"whatever": siglip_dummy},
      "transformer/embedder/mm_patch": {"kernel": mm_dummy},

      # Layer 0 (tiny shapes)
      "transformer/layer_0/attn/_key_norm":     {"scale": np.arange(2, dtype=np.float32)},
      "transformer/layer_0/attn/_query_norm":   {"scale": np.arange(2, dtype=np.float32)},
      "transformer/layer_0/attn/attn_vec_einsum": {"w": o_w},
      "transformer/layer_0/attn/kv_einsum":       {"w": kv_w},
      "transformer/layer_0/attn/q_einsum":        {"w": q_w},
      "transformer/layer_0/mlp/gating_einsum":    {"w": gate_up},
      "transformer/layer_0/mlp/linear":           {"w": down},
      "transformer/layer_0/post_attention_norm":  {"scale": post_attn},
      "transformer/layer_0/post_ffw_norm":        {"scale": post_ffw},
      "transformer/layer_0/pre_attention_norm":   {"scale": pre_attn},
      "transformer/layer_0/pre_ffw_norm":         {"scale": pre_ffw},
    }

    mapped = gemma3_params_lib.map_from_upstream_checkpoint(upstream, model_type)
    flat_m = flatten_dict(mapped)  # tuple keys

    # --- Keys & shapes we expect after mapping (tiny) ---
    expected = {
      ('embedder', 'input_embedding'):              (5, 3),
      ('final_norm', 'scale'):                      (3,),

      ('layers', 0, 'attn', '_key_norm', 'scale'):  (2,),
      ('layers', 0, 'attn', '_query_norm', 'scale'):(2,),
      ('layers', 0, 'attn', 'attn_vec_einsum', 'w'):(4, 2, 3),
      ('layers', 0, 'attn', 'kv_einsum', 'w'):      (2, 1, 3, 2),
      ('layers', 0, 'attn', 'q_einsum', 'w'):       (4, 3, 2),

      ('layers', 0, 'mlp', 'down_proj', 'kernel'):  (6, 3),
      ('layers', 0, 'mlp', 'gate_proj', 'kernel'):  (3, 6),  # from gate_up[0].T
      ('layers', 0, 'mlp', 'up_proj', 'kernel'):    (3, 6),  # from gate_up[1].T

      ('layers', 0, 'post_attn_norm' if model_type=="gemma2" else 'post_attention_norm', 'scale'):     (3,),
      ('layers', 0, 'post_ffw_norm', 'scale'):      (3,),
      ('layers', 0, 'pre_attention_norm', 'scale'): (3,),
      ('layers', 0, 'pre_ffw_norm', 'scale'):       (3,),
    }

    # 1) keys and shapes
    for k, shp in expected.items():
      assert k in flat_m, f"Missing key {k}"
      assert flat_m[k].shape == shp, f"Shape mismatch for {k}: got {flat_m[k].shape}, want {shp}"

    # 2) value checks for transforms & pass-through
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'mlp', 'gate_proj', 'kernel')],
      upstream["transformer/layer_0/mlp/gating_einsum"]["w"][0].T,
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'mlp', 'up_proj', 'kernel')],
      upstream["transformer/layer_0/mlp/gating_einsum"]["w"][1].T,
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'mlp', 'down_proj', 'kernel')],
      upstream["transformer/layer_0/mlp/linear"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'attn', 'attn_vec_einsum', 'w')],
      upstream["transformer/layer_0/attn/attn_vec_einsum"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'attn', 'kv_einsum', 'w')],
      upstream["transformer/layer_0/attn/kv_einsum"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'attn', 'q_einsum', 'w')],
      upstream["transformer/layer_0/attn/q_einsum"]["w"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'post_attn_norm', 'scale') if model_type=="gemma2" else
      ('layers', 0, 'post_attention_norm', 'scale')],
      upstream["transformer/layer_0/post_attention_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'post_ffw_norm', 'scale')],
      upstream["transformer/layer_0/post_ffw_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'pre_attention_norm', 'scale')],
      upstream["transformer/layer_0/pre_attention_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('layers', 0, 'pre_ffw_norm', 'scale')],
      upstream["transformer/layer_0/pre_ffw_norm"]["scale"],
    )
    np.testing.assert_array_equal(
      flat_m[('embedder', 'input_embedding')],
      upstream["transformer/embedder"]["input_embedding"],
    )
    np.testing.assert_array_equal(
      flat_m[('final_norm', 'scale')],
      upstream["transformer/final_norm"]["scale"],
    )

    # 3) ensure skipped subtrees absent
    assert not any(k[0] == 'siglip_encoder' for k in flat_m.keys())
    assert ('embedder', 'mm_patch') not in mapped.get('embedder', {})

if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  try:
    get_ipython()
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(Llama3ParamsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  except NameError:
    # Running as a script - use absltest.main()
    absltest.main()
