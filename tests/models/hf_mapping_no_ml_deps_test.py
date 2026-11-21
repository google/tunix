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

"""Zero-ML-dependency unit tests for `tunix.models.hf_mapping`."""

import subprocess
import sys

from absl.testing import absltest
from tunix.models import hf_mapping


class HfMappingNoMlDepsTest(absltest.TestCase):
  """Verifies 12-Op schema, algebraic inversion, and zero JAX/Torch imports."""

  def test_no_jax_or_torch_imported(self):
    if sys.executable:
      subprocess.run(
          [
              sys.executable,
              "-c",
              (
                  "import importlib.util, sys; spec ="
                  " importlib.util.spec_from_file_location('hf_mapping',"
                  f" {hf_mapping.__file__!r}); mod ="
                  " importlib.util.module_from_spec(spec);"
                  " sys.modules['hf_mapping'] = mod;"
                  " spec.loader.exec_module(mod); assert 'jax' not in"
                  " sys.modules and 'torch' not in sys.modules"
              ),
          ],
          check=True,
      )
    else:
      self.assertNotIn("jax", sys.modules)
      self.assertNotIn("torch", sys.modules)

  def test_all_12_ops_invert_round_trip(self):
    manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "model_type": "test_12_ops",
        "rules": [
            {
                "source_keys": ["k1"],
                "target_keys": ["t1"],
                "ops": [
                    {"transpose": {"axes": [2, 0, 1]}},
                    {
                        "reshape": {
                            "src_shape": [4, 2, 3],
                            "dst_shape": [8, 3],
                        }
                    },
                    {"pad": {"axis": 0, "sizes": [8, 2]}},
                    {"unpad": {"axis": 0, "sizes": [8, 2]}},
                    {"repeat": {"axis": 1, "times": 2}},
                    {"unrepeat": {"axis": 1, "times": 2}},
                    {
                        "cast": {
                            "src_dtype": "float32",
                            "dst_dtype": "bfloat16",
                        }
                    },
                ],
                "sharding": ["tp", None],
            },
            {
                "source_keys": ["a", "b"],
                "target_keys": ["ab_concat"],
                "ops": [{"concat": {"axis": 0, "sizes": [4, 4]}}],
            },
            {
                "source_keys": ["ab_split"],
                "target_keys": ["s0", "s1"],
                "ops": [{"split": {"axis": 0, "sizes": [3, 5]}}],
            },
            {
                "source_keys": ["e0", "e1"],
                "target_keys": ["e_stack"],
                "ops": [{"stack": {"axis": 0}}],
            },
            {
                "source_keys": ["e_unstack"],
                "target_keys": ["u0", "u1"],
                "ops": [{"unstack": {"axis": 0}}],
            },
            {
                "source_keys": ["id_src"],
                "target_keys": ["id_dst"],
                "ops": [{"scale": {"factor": 2.0}}],
            },
        ],
        "tied_keys": [["embed", "lm_head"]],
        "dropped_keys": ["rotary_emb.inv_freq"],
    }
    inv = hf_mapping.invert_manifest(manifest)
    self.assertEqual(inv["tied_keys"], [])
    self.assertIn("lm_head", inv["dropped_keys"])
    self.assertNotIn("sharding", inv["rules"][0])
    round_trip = hf_mapping.invert_manifest(inv)
    expected_rules = [
        {k: v for k, v in r.items() if k != "sharding"}
        for r in manifest["rules"]
    ]
    self.assertEqual(round_trip["rules"], expected_rules)

  def test_spec_dataclass_and_parse_manifest(self):
    spec = hf_mapping.WeightMappingSpec(
        rules=(
            hf_mapping.ParamRule(
                source_keys=("hf.w",),
                target_keys=("tunix.w",),
                ops=({"transpose": {"axes": [1, 0]}},),
                sharding=("tp", None),
            ),
            hf_mapping.ParamRule(
                source_keys=("shared.norm",),
                target_keys=("shared.norm",),
                ops=({"scale": {"factor": 2.0}},),
            ),
        ),
        tied_keys=(("embed", "lm_head"),),
        dropped_keys=("unused",),
        model_type="test_model",
    )
    as_dict = spec.to_dict()
    self.assertEqual(as_dict["model_type"], "test_model")
    self.assertTrue(spec.to_json().startswith('{"dropped_keys":'))
    self.assertEqual(hf_mapping.parse_manifest(spec), as_dict)
    self.assertEqual(hf_mapping.parse_manifest(spec.to_json()), as_dict)
    self.assertEqual(
        hf_mapping.WeightMappingSpec.from_json(spec.to_json()), spec
    )

    inv = hf_mapping.invert_manifest(spec)
    self.assertEqual(inv["rules"][0]["source_keys"], ["tunix.w"])
    self.assertEqual(inv["rules"][0]["target_keys"], ["hf.w"])
    self.assertEqual(inv["rules"][1]["ops"], [{"scale": {"factor": 0.5}}])
    self.assertEqual(inv["tied_keys"], [])
    self.assertIn("lm_head", inv["dropped_keys"])

  def test_validation_rejects_invalid_ops_and_rules(self):
    for op_name, bad_kw in (
        ("unknown_op", {}),
        ("stack", {"axis": "0"}),
        ("stack", {"axis": True}),
        ("unstack", {"axis": "0"}),
        ("unstack", {"axis": False}),
        ("transpose", {"axes": []}),
        ("transpose", {"axes": [0, 0]}),
        ("reshape", {"src_shape": [], "dst_shape": []}),
        ("reshape", {"src_shape": [2], "dst_shape": []}),
        ("reshape", {"src_shape": [-2, -3], "dst_shape": [6]}),
        ("reshape", {"src_shape": [6], "dst_shape": [-2, -3]}),
        ("reshape", {"src_shape": [2, 3], "dst_shape": [2, 2]}),
        ("concat", {"axis": "0", "sizes": [2, 2]}),
        ("concat", {"axis": True, "sizes": [2, 2]}),
        ("concat", {"axis": 0, "sizes": []}),
        ("concat", {"axis": 0, "sizes": [0, 2]}),
        ("split", {"axis": "0", "sizes": [2, 2]}),
        ("split", {"axis": True, "sizes": [2, 2]}),
        ("split", {"axis": 0, "sizes": []}),
        ("split", {"axis": 0, "sizes": [0, 2]}),
        ("pad", {"axis": "0", "sizes": [1, 1]}),
        ("pad", {"axis": True, "sizes": [1, 1]}),
        ("pad", {"axis": 0, "sizes": [4]}),
        ("pad", {"axis": 0, "sizes": [1, -1]}),
        ("unpad", {"axis": "0", "sizes": [1, 1]}),
        ("unpad", {"axis": True, "sizes": [1, 1]}),
        ("unpad", {"axis": 0, "sizes": [4]}),
        ("unpad", {"axis": 0, "sizes": [1, -1]}),
        ("repeat", {"axis": "0", "times": 2}),
        ("repeat", {"axis": True, "times": 2}),
        ("repeat", {"axis": 0, "times": True}),
        ("repeat", {"axis": 0, "times": 0}),
        ("unrepeat", {"axis": "0", "times": 2}),
        ("unrepeat", {"axis": True, "times": 2}),
        ("unrepeat", {"axis": 0, "times": False}),
        ("unrepeat", {"axis": 0, "times": 0}),
        ("cast", {"src_dtype": "float32", "dst_dtype": ""}),
        ("scale", {"factor": 0.0}),
        ("scale", {"factor": float("inf")}),
        ("scale", {"factor": float("nan")}),
    ):
      with self.assertRaises(ValueError):
        hf_mapping.validate_op(op_name, bad_kw)
      with self.assertRaises(ValueError):
        hf_mapping.invert_op(op_name, bad_kw)

    with self.assertRaises(ValueError):
      hf_mapping.ParamRule(source_keys=(), target_keys=("a",))
    with self.assertRaises(ValueError):
      hf_mapping.ParamRule.from_dict(
          {"source_keys": ["a"], "target_keys": ["b"], "ops": [], "extra": 1}
      )

    for bad_rule in (
        {"source_keys": [], "target_keys": ["a"], "ops": []},
        {"source_keys": "abc", "target_keys": ["a"], "ops": []},
        {
            "source_keys": ["a"],
            "target_keys": ["b"],
            "ops": [],
            "sharding": "tp",
        },
        {"source_keys": ["a"], "target_keys": ["b"], "ops": "not_a_list"},
        {"source_keys": ["a"], "target_keys": ["b"], "ops": [{}]},
        {"source_keys": ["a"], "target_keys": ["b"], "ops": [{"unknown": {}}]},
    ):
      with self.assertRaises(ValueError):
        hf_mapping.validate_rule(bad_rule)

    for bad_manifest in (
        {"schema_version": "0.0", "rules": []},
        "[]",
        "123",
        {"rules": [], "dropped_keys": "abc"},
        {"rules": [], "tied_keys": ["ab"]},
        {"rules": "not_a_list"},
        {"rules": [{"source_keys": [], "target_keys": ["a"], "ops": []}]},
    ):
      with self.assertRaises(ValueError):
        hf_mapping.parse_manifest(bad_manifest)


if __name__ == "__main__":
  absltest.main()
