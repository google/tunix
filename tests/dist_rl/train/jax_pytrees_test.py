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

"""Tests for the JAX pytree registration of trainer payloads."""

import dataclasses

from absl.testing import absltest
import jax
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import jax_pytrees  # pylint: disable=unused-import


def _rl_payload(**overrides) -> datatypes.RLTrainerPayload:
  fields = dict(
      prompt_ids=np.array([1, 2], dtype=np.int32),
      prompt_mask=np.ones(2, dtype=np.float32),
      completion_ids=np.array([3, 4], dtype=np.int32),
      completion_mask=np.ones(2, dtype=np.float32),
      advantages=np.array([1.0, 2.0], dtype=np.float32),
      num_segments=5,
      metadata={"key": "value"},
  )
  fields.update(overrides)
  return datatypes.RLTrainerPayload(**fields)


class JaxPytreesTest(absltest.TestCase):

  def test_rl_payload_static_fields_are_not_leaves(self):
    payload = _rl_payload()
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    for leaf in leaves:
      self.assertIsInstance(leaf, np.ndarray)
    # 5 required arrays; the 8 optional arrays are None (empty subtrees).
    self.assertLen(leaves, 5)

    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(restored.num_segments, 5)
    self.assertFalse(restored.sampler_agreement_applied)
    self.assertEqual(restored.metadata, {"key": "value"})
    np.testing.assert_allclose(restored.advantages, [1.0, 2.0])

  def test_static_fields_match_dataclass_metadata(self):
    static = {
        f.name
        for f in dataclasses.fields(datatypes.RLTrainerPayload)
        if f.metadata == datatypes.STATIC_FIELD_METADATA
    }
    self.assertEqual(
        static, {"num_segments", "sampler_agreement_applied", "metadata"}
    )
    # Changing a static field changes the treedef (it is part of the
    # structure, i.e. a jit cache key), changing an array field does not.
    base = jax.tree.structure(_rl_payload())
    self.assertNotEqual(base, jax.tree.structure(_rl_payload(num_segments=6)))
    self.assertEqual(
        base,
        jax.tree.structure(
            _rl_payload(advantages=np.array([9.0, 9.0], dtype=np.float32))
        ),
    )

  def test_logprobs_scoring_payload_is_a_pytree_with_static_scoring_fields(
      self,
  ):
    static = {
        f.name
        for f in dataclasses.fields(datatypes.LogprobsScoringPayload)
        if f.metadata == datatypes.STATIC_FIELD_METADATA
    }
    self.assertEqual(static, {"pad_id", "eos_id", "temperature"})
    base = datatypes.LogprobsScoringPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        pad_id=0,
        eos_id=2,
        temperature=1.0,
    )
    self.assertLen(jax.tree.leaves(base), 2)
    base_struct = jax.tree.structure(base)
    self.assertNotEqual(
        base_struct,
        jax.tree.structure(dataclasses.replace(base, temperature=0.7)),
    )
    self.assertEqual(
        base_struct,
        jax.tree.structure(
            dataclasses.replace(
                base, prompt_ids=np.array([[9, 9]], dtype=np.int32)
            )
        ),
    )

  def test_tree_map_preserves_type_and_statics(self):
    payload = _rl_payload()
    doubled = jax.tree.map(lambda x: x * 2, payload)
    self.assertIsInstance(doubled, datatypes.RLTrainerPayload)
    np.testing.assert_allclose(doubled.advantages, [2.0, 4.0])
    self.assertEqual(doubled.num_segments, 5)
    self.assertIsNone(doubled.segment_ids)

  def test_sft_payload_is_a_pytree(self):
    payload = datatypes.SFTTrainerPayload(
        token_ids=np.array([[1, 2]], dtype=np.int32),
        token_mask=np.ones((1, 2), dtype=np.float32),
    )
    self.assertLen(jax.tree.leaves(payload), 2)

  def test_payload_crosses_jit_boundary(self):
    @jax.jit
    def total(p: datatypes.RLTrainerPayload):
      return p.advantages.sum() + p.num_segments

    self.assertEqual(float(total(_rl_payload())), 3.0 + 5)


if __name__ == "__main__":
  absltest.main()
