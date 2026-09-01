# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Router replay on the non-experimental RL stack, with a MaxText model.

`tunix/rl` reaches the model through `compute_per_token_logps`, which only
forwards `forced_routed_experts` to models that advertise the kwarg. MaxText's
`TunixMaxTextAdapter` does; tunix's own models do not. These tests pin both
halves of that, and that the replay actually changes what the trainer computes.

Skipped unless MaxText is importable, since tunix does not depend on it.
"""

import os
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
import jax
import numpy as np
from tunix.rl import common

try:
  from flax import nnx
  from jax.sharding import Mesh

  from maxtext.configs import pyconfig
  from maxtext.integration.tunix import tunix_adapter
  from maxtext.models import models
  from maxtext.utils import maxtext_utils

  MAXTEXT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
  MAXTEXT_AVAILABLE = False

PROMPT_LEN = 4
COMPLETION_LEN = 4
SEQ_LEN = PROMPT_LEN + COMPLETION_LEN
NUM_LAYERS = 2
TOP_K = 2
NUM_EXPERTS = 4
PAD_ID = 0
EOS_ID = 1


class ModelCallGateTest(absltest.TestCase):
  """Only models that accept the kwarg may be handed replayed routing."""

  def test_gate_matches_the_call_signature(self):

    class WithReplay:

      def __call__(self, x, forced_routed_experts=None):
        del x, forced_routed_experts

    class WithoutReplay:

      def __call__(self, x):
        del x

    self.assertTrue(
        common.model_call_contains(WithReplay(), "forced_routed_experts")
    )
    self.assertFalse(
        common.model_call_contains(WithoutReplay(), "forced_routed_experts")
    )

  @absltest.skipUnless(MAXTEXT_AVAILABLE, "requires MaxText")
  def test_maxtext_adapter_accepts_replay(self):
    """The gate is worthless if MaxText's real adapter does not pass it."""
    self.assertTrue(
        common._call_contains_by_type(  # pylint: disable=protected-access
            tunix_adapter.TunixMaxTextAdapter, "forced_routed_experts"
        )
    )


@absltest.skipUnless(MAXTEXT_AVAILABLE, "requires MaxText")
class ReplayThroughMaxTextTest(absltest.TestCase):
  """Replayed routing must change the log-probs the trainer computes."""

  def setUp(self):
    super().setUp()
    self.enterContext(
        mock.patch.dict(
            os.environ, {"NEW_MODEL_DESIGN": "1", "SKIP_JAX_PRECOMPILE": "1"}
        )
    )
    base_yml = os.path.join(
        os.path.dirname(maxtext_utils.__file__), "..", "configs", "base.yml"
    )
    cfg = pyconfig.initialize(
        [sys.argv[0], base_yml],
        enable_checkpointing=False,
        log_config=False,
        skip_jax_distributed_system=True,
        override_model_config=True,
        model_name="qwen3.5-35b-a3b",
        attention="dot_product",
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        base_emb_dim=128,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=128,
        # Qwen3.5 enables MRoPE, whose mrope_section would pin head_dim to the
        # full model's 256. Irrelevant to routing.
        use_mrope=False,
        partial_rotary_factor=0.25,
        base_mlp_dim=128,
        base_moe_mlp_dim=128,
        vocab_size=200,
        max_target_length=SEQ_LEN,
        max_prefill_predict_length=PROMPT_LEN,
        per_device_batch_size=1.0,
        run_name="rl_router_replay_test",
        base_output_directory=os.path.join(
            tempfile.gettempdir(), "rl_router_replay_test"
        ),
        base_num_decoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        scan_layers=False,
        enable_dropout=False,
        weight_dtype="float32",
        dtype="float32",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    base = models.Transformer(
        config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0)
    )
    self.model = tunix_adapter.TunixMaxTextAdapter(base, pad_id=PAD_ID)
    self.prompt = np.arange(10, 10 + PROMPT_LEN, dtype=np.int32)[None, :]
    self.completion = np.arange(20, 20 + COMPLETION_LEN, dtype=np.int32)[
        None, :
    ]

  def _logps(self, routed):
    graphdef, state = nnx.split(self.model)
    return np.asarray(
        common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens=jax.numpy.asarray(self.prompt),
            completion_tokens=jax.numpy.asarray(self.completion),
            pad_id=PAD_ID,
            eos_id=EOS_ID,
            routed_experts=routed,
        )
    )

  def _routing(self, first, second):
    routed = np.zeros((1, SEQ_LEN, NUM_LAYERS, TOP_K), dtype=np.int32)
    routed[..., 0] = first
    routed[..., 1] = second
    return jax.numpy.asarray(routed)

  def test_replay_changes_the_logps(self):
    """Two different replays must disagree.

    Comparing log-probs rather than just asserting they are finite: dropping
    the routing would still produce perfectly good log-probs, just the
    normally-routed ones. Note the expert *set* has to differ -- permuting the
    top-k slots yields identical output, since the MoE scatter accumulates
    both experts' contributions.
    """
    logps_a = self._logps(self._routing(1, 3))
    logps_b = self._logps(self._routing(0, 2))
    self.assertFalse(np.isnan(logps_a).any(), "replay produced NaN log-probs")
    self.assertFalse(
        np.allclose(logps_a, logps_b),
        "forcing different experts gave identical log-probs, so the replayed "
        "routing is not reaching the MoE layers",
    )

  def test_no_replay_still_works(self):
    """Replay stays opt-in: without routing this is an ordinary forward."""
    logps = self._logps(None)
    self.assertEqual(logps.shape[0], 1)
    self.assertFalse(np.isnan(logps).any())


if __name__ == "__main__":
  absltest.main()
