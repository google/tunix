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

"""End to end on CPU: MaxText MoE drop audit and dropless replay through Tunix's GRPO loss.

The trainer is built the way the e2e run builds it, minus the checkpoint: MaxText's training engine on a
`TunixMaxTextAdapter` model (ring-of-experts ragged MoE, cp-as-ep mesh, 8 CPU devices, rbf 2.5,
retry_when_tokens_dropped, log_moe_drops), wrapped in Tunix's `TrainerWorker`, configured as
`DistributedRLEngine.configure_worker` does (`GRPOAdapter.loss_fn()`, `build_gen_model_input_fn`), driven with
`fwd_bwd(request=...)` and `update()`.

Two payloads, both from Tunix's assembler:
- the TRAINER_WARMUP_COMPILE payload (`trainer_warmup.build_warmup_payload`): every token the same id and every
  routed expert UNSET, so the router picks the same experts for every token. Its MoE load is concentrated on a
  few shards and must overflow at rbf 2.5, be replayed dropless, and be logged with a nonzero discarded count and
  `dropped_assignments_used=0`;
- a replayed-routing payload whose routed experts are spread evenly over the expert shards (load ratio 1.0): no
  overflow, no replay.

Skipped unless MaxText is importable.
"""

import os
import subprocess
import sys

from absl.testing import absltest

_PASS = "MOE_DROP_AUDIT_E2E_PASSED"
_NUM_DEVICES = 8

try:
  import maxtext  # pylint: disable=unused-import

  MAXTEXT_AVAILABLE = True
except ImportError:  # pragma: no cover
  MAXTEXT_AVAILABLE = False


class MoeDropAuditE2ETest(absltest.TestCase):

  @absltest.skipUnless(MAXTEXT_AVAILABLE, "requires MaxText")
  def test_grpo_loss_through_trainer_worker(self):
    env = os.environ.copy()
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
    env["JAX_PLATFORMS"] = "cpu"
    env["NEW_MODEL_DESIGN"] = "1"
    result = subprocess.run(
        [sys.executable, __file__, "--child"], env=env, capture_output=True, text=True, check=False
    )
    print(result.stdout[-6000:])
    lines = [l for l in (result.stdout + result.stderr).splitlines() if "MOE_DROP_AUDIT" in l]
    print("\n".join(lines))
    self.assertEqual(result.returncode, 0, result.stderr[-8000:])
    self.assertIn(_PASS, result.stdout)


def _child():
  # pylint: disable=g-import-not-at-top
  import logging
  import re
  import tempfile
  from unittest import mock

  from absl import logging as absl_logging
  from flax import nnx
  import jax
  from jax.sharding import Mesh
  import numpy as np

  from maxtext.configs import pyconfig
  from maxtext.integration.tunix import tunix_adapter
  from maxtext.models import models
  from maxtext.training_engine import maxtext_engine
  from maxtext.utils import maxtext_utils
  from tunix.experimental.common import datatypes
  from tunix.experimental.examples.deepswe_dist import trainer_warmup
  from tunix.experimental.orchestrator import algorithm_adapter
  from tunix.experimental.orchestrator import batch_assembly
  from tunix.experimental.worker import trainer_worker
  from tunix.rl import algorithm_config

  seq = 64
  num_layers = 2
  num_experts = 16
  top_k = 4
  base_yml = os.path.join(os.path.dirname(maxtext_utils.__file__), "..", "configs", "base.yml")
  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml],
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      override_model_config=True,
      model_name="qwen3.5-35b-a3b",
      attention="dot_product",
      num_experts=num_experts,
      num_experts_per_tok=top_k,
      base_emb_dim=128,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=128,
      use_mrope=False,
      partial_rotary_factor=0.25,
      base_mlp_dim=128,
      base_moe_mlp_dim=128,
      vocab_size=256,
      max_target_length=seq,
      max_prefill_predict_length=seq,
      per_device_batch_size=0.25,
      run_name="moe_drop_audit_e2e",
      base_output_directory=os.path.join(tempfile.gettempdir(), "moe_drop_audit_e2e"),
      base_num_decoder_layers=num_layers,
      num_decoder_layers=num_layers,
      inhomogeneous_layer_cycle_interval=1,
      scan_layers=False,
      enable_dropout=False,
      weight_dtype="float32",
      dtype="bfloat16",
      megablox=True,
      sparse_matmul=True,
      use_tokamax_gmm=False,
      use_gmm_v2=False,
      use_ring_of_experts=True,
      use_ragged_sort=True,
      use_custom_sort_vjp=False,
      custom_mesh_and_rule="cp-as-ep",
      ici_fsdp_parallelism=1,
      ici_context_parallelism=4,
      ici_expert_parallelism=2,
      context_parallel_load_balance=False,
      ring_of_experts_local_routing=True,
      packing=True,
      ragged_buffer_factor=2.5,
      retry_when_tokens_dropped=True,
      retry_num_moe_token_chunks=2,
      log_moe_drops=True,
  )
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  print("mesh", dict(mesh.shape), flush=True)
  base = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
  adapter = tunix_adapter.TunixMaxTextAdapter(base, pad_id=0)
  with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=adapter):
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
  worker = trainer_worker.TrainerWorker(trainer_factory=lambda: engine, worker_id="e2e", execution_context=mesh)
  worker.initialize()

  algo = algorithm_adapter.GRPOAdapter(
      algo_config=algorithm_config.GRPOConfig(num_generations=2, beta=0.0, loss_agg_mode="token-mean", temperature=1.0),
      mini_batch_size=2,
      train_micro_batch_size=2,
      max_packed_len=seq,
      max_response_length=seq // 2,
  )
  batch_config = batch_assembly.BatchConfig(
      pad_id=0,
      max_prompt_length=seq // 2,
      max_response_length=seq // 2,
      max_seq_token_per_tpu=seq,
      max_segments_per_packed_row=2,
      trainer_fsdp=1,
      trainer_dp=1,
      trainer_expert=2,
  )
  assembler = batch_assembly.create_batch_assembler(
      num_generations=2, mini_batch_size=2, train_micro_batch_size=2, batch_config=batch_config
  )
  # configure_worker, ACTOR role.
  a_pad = getattr(assembler, "pad_id", 0)
  worker.with_loss_fn(algo.loss_fn(), has_aux=True)
  worker.with_gen_model_input_fn(algo.build_gen_model_input_fn(pad_id=a_pad, eos_id=getattr(assembler, "eos_id", a_pad)))

  warm = trainer_warmup.build_warmup_payload(
      algo=algo,
      batch_config=batch_config,
      num_generations=2,
      mini_batch_size=2,
      train_micro_batch_size=2,
      routed_experts_shape=(num_layers, top_k),
  )
  print("payload", batch_assembly.describe_payload(warm), flush=True)
  routed = np.asarray(warm.routed_experts)
  # Balanced replayed routing: token t uses experts {(t*top_k + j) % E}, i.e. every expert shard gets the same load.
  t = np.arange(routed.shape[1])[:, None, None]
  j = np.arange(top_k)[None, None, :]
  balanced_routed = np.broadcast_to(((t * top_k + j) % num_experts).astype(routed.dtype), routed.shape[1:])
  balanced = warm.replace(routed_experts=np.broadcast_to(balanced_routed, routed.shape).copy()) if hasattr(
      warm, "replace") else __import__("dataclasses").replace(
          warm, routed_experts=np.broadcast_to(balanced_routed, routed.shape).copy())

  worker.compile(dummy_data=warm)

  records = []

  class _Grab(logging.Handler):

    def emit(self, record):
      records.append(record.getMessage())

  absl_logging.set_verbosity(absl_logging.INFO)
  absl_logging.get_absl_logger().setLevel(logging.INFO)
  absl_logging.get_absl_logger().addHandler(_Grab())

  def step(payload, n):
    start = len(records)
    for m in range(n):
      worker.fwd_bwd(request=datatypes.TrainRequest(request_id=f"r{m}", payload=payload, metadata={}))
    worker.update()
    return [r for r in records[start:] if r.startswith("MOE_DROP_AUDIT")]

  warm_lines = step(warm, 2)
  bal_lines = step(balanced, 2)
  print("WARM:\n" + "\n".join(warm_lines), flush=True)
  print("BALANCED:\n" + "\n".join(bal_lines), flush=True)

  def field(line, name):
    return re.search(rf"{name}=(-?[0-9.]+)", line).group(1)

  audit = [l for l in warm_lines if l.startswith("MOE_DROP_AUDIT step=")]
  assert len(audit) == 2, warm_lines
  for l in audit:
    assert "replayed=True" in l and field(l, "dropped_assignments_used") == "0", l
  disc = [l for l in warm_lines if l.startswith("MOE_DROP_AUDIT_DISCARDED")]
  assert len(disc) == 2 and all(int(field(l, "dropped_assignments_discarded")) > 0 for l in disc), warm_lines
  assert all(float(field(l, "max_ratio")) > 2.5 for l in disc), disc
  step_line = [l for l in warm_lines if l.startswith("MOE_DROP_AUDIT_STEP")][0]
  assert "dropped_assignments_used=0 dropless_replays=2 " in step_line, step_line

  audit = [l for l in bal_lines if l.startswith("MOE_DROP_AUDIT step=")]
  assert len(audit) == 2, bal_lines
  for l in audit:
    assert "replayed=False" in l and field(l, "dropped_assignments_used") == "0", l
    assert abs(float(field(l, "max_ratio")) - 1.0) < 1e-3, l
  step_line = [l for l in bal_lines if l.startswith("MOE_DROP_AUDIT_STEP")][0]
  assert "dropped_assignments_used=0 dropless_replays=0 " in step_line, step_line
  print(_PASS, flush=True)


if __name__ == "__main__":
  if "--child" in sys.argv:
    _child()
  else:
    absltest.main()
