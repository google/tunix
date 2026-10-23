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

"""Synthetic forward/backward smoke test for eight-device MaxText pipelines.

Uses a tiny randomly initialized Qwen3-shaped model, not pretrained weights.
Run each layout in a separate process; see maxtext_pipeline_smoke.ipynb.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--pipeline-stages", type=int, choices=(2, 4), required=True
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--targeted-imports",
      action="store_true",
      help=(
          "Load real submodules without executing the unrelated top-level API."
      ),
  )
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  report = {
      "ok": False,
      "scope": "targeted_modules" if args.targeted_imports else "full_import",
      "weights": "random_tiny_qwen3_shape",
      "data": "synthetic_integer_tokens",
      "training_throughput_claim": False,
      "pipeline_stages": args.pipeline_stages,
      "tensor_parallelism": 8 // args.pipeline_stages,
      "tunix_commit": (
          subprocess.check_output(
              ["git", "rev-parse", "HEAD"], cwd=root, text=True
          ).strip()
      ),
      "source_sha256": {
          name: hashlib.sha256((root / name).read_bytes()).hexdigest()
          for name in (
              "tunix/models/automodel.py",
              "tunix/models/maxtext_parallelism.py",
          )
      },
  }
  try:
    if args.targeted_imports:
      package = types.ModuleType("tunix")
      package.__path__ = [str(root / "tunix")]
      package.__spec__ = importlib.machinery.ModuleSpec(
          "tunix", loader=None, is_package=True
      )
      sys.modules["tunix"] = package
    sys.path.insert(0, str(root))
    from flax import linen as nn  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import maxtext  # pylint: disable=g-import-not-at-top
    import optax  # pylint: disable=g-import-not-at-top
    from tunix.models import automodel  # pylint: disable=g-import-not-at-top
    from tunix.models import maxtext_parallelism  # pylint: disable=g-import-not-at-top

    devices = jax.devices()
    report["environment"] = {
        "jax": jax.__version__,
        "flax": importlib.metadata.version("flax"),
        "maxtext": maxtext.__version__,
        "maxtext_direct_url": json.loads(
            importlib.metadata.distribution("maxtext").read_text(
                "direct_url.json"
            )
            or "null"
        ),
        "device_count": len(devices),
        "device_kinds": [device.device_kind for device in devices],
    }
    assert len(devices) == 8 and all(d.platform == "tpu" for d in devices)
    layout = maxtext_parallelism.MaxTextPipelineConfig(
        pipeline_parallelism=args.pipeline_stages,
        tensor_parallelism=8 // args.pipeline_stages,
        num_layers_per_pipeline_stage=4 // args.pipeline_stages,
        num_pipeline_microbatches=8,
        pipeline_parallel_layers=4,
    )
    layout.validate_batch_size(8)
    mesh = layout.create_mesh(devices=devices)
    with jax.set_mesh(mesh):
      model, _ = automodel.AutoModel.from_pretrained(
          "Qwen/Qwen3-1.7B",
          mesh=mesh,
          model_source=automodel.ModelSource.MAXTEXT,
          maxtext_pipeline_config=layout,
          override_model_config=True,
          base_emb_dim=256,
          base_num_query_heads=4,
          base_num_kv_heads=4,
          base_mlp_dim=512,
          base_num_decoder_layers=4,
          head_dim=64,
          vocab_size=512,
          per_device_batch_size=1,
          max_target_length=32,
          max_prefill_predict_length=16,
          steps=1,
          dataset_type="synthetic",
          convert_checkpoint_if_possible=False,
          enable_checkpointing=False,
          enable_dropout=False,
          attention="dot_product",
          remat_policy="full",
          scan_layers=False,
          scan_layers_per_stage=False,
          weight_dtype="float32",
      )
      config = model.base.config
      report["model_shape"] = {
          "layers": config.num_decoder_layers,
          "embedding_dim": config.emb_dim,
          "vocab_size": config.vocab_size,
      }
      assert config.num_decoder_layers == 4 and config.emb_dim == 256
      assert config.vocab_size == 512
      report["mesh_axes"] = list(mesh.axis_names)
      report["maxtext_mesh_axes"] = list(config.mesh_axes)
      report["mesh_shape"] = dict(mesh.shape)
      assert tuple(config.mesh_axes) == tuple(mesh.axis_names)
      leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
      for axis in ("stage", "tensor"):
        count = sum(
            axis in str(getattr(value.sharding, "spec", "")) for value in leaves
        )
        report[f"{axis}_sharded_leaves"] = count
        assert count > 0, f"No parameters are sharded over {axis}"
      tokens = jnp.arange(8 * 32, dtype=jnp.int32).reshape(8, 32) % 511 + 1
      positions = jnp.broadcast_to(jnp.arange(32), tokens.shape)
      segments = jnp.ones_like(tokens)
      targets = jnp.roll(tokens, -1, axis=1)

      def loss_fn(module):
        logits, _ = module(
            tokens, positions, None, None, decoder_segment_ids=segments
        )
        return optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1].astype(jnp.float32), targets[:, :-1]
        ).mean()

      @nnx.jit
      def train_step(module, optimizer):
        before = nnx.state(module, nnx.Param)
        loss, grads = nnx.value_and_grad(loss_fn)(module)
        grad_norm = optax.global_norm(grads)
        optimizer.update(module, grads)
        deltas = jax.tree.map(
            lambda old, new: jnp.sum(jnp.abs(new - old)),
            before,
            nnx.state(module, nnx.Param),
        )
        return loss, grad_norm, sum(jax.tree.leaves(deltas))

      optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)
      with nn.partitioning.axis_rules(config.logical_axis_rules):
        forward_loss = nnx.jit(loss_fn)(model)
        jax.block_until_ready(forward_loss)
        report["forward_loss"] = float(forward_loss)
        loss, grad_norm, delta = train_step(model, optimizer)
        jax.block_until_ready((loss, grad_norm, delta))
        report.update(
            loss=float(loss),
            gradient_norm=float(grad_norm),
            parameter_update_l1=float(delta),
        )
      assert bool(jnp.isfinite(forward_loss))
      assert all(bool(jnp.isfinite(x)) for x in (loss, grad_norm, delta))
      assert float(grad_norm) > 0 and float(delta) > 0
      report["ok"] = True
  except Exception as exc:
    report["error_type"] = type(exc).__name__
    report["error"] = str(exc)
    raise
  finally:
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("MAXTEXT_PIPELINE_SMOKE " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
