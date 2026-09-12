"""Source-bound installed methods, not real weights or TPU kernel parity."""

import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from examples.frozenlake.gemma4_tim import artifacts, recipe, reference
from tunix.models.gemma4 import model as trainer
from tpu_inference.models.jax import gemma4 as serving


class InstalledTest(unittest.TestCase):
  def test_real_installed_methods_and_stock_ple_seam(self):
    recipe.require_versions()
    self.assertTrue(all(d.platform == "cpu" for d in jax.devices()))
    source_sha = artifacts.file_sha(Path(inspect.getfile(serving)))
    self.assertEqual(source_sha, "5b93b3b3dd5d42374c0115a4523661ed584191fb0cb9fc9639ae2248b3896fbd")
    config = NS(num_hidden_layers=35, num_kv_shared_layers=20,
                layer_types=["full_attention" if kind == "global" else "sliding_attention"
                             for kind in reference.E2B_LAYER_TYPES])
    shared = serving.compute_kv_share_map(config)
    self.assertEqual(tuple(shared.get(i, i) for i in range(35)), reference.E2B_KV_PRODUCERS)
    rng = np.random.default_rng(91)
    x = jnp.asarray(rng.normal(size=(4, 24)), jnp.float32)
    gate, up = [jnp.asarray(rng.normal(size=(24, 32)), jnp.float32) for _ in range(2)]
    down = jnp.asarray(rng.normal(size=(32, 24)), jnp.float32)
    native = NS(gate_proj=lambda z: z @ gate, up_proj=lambda z: z @ up,
                down_proj=lambda z: z @ down)
    engine = NS(gate_up_proj=lambda z: jnp.concatenate((z @ gate, z @ up), -1),
                down_proj=native.down_proj, act_fn=nnx.gelu)
    native_mlp = lambda z: trainer.FeedForward.block(native, z)
    engine_mlp = lambda z: serving.Gemma4MLP.__call__(engine, z)
    np.testing.assert_array_equal(native_mlp(x), engine_mlp(x))
    np.testing.assert_array_equal(jax.grad(lambda z: native_mlp(z).sum())(x),
                                  jax.grad(lambda z: engine_mlp(z).sum())(x))
    wrong = (jax.nn.silu(x @ gate) * (x @ up)) @ down
    self.assertFalse(np.array_equal(native_mlp(x), wrong))

    # Native scales the projection weight before matmul, engine scales its
    # output afterwards. Retain both real methods and expose the disagreement.
    layers, ple = 35, 8
    weights = jnp.asarray(rng.normal(size=(24, layers * ple)), jnp.float32)
    embedding = jnp.asarray(rng.normal(size=(19, layers * ple)), jnp.float32)
    ids = jnp.array([2, 4, 7, 9], jnp.int32)
    norm = lambda z: z * jax.lax.rsqrt(jnp.mean(z * z, axis=-1, keepdims=True) + 1e-6)
    einsum = NS(w=NS(value=weights), w_scale=24**-.5, dtype=jnp.float32,
                einsum_str="BTD,DX->BTX")
    native_ple = NS(vocab_size=19, config=NS(num_layers=layers, per_layer_input_dim=ple),
                    per_layer_model_projection=lambda z: trainer.Einsum.__call__(einsum, z),
                    per_layer_projection_norm=norm, per_layer_input_embedding=NS(value=embedding))
    engine_ple = NS(hidden_size_per_layer_input=ple, embed_tokens_per_layer=lambda z: embedding[z],
                    num_hidden_layers=layers, vocab_size_per_layer_input=19,
                    embed_scale_per_layer=ple**.5, per_layer_model_projection=lambda z: z @ weights,
                    per_layer_projection_scale=24**-.5, per_layer_projection_norm=norm,
                    per_layer_input_scale=2**-.5)
    a = np.asarray(trainer.Embedder.encode_per_layer_input(native_ple, x[None], ids[None])[0])
    b = np.asarray(serving.Gemma4Model.compute_per_layer_inputs(engine_ple, ids, x))
    self.assertEqual(a.shape, (4, 35, 8))
    self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
    difference = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    self.assertGreater(difference, 0)  # Live first-seam diagnostic, NOT parity.
    print("GEMMA4_E2B_INSTALLED_METHODS " + recipe.json_bytes({
        "source_sha256": source_sha, "kv_graph": "EXACT_35",
        "controlled_mlp": "EXACT_FORWARD_BACKWARD", "silu_fault": "DETECTED",
        "ple": {"differing_elements": difference, "max_abs": float(np.max(np.abs(a - b))),
                "classification": "STOCK_PROGRAMS_DIFFER"},
        "claim": "CPU-controlled-methods-only", "target": "NOT_RUN",
    }).decode().strip(), flush=True)


if __name__ == "__main__":
  unittest.main()
