# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real tiny-Qwen adapter training, base isolation and native serving sync."""

import asyncio
import dataclasses
import json
from pathlib import Path

from flax import nnx
import jax
import numpy as np
import optax
import pytest
from tokenizers import models
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from examples.r2orch_math import backends
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.rollout import sampler
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.models import automodel
from tunix.models.qwen3 import model as qwen_model


def snapshot(model, selector=nnx.Param):
  return [
      np.asarray(x).copy() for x in jax.tree.leaves(nnx.state(model, selector))
  ]


def assert_same(before, after):
  assert len(before) == len(after)
  for a, b in zip(before, after, strict=True):
    np.testing.assert_array_equal(a, b)


def test_qwen35_9b_preset_matches_pinned_official_text_configuration():
  cfg = automodel.call_model_config("qwen3.5-9b")
  assert (cfg.num_layers, cfg.embed_dim, cfg.hidden_dim) == (32, 4096, 12288)
  assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (16, 4, 256)
  assert (cfg.linear_num_key_heads, cfg.linear_num_value_heads) == (16, 32)
  assert cfg.vocab_size == 248320 and not cfg.use_tied_embedding
  assert cfg.partial_rotary_factor == 0.25 and len(cfg.layer_types) == 32


@pytest.mark.parametrize(
    "lora",
    [
        {"rank": 4, "alpha": 4, "dropout": 0.05},
        {"rank": 0, "alpha": 4},
        {"rank": True, "alpha": 4},
        {"rank": 4, "alpha": float("nan")},
        {"rank": 4, "alpha": 4, "unexpected": True},
    ],
)
def test_invalid_or_stochastic_adapters_fail_closed(lora):
  with pytest.raises(ValueError):
    backends.validate_lora(lora)


def test_snapshot_pin_is_required_before_model_load(tmp_path):
  cfg = json.loads(
      (
          Path(__file__).parents[3]
          / "examples/r2orch_math/configs/qwen3_5_9b.json"
      ).read_text()
  )
  path = tmp_path / cfg["model_revision"]
  path.mkdir()
  backends.validate_backend_config(cfg, path)
  with pytest.raises(ValueError, match="snapshot directory"):
    backends.validate_backend_config(cfg, tmp_path)
  with pytest.raises(ValueError, match="immutable"):
    backends.validate_backend_config({**cfg, "model_revision": "main"}, path)


def test_native_trainer_updates_only_lora_and_publishes_adapter_snapshot():
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("fsdp", "tp")
  )
  tokenizer = PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(
          models.WordLevel(
              {
                  "[EOS]": 0,
                  "[UNK]": 1,
                  "a": 2,
                  "b": 3,
                  "c": 4,
                  "d": 5,
                  "e": 6,
                  "f": 7,
              },
              unk_token="[UNK]",
          )
      ),
      eos_token="[EOS]",
      pad_token="[EOS]",
      unk_token="[UNK]",
  )
  with mesh:
    base = qwen_model.Qwen3(
        qwen_model.ModelConfig(
            num_layers=1,
            vocab_size=8,
            embed_dim=16,
            hidden_dim=32,
            num_heads=2,
            head_dim=8,
            num_kv_heads=1,
            norm_eps=1e-6,
            rope_theta=1000000,
        ),
        rngs=nnx.Rngs(42),
    )
    frozen = nnx.clone(base)
    nnx.update(frozen, jax.tree.map(lambda x: x.copy(), nnx.state(base)))
    actor = backends.apply_lora(
        base, {"rank": 2, "alpha": 2, "dropout": 0}, mesh=mesh
    )
    base_selector = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    original_base = snapshot(actor, base_selector)
    original_adapters = snapshot(actor, nnx.LoRAParam)
    frozen_before = snapshot(frozen)
    assert len(original_adapters) == 14
    cfg = {
        "max_prompt_length": 4,
        "worker_max_prompt_length": 3,
        "max_response_length": 2,
        "worker_max_tokens": 2,
        "train_temperature": 1.0,
        "eval_temperature": 1.0,
        "worker_temperature": 0.3,
        "seed": 42,
        "rollout_batch_size": 1,
    }
    ds, es, ws, sync = backends.make_native_samplers(
        actor, frozen, tokenizer, cfg, eos_tokens=[0]
    )
    assert ds.config.max_prompt_length == 4
    assert ws.config.max_prompt_length == 3
    assert ds.config.kv_cache_size == 6 and ws.config.kv_cache_size == 5
    native = peft_trainer_v2.PeftTrainer(
        actor,
        optax.sgd(0.01),
        peft_trainer_v2.TrainingConfig(
            max_steps=2,
            eval_every_n_steps=3,
            gradient_accumulation_steps=1,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    trainer = trainer_worker.TrainerWorker(
        lambda: native, execution_context=mesh
    )
    trainer.initialize()
    engine = local_engine.LocalSwarmEngine(trainer, sync)
    engine.configure(
        swarm_program.shared_grpo_config(2),
        swarm_program.SwarmConfig(group_size=2),
    )
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], np.int32),
        prompt_mask=np.ones((1, 2)),
        completion_ids=np.array([[3, 4]], np.int32),
        completion_mask=np.ones((1, 2)),
        completion_attention_mask=np.ones((1, 2)),
        advantages=np.ones((1, 2)),
        old_per_token_logps=np.zeros((1, 2)),
    )

    async def update():
      old = await engine.score(payload)
      scored = dataclasses.replace(
          payload, old_per_token_logps=old, sampler_agreement_applied=True
      )
      await engine.train(scored, apply_optimizer=True)
      await engine.publish()
      response = await ds.sample(
          sampler.SamplingRequest(
              request_id="native-adapter",
              prompt=np.array([1, 2], np.int32),
              sampling_params=sampler.SamplingParams(
                  max_tokens=2, temperature=1.0, top_p=1.0, return_logprobs=True
              ),
          )
      )
      assert response.error is None and response.logprobs is not None

    try:
      asyncio.run(update())
      assert_same(original_base, snapshot(actor, base_selector))
      assert_same(frozen_before, snapshot(frozen))
      assert any(
          not np.array_equal(a, b)
          for a, b in zip(
              original_adapters, snapshot(actor, nnx.LoRAParam), strict=True
          )
      )
      assert_same(snapshot(actor), snapshot(ds.sampler.transformer))
      assert_same(snapshot(actor), snapshot(es.sampler.transformer))
      assert_same(frozen_before, snapshot(ws.sampler.transformer))
    finally:
      native.close()
