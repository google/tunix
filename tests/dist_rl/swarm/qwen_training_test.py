"""Native Qwen3 attention/KV sampling and dist_rl training on a CPU-sized model.

Uses random tiny weights and a fixture vocabulary, not the pretrained 4B model.
The categorical policy tests provide the independent upstream gradient oracle.
"""

import json

from flax import nnx
import jax
import numpy as np
import optax
from tokenizers import models
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from examples.swarm_math import cpu_fixture
from examples.swarm_math import train
from tunix.common import configs
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.models.qwen3 import model as qwen3_model


def test_native_qwen_sampler_and_trainer_two_rounds():
  vocab = {text: i for i, text in enumerate(cpu_fixture.VOCAB)}
  del vocab[""]
  vocab["[EOS]"] = 0
  tokenizer = PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="reason")),
      eos_token="[EOS]",
      pad_token="[EOS]",
      unk_token="reason",
  )
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("fsdp", "tp")
  )
  with mesh, jax.default_device(jax.devices("cpu")[0]):
    actor = qwen3_model.Qwen3(
        qwen3_model.ModelConfig(
            num_layers=1,
            vocab_size=8,
            embed_dim=16,
            hidden_dim=32,
            num_heads=2,
            head_dim=8,
            num_kv_heads=1,
            rope_theta=1_000_000,
            norm_eps=1e-6,
        ),
        rngs=nnx.Rngs(42),
    )
    sampler = train.NativeMathSampler(
        actor,
        tokenizer,
        configs.RolloutConfig(
            max_prompt_length=16,
            max_tokens_to_generate=3,
            kv_cache_size=19,
            temperature=1.0,
            top_p=1.0,
            seed=42,
            return_logprobs=True,
        ),
        "train",
    )
    native = peft_trainer_v2.PeftTrainer(
        actor,
        optax.sgd(0.02),
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=100,
            max_steps=2,
            gradient_accumulation_steps=2,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: native, execution_context=mesh
    )
    worker.initialize()
    snapshots = []

    def sync():
      sampler.sync()
      values = np.concatenate([
          np.asarray(x).ravel()
          for x in jax.tree.leaves(nnx.state(actor, nnx.Param))
      ])
      assert np.isfinite(values).all()
      snapshots.append(values.copy())

    collector = train.MathCollector(
        sampler,
        cpu_fixture.encode,
        train.load_tasks(train.ROOT / "data/cpu_train.jsonl"),
        config,
    )
    program = swarm_program.SwarmRLProgram(
        collector,
        swarm_program.SwarmConfig(
            group_size=8,
            micro_batch_size=4,
            max_prompt_length=16,
            max_response_length=3,
        ),
    )
    try:
      program.run(local_engine.LocalSwarmEngine(worker, sync), num_steps=2)
      assert program.policy_version == 2 and len(snapshots) == 3
      assert all(
          np.linalg.norm(b - a) > 0
          for a, b in zip(snapshots[:-1], snapshots[1:], strict=True)
      )
      assert all(h["max_logprob_error"] < 1e-5 for h in program.history)
      print({
          "tiny_qwen_history": program.history,
          "parameter_deltas": [
              float(np.linalg.norm(b - a))
              for a, b in zip(snapshots[:-1], snapshots[1:], strict=True)
          ],
      })
    finally:
      native.close()
