"""Two real CPU optimizer updates checked against upstream PyTorch gradients."""

import asyncio
import dataclasses
import json

from flax import nnx
import jax
import numpy as np
import optax
import pytest
from examples.swarm_math import cpu_fixture
from examples.swarm_math import train
from tests.dist_rl.swarm import reference
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker


def concatenate(payloads):
  fields = (
      "prompt_ids",
      "prompt_mask",
      "completion_ids",
      "completion_mask",
      "completion_attention_mask",
      "advantages",
      "old_per_token_logps",
  )
  return dataclasses.replace(
      payloads[0],
      **{
          f: np.concatenate([getattr(p, f) for p in payloads], axis=0)
          for f in fields
      }
  )


@pytest.mark.parametrize(
    "normalization,loss_reduction",
    [
        ("role", "global_token_mean"),
        ("episode", "global_token_mean"),
        ("role", "microbatch_token_mean"),
    ],
)
def test_two_updates_match_original_gradient_and_publish_shared_weights(
    normalization, loss_reduction
):
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  config["normalization"] = normalization
  task = train.load_tasks(train.ROOT / "data/cpu_train.jsonl")
  mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("fsdp",))
  with mesh, jax.default_device(jax.devices("cpu")[0]):
    actor = cpu_fixture.TinyPolicy()
    rollout = nnx.clone(actor)
    native = peft_trainer_v2.PeftTrainer(
        actor,
        optax.sgd(config["learning_rate"]),
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
    snapshots, episodes_by_round, errors, losses = [], [], [], []

    def sync():
      nnx.update(rollout, nnx.state(actor))
      snapshots.append(np.asarray(rollout.logits[...]).copy())

    class CheckedEngine(local_engine.LocalSwarmEngine):

      def __init__(self):
        super().__init__(worker, sync)
        self.pending = []

      async def train(self, payload, *, apply_optimizer):
        self.pending.append(payload)
        before = np.asarray(actor.logits[...]).copy()
        result = await super().train(payload, apply_optimizer=apply_optimizer)
        after = np.asarray(actor.logits[...]).copy()
        if not apply_optimizer:
          np.testing.assert_array_equal(before, after)
        else:
          # Microbatch weights live in the payload advantages, so the native
          # update must equal the upstream objective on the weighted rows.
          all_rows = concatenate(self.pending)
          old_loss, gradient = reference.loss_and_gradient(before, all_rows)
          expected = before - config["learning_rate"] * gradient
          errors.append(float(np.max(np.abs(after - expected))))
          np.testing.assert_allclose(after, expected, atol=2e-6, rtol=2e-5)
          assert np.linalg.norm(gradient) > 0
          new_loss = reference.loss_and_gradient(after, all_rows)[0]
          assert new_loss < old_loss
          losses.append((old_loss, new_loss))
          self.pending.clear()
        return result

    collector = train.MathCollector(
        cpu_fixture.TinySampler(rollout), cpu_fixture.encode, task, config
    )

    async def collect(version, step):
      np.testing.assert_array_equal(rollout.logits[...], actor.logits[...])
      episodes = await collector(version, step)
      episodes_by_round.append(episodes)
      return episodes

    program = swarm_program.SwarmRLProgram(
        collect,
        swarm_program.SwarmConfig(
            group_size=8,
            micro_batch_size=4,
            max_prompt_length=16,
            max_response_length=3,
            normalization=normalization,
            loss_reduction=loss_reduction,
        ),
    )
    try:
      program.run(CheckedEngine(), num_steps=2)
      assert program.step == program.policy_version == 2
      assert len(snapshots) == 3 and len(errors) == 2
      assert all(
          np.linalg.norm(b - a) > 0
          for a, b in zip(snapshots[:-1], snapshots[1:], strict=True)
      )
      assert len({len(e.calls) for e in episodes_by_round[0]}) == 2
      assert [
          len({c.policy_id for e in round_ for c in e.calls})
          for round_ in episodes_by_round
      ] == [1, 1]
      assert [
          {c.policy_version for e in round_ for c in e.calls}
          for round_ in episodes_by_round
      ] == [{0}, {1}]
      # Unequal microbatch token counts and a partial final batch are covered.
      assert program.history[0]["calls"] % 4 != 0
      tm = program.history[0]["trainer_metrics"]
      assert {"loss", "grad_norm", "entropy", "ppo_kl"} <= set(tm)
      assert tm["grad_norm"] > 0 and tm["entropy"] > 0
      json.dumps(program.history)
      print({
          "normalization": normalization,
          "loss_reduction": loss_reduction,
          "max_parameter_errors": errors,
          "loss_before_after": losses,
      })
    finally:
      native.close()


@pytest.mark.parametrize("failure", ["partial", "stale", "mismatch"])
def test_fail_before_any_optimizer_update(failure):
  from tests.dist_rl.swarm.credit_test import episode  # pylint: disable=g-import-not-at-top

  a, b = episode("a", 1), episode("b", 0)
  if failure == "stale":
    b = dataclasses.replace(
        b,
        calls=tuple(dataclasses.replace(c, policy_version=1) for c in b.calls),
    )

  class Engine:
    trained = False

    def configure(self, *_):
      pass

    async def prepare(self):
      return 0

    async def score(self, payload):
      return payload.old_per_token_logps + 1

    async def train(self, *args, **kwargs):
      self.trained = True

    async def publish(self):
      raise AssertionError("Should not publish")

  async def collect(*_):
    return [a] if failure == "partial" else [a, b]

  engine = Engine()
  program = swarm_program.SwarmRLProgram(
      collect,
      swarm_program.SwarmConfig(
          group_size=2,
          micro_batch_size=2,
          max_prompt_length=4,
          max_response_length=4,
      ),
  )
  with pytest.raises(ValueError):
    asyncio.run(program.run_async(engine, num_steps=1))
  assert not engine.trained
  assert program.step == 0 and not program._is_running


def test_zero_signal_does_not_decay_weights_or_advance_policy():
  from tests.dist_rl.swarm.credit_test import episode

  class Engine:

    def configure(self, *_):
      pass

    async def prepare(self):
      return 0

    async def score(self, *_):
      raise AssertionError("No scoring for zero signal")

    async def train(self, *args, **kwargs):
      raise AssertionError("No optimizer for zero signal")

    async def publish(self):
      raise AssertionError("No publication for zero signal")

  async def collect(*_):
    return [episode("a", 1), episode("b", 1)]

  program = swarm_program.SwarmRLProgram(
      collect, swarm_program.SwarmConfig(group_size=2)
  )
  program.run(Engine(), num_steps=1)
  assert program.step == 1 and program.policy_version == 0
  assert program.history[0]["skip_reason"] == "zero_signal"


def test_rejects_unknown_loss_reduction():
  with pytest.raises(ValueError, match="loss_reduction"):
    swarm_program.SwarmConfig(loss_reduction="sequence_mean")
  with pytest.raises(ValueError, match="sampler_is"):
    swarm_program.SwarmConfig(sampler_is="geometric")
  with pytest.raises(ValueError, match="sampler_is_threshold"):
    swarm_program.SwarmConfig(sampler_is_threshold=0.0)


def test_sampler_is_token_uses_on_policy_ratio_and_downweights_outlier():
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  task = train.load_tasks(train.ROOT / "data/cpu_train.jsonl")
  mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("fsdp",))
  with mesh, jax.default_device(jax.devices("cpu")[0]):
    actor = cpu_fixture.TinyPolicy()
    rollout = nnx.clone(actor)
    native = peft_trainer_v2.PeftTrainer(
        actor,
        optax.sgd(config["learning_rate"]),
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=100,
            max_steps=1,
            gradient_accumulation_steps=2,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: native, execution_context=mesh
    )
    worker.initialize()
    collector = train.MathCollector(
        cpu_fixture.TinySampler(rollout), cpu_fixture.encode, task, config
    )

    async def collect_with_outlier(version, step):
      episodes = list(await collector(version, step))
      c0, c1 = episodes[0].calls[0], episodes[0].calls[1]
      expected_min_weight["val"] = float(np.exp(c0.rollout_logprobs[0] + 1e-4))
      bad_c0 = dataclasses.replace(
          c0,
          rollout_logprobs=(-1e-4,) + tuple(c0.rollout_logprobs[1:]),
      )
      bad_c1 = dataclasses.replace(
          c1,
          rollout_logprobs=(float(c1.rollout_logprobs[0]) - 10.0,)
          + tuple(c1.rollout_logprobs[1:]),
      )
      episodes[0] = dataclasses.replace(
          episodes[0],
          calls=(bad_c0, bad_c1) + episodes[0].calls[2:],
      )
      return episodes

    expected_min_weight = {}
    program = swarm_program.SwarmRLProgram(
        collect_with_outlier,
        swarm_program.SwarmConfig(
            group_size=8,
            micro_batch_size=4,
            max_prompt_length=16,
            max_response_length=3,
            max_logprob_error=20.0,
            sampler_is="token",
            sampler_is_threshold=2.0,
        ),
    )
    try:
      program.run(
          local_engine.LocalSwarmEngine(
              worker, lambda: nnx.update(rollout, nnx.state(actor))
          ),
          num_steps=1,
      )
      tm = program.history[0]["trainer_metrics"]
      assert tm["is_ratio/max"] == pytest.approx(1.0)
      assert tm["pg_clipfrac"] == pytest.approx(0.0)
      assert tm["ppo_kl"] == pytest.approx(0.0, abs=1e-6)
      assert tm["sampler_is/weight_min"] == pytest.approx(
          expected_min_weight["val"], abs=1e-4
      )
      assert tm["sampler_is/frac_clipped_at_threshold"] > 0.0
      assert tm["sampler_trainer/logp_diff_max"] == pytest.approx(
          10.0, abs=1e-4
      )
    finally:
      native.close()
