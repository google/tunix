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

"""Independent categorical Proposer gradient oracle and native optimizer checks."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from examples.swarm_math import cpu_fixture
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.swarm import proposer_loss
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker


def torch_objective(logits, reference, advantages, beta, gap_clip):
  current = torch.log_softmax(logits, dim=-1)
  reference = torch.tensor(np.array(reference), dtype=torch.float32)
  advantages = torch.as_tensor(advantages, dtype=torch.float32)
  losses = []
  for p, r, a in zip(current, reference, advantages):
    indices = torch.nonzero(a).flatten()
    kl = torch.sum(torch.exp(p) * (p - r))
    term = torch.zeros_like(kl)
    if len(indices):
      first, second = indices.tolist()
      gap = float(((p[first] - p[second]) - (r[first] - r[second])).detach())
      clipped = (
          gap_clip > 0 and float(a[first]) * gap > 0 and abs(gap) >= gap_clip
      )
      if not clipped:
        term = -a[first] * (p[first] - p[second])
    losses.append(term + beta * kl)
  return torch.stack(losses).mean()


@pytest.mark.parametrize(
    "values,advantage",
    [
        ([0.2, -0.3, 0.9], [1, -1, 0]),
        ([6, -6, 0], [1, -1, 0]),
        ([-6, 6, 0], [-1, 1, 0]),
        ([6, -6, 0], [-1, 1, 0]),
        ([0.2, -0.3, 0.9], [0, 0, 0]),
    ],
)
def test_pair_loss_and_gradient_match_independent_torch(values, advantage):
  logits = np.asarray([values], dtype=np.float32)
  reference = np.asarray(jax.nn.log_softmax(jnp.array([[0.3, -0.2, 0.1]])))
  advantages = np.asarray([advantage], dtype=np.float32)

  def loss(x):
    return proposer_loss.pair_objective(
        jax.nn.log_softmax(x), reference, advantages, beta=0.3, gap_clip=4
    )[0].mean()

  value, gradient = jax.value_and_grad(loss)(jnp.asarray(logits))
  independent = torch.tensor(logits, requires_grad=True)
  expected = torch_objective(independent, reference, advantages, 0.3, 4)
  expected.backward()
  np.testing.assert_allclose(value, expected.detach(), atol=2e-6)
  np.testing.assert_allclose(gradient, independent.grad, atol=2e-6, rtol=2e-5)


def payload_for(model):
  payload = datatypes.RLTrainerPayload(
      prompt_ids=np.array([[1, 0, 2], [1, 0, 3]], dtype=np.int32),
      prompt_mask=np.ones((2, 3), dtype=np.float32),
      completion_ids=np.array([[4, 5, 6], [1, 6, 7]], dtype=np.int32),
      completion_mask=np.ones((2, 3), dtype=np.float32),
      completion_attention_mask=np.ones((2, 3), dtype=np.float32),
      advantages=np.array([[1, -1, 0], [0, -0.5, 0.5]], dtype=np.float32),
  )
  return dataclasses.replace(
      payload,
      ref_per_token_logps=np.asarray(
          proposer_loss.candidate_logps(model, payload)
      ),
  )


def test_candidates_are_independent_final_prompt_predictions():
  model = cpu_fixture.TinyPolicy(
      np.arange(64, dtype=np.float32).reshape(8, 8) / 20
  )
  payload = proposer_loss.validate_payload(payload_for(model))
  raw = np.asarray(model.logits[...])[[2, 3]][
      np.arange(2)[:, None], payload.completion_ids
  ]
  expected = np.asarray(jax.nn.log_softmax(jnp.asarray(raw), axis=-1))
  np.testing.assert_allclose(
      proposer_loss.candidate_logps(model, payload), expected, atol=1e-6
  )


@pytest.mark.parametrize(
    "change",
    [
        {"completion_ids": np.array([[4, 4, 6], [1, 6, 7]], dtype=np.int32)},
        {"completion_mask": np.array([[1, 1, 0], [1, 1, 1]])},
        {"advantages": np.array([[1, -2, 0], [0, -0.5, 0.5]])},
        {"advantages": np.array([[1, -0.5, -0.5], [0, -0.5, 0.5]])},
        {"ref_per_token_logps": np.zeros((2, 3))},
        {"prompt_mask": np.array([[1, 1, 0], [1, 1, 1]])},
    ],
)
def test_payload_admission_rejects_invalid_pair_semantics(change):
  payload = payload_for(cpu_fixture.TinyPolicy())
  with pytest.raises(ValueError):
    proposer_loss.validate_payload(dataclasses.replace(payload, **change))


def test_two_native_proposer_updates_match_torch_and_leave_reference_frozen():
  mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("fsdp",))
  with mesh, jax.default_device(jax.devices("cpu")[0]):
    actor = cpu_fixture.TinyPolicy(
        np.arange(64, dtype=np.float32).reshape(8, 8) / 20
    )
    # Allocate independent buffers: nnx.clone alone aliases donated arrays.
    reference = cpu_fixture.TinyPolicy(np.asarray(actor.logits[...]).copy())
    frozen = np.asarray(reference.logits[...]).copy()
    payload = proposer_loss.validate_payload(payload_for(reference))
    native = peft_trainer_v2.PeftTrainer(
        actor,
        optax.sgd(0.05),
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=100,
            max_steps=2,
            gradient_accumulation_steps=1,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: native, execution_context=mesh
    )
    worker.initialize()

    def loss(model, train_example):
      return proposer_loss.proposer_pair_loss(
          model, train_example, SimpleNamespace(beta=0.3, gap_clip=4), 0, 0
      )

    worker.with_loss_fn(loss, has_aux=True)
    worker.with_gen_model_input_fn(lambda item: {"train_example": item})
    try:
      for _ in range(2):
        before = np.asarray(actor.logits[...]).copy()
        oracle = torch.tensor(before, requires_grad=True)
        labels = torch.as_tensor(payload.completion_ids, dtype=torch.long)
        selected = oracle[torch.tensor([2, 3])].gather(1, labels)
        expected_loss = torch_objective(
            selected, payload.ref_per_token_logps, payload.advantages, 0.3, 4
        )
        expected_loss.backward()
        worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
        np.testing.assert_array_equal(actor.logits[...], before)
        worker.update()
        np.testing.assert_allclose(
            actor.logits[...],
            before - 0.05 * oracle.grad.numpy(),
            atol=2e-6,
            rtol=2e-5,
        )
        assert np.max(np.abs(np.asarray(actor.logits[...]) - before)) > 0
        np.testing.assert_array_equal(reference.logits[...], frozen)
    finally:
      native.close()
