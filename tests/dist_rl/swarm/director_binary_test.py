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

"""Original off/on policy support, full categorical KL and mixed-call gradients."""

import dataclasses

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from examples.swarm_math import cpu_fixture
from tests.dist_rl.swarm.gradient_test import payload_and_weights
from tests.dist_rl.swarm.r2orch_gradient_test import director_config
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.swarm import director_loss
from tunix.dist_rl.swarm import trace


def test_mixed_binary_and_autoregressive_calls_match_torch_every_gradient():
  p, weights = payload_and_weights()
  mask = p.completion_mask.copy()
  mask[0] = [1, 0, 0]
  support = np.array([[4, 6], [-1, -1], [-1, -1]], np.int32)
  ref_support = np.array(
      [[np.log(0.2), np.log(0.8)], [0, 0], [0, 0]], np.float32
  )
  old = p.old_per_token_logps.copy()
  raw = weights[2, [4, 6]]
  old[0, 0] = raw[0] - np.log(np.exp(raw).sum())
  reference = old - 0.7
  reference[0, 0] = ref_support[0, 0]
  correction = np.full(mask.shape, 0.6, np.float32)
  p = dataclasses.replace(
      p,
      completion_mask=mask,
      completion_attention_mask=np.where(
          np.arange(3)[None, :] < np.array([1, 3, 3])[:, None], 1.0, 0.0
      ),
      old_per_token_logps=old,
      ref_per_token_logps=reference,
      action_support_ids=support,
      reference_support_logps=ref_support,
      sampler_is_weights=correction,
  )
  config = director_config(0.2)

  def native(model):
    return director_loss.director_loss_fn(
        model, p, config, 0, 0
    ).primary_loss.compute()

  loss, grad = nnx.value_and_grad(native)(cpu_fixture.TinyPolicy(weights))

  w = torch.tensor(weights, requires_grad=True)
  previous = np.concatenate(
      [p.prompt_ids[:, -1:], p.completion_ids[:, :-1]], -1
  )
  standard = torch.log_softmax(w[torch.tensor(previous, dtype=torch.long)], -1)
  ordinary = standard.gather(
      -1, torch.tensor(p.completion_ids, dtype=torch.long)[..., None]
  )[..., 0]
  binary = torch.log_softmax(w[2, [4, 6]], -1)
  current = torch.cat(
      [torch.cat([binary[:1], ordinary[0, 1:]])[None], ordinary[1:]], 0
  )
  ratio = (current - torch.tensor(old)).clamp(-20, 20).exp()
  advantage = torch.tensor(p.advantages)
  pg = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.28))
  diff = (torch.tensor(reference) - current).clamp(-20, 20)
  sampled = diff.exp() - diff - 1
  exact = (binary.exp() * (binary - torch.tensor(ref_support[0]))).sum()
  kl = torch.cat(
      [torch.cat([exact[None], sampled[0, 1:]])[None], sampled[1:]], 0
  )
  expected = (pg * torch.tensor(correction) + 0.2 * kl)[
      torch.tensor(mask).bool()
  ].mean()
  expected.backward()
  np.testing.assert_allclose(loss, expected.detach(), atol=2e-6)
  np.testing.assert_allclose(grad["logits"][...], w.grad, atol=2e-6, rtol=2e-5)
  assert not np.isclose(float(exact.detach()), float(sampled[0, 0].detach()))


def test_padded_binary_support_preserved_and_packing_rejected():
  p, _ = payload_and_weights()

  def row(index, binary=False):
    names = (
        "prompt_ids",
        "prompt_mask",
        "completion_ids",
        "completion_mask",
        "completion_attention_mask",
        "advantages",
        "old_per_token_logps",
    )
    fields = {
        name: getattr(p, name)[index : index + 1].copy() for name in names
    }
    if binary:
      for key in names[2:]:
        fields[key] = fields[key][:, :1]
      fields["action_support_ids"] = np.array([[4, 6]], np.int32)
    return dataclasses.replace(p, **fields)

  assembler = batch_assembly.PaddedBatchAssembler(
      batch_size=3,
      max_prompt_length=4,
      max_response_length=3,
      pad_id=0,
      num_generations=2,
      mini_batch_size=1,
  )
  result = assembler.pack([row(0, True), row(1)])[0]
  np.testing.assert_array_equal(
      result.action_support_ids, [[4, 6], [-1, -1], [-1, -1]]
  )
  assert result.completion_mask[0].tolist() == [1, 0, 0]
  with pytest.raises(ValueError, match="padded"):
    batch_assembly.to_pack_item(row(0, True))
  with pytest.raises(ValueError, match="Invalid binary"):
    assembler.pack([
        dataclasses.replace(row(0, True), action_support_ids=np.array([[5, 6]]))
    ])


def test_large_sampled_kl_is_not_silently_capped_at_ten():
  loss, kl = director_loss.token_loss(
      jnp.array([[-8.0]]),
      jnp.array([[-8.0]]),
      jnp.array([[-1.0]]),
      jnp.zeros((1, 1)),
      jnp.ones((1, 1)),
      beta=0.005,
  )
  assert float(kl[0, 0]) > 1000
  np.testing.assert_allclose(loss, 0.005 * (np.exp(7) - 8), rtol=1e-6)


def test_paper_cap_and_reference_stride_apply_only_to_the_intended_terms():
  loss, kl = director_loss.token_loss(
      jnp.array([[-8.0]]),
      jnp.array([[-8.0]]),
      jnp.array([[-1.0]]),
      jnp.ones((1, 1)),
      jnp.ones((1, 1)),
      beta=0.005,
      kl_term_cap=10.0,
      reference_kl_weights=jnp.full((1, 1), 4.0),
  )
  np.testing.assert_allclose(kl, 40.0)
  np.testing.assert_allclose(loss, -0.8)
  loss, kl = director_loss.token_loss(
      jnp.zeros((1, 1)),
      jnp.zeros((1, 1)),
      jnp.array([[-30.0]]),
      jnp.ones((1, 1)),
      jnp.ones((1, 1)),
      beta=0.005,
      kl_term_cap=10.0,
      reference_kl_weights=jnp.full((1, 1), 4.0),
      support_ids=jnp.array([[1, 2]]),
      support_logps=jnp.array([[0.0, -100.0]]),
      reference_support_logps=jnp.array([[-30.0, 0.0]]),
  )
  np.testing.assert_allclose(kl, 120.0)
  np.testing.assert_allclose(loss, -0.4, atol=1e-7)


def test_trace_rejects_unsupported_token_or_multitoken_relation():
  kwargs = dict(
      call_id="r",
      role="director",
      policy_id="d",
      policy_version=0,
      prompt_ids=(1,),
      completion_ids=(4,),
      rollout_logprobs=(-0.5,),
      loss_mask=(1,),
      text="off",
  )
  with pytest.raises(ValueError, match="binary"):
    trace.CallTrace(**kwargs, action_support_ids=(5, 6))
  assert trace.CallTrace(
      **kwargs, action_support_ids=(4, 6)
  ).action_support_ids == (4, 6)


def test_ignored_nan_support_rows_cannot_poison_gradient():
  import jax

  ids = jnp.array([[2, 3], [-1, -1], [2, 3]])
  support = jnp.array(
      [[jnp.log(0.4), jnp.log(0.6)], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]
  )
  references = jnp.array(
      [[jnp.log(0.3), jnp.log(0.7)], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]
  )
  mask = jnp.array([[1.0], [1.0], [0.0]])

  def loss(values):
    terms, _ = director_loss.token_loss(
        jnp.full((3, 1), -0.5),
        jnp.full((3, 1), -0.5),
        jnp.full((3, 1), -0.6),
        jnp.ones((3, 1)),
        mask,
        beta=0.1,
        support_ids=ids,
        support_logps=values,
        reference_support_logps=references,
    )
    return terms.sum()

  value, gradients = jax.value_and_grad(loss)(support)
  assert np.isfinite(value) and np.isfinite(gradients).all()
  np.testing.assert_array_equal(gradients[1:], 0)
  assert np.linalg.norm(gradients[0]) > 0


@pytest.mark.parametrize(
    "field", ["segment_ids", "segment_positions", "num_segments"]
)
def test_ordinary_director_calls_reject_packing_metadata(field):
  p, weights = payload_and_weights()
  packed = dataclasses.replace(
      p, **{field: 2 if field == "num_segments" else np.ones((3, 6), np.int32)}
  )
  with pytest.raises(ValueError, match="unpacked"):
    director_loss.policy_logprobs(cpu_fixture.TinyPolicy(weights), packed)


def test_chunked_binary_and_proposer_qwen_gradients_match_full_head(
    monkeypatch,
):
  import jax
  from tunix.dist_rl.swarm import proposer_loss
  from tunix.models.qwen3 import model as qwen_model

  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("fsdp", "tp")
  )
  projected_lengths = []
  original_head = qwen_model.Qwen3.compute_final_logits

  def head(self, values):
    projected_lengths.append(values.shape[1])
    return original_head(self, values)

  monkeypatch.setattr(qwen_model.Qwen3, "compute_final_logits", head)
  with mesh, jax.default_device(jax.devices("cpu")[0]):
    model = qwen_model.Qwen3(
        qwen_model.ModelConfig(
            num_layers=1,
            vocab_size=8,
            embed_dim=16,
            hidden_dim=32,
            num_heads=2,
            head_dim=8,
            num_kv_heads=1,
            rope_theta=1000000,
            norm_eps=1e-6,
        ),
        rngs=nnx.Rngs(42),
    )
    p, _ = payload_and_weights()
    p = dataclasses.replace(
        p,
        completion_ids=np.array([[4, 6], [1, 6], [2, 7]], np.int32),
        completion_mask=np.ones((3, 2), np.float32),
        completion_attention_mask=np.ones((3, 2), np.float32),
    )
    coefficients = jnp.array([[0.2, -0.3], [0.7, -0.5], [-0.3, 0.6]])

    def objective(current, chunk):
      values = proposer_loss.candidate_logps(current, p, chunk_size=chunk)
      return jnp.sum(values * coefficients)

    full, full_grad = nnx.value_and_grad(objective)(model, 0)
    assert max(projected_lengths) == p.prompt_ids.shape[1] + 1
    projected_lengths.clear()
    chunked, chunked_grad = nnx.value_and_grad(objective)(model, 1)
    assert projected_lengths and set(projected_lengths) == {1}
    np.testing.assert_allclose(chunked, full, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(
        jax.tree.leaves(chunked_grad), jax.tree.leaves(full_grad), strict=True
    ):
      np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
    expected = proposer_loss.candidate_logps(model, p, chunk_size=1)[0]
    projected_lengths.clear()
    actual = director_loss.binary_logprobs(
        model, p.prompt_ids[0, 1:], p.completion_ids[0], chunk_size=1
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert set(projected_lengths) == {1}

    # The mixed-call Director path must also forward the chunk size into its
    # restricted-support scorer; only one hidden position is needed there.
    p = dataclasses.replace(
        p,
        completion_ids=p.completion_ids[:, :1],
        completion_mask=np.ones((3, 1), np.float32),
        completion_attention_mask=np.ones((3, 1), np.float32),
        action_support_ids=np.array([[4, 6], [-1, -1], [2, 7]], np.int32),
    )
    projected_lengths.clear()
    values, _, support = director_loss.policy_logprobs(model, p, chunk_size=1)
    assert np.isfinite(values).all() and np.isfinite(support).all()
    assert set(projected_lengths) == {1}
