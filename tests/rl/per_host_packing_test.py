# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-host sequence packing (`utils.pack_sequences_per_host`).

Under the per-host learner (TUNIX_PER_HOST_LEARNER=1) every process packs its
own rollouts; the processes agree on the packed shape with one allgather. These
tests check the packed layout, that the GRPO loss, its gradient and the
per-token log-probs equal the unpacked ones (for one packed step and for
accumulated capped chunks), that segments do not attend to each other, that
processes agree on the plan and abort together, and (2-process CPU job) that the
per-host learner trains the same with and without packing.
"""

# pylint: disable=protected-access

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
import types

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core
from tunix.rl import common
from tunix.rl import utils

_PAD = 0
_VOCAB = 32
_P, _C, _BUDGET = 6, 10, 16
# (prompt_len, completion_len); under FFD with budget 16 the rows are
# [16], [14, 2], [10, 4, 2], [10, 4], [8, 8].
_LENGTHS = [
    (6, 10), (5, 9), (3, 7), (1, 1), (2, 2),
    (1, 3), (4, 4), (2, 6), (6, 4), (1, 1),
]


def _batch(lengths, seed=0, old_logps=True):
  """Unpacked TrainExample; row i's first prompt token is the marker i + 1."""
  rng = np.random.default_rng(seed)
  n = len(lengths)
  prompt_ids = np.zeros((n, _P), np.int32)
  completion_ids = np.zeros((n, _C), np.int32)
  for i, (p_len, c_len) in enumerate(lengths):
    p = rng.integers(1, _VOCAB, size=p_len).astype(np.int32)
    p[0] = i + 1
    prompt_ids[i, _P - p_len :] = p
    completion_ids[i, :c_len] = rng.integers(1, _VOCAB, size=c_len)
  completion_mask = completion_ids != _PAD
  old = None
  if old_logps:
    old = np.where(
        completion_mask,
        -np.log(_VOCAB) + 0.3 * rng.normal(size=(n, _C)),
        0.0,
    ).astype(np.float32)
  return common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_ids != _PAD,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      advantages=rng.normal(size=n).astype(np.float32),
      ref_per_token_logps=None,
      old_per_token_logps=old,
  )


def _pack(examples, budget=_BUDGET, **kwargs):
  """Packs `examples` as one process-local mini-batch; returns the chunks."""
  n = sum(int(np.shape(e.completion_ids)[0]) for e in examples)
  kwargs.setdefault("pad_id", _PAD)
  kwargs.setdefault("data_axis_size", 1)
  kwargs.setdefault("leading_dim_factor", 1)
  kwargs.setdefault("allgather_fn", lambda row: np.asarray(row)[None, :])
  kwargs.setdefault("process_index", 0)
  out = utils.pack_sequences_per_host(
      iter([list(examples)]), budget, n, **kwargs
  )
  return [chunk for [chunk] in out]


def _segments(chunk):
  """Yields (row, start, stop, source_row) for every packed segment."""
  seg = np.asarray(chunk.segment_ids)
  ids = np.asarray(chunk.completion_ids)
  for r in range(seg.shape[0]):
    for s in sorted(set(seg[r].tolist()) - {0}):
      idx = np.flatnonzero(seg[r] == s)
      assert idx[-1] - idx[0] + 1 == idx.size, "segment is not contiguous"
      yield r, int(idx[0]), int(idx[-1]) + 1, int(ids[r, idx[0]]) - 1


class _TinyCausalLM(nnx.Module):
  """One causal attention block with MaxText's segment semantics.

  As in `TunixMaxTextAdapter`, explicit `segment_ids` win over
  `attention_mask`: a query attends to the keys of its own segment at or before
  its position. Learned absolute positions stand in for RoPE.
  """

  def __init__(self, vocab=_VOCAB, dim=16, max_len=64, *, rngs):
    self.emb = nnx.Embed(vocab, dim, rngs=rngs)
    self.pos = nnx.Embed(max_len, dim, rngs=rngs)
    self.q = nnx.Linear(dim, dim, rngs=rngs)
    self.k = nnx.Linear(dim, dim, rngs=rngs)
    self.v = nnx.Linear(dim, dim, rngs=rngs)
    self.mlp = nnx.Linear(dim, dim, rngs=rngs)
    self.head = nnx.Linear(dim, vocab, rngs=rngs)

  def __call__(
      self, x, positions, cache=None, attention_mask=None, segment_ids=None
  ):
    h = self.emb(x) + self.pos(positions)
    length = x.shape[1]
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))[None]
    if segment_ids is not None:
      mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & causal
    elif attention_mask is not None:
      mask = attention_mask.astype(bool)
    else:
      mask = jnp.broadcast_to(causal, (x.shape[0], length, length))
    scores = jnp.einsum("bqd,bkd->bqk", self.q(h), self.k(h))
    scores = jnp.where(mask, scores / np.sqrt(h.shape[-1]), -1e9)
    attn = jax.nn.softmax(scores, axis=-1)
    h = h + jnp.einsum("bqk,bkd->bqd", attn, self.v(h))
    h = h + jnp.tanh(self.mlp(h))
    return self.head(h), cache


def _config(mode, algo):
  return types.SimpleNamespace(
      beta=0.0,
      epsilon=0.2,
      epsilon_high=0.28,
      epsilon_c=None,
      loss_algo=algo,
      loss_agg_mode=mode,
      temperature=1.0,
      kl_loss_mode="low_var_kl",
      kl_clamp_value=None,
      sampler_is=None,
      sampler_is_threshold=2.0,
      sampler_rs=None,
      sampler_rs_min=None,
      sampler_rs_max=None,
      seq_logprob_error_threshold=None,
      use_rollout_logps=True,
  )


def _logps(model, example):
  graphdef, state = nnx.split(model)
  kwargs = {}
  if getattr(example, "segment_ids", None) is not None:
    kwargs = dict(
        segment_ids=jnp.asarray(example.segment_ids),
        segment_positions=jnp.asarray(example.segment_positions),
    )
  return np.asarray(
      common.compute_per_token_logps(
          graphdef,
          state,
          prompt_tokens=jnp.asarray(example.prompt_ids),
          completion_tokens=jnp.asarray(example.completion_ids),
          pad_id=_PAD,
          eos_id=-1,
          **kwargs,
      )
  )


def _assert_trees_close(test, a, b, rtol):
  la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
  test.assertLen(lb, len(la))
  scale = max(float(np.max(np.abs(np.asarray(x)))) for x in la) + 1e-12
  for x, y in zip(la, lb):
    np.testing.assert_allclose(
        np.asarray(x), np.asarray(y), rtol=0, atol=rtol * scale
    )


class PlanTest(parameterized.TestCase):

  @parameterized.parameters(
      (128, 16, 8), (128, 1, 128), (64, 16, 4), (6, 4, 3), (1, 1, 1)
  )
  def test_unit(self, data_axis, factor, want):
    self.assertEqual(utils.per_host_pack_unit(data_axis, factor), want)

  @parameterized.parameters(
      (78, 8, 0, 1, (1, 80)),
      (78, 8, 2, 1, (5, 16)),  # O06 with a cap of 2 rows per data shard
      (78, 8, 0, 4, (1, 96)),
      (78, 8, 4, 2, (3, 32)),
      (0, 8, 0, 1, (1, 8)),
      (19, 8, 0, 1, (1, 24)),
  )
  def test_plan(self, rows, unit, cap, multiple, want):
    self.assertEqual(utils.per_host_pack_plan(rows, unit, cap, multiple), want)

  def test_plan_rejects_bad_knobs(self):
    with self.assertRaises(ValueError):
      utils.per_host_pack_plan(10, 8, 3, 2)  # multiple must divide the cap
    with self.assertRaises(ValueError):
      utils.per_host_pack_plan(10, 0)
    with self.assertRaises(ValueError):
      utils.per_host_pack_unit(0, 1)


class PackSequencesPerHostTest(parameterized.TestCase):

  @parameterized.product(cap=(0, 1, 2), old_logps=(True, False))
  def test_round_trip(self, cap, old_logps):
    ex = _batch(_LENGTHS, old_logps=old_logps)
    chunks = _pack([ex], max_rows_per_shard=cap)
    flags = [bool(np.asarray(c.is_update_step)) for c in chunks]
    self.assertEqual(flags, [False] * (len(chunks) - 1) + [True])
    self.assertLen({np.shape(c.completion_ids) for c in chunks}, 1)
    seen = set()
    for chunk in chunks:
      rows = np.shape(chunk.completion_ids)[0]
      self.assertEqual(np.shape(chunk.prompt_ids), (rows, 0))
      self.assertEqual(chunk.num_segments, _BUDGET + 1)
      self.assertEqual(np.shape(np.asarray(chunk.is_update_step)), ())
      ids = np.asarray(chunk.completion_ids)
      seg = np.asarray(chunk.segment_ids)
      pos = np.asarray(chunk.segment_positions)
      mask = np.asarray(chunk.completion_mask)
      adv = np.asarray(chunk.advantages)
      for r, a, b, i in _segments(chunk):
        self.assertNotIn(i, seen)
        seen.add(i)
        p = ex.prompt_ids[i][ex.prompt_mask[i]]
        c_len = int(ex.completion_mask[i].sum())
        np.testing.assert_array_equal(
            ids[r, a:b], np.concatenate([p, ex.completion_ids[i, :c_len]])
        )
        np.testing.assert_array_equal(pos[r, a:b], np.arange(b - a))
        self.assertFalse(mask[r, a : a + p.size].any())
        self.assertTrue(mask[r, a + p.size : b].all())
        np.testing.assert_allclose(adv[r, a + p.size : b], ex.advantages[i])
        self.assertFalse(adv[r, a : a + p.size].any())
        if old_logps:
          np.testing.assert_allclose(
              np.asarray(chunk.old_per_token_logps)[r, a + p.size : b],
              ex.old_per_token_logps[i, :c_len],
          )
        else:
          self.assertIsNone(chunk.old_per_token_logps)
      off = seg == 0
      self.assertFalse(mask[off].any())
      self.assertFalse(adv[off].any())
    self.assertEqual(seen, set(range(len(_LENGTHS))))

  def test_rows_are_padded_to_a_multiple_of_the_unit(self):
    # 3 rows of 16 tokens; unit = 8 // gcd(8, 2) = 4 -> one dummy row.
    ex = _batch([(6, 10), (6, 10), (6, 10)])
    [chunk] = _pack([ex], data_axis_size=8, leading_dim_factor=2)
    self.assertEqual(np.shape(chunk.completion_ids), (4, _BUDGET))
    np.testing.assert_array_equal(np.asarray(chunk.segment_ids)[3], 0)
    self.assertFalse(np.asarray(chunk.completion_mask)[3].any())

  def test_max_segments_per_packed_row(self):
    ex = _batch([(1, 1)] * 6)
    [chunk] = _pack([ex], max_segments_per_packed_row=2)
    self.assertEqual(chunk.num_segments, 3)
    self.assertLessEqual(int(np.asarray(chunk.segment_ids).max()), 2)

  def test_loss_mask_zero_inside_a_completion_stays_context(self):
    ex = _batch(_LENGTHS)
    mask = np.array(ex.completion_mask)
    mask[0, 3] = False  # e.g. a rejected token: not scored, still attended
    ex = ex.replace(completion_mask=mask)
    chunks = _pack([ex])
    lengths = {i: b - a for c in chunks for _, a, b, i in _segments(c)}
    self.assertEqual(lengths[0], _LENGTHS[0][0] + _LENGTHS[0][1])

  def test_hole_aborts(self):
    ex = _batch(_LENGTHS)
    ids = np.array(ex.completion_ids)
    ids[0, 3] = _PAD  # a pad-valued token inside a completion
    ex = ex.replace(completion_ids=ids, completion_mask=ids != _PAD)
    with self.assertRaisesRegex(RuntimeError, "pad-valued token"):
      _pack([ex])

  def test_too_long_aborts(self):
    with self.assertRaisesRegex(RuntimeError, "longer than"):
      _pack([_batch(_LENGTHS)], budget=12)

  def test_empty_prompt_aborts(self):
    ex = _batch(_LENGTHS)
    pm = np.array(ex.prompt_mask)
    pm[2] = False
    with self.assertRaisesRegex(RuntimeError, "empty prompt"):
      _pack([ex.replace(prompt_mask=pm)])

  def test_routed_experts_abort(self):
    experts = np.zeros((len(_LENGTHS), _P + _C, 1, 1), np.int32)
    ex = _batch(_LENGTHS).replace(routed_experts=experts)
    with self.assertRaisesRegex(RuntimeError, "routed_experts"):
      _pack([ex])

  def test_chunks_keep_the_sampler_agreement_flag(self):
    for applied in (False, True):
      ex = _batch(_LENGTHS).replace(sampler_agreement_applied=applied)
      chunks = _pack([ex], max_rows_per_shard=1)
      self.assertGreater(len(chunks), 1)
      self.assertEqual({c.sampler_agreement_applied for c in chunks}, {applied})

  def test_stream_must_end_on_a_mini_batch_boundary(self):
    gen = utils.pack_sequences_per_host(
        iter([[_batch(_LENGTHS)]]),
        _BUDGET,
        len(_LENGTHS) + 1,
        pad_id=_PAD,
        data_axis_size=1,
        leading_dim_factor=1,
        allgather_fn=lambda row: np.asarray(row)[None, :],
        process_index=0,
    )
    with self.assertRaisesRegex(ValueError, "mid-mini-batch"):
      list(gen)


def _run_processes(batches, budget, **kwargs):
  """Runs one packer per thread; a barrier allgather stands in for JAX's."""
  n = len(batches)
  rows, results = [None] * n, [None] * n
  barrier = threading.Barrier(n, timeout=60)

  def gather_for(p):
    def gather(row):
      rows[p] = np.asarray(row).copy()
      barrier.wait()
      table = np.stack(rows)
      barrier.wait()
      return table

    return gather

  def run(p):
    try:
      results[p] = _pack(
          [batches[p]],
          budget=budget,
          allgather_fn=gather_for(p),
          process_index=p,
          **kwargs,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      results[p] = e

  threads = [threading.Thread(target=run, args=(p,)) for p in range(n)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(120)
  assert not any(t.is_alive() for t in threads), "a packer hung"
  return results


class MultiProcessAgreementTest(absltest.TestCase):

  def test_processes_agree_on_steps_and_shapes(self):
    short = _batch([(2, 1), (1, 2), (3, 3), (2, 2)], seed=1)
    long = _batch([(6, 10), (5, 9), (4, 8), (6, 6)], seed=2)
    mid = _batch([(3, 7), (2, 6), (4, 4), (1, 9)], seed=3)
    for cap in (0, 1):
      res = _run_processes(
          [short, long, mid],
          _BUDGET,
          data_axis_size=6,
          leading_dim_factor=3,
          max_rows_per_shard=cap,
      )
      for r in res:
        self.assertNotIsInstance(r, Exception)
      shapes = [[np.shape(c.completion_ids) for c in r] for r in res]
      self.assertEqual(shapes[0], shapes[1])
      self.assertEqual(shapes[0], shapes[2])
      self.assertEqual(shapes[0][0][0] % 2, 0)  # unit = 6 // gcd(6, 3) = 2

  def test_a_failure_on_one_process_aborts_every_process(self):
    short = _batch([(2, 1), (1, 2), (3, 3), (2, 2)], seed=1)
    long = _batch([(6, 10), (5, 9), (4, 8), (6, 6)], seed=2)
    res = _run_processes(
        [short, long], 12, data_axis_size=2, leading_dim_factor=2
    )
    for r in res:
      self.assertIsInstance(r, RuntimeError)
      self.assertIn("process(es) {1:", str(r))


class PackedLossEquivalenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model = _TinyCausalLM(rngs=nnx.Rngs(0))
    self.ex = _batch(_LENGTHS)

  def test_per_token_logps(self):
    [packed] = _pack([self.ex])
    lp_u, lp_p = _logps(self.model, self.ex), _logps(self.model, packed)
    for r, a, b, i in _segments(packed):
      p_len = int(self.ex.prompt_mask[i].sum())
      c_len = int(self.ex.completion_mask[i].sum())
      np.testing.assert_allclose(
          lp_p[r, a + p_len : b], lp_u[i, :c_len], rtol=0, atol=2e-5
      )

  def test_segments_do_not_attend_to_each_other(self):
    [packed] = _pack([self.ex])
    segs = list(_segments(packed))
    rows = [r for r, *_ in segs]
    row = next(r for r in rows if rows.count(r) >= 2)
    (a0, b0), *others = [(a, b) for r, a, b, _ in segs if r == row]
    ids = np.array(packed.completion_ids)
    ids[row, a0 + 1] = ids[row, a0 + 1] % (_VOCAB - 1) + 1
    before = _logps(self.model, packed)
    after = _logps(self.model, packed.replace(completion_ids=ids))
    self.assertGreater(np.max(np.abs(after[row, a0:b0] - before[row, a0:b0])), 1e-6)
    for a, b in others:
      # logp[a] (the segment's first token, a prompt token outside the loss) is predicted from logit[a-1], the previous
      # segment's last position, by construction of packed logps; isolation applies to positions a+1..b-1.
      np.testing.assert_allclose(after[row, a + 1:b], before[row, a + 1:b], atol=1e-6)

  @parameterized.parameters(
      ("sequence-mean-token-mean", "grpo"),
      ("token-mean", "grpo"),
      ("seq-mean-token-sum", "grpo"),
      ("sequence-mean-token-mean", "gspo-token"),
  )
  def test_loss_and_grads(self, mode, algo):
    cfg = _config(mode, algo)

    def loss(m, e):
      out = algo_core.grpo_loss_fn(m, e, cfg, pad_id=_PAD, eos_id=-1)
      return out.primary_loss.compute()

    def parts(m, e):
      out = algo_core.grpo_loss_fn(m, e, cfg, pad_id=_PAD, eos_id=-1)
      return out.primary_loss.unreduced_sum, out.primary_loss.denominator

    loss_u, grads_u = nnx.value_and_grad(loss)(self.model, self.ex)
    [packed] = _pack([self.ex])
    loss_p, grads_p = nnx.value_and_grad(loss)(self.model, packed)
    np.testing.assert_allclose(float(loss_p), float(loss_u), rtol=2e-5)
    _assert_trees_close(self, grads_u, grads_p, 2e-4)

    # Capped chunks, accumulated as the trainer does: sum of the gradients of
    # the unreduced sums over the sum of the denominators.
    chunks = _pack([self.ex], max_rows_per_shard=2)
    self.assertGreater(len(chunks), 1)
    acc, num, den = None, 0.0, 0.0
    for chunk in chunks:
      (s, d), g = nnx.value_and_grad(parts, has_aux=True)(self.model, chunk)
      num, den = num + float(s), den + float(d)
      acc = g if acc is None else jax.tree.map(lambda x, y: x + y, acc, g)
    np.testing.assert_allclose(num / den, float(loss_u), rtol=2e-5)
    _assert_trees_close(
        self, grads_u, jax.tree.map(lambda x: x / den, acc), 2e-4
    )


# -----------------------------------------------------------------------------
# 2-process CPU job: the per-host learner trains the same packed and unpacked.
# Modeled on tests/rl/per_host_metrics_test.py.
# -----------------------------------------------------------------------------
_CHILD_ENV = "TUNIX_PER_HOST_PACKING_TEST_CHILD"
_NUM_PROCESSES = 2
_TRAIN_STEPS = 2
_PROMPTS = ["hello world", "My name", "input string", "hello there"]
_BASE_LENGTH = {"hello world": 1, "My name": 6, "input string": 3, "hello there": 8}


def _run_child():
  """One process of the 2-process per-host GRPO job (see `_run_job`)."""
  args = json.loads(os.environ[_CHILD_ENV])
  jax.config.update("jax_cpu_collectives_implementation", "gloo")
  jax.distributed.initialize(
      f"localhost:{args['port']}",
      num_processes=_NUM_PROCESSES,
      process_id=args["process_id"],
  )
  # pylint: disable=g-import-not-at-top
  from absl import logging as absl_logging
  from jax.experimental import multihost_utils
  import optax
  from tunix.rl import rl_cluster as rl_engine_lib
  from tunix.rl.grpo import grpo_learner as grpo_lib
  from tunix.rl.rollout import base_rollout
  from tunix.tests import test_common as tc
  # pylint: enable=g-import-not-at-top

  absl_logging.set_verbosity(absl_logging.INFO)
  vocab = tc.MockVocab()

  class ShardRollout(base_rollout.BaseRollout):
    """Host-local rollout; completion length depends on prompt and sample."""

    def __init__(self, rollout_actor=None, **kwargs):
      del kwargs
      self._model = rollout_actor

    def generate(self, prompts, rollout_config, **kwargs):
      del kwargs
      samples, tokens, texts = {}, [], []
      prompt_tokens = np.zeros(
          (len(prompts), rollout_config.max_prompt_length), np.int32
      )
      for row, prompt in enumerate(prompts):
        prompt = str(prompt)
        sample = samples.get(prompt, 0)
        samples[prompt] = sample + 1
        length = _BASE_LENGTH[prompt] + sample
        tokens.append(np.full((length,), vocab.DEFAULT_MAPPING["hello"]))
        texts.append(" ".join(["hello"] * length))
        ids = [vocab.DEFAULT_MAPPING[w] for w in prompt.split()]
        prompt_tokens[row, -len(ids) :] = ids
      return base_rollout.RolloutOutput(
          text=texts,
          logits=None,
          tokens=tokens,
          left_padded_prompt_tokens=prompt_tokens,
          logprobs=None,
      )

    def get_per_token_logps(self, *args, **kwargs):
      raise NotImplementedError()

    def update_params(self, params, filter_types=None):
      del params, filter_types

    def pad_id(self):
      return vocab.pad_id()

    def eos_id(self):
      return vocab.eos_id()

    def model(self):
      return self._model

  def reward(prompts, completions, **kwargs):
    del kwargs
    return [
        (1.0 if str(p) in ("hello world", "input string") else 0.0)
        + 0.1 * len(c.split())
        for p, c in zip(prompts, completions)
    ]

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()).reshape(2, 2), ("fsdp", "tp")
  )
  budget = args["budget"]
  cluster_config = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: mesh,
          rl_engine_lib.Role.REFERENCE: mesh,
          rl_engine_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine=ShardRollout,
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-2),
          eval_every_n_steps=1,
          max_steps=_TRAIN_STEPS,
          max_seq_token_per_tpu=budget or None,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=10, max_prompt_length=8, kv_cache_size=64
      ),
  )
  rl_engine = rl_engine_lib.RLEngine(
      actor=_TinyCausalLM(vocab=vocab.GetPieceSize(), rngs=nnx.Rngs(0)),
      reference=None,
      tokenizer=vocab,
      cluster_config=cluster_config,
  )
  learner = grpo_lib.GRPOLearner(
      rl_engine=rl_engine,
      reward_fns=reward,
      algo_config=grpo_lib.GRPOConfig(
          num_generations=2, num_iterations=1, beta=0.0
      ),
  )
  learner.train([{"prompts": np.array(_PROMPTS)}] * _TRAIN_STEPS)
  actor_logger = rl_engine.actor_trainer.metrics_logger
  trainer = {
      name: np.asarray(
          actor_logger.get_metric_history("actor", name, "train")
      ).tolist()
      for name in ("loss", "grad_norm")
  }
  params = [
      np.asarray(multihost_utils.process_allgather(x, tiled=True))
      for x in jax.tree.leaves(
          nnx.state(rl_engine.actor_trainer.model, nnx.Param)
      )
  ]
  if args["process_id"] == 0:
    np.savez(os.path.join(args["out_dir"], "params.npz"), *params)
  with open(
      os.path.join(args["out_dir"], f"process{args['process_id']}.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump({"trainer": trainer}, f)


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def _run_job(out_dir: str, budget: int, max_rows_per_shard: int = 0):
  """Runs the 2-process per-host job; returns (results, params, logs)."""
  port = _free_port()
  env = {
      k: v
      for k, v in os.environ.items()
      if k.lower() not in ("http_proxy", "https_proxy", "no_proxy")
  }
  repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  env.update(
      JAX_PLATFORMS="cpu",
      XLA_FLAGS="--xla_force_host_platform_device_count=2",
      TUNIX_PER_HOST_LEARNER="1",
      TUNIX_PER_HOST_SAFE_LOGPS="1",
      TUNIX_PER_HOST_PACK_MAX_ROWS_PER_SHARD=str(max_rows_per_shard),
      PYTHONPATH=os.pathsep.join(
          [os.path.abspath(repo_root), env.get("PYTHONPATH", "")]
      ),
  )
  procs = []
  for process_id in range(_NUM_PROCESSES):
    env[_CHILD_ENV] = json.dumps({
        "process_id": process_id,
        "port": port,
        "out_dir": out_dir,
        "budget": budget,
    })
    procs.append(
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    )
  logs = [p.communicate(timeout=900)[0] for p in procs]
  if any(p.returncode for p in procs):
    raise AssertionError(
        "per-host job failed:\n" + "\n".join(log[-4000:] for log in logs)
    )
  results = []
  for process_id in range(_NUM_PROCESSES):
    with open(
        os.path.join(out_dir, f"process{process_id}.json"), encoding="utf-8"
    ) as f:
      results.append(json.load(f))
  with np.load(os.path.join(out_dir, "params.npz")) as f:
    params = [f[k] for k in sorted(f.files, key=lambda k: int(k[4:]))]
  return results, params, logs


class PerHostPackedJobTest(absltest.TestCase):

  def test_packed_training_matches_unpacked(self):
    runs = {}
    for name, budget, cap in (
        ("unpacked", 0, 0),
        ("packed", 18, 0),  # 8 prompt + 10 completion tokens per row
        ("packed_capped", 18, 1),  # one row per data shard per train step
    ):
      with tempfile.TemporaryDirectory() as out_dir:
        runs[name] = _run_job(out_dir, budget, cap)
    ref_results, ref_params, _ = runs["unpacked"]
    for name in ("packed", "packed_capped"):
      results, params, logs = runs[name]
      # Process 1 rolls out the long completions, so it packs more rows than
      # process 0; both processes report the same agreed plan.
      self.assertTrue(
          any("[TUNIX-PH-PACK]" in log for log in logs), f"{name}: no receipt"
      )
      self.assertTrue(any("Per-host packing:" in log for log in logs[:1]))
      np.testing.assert_allclose(
          results[0]["trainer"]["loss"],
          ref_results[0]["trainer"]["loss"],
          rtol=1e-5,
          err_msg=name,
      )
      self.assertLen(params, len(ref_params))
      for got, want in zip(params, ref_params):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  if _CHILD_ENV in os.environ:
    exit_code = 0
    try:
      _run_child()
    except Exception:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip the distributed shutdown barrier; the peer may already be gone.
    os._exit(exit_code)
  absltest.main()
