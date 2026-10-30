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

"""CPU OPD training through HTTP, program stages, native loss and optimizer.

The in-process engine replaces RPC/Raiden with a queue and explicit NNX state
copy. Its seeded rollout samples the synced policy's full categorical
distribution. The HTTP server implements SGLang's scoring response with an
independent NumPy teacher; it is not a running SGLang inference engine.
"""

import asyncio
import copy
import http.server
import json
import threading
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import models
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.distillation import on_policy
from tunix.dist_rl.distillation import teacher
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algorithm_config

_VOCAB = 6
_GROUP_SIZE = 8
_LEARNING_RATE = 0.2


def _log_softmax(values):
  """NumPy oracle independent of the production scoring/loss helpers."""
  values = np.asarray(values, dtype=np.float64)
  shifted = values - values.max(axis=-1, keepdims=True)
  return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _dense_kl(student, fixed_teacher):
  log_p, log_q = _log_softmax(student), _log_softmax(fixed_teacher)
  return float(np.mean(np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)))


def _tokenizer():
  vocab = {"[EOS]": 0, "[BOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5}
  return PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[EOS]")),
      unk_token="[EOS]",
      pad_token="[EOS]",
      eos_token="[EOS]",
      bos_token="[BOS]",
  )


class _PositionPolicy(nnx.Module):
  """A trainable distribution for previous token and absolute-position parity."""

  def __init__(self, logits):
    self.logits = nnx.Param(jnp.asarray(logits, dtype=jnp.float32))

  def __call__(
      self, tokens, positions, cache=None, attention_mask=None, segment_ids=None
  ):
    del attention_mask, segment_ids
    return self.logits[tokens, positions % 2], cache


class _TeacherServer(http.server.ThreadingHTTPServer):
  daemon_threads = True

  def __init__(self, logits):
    super().__init__(("127.0.0.1", 0), _TeacherHandler)
    self.logits = np.asarray(logits, dtype=np.float64).copy()
    self.logits.flags.writeable = False
    self.shift_scores = False
    self.requests = []
    self.lock = threading.Lock()


class _TeacherHandler(http.server.BaseHTTPRequestHandler):

  def log_message(self, *args):
    del args

  def do_POST(self):  # pylint: disable=invalid-name
    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    ids = body["input_ids"]
    log_probs = _log_softmax(self.server.logits)
    values = [[None, ids[0], ""]]
    for position in range(1, len(ids)):
      score = log_probs[ids[position - 1], (position - 1) % 2, ids[position]]
      values.append([float(score), ids[position], ""])
    if self.server.shift_scores:
      # Malformed scores retain their token IDs, defeating ID-only checks.
      shifted = [value[0] for value in values[1:]]
      shifted = shifted[1:] + shifted[:1]
      for value, score in zip(values[1:], shifted):
        value[0] = score
    with self.server.lock:
      self.server.requests.append(copy.deepcopy(body))
    response = json.dumps(
        {"meta_info": {"input_token_logprobs": values}}
    ).encode()
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(response)))
    self.end_headers()
    self.wfile.write(response)


class _CpuEngine:
  """Local transport adapter around a real native trainer and worker."""

  def __init__(self, initial, mesh):
    self.trainer = peft_trainer_v2.PeftTrainer(
        _PositionPolicy(initial),
        optax.sgd(_LEARNING_RATE),
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=100,
            max_steps=2,
            gradient_accumulation_steps=_GROUP_SIZE // 2,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    self.worker = trainer_worker.TrainerWorker(
        lambda: self.trainer, execution_context=mesh
    )
    self.worker.initialize()
    self.rollout = _PositionPolicy(initial)
    self.pending = asyncio.Queue()
    self.version = 0
    self.generations = []
    self.batches = []
    self.before_updates = []
    self.after_updates = []
    self.checkpoints = []
    self.restored_checkpoint_metadata = None

  async def resume_from_checkpoint(self, **kwargs):
    del kwargs
    return 0

  def configure_worker(self, *, algo, assembler, **kwargs):
    del kwargs
    self.worker.with_loss_fn(algo.loss_fn(), has_aux=True)
    self.worker.with_gen_model_input_fn(
        algo.build_gen_model_input_fn(pad_id=assembler.pad_id, eos_id=0)
    )

  async def prepare_rollout_policy(self, **kwargs):
    del kwargs
    nnx.update(self.rollout, nnx.state(self.trainer.model))
    return self.version

  async def dispatch_rollouts(
      self, prompts, *, num_generations, policy_version, **kwargs
  ):
    del kwargs
    if policy_version != self.version:
      raise AssertionError("Rollout was dispatched against a stale policy.")
    snapshot = np.asarray(self.rollout.logits[...]).copy()
    log_probs = _log_softmax(snapshot)
    rng = np.random.default_rng(7 + self.version)
    items = []
    for prompt in prompts:
      generated = []
      for group_index in range(num_generations):
        # A real ID-0 token occurs in the prompt even though pad=eos=0.
        prefix = [1, 0]
        completion, mask, scores = [], [], []
        for step in range(3):
          distribution = log_probs[prefix[-1], (len(prefix) - 1) % 2]
          token = int(rng.choice(_VOCAB, p=np.exp(distribution)))
          completion.append(token)
          mask.append(1.0)
          scores.append(float(distribution[token]))
          prefix.append(token)
          if token == 0:
            break
          if step == 0 and group_index % 2:
            # Simulated tool observation: no policy loss, but valid context.
            completion.append(0)
            mask.append(0.0)
            scores.append(0.0)
            prefix.append(0)
        item = datatypes.TrajectoryItem(
            prompt_id=prompt["prompt_id"],
            group_index=group_index,
            metadata={**prompt["metadata"], "policy_version": self.version},
            traj={
                "prompt_tokens": np.asarray([1, 0], dtype=np.int32),
                "conversation_tokens": np.asarray(completion, dtype=np.int32),
                "conversation_masks": np.asarray(mask, dtype=np.float32),
                "old_logprobs": np.asarray(scores, dtype=np.float32),
                "trajectory_reward": 1.0,
                "conversation_text": [
                    {"role": "user", "content": "fixed context"},
                    {"role": "assistant", "content": "sampled response"},
                ],
                "status": datatypes.TrajectoryStatus.SUCCEEDED,
                "policy_version": self.version,
            },
        )
        items.append(item)
        generated.append(item)
      self.generations.append((self.version, snapshot, generated))
    await self.pending.put(items)
    return [f"{item.prompt_id}-{item.group_index}" for item in items]

  async def poll_rollouts(self):
    try:
      return await asyncio.wait_for(self.pending.get(), timeout=0.02)
    except TimeoutError:
      return []

  async def per_token_logps(self, role, items, **kwargs):
    del role, kwargs
    return self.worker.per_token_logps(items)

  async def train_step(self, payload, *, apply_optimizer, **kwargs):
    del kwargs
    if not self.batches or self.batches[-1][0] != self.version:
      self.before_updates.append(
          np.asarray(self.trainer.model.logits[...]).copy()
      )
    self.batches.append((self.version, payload))
    self.worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
    if apply_optimizer:
      step = self.worker.update()
      self.after_updates.append(
          np.asarray(self.trainer.model.logits[...]).copy()
      )
      return {"train_step": step}
    return None

  async def get_metrics(self, **kwargs):
    del kwargs
    return self.worker.get_metrics()

  async def save_checkpoint(self, *, metadata, **kwargs):
    del kwargs
    # Metadata is recorded; disk checkpoint/recovery is outside this test.
    self.checkpoints.append(copy.deepcopy(metadata))

  async def sync_weights(self, **kwargs):
    del kwargs
    nnx.update(self.rollout, nnx.state(self.trainer.model))
    self.version += 1
    return self.version


class OPDTrainingTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    rng = np.random.default_rng(7)
    self.initial = rng.normal(0, 0.25, (_VOCAB, 2, _VOCAB)).astype(np.float32)
    self.initial[..., 0] -= 3.0
    target = self.initial.astype(np.float64).copy()
    target[:, 0, 3] += 1.0
    target[:, 1, 4] += 1.4
    self.server = _TeacherServer(target)
    self.thread = threading.Thread(
        target=lambda: self.server.serve_forever(poll_interval=0.01),
        daemon=True,
    )
    self.thread.start()

  async def asyncTearDown(self):
    await asyncio.to_thread(self.server.shutdown)
    self.server.server_close()
    self.thread.join(timeout=2.0)

  async def _run(self, coefficient, steps=2):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("fsdp",))
    with mesh, jax.default_device(jax.devices("cpu")[0]):
      engine = _CpuEngine(self.initial, mesh)
      adapter = algorithm_adapter.GRPOAdapter(
          algo_config=algorithm_config.GRPOConfig(
              num_generations=_GROUP_SIZE,
              beta=0.0,
              reward_mode="env",
              use_rollout_logps=True,
              loss_agg_mode="token-mean",
          ),
          mini_batch_size=1,
          train_micro_batch_size=2,
          max_response_length=4,
          max_packed_len=8,
      )
      scorer = teacher.SGLangTeacherScorer(
          f"http://127.0.0.1:{self.server.server_port}",
          student_tokenizer=_tokenizer(),
          teacher_tokenizer=_tokenizer(),
          timeout_seconds=5.0,
          max_concurrency=3,
      )
      program = rl_program.StandardRLProgram(
          algo=adapter,
          dataset=[
              {"prompt_id": f"p{step}", "prompt": "fixed"}
              for step in range(steps)
          ],
          max_steps=steps,
          batch_size=1,
          precompile_trainer=False,
          assembler=batch_assembly.PaddedBatchAssembler(
              batch_size=2,
              num_generations=_GROUP_SIZE,
              mini_batch_size=1,
              max_prompt_length=3,
              max_response_length=4,
              pad_id=0,
          ),
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, top_p=1.0, top_k=0
          ),
          opd_config=on_policy.OnPolicyDistillationConfig(
              coefficient=coefficient
          ),
          teacher_scorer=scorer,
      )
      try:
        await asyncio.wait_for(program.run_async(engine), timeout=30.0)
      finally:
        program.close()
        engine.worker.stop()
    return engine

  def _expected_after_update(
      self, engine, version, coefficient, *, verify_signals=True
  ):
    teacher_scores = _log_softmax(self.server.logits)
    before = engine.before_updates[version]
    student_scores = _log_softmax(before)
    gradient = np.zeros_like(before, dtype=np.float64)
    count = 0
    for batch_version, batch in engine.batches:
      if batch_version != version:
        continue
      for row in range(len(batch.prompt_ids)):
        prompt = np.asarray(batch.prompt_ids[row])[
            np.asarray(batch.prompt_mask[row], bool)
        ]
        length = int(np.sum(batch.completion_attention_mask[row]))
        completion = np.asarray(batch.completion_ids[row])[:length]
        prefix = list(prompt)
        for index, token in enumerate(completion):
          context, parity = prefix[-1], (len(prefix) - 1) % 2
          if batch.completion_mask[row, index] > 0:
            log_p = student_scores[context, parity]
            advantage = coefficient * (
                teacher_scores[context, parity, token] - log_p[token]
            )
            if verify_signals:
              np.testing.assert_allclose(
                  batch.old_per_token_logps[row, index], log_p[token], atol=2e-6
              )
              np.testing.assert_allclose(
                  batch.advantages[row, index], advantage, atol=2e-6
              )
            term = np.exp(log_p)
            term[token] -= 1.0
            gradient[context, parity] += advantage * term
            count += 1
          else:
            self.assertEqual(batch.advantages[row, index], 0.0)
          prefix.append(token)
    self.assertGreater(count, 0)
    return before - _LEARNING_RATE * gradient / count

  def _assert_sample_gradient(self, engine, coefficient):
    for version, after in enumerate(engine.after_updates):
      expected = self._expected_after_update(engine, version, coefficient)
      np.testing.assert_allclose(after, expected, atol=2e-6, rtol=2e-6)

  def _assert_boundary_coverage(self, engine):
    sampled_eos = tool_zero = 0
    lengths = set()
    for _, _, items in engine.generations:
      for item in items:
        tokens = item.traj["conversation_tokens"]
        mask = item.traj["conversation_masks"]
        sampled_eos += int(np.sum((tokens == 0) & (mask > 0)))
        tool_zero += int(np.sum((tokens == 0) & (mask == 0)))
        lengths.add(len(tokens))
    self.assertGreater(sampled_eos, 0, "Fixture must sample a real EOS action.")
    self.assertGreater(tool_zero, 0, "Fixture must contain a masked tool zero.")
    self.assertGreaterEqual(len(lengths), 2)
    batched_eos = batched_tool_zero = padding = 0
    for _, batch in engine.batches:
      tokens = np.asarray(batch.completion_ids)
      action = np.asarray(batch.completion_mask) > 0
      attention = np.asarray(batch.completion_attention_mask, bool)
      self.assertTrue(np.all(attention[action]))
      batched_eos += int(np.sum((tokens == 0) & action & attention))
      batched_tool_zero += int(np.sum((tokens == 0) & ~action & attention))
      padding += int(np.sum(~attention))
      self.assertFalse(np.any(action & ~attention))
    self.assertEqual(batched_eos, sampled_eos)
    self.assertEqual(batched_tool_zero, tool_zero)
    self.assertGreater(padding, 0)
    return sampled_eos, tool_zero, sorted(lengths), padding

  async def test_http_pure_opd_trains_and_syncs_two_real_optimizer_steps(self):
    teacher_before = self.server.logits.copy()
    engine = await self._run(coefficient=1.0)

    self.assertEqual([version for version, _, _ in engine.generations], [0, 1])
    self.assertEqual(engine.version, 2)
    self.assertEqual(len(engine.before_updates), 2)
    self.assertEqual(len(self.server.requests), 2 * _GROUP_SIZE)
    eos, tool_zero, lengths, padding = self._assert_boundary_coverage(engine)
    self._assert_sample_gradient(engine, 1.0)
    np.testing.assert_array_equal(self.server.logits, teacher_before)
    np.testing.assert_array_equal(
        engine.generations[1][1], engine.after_updates[0]
    )
    np.testing.assert_array_equal(
        engine.before_updates[1], engine.after_updates[0]
    )
    self.assertFalse(
        np.array_equal(engine.generations[0][1], engine.generations[1][1])
    )
    for body in self.server.requests:
      self.assertEqual(body["input_ids"][:2], [1, 0])
      self.assertEqual(body["sampling_params"]["max_new_tokens"], 0)
    before = _dense_kl(self.initial, teacher_before)
    after = _dense_kl(engine.after_updates[-1], teacher_before)
    self.assertLess(after, before)
    score_change = float(
        np.max(
            np.abs(
                _log_softmax(engine.generations[1][1])
                - _log_softmax(engine.generations[0][1])
            )
        )
    )
    self.assertGreater(score_change, 1e-4)
    print(
        f"CPU OPD: dense KL {before:.9f} -> {after:.9f}; max rollout logprob"
        f" change {score_change:.9f}; versions [0, 1] -> 2; sampled EOS {eos},"
        f" tool zeros {tool_zero}, response lengths {lengths}, padding"
        f" {padding}"
    )

  async def test_parameter_oracle_rejects_shifted_teacher_scores_with_valid_ids(
      self,
  ):
    self.server.shift_scores = True
    engine = await self._run(coefficient=1.0, steps=1)
    self.assertEqual(len(self.server.requests), _GROUP_SIZE)
    # Scoring succeeds because IDs agree. The independent parameter oracle must
    # still detect semantic misalignment without inspecting stored advantages.
    expected = self._expected_after_update(
        engine, version=0, coefficient=1.0, verify_signals=False
    )
    actual = engine.after_updates[0]
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    error = float(np.max(np.abs(actual - expected)))
    self.assertGreater(error, 1e-4)
    print(
        f"CPU OPD negative control: shifted-score parameter error {error:.9f}"
    )

  async def test_zero_coefficient_pure_opd_has_no_update_or_teacher_requests(
      self,
  ):
    engine = await self._run(coefficient=0.0, steps=1)
    self.assertEqual(self.server.requests, [])
    np.testing.assert_array_equal(engine.after_updates[0], self.initial)
    self._assert_sample_gradient(engine, 0.0)


if __name__ == "__main__":
  unittest.main()
