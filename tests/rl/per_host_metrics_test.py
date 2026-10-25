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

"""Global RL metrics under the per-host learner (TUNIX_PER_HOST_LEARNER=1).

Every process of the per-host learner rolls out and scores its own prompt
shard, and only process 0 exports metrics. These tests check that the
exported rewards, completion lengths and metric fn values are the ones of the
union of all processes' examples, including uneven completion lengths and a
process without valid items, and that training itself is unchanged.
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
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.sft import metrics_logger
from tunix.sft import utils as sft_utils

_CHILD_ENV = "TUNIX_PER_HOST_METRICS_TEST_CHILD"
_NUM_PROCESSES = 2
_TRAIN_STEPS = 2

# The global batch is strided over the processes ([process::2]), so process 0
# gets the short completions with high rewards and process 1 the long ones.
_PROMPTS = ["hello world", "My name", "input string", "hello there"]
_BASE_LENGTH = {
    "hello world": 1,
    "My name": 6,
    "input string": 3,
    "hello there": 8,
}
_HIGH_REWARD = ("hello world", "input string")
_LONG = 5  # Completions longer than this are "valid"; process 0 has none.


def _apply(values, op):
  """Reduces buffered values the way `RLEngine._log_metrics` does."""
  return op(np.array(values))


class MergeProcessMetricsTest(absltest.TestCase):

  def test_per_example_values_of_uneven_and_empty_shards(self):
    # Process 0 has no valid items (empty array), the others differ in size.
    shards = [np.array([]), np.array([1.0, 3.0]), np.array([2.0, 6.0, 7.0])]
    union = np.concatenate(shards)
    for op in (np.mean, np.min, np.max, np.sum):
      merged = rl_engine_lib._merge_process_metrics(
          [{"m": ([shard], op)} for shard in shards], {"m": op}
      )
      self.assertAlmostEqual(_apply(*merged["m"]), op(union))

  def test_equal_shapes_keep_the_buffered_structure(self):
    merged = rl_engine_lib._merge_process_metrics(
        [{"m": ([1.0, 2.0], np.mean)}, {"m": ([4.0, 5.0], np.mean)}],
        {"m": np.mean},
    )
    self.assertEqual(merged["m"][0], [1.0, 2.0, 4.0, 5.0])

  def test_metric_only_other_processes_buffered(self):
    merged = rl_engine_lib._merge_process_metrics(
        [{}, {"m": ([np.array([3.0, 5.0])], np.max)}], {}
    )
    self.assertEqual(_apply(*merged["m"]), 5.0)

  def test_unknown_reducer_is_skipped(self):
    merged = rl_engine_lib._merge_process_metrics(
        [{}, {"m": ([1.0], None)}], {}
    )
    self.assertNotIn("m", merged)

  def test_weighted_metric_keeps_real_denominators(self):
    # Process 0 has no valid tokens (denominator 0).
    shards = [
        sft_utils.WeightedMetric(np.float32(0.0), np.float32(0.0)),
        sft_utils.WeightedMetric(np.float32(6.0), np.float32(2.0)),
        sft_utils.WeightedMetric(np.float32(30.0), np.float32(4.0)),
    ]
    merged = rl_engine_lib._merge_process_metrics(
        [{"m": ([s], common.global_weighted_mean)} for s in shards],
        {"m": common.global_weighted_mean},
    )
    self.assertAlmostEqual(_apply(*merged["m"]), 36.0 / 6.0)

  def test_pearson_metric_of_the_union(self):
    def pearson(x, y):
      if not x.size:
        return common.make_pearson_metric(*np.zeros(6, np.float32))
      mx, my = x.mean(), y.mean()
      return common.make_pearson_metric(
          ((x - mx) * (y - my)).sum(),
          ((x - mx) ** 2).sum(),
          ((y - my) ** 2).sum(),
          np.float32(x.size),
          mx,
          my,
      )

    rng = np.random.default_rng(0)
    x = rng.normal(size=9).astype(np.float32)
    y = (0.5 * x + rng.normal(size=9)).astype(np.float32)
    # Process 0 has no valid tokens; the others hold 4 and 5 of them.
    shards = [
        pearson(x[:0], y[:0]),
        pearson(x[:4], y[:4]),
        pearson(x[4:], y[4:]),
    ]
    merged = rl_engine_lib._merge_process_metrics(
        [{"m": ([s], sft_utils.weighted_metric_mean)} for s in shards],
        {"m": sft_utils.weighted_metric_mean},
    )
    self.assertAlmostEqual(
        _apply(*merged["m"]), np.corrcoef(x, y)[0, 1], places=5
    )


def _engine(cross_process: bool):
  """An RLEngine with only the metric buffering state."""
  engine = rl_engine_lib.RLEngine.__new__(rl_engine_lib.RLEngine)
  engine._rl_metrics_logger = metrics_logger.MetricsLogger(None)
  engine._buffered_train_metrics = []
  engine._buffered_eval_metrics = []
  engine._external_metrics_logger = None
  engine._cross_process_metrics = False
  engine.global_steps = 0
  if cross_process:
    engine.enable_cross_process_metrics()
  return engine


def _flush(engine, buffers):
  """Logs the oldest buffer of `buffers` like a step flush, synchronously."""
  engine._log_metrics(buffers.pop(0))


def _history(engine, prefix, name, mode="train"):
  return [
      float(np.asarray(v))
      for v in engine._rl_metrics_logger.get_metric_history(prefix, name, mode)
  ]


class RLEngineCrossProcessMetricsTest(absltest.TestCase):

  def test_flags_off_logging_is_unchanged(self):
    engine = _engine(cross_process=False)
    engine.buffer_metrics({
        "rewards/mean": (np.array([1.0, 3.0]), np.mean),
        "rewards/max": (np.array([1.0, 3.0]), np.max),
        "prompts": (["a", "b"], None),
    })
    with self.assertLogs(level="INFO") as logs:
      _flush(engine, engine._buffered_train_metrics)
    self.assertEqual(_history(engine, "rewards", "mean"), [2.0])
    self.assertEqual(_history(engine, "rewards", "max"), [3.0])
    self.assertIn(
        "Rollout string metric prompts: [['a' 'b']]", "\n".join(logs.output)
    )

  def test_flags_off_reduce_is_a_noop(self):
    engine = _engine(cross_process=False)
    engine.buffer_metrics({"rewards/mean": (np.array([1.0, 3.0]), np.mean)})
    engine.global_steps = 1
    with mock.patch.object(rl_engine_lib, "_allgather_bytes") as gather:
      engine.reduce_metrics_across_processes()
      engine.reduce_metrics_across_processes(include_current=True)
    gather.assert_not_called()
    self.assertNotIsInstance(
        engine._buffered_train_metrics[0].metrics, rl_engine_lib._GlobalMetrics
    )

  def test_reduced_buffer_holds_every_process_values(self):
    engine = _engine(cross_process=True)
    remote = {
        ("train", 0, 0): {
            "rewards/mean": ([np.array([0.0, 0.0])], np.mean),
            "only_remote": ([np.array([7.0])], np.max),
        },
        ("eval", 0, 0): {"rewards/mean": ([np.array([4.0])], np.mean)},
    }
    payloads = []

    def fake_gather(payload):
      payloads.append(rl_engine_lib.pickle.loads(payload))
      other = remote if payloads[-1] else {}
      return [payload, rl_engine_lib.pickle.dumps(other)]

    engine.buffer_metrics({
        "rewards/mean": (np.array([1.0, 1.0]), np.mean),
        "prompts": (["a", "b"], None),
    })
    engine.buffer_metrics(
        {"rewards/mean": (np.array([2.0]), np.mean)},
        mode=rl_engine_lib.Mode.EVAL,
    )
    with mock.patch.object(rl_engine_lib, "_allgather_bytes", fake_gather):
      # Every call gathers (all processes must take part), but only buffers
      # of completed steps that were not reduced yet are sent.
      engine.reduce_metrics_across_processes()  # Step 0 is not completed.
      engine.global_steps = 1
      engine.reduce_metrics_across_processes()
      engine.reduce_metrics_across_processes()  # Already reduced.
    self.assertLen(payloads, 3)
    self.assertEqual(payloads[0], {})
    self.assertEqual(sorted(payloads[1]), [("eval", 0, 0), ("train", 0, 0)])
    self.assertNotIn("prompts", payloads[1][("train", 0, 0)])  # Host-local.
    self.assertEqual(payloads[2], {})
    _flush(engine, engine._buffered_train_metrics)
    _flush(engine, engine._buffered_eval_metrics)
    self.assertEqual(_history(engine, "rewards", "mean"), [0.5])
    self.assertEqual(_history(engine, "rewards", "mean", "eval"), [3.0])
    self.assertEqual(_history(engine, "global", "only_remote"), [7.0])

  def test_unreduced_buffer_is_not_logged_under_the_global_name(self):
    engine = _engine(cross_process=True)
    engine.buffer_metrics({"rewards/mean": (np.array([1.0]), np.mean)})
    with self.assertLogs(level="WARNING") as logs:
      engine._log_metrics(engine._buffered_train_metrics[0])
    self.assertFalse(
        engine._rl_metrics_logger.metric_exists("rewards", "mean", "train")
    )
    self.assertIn("not reduced across processes", "\n".join(logs.output))


def _run_child():
  """One process of the 2-process per-host GRPO job (see `_run_job`)."""
  args = json.loads(os.environ[_CHILD_ENV])
  import jax  # pylint: disable=g-import-not-at-top

  jax.config.update("jax_cpu_collectives_implementation", "gloo")
  jax.distributed.initialize(
      f"localhost:{args['port']}",
      num_processes=_NUM_PROCESSES,
      process_id=args["process_id"],
  )
  # pylint: disable=g-import-not-at-top
  from flax import nnx
  from jax.experimental import multihost_utils
  import optax
  from tunix.rl.grpo import grpo_learner as grpo_lib
  from tunix.rl.rollout import base_rollout
  from tunix.tests import test_common as tc
  # pylint: enable=g-import-not-at-top

  vocab = tc.MockVocab()

  class ShardRollout(base_rollout.BaseRollout):
    """Host-local rollout; completion length depends on prompt and sample."""

    def __init__(self, rollout_actor=None, **kwargs):
      del kwargs
      self._model = rollout_actor

    def generate(self, prompts, rollout_config, **kwargs):
      del kwargs
      samples = {}
      tokens, texts = [], []
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

  local_examples = []

  def shard_reward(prompts, completions, mode, **kwargs):
    del kwargs
    lengths = [len(c.split()) for c in completions]
    rewards = [
        (1.0 if str(p) in _HIGH_REWARD else 0.0) + 0.1 * n
        for p, n in zip(prompts, lengths)
    ]
    local_examples.append(
        {"mode": mode, "lengths": lengths, "rewards": rewards}
    )
    return rewards

  def valid_item_metrics(completions, rewards, **kwargs):
    del kwargs
    lengths = np.array([len(c.split()) for c in completions], np.float32)
    valid = lengths > _LONG
    metrics = {
        "custom/valid_success": (np.asarray(rewards)[valid] > 0.75, np.mean),
        "custom/len_ge3": (lengths[lengths >= 3], np.mean),
        "custom/valid_len": (
            sft_utils.WeightedMetric(
                np.float32(lengths[valid].sum()), np.float32(valid.sum())
            ),
            common.global_weighted_mean,
        ),
        "custom/valid_count": (float(valid.sum()), np.sum),
    }
    if valid.any():  # Not buffered at all by a process without valid items.
      metrics["custom/valid_max_len"] = (lengths[valid], np.max)
    return metrics

  gather_threads = []
  if hasattr(rl_engine_lib, "_allgather_bytes"):
    allgather_bytes = rl_engine_lib._allgather_bytes

    def recording_allgather(payload):
      gather_threads.append(threading.current_thread().name)
      return allgather_bytes(payload)

    rl_engine_lib._allgather_bytes = recording_allgather
  if args["disable_reduction"]:
    rl_engine_lib.RLEngine.reduce_metrics_across_processes = (
        lambda self, include_current=False: None
    )

  mesh = jax.sharding.Mesh(
      np.array(jax.devices()).reshape(2, 2), ("fsdp", "tp")
  )
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
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=10, max_prompt_length=8, kv_cache_size=64
      ),
  )
  rl_engine = rl_engine_lib.RLEngine(
      actor=tc.ToyTransformer(
          config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
          rngs=nnx.Rngs(0),
      ),
      reference=None,
      tokenizer=vocab,
      cluster_config=cluster_config,
  )
  learner = grpo_lib.GRPOLearner(
      rl_engine=rl_engine,
      reward_fns=shard_reward,
      algo_config=grpo_lib.GRPOConfig(
          num_generations=2, num_iterations=1, beta=0.0
      ),
      metric_fns=[valid_item_metrics],
  )
  batch = {"prompts": np.array(_PROMPTS)}
  learner.train([batch] * _TRAIN_STEPS, [batch])

  exported = {}
  rl_logger = rl_engine._rl_metrics_logger
  for prefix, modes in rl_logger._metrics.items():
    for mode, metrics in modes.items():
      for name in metrics:
        exported[f"{mode}:{prefix}/{name}"] = [
            np.asarray(v).tolist()
            for v in rl_logger.get_metric_history(prefix, name, mode)
        ]
  actor_logger = rl_engine.actor_trainer.metrics_logger
  trainer = {
      name: (
          np.asarray(
              actor_logger.get_metric_history("actor", name, "train")
          ).tolist()
      )
      for name in ("loss", "grad_norm")
  }
  params = [
      np.asarray(multihost_utils.process_allgather(x, tiled=True))
      for x in jax.tree.leaves(
          nnx.state(rl_engine.actor_trainer.model, nnx.Param)
      )
  ]
  out_dir = args["out_dir"]
  if args["process_id"] == 0:
    np.savez(os.path.join(out_dir, "params.npz"), *params)
  with open(
      os.path.join(out_dir, f"process{args['process_id']}.json"),
      "w",
      encoding="utf-8",
  ) as f:
    json.dump(
        {
            "examples": local_examples,
            "exported": exported,
            "trainer": trainer,
            "gather_threads": gather_threads,
            "main_thread": threading.main_thread().name,
        },
        f,
    )


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def _run_job(out_dir: str, disable_reduction: bool = False) -> list[dict]:
  """Runs a 2-process per-host GRPO job on CPU (gloo collectives)."""
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
        "disable_reduction": disable_reduction,
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
  logs = [p.communicate(timeout=600)[0] for p in procs]
  if any(p.returncode for p in procs):
    raise AssertionError(
        "per-host job failed:\n" + "\n".join(log[-3000:] for log in logs)
    )
  results = []
  for process_id in range(_NUM_PROCESSES):
    with open(
        os.path.join(out_dir, f"process{process_id}.json"), encoding="utf-8"
    ) as f:
      results.append(json.load(f))
  return results


def _reference(results, mode: str) -> dict[str, list[float]]:
  """Metrics of the union of every process's examples, per call."""
  calls = [[e for e in r["examples"] if e["mode"] == mode] for r in results]
  assert len({len(c) for c in calls}) == 1, "processes made different calls"
  ref = {}
  for per_process in zip(*calls):
    rewards = np.concatenate([e["rewards"] for e in per_process])
    lengths = np.concatenate([e["lengths"] for e in per_process])
    valid = lengths > _LONG
    values = {
        "rewards/sum": rewards.mean(),
        "rewards/mean": rewards.mean(),
        "rewards/shard_reward": rewards.mean(),
        "rewards/min": rewards.min(),
        "rewards/max": rewards.max(),
        "rewards/score/mean": rewards.mean(),
        "rewards/score/min": rewards.min(),
        "rewards/score/max": rewards.max(),
        "completions/mean_length": lengths.mean(),
        "completions/min_length": lengths.min(),
        "completions/max_length": lengths.max(),
        "custom/valid_success": (rewards[valid] > 0.75).mean(),
        "custom/len_ge3": lengths[lengths >= 3].mean(),
        "custom/valid_len": lengths[valid].sum() / valid.sum(),
        "custom/valid_count": valid.sum(),
        "custom/valid_max_len": lengths[valid].max(),
    }
    for name, value in values.items():
      ref.setdefault(name, []).append(float(value))
  return ref


class PerHostLearnerMetricsTest(absltest.TestCase):

  def test_process0_exports_global_metrics_and_training_is_unchanged(self):
    with tempfile.TemporaryDirectory() as out_dir:
      results = _run_job(out_dir)
      with np.load(os.path.join(out_dir, "params.npz")) as f:
        params = [f[k] for k in sorted(f.files, key=lambda k: int(k[4:]))]
    exported = results[0]["exported"]
    for mode in ("train", "eval"):
      reference = _reference(results, mode)
      # The shards really differ: process 0 alone would report other values.
      local = [
          np.mean(e["rewards"])
          for e in results[0]["examples"]
          if e["mode"] == mode
      ]
      self.assertNotAlmostEqual(local[0], reference["rewards/mean"][0])
      for name, want in reference.items():
        got = exported.get(f"{mode}:{name}")
        self.assertIsNotNone(got, f"{mode}:{name} was not exported")
        np.testing.assert_allclose(
            np.asarray(got, np.float64),
            want,
            rtol=1e-6,
            err_msg=f"{mode}:{name}",
        )
    self.assertLen(exported["train:rewards/mean"], _TRAIN_STEPS)

    # One gather per global step plus one at the end, on every process, all
    # on the main thread (never on a background metrics thread).
    for result in results:
      self.assertEqual(
          result["gather_threads"], [result["main_thread"]] * (_TRAIN_STEPS + 1)
      )

    # The reduction only touches metric buffers: the same job without it
    # trains bit-identically.
    with tempfile.TemporaryDirectory() as out_dir:
      unreduced = _run_job(out_dir, disable_reduction=True)
      with np.load(os.path.join(out_dir, "params.npz")) as f:
        unreduced_params = [
            f[k] for k in sorted(f.files, key=lambda k: int(k[4:]))
        ]
    self.assertEqual(results[0]["trainer"], unreduced[0]["trainer"])
    self.assertLen(params, len(unreduced_params))
    for got, want in zip(params, unreduced_params):
      np.testing.assert_array_equal(got, want)


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
