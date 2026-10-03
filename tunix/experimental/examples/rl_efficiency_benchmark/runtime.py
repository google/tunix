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

"""Opt-in host-side tracing for RL efficiency experiments.

Only benchmark entry points install these hooks. Stage mode waits for device
completion at module boundaries; pipeline mode preserves intra-step overlap.
Both include RPC waits and framework dispatch, not just TPU kernel durations.
"""

import atexit
import functools
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

FROZENLAKE_ENV_TOKEN_RESERVE = 512


def _length(value):
  if value is None:
    return 0
  shape = getattr(value, "shape", None)
  if shape is not None:
    return math.prod(int(dimension) for dimension in shape)
  try:
    return len(value)
  except TypeError:
    return 0


def _sum(value):
  if value is None:
    return 0.0
  if isinstance(value, dict):
    return sum(_sum(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return sum(_sum(item) for item in value)
  array_sum = getattr(value, "sum", None)
  if callable(array_sum):
    return float(array_sum())
  try:
    return float(value)
  except (TypeError, ValueError):
    return 0.0


def _identity(engine, result=None):
  env = getattr(engine, "env", None)
  task = getattr(env, "task", {}) or {}
  extra = getattr(env, "extra_kwargs", {}) or {}
  result = result if isinstance(result, dict) else {}
  step = result.get("policy_version")
  if step is None:
    step = task.get("policy_version", extra.get("benchmark_policy_version", -1))
  return {
      "step": int(step) if step is not None else -1,
      "group_id": result.get("group_id", extra.get("group_id")),
      "pair_index": extra.get("pair_index"),
  }


def _generation_counts(output):
  generated = sum(_length(row) for row in getattr(output, "tokens", []) or [])
  prompt_lengths = getattr(output, "prompt_lengths", None)
  if prompt_lengths is not None:
    prompt = int(_sum(prompt_lengths))
  else:
    # Both benchmark paths issue one request per model call. In that case the
    # left-padded width is the real prompt length, not cross-request padding.
    prompt = _length(getattr(output, "left_padded_prompt_tokens", None))
  return prompt, generated


def _trajectory_fields(engine, result):
  result = result if isinstance(result, dict) else {}
  masks = result.get("conversation_masks")
  conversation_tokens = _length(result.get("conversation_tokens"))
  generated_tokens = int(_sum(masks)) if masks is not None else 0
  prompt_length = result.get("prompt_length")
  if prompt_length is None:
    prompt_length = _length(result.get("prompt_tokens"))
  status = result.get("status", "")
  if hasattr(status, "name"):
    status = status.name
  trajectory = getattr(getattr(engine, "agent", None), "trajectory", None)
  reward = result.get("trajectory_reward")
  if reward is None:
    reward = getattr(trajectory, "reward", None)
  return {
      **_identity(engine, result),
      "exact_token_continuity": bool(engine.exact_token_continuity),
      "prompt_tokens": int(prompt_length),
      "generated_tokens": generated_tokens,
      "environment_tokens": max(0, conversation_tokens - generated_tokens),
      "conversation_tokens": conversation_tokens,
      "turns": len(getattr(trajectory, "steps", []) or []),
      "environment_seconds": _sum(result.get("env_time")),
      "reward_seconds": _sum(result.get("reward_time")),
      "reward": None if reward is None else float(reward),
      "status": str(status),
  }


def record_dataset(rows):
  directory = os.environ.get("TUNIX_BENCHMARK_DIR")
  if directory:
    value = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    with (Path(directory) / "dataset.json").open("x") as file:
      json.dump(
          {
              "rows": len(rows),
              "sha256": hashlib.sha256(value.encode()).hexdigest(),
          },
          file,
      )


def batch_shapes(batches):
  """Inspects shapes only; never copies TPU arrays to the host."""
  return [
      {
          "prompt": list(b.prompt_ids.shape),
          "completion": list(b.completion_ids.shape),
      }
      for b in batches
  ]


def token_shapes(prompt_tokens, completion_tokens):
  """Returns token-array shapes without transferring array contents."""
  return [{
      "prompt": list(prompt_tokens.shape),
      "completion": list(completion_tokens.shape),
  }]


def _wait_for_result(args, kwargs, result):
  import jax

  del args, kwargs
  jax.block_until_ready(result)


def _wait_for_trainer(trainer):
  from flax import nnx
  import jax

  # Waiting on the returned integer step would not wait for device work.
  jax.block_until_ready(
      tuple(
          nnx.state(node)
          for node in (
              trainer.model,
              trainer.optimizer,
              trainer.grad_accumulator,
          )
      )
  )


def _wait_for_actor(args, kwargs, result):
  del kwargs, result
  _wait_for_trainer(args[0].actor_trainer)


def _wait_for_sampler(args, kwargs, result):
  import jax

  del kwargs, result
  jax.block_until_ready(args[0].transformer_state)


def _wait_for_sync(args, kwargs, result):
  import jax

  _wait_for_actor(args, kwargs, result)
  jax.block_until_ready(args[0]._anchor_policy_state)


def _after_call(cls, method, wait):
  """Installs a completion fence inside an enclosing recorded API span."""
  original = getattr(cls, method)

  @functools.wraps(original)
  def wrapped(*args, **kwargs):
    result = original(*args, **kwargs)
    wait(args, kwargs, result)
    return result

  setattr(cls, method, wrapped)


def trainer_main(*args, **kwargs):
  """Benchmark entry point; fences execute in the process owning TPU arrays."""
  from tunix.experimental.examples.common import run_trainer_node

  install("trainer-worker")
  return run_trainer_node.main(*args, **kwargs)


def rollout_main(*args, **kwargs):
  """Record rollout and weight-install completion in the rollout process."""
  from tunix.experimental.examples.common import run_rollout_node

  install("rollout-worker", "vllm-rollout-0")
  return run_rollout_node.main(*args, **kwargs)


def dist_main(*args, **kwargs):
  """Record the distributed run without changing the FrozenLake recipe."""
  from tunix.experimental.examples.frozenlake_dist import frozenlake
  from tunix.experimental.examples.frozenlake_dist import run_frozenlake_dist
  from tpu_sync.rpc import raiden_controller

  if "host_subgrid" not in inspect.signature(
      raiden_controller.RaidenController.register_work_unit
  ).parameters:
    raise RuntimeError(
        "Benchmark requires Raiden with host_subgrid support; loaded "
        f"{raiden_controller.__file__}"
    )

  create_dataset = frozenlake.create_dataset

  @functools.wraps(create_dataset)
  def recorded_dataset(*dataset_args, **dataset_kwargs):
    rows = create_dataset(*dataset_args, **dataset_kwargs)
    record_dataset(rows)
    return rows

  frozenlake.create_dataset = recorded_dataset
  install("dist")
  return run_frozenlake_dist.main(*args, **kwargs)


class Recorder:
  """Records successful sync boundaries and API spans on one host clock."""

  def __init__(self, path, clock=time.monotonic):
    self.clock = clock
    self.file = Path(path).open("x", encoding="utf-8", buffering=1)
    self.lock = threading.Lock()
    self.previous_sync = clock()
    self.pending_trajectories = 0
    self.step = 0

  def emit(self, kind, **fields):
    with self.lock:
      if self.file.closed:
        return
      self.file.write(
          json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n"
      )

  def close(self):
    with self.lock:
      self.file.close()

  def finish(self, name, start, end, trajectories=0, **fields):
    self.emit(
        "span",
        name=name,
        step=self.step,
        start=start,
        end=end,
        seconds=end - start,
        trained_trajectories=trajectories,
        **fields,
    )
    if fields.get("ok", True):
      self.pending_trajectories += trajectories
      if name == "weight_sync":
        if self.pending_trajectories:
          self.emit(
              "training_step",
              step=self.step,
              start=self.previous_sync,
              end=end,
              seconds=end - self.previous_sync,
              trained_trajectories=self.pending_trajectories,
          )
          self.step += 1
          self.pending_trajectories = 0
        else:
          self.emit("initial_sync", end=end)
        self.previous_sync = end

  def wrap(
      self,
      cls,
      method,
      name,
      trajectories_fn=None,
      shapes_fn=None,
      wait_fn=None,
  ):
    original = getattr(cls, method)

    def finish(start, args, kwargs, ok):
      event_name = name(args, kwargs) if callable(name) else name
      trajectories = (
          trajectories_fn(args, kwargs) if ok and trajectories_fn else 0
      )
      end = self.clock()
      shapes = shapes_fn(args, kwargs) if ok and shapes_fn else None
      self.finish(
          event_name,
          start,
          end,
          trajectories=trajectories,
          ok=ok,
          device_complete=ok and wait_fn is not None,
          shapes=shapes,
      )

    if inspect.iscoroutinefunction(original):

      @functools.wraps(original)
      async def wrapped(*args, **kwargs):
        start = self.clock()
        ok = False
        try:
          result = await original(*args, **kwargs)
          if wait_fn is not None:
            wait_fn(args, kwargs, result)
          ok = True
          return result
        finally:
          finish(start, args, kwargs, ok)

    else:

      @functools.wraps(original)
      def wrapped(*args, **kwargs):
        start = self.clock()
        ok = False
        try:
          result = original(*args, **kwargs)
          if wait_fn is not None:
            wait_fn(args, kwargs, result)
          ok = True
          return result
        finally:
          finish(start, args, kwargs, ok)

    setattr(cls, method, wrapped)


def _install_trajectory_hooks(recorder):
  """Records the same episode/model-call boundaries in both implementations."""
  from tunix.rl.agentic.trajectory import trajectory_collect_engine

  TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine

  original_collect = TrajectoryCollectEngine.collect
  if getattr(original_collect, "_tunix_benchmark_hook", False):
    raise RuntimeError("Trajectory benchmark hooks were installed twice.")

  original_timed_call = TrajectoryCollectEngine._run_with_timing

  @functools.wraps(original_timed_call)
  async def timed_call(engine, *args, **kwargs):
    try:
      return await original_timed_call(engine, *args, **kwargs)
    except TimeoutError:
      # collect() catches some environment timeouts, including close(). A
      # returned trajectory then does not prove the environment work finished.
      engine._benchmark_environment_timeout = True
      raise

  TrajectoryCollectEngine._run_with_timing = timed_call

  def instrument_model_call(engine, model_call):
    def finish(start, output, ok):
      end = recorder.clock()
      prompt, generated = _generation_counts(output) if ok else (0, 0)
      recorder.emit(
          "generation",
          start=start,
          end=end,
          seconds=end - start,
          ok=ok,
          prompt_tokens=prompt,
          generated_tokens=generated,
          **_identity(engine),
      )

    if inspect.iscoroutinefunction(model_call):

      @functools.wraps(model_call)
      async def async_call(*args, **kwargs):
        start = recorder.clock()
        output = None
        ok = False
        try:
          output = await model_call(*args, **kwargs)
          ok = True
          return output
        finally:
          finish(start, output, ok)

      return async_call

    @functools.wraps(model_call)
    def sync_call(*args, **kwargs):
      start = recorder.clock()
      output = None
      ok = False
      try:
        output = model_call(*args, **kwargs)
        ok = True
        return output
      finally:
        finish(start, output, ok)

    return sync_call

  @functools.wraps(original_collect)
  async def collect(engine, *args, **kwargs):
    start = recorder.clock()
    engine._benchmark_environment_timeout = False
    result = None
    ok = False
    model_call = engine.model_call
    response_budget = engine.max_response_length
    # The final FrozenLake observation can add tokens after generation. Leave
    # room for it within the learner's unchanged completion-padding budget.
    if response_budget is not None:
      engine.max_response_length = (
          response_budget - FROZENLAKE_ENV_TOKEN_RESERVE
      )
    engine.model_call = instrument_model_call(engine, model_call)
    try:
      result = await original_collect(engine, *args, **kwargs)
      ok = not engine._benchmark_environment_timeout
      return result
    finally:
      engine.max_response_length = response_budget
      engine.model_call = model_call
      end = recorder.clock()
      fields = _trajectory_fields(engine, result) if ok else _identity(engine)
      recorder.emit(
          "rollout_trajectory",
          start=start,
          end=end,
          seconds=end - start,
          ok=ok,
          **fields,
      )

  collect._tunix_benchmark_hook = True
  TrajectoryCollectEngine.collect = collect


def _install_dist_rollout_identity_hook():
  """Makes request identity visible to the shared inner collector hook."""
  from tunix.experimental.rollout.collector import TrajectoryCollectorEngine

  original = TrajectoryCollectorEngine.run_episode

  @functools.wraps(original)
  async def run_episode(collector, *args, **kwargs):
    extra = getattr(collector.env, "extra_kwargs", None)
    if isinstance(extra, dict):
      extra["benchmark_policy_version"] = (
          collector.request.target_policy_version
      )
      extra["group_id"] = collector.request.prompt_id
      extra["pair_index"] = collector.request.group_index
    return await original(collector, *args, **kwargs)

  TrajectoryCollectorEngine.run_episode = run_episode


def install(mode, label=None):
  """Installs benchmark-only hooks after the entry point's normal imports."""
  directory = os.environ.get("TUNIX_BENCHMARK_DIR")
  if not directory:
    return
  timing_mode = os.environ.get("TUNIX_BENCHMARK_TIMING", "stages")
  if timing_mode not in ("stages", "pipeline"):
    raise ValueError(f"Unknown benchmark timing mode: {timing_mode}")
  stages = timing_mode == "stages"
  filename = "events.jsonl"
  if mode == "trainer-worker":
    filename = "events-trainer.jsonl"
  elif mode == "rollout-worker":
    safe_label = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(label or "0")
    )
    filename = f"events-rollout-{safe_label}.jsonl"
  recorder = Recorder(Path(directory) / filename)
  atexit.register(recorder.close)
  recorder.emit("timing_contract", process=mode, timing_mode=timing_mode)
  if mode == "trainer-worker":
    if stages:
      from tunix.experimental.worker.trainer_worker import TrainerWorker

      def completed(method):
        def wait(args, kwargs, result):
          if method == "per_token_logps":
            _wait_for_result(args, kwargs, result.per_token_logps)
          else:
            _wait_for_trainer(args[0]._trainer)
          recorder.emit(
              "device_completion", method=method, time=recorder.clock()
          )

        return wait

      # A microbatch may be followed by log-probs/preprocessing for the next
      # one. Fence each training RPC so its work cannot leak into that stage.
      # Record the worker-side span as well as the orchestrator RPC span, so
      # their difference can expose transport/dispatch overhead.
      for method in ("fwd_bwd", "update", "per_token_logps"):
        recorder.wrap(
            TrainerWorker,
            method,
            "trainer_worker_" + method,
            wait_fn=completed(method),
        )
    return
  if mode == "rollout-worker":
    from tunix.experimental.weight_sync.raiden_synchronizer import RaidenSynchronizer

    def installed(args, kwargs, result):
      synchronizer = args[0]
      if not synchronizer._is_proxy and synchronizer._sync is None:
        raise RuntimeError("Weight installation has no active transport.")
      # h2d() awaits the native transfer future (or FFI output readiness),
      # not just the readiness of the old destination arrays.
      recorder.emit(
          "device_completion", method="weight_install", time=recorder.clock()
      )

    _after_call(RaidenSynchronizer, "h2d", installed)
    _install_dist_rollout_identity_hook()
    _install_trajectory_hooks(recorder)
    return
  if mode == "dist":
    from tunix.experimental.orchestrator import distributed_rl_engine

    DistributedRLEngine = distributed_rl_engine.DistributedRLEngine

    def trajectories(args, kwargs):
      payload = args[1] if len(args) > 1 else kwargs["payload"]
      return int(payload.completion_ids.shape[0])

    recorder.wrap(
        DistributedRLEngine,
        "train_step",
        "training",
        trajectories,
        lambda args, kwargs: batch_shapes(
            [args[1] if len(args) > 1 else kwargs["payload"]]
        ),
    )
    for method, name in (
        ("dispatch_rollouts", "rollout_dispatch"),
        ("poll_rollouts", "rollout_poll"),
        ("sync_weights", "weight_sync"),
        ("save_checkpoint", "checkpoint_save"),
        ("get_metrics", "metrics_fetch"),
    ):
      recorder.wrap(DistributedRLEngine, method, name)
    recorder.wrap(
        DistributedRLEngine,
        "_invoke_worker",
        lambda args, kwargs: "worker_rpc_"
        + str(args[2] if len(args) > 2 else kwargs["method_name"]),
    )
    recorder.wrap(
        DistributedRLEngine,
        "per_token_logps",
        "actor_log_probs",
        shapes_fn=lambda args, kwargs: token_shapes(
            (args[2] if len(args) > 2 else kwargs["items"]).prompt_tokens,
            (args[2] if len(args) > 2 else kwargs["items"]).completion_tokens,
        ),
    )
  else:
    from tunix.generate.vllm_sampler import VllmSampler
    from tunix.rl.rl_cluster import RLEngine

    # Even pipeline mode needs a completed rollout-weight boundary. Dist's
    # Raiden coordinator already awaits installation on the destination.
    _after_call(VllmSampler, "update_params", _wait_for_sampler)

    def trajectories(args, kwargs):
      batches = args[1] if len(args) > 1 else kwargs["train_ds"]
      return sum(int(batch.completion_ids.shape[0]) for batch in batches)

    recorder.wrap(
        RLEngine,
        "update_actor",
        "training",
        trajectories,
        lambda args, kwargs: batch_shapes(
            args[1] if len(args) > 1 else kwargs["train_ds"]
        ),
        wait_fn=_wait_for_actor if stages else None,
    )
    recorder.wrap(
        RLEngine,
        "get_actor_per_token_logps",
        "actor_log_probs",
        shapes_fn=lambda args, kwargs: token_shapes(
            args[1] if len(args) > 1 else kwargs["prompt_tokens"],
            args[2] if len(args) > 2 else kwargs["completion_tokens"],
        ),
        wait_fn=_wait_for_result if stages else None,
    )
    recorder.wrap(
        RLEngine, "sync_weights", "weight_sync", wait_fn=_wait_for_sync
    )
    _install_trajectory_hooks(recorder)

  from tunix.rl import common as rl_common

  recorder.wrap(
      rl_common,
      "sampler_trainer_agreement",
      "sampler_trainer_agreement",
      wait_fn=_wait_for_result if stages else None,
  )
