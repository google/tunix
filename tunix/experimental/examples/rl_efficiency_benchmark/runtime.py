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
import math
import os
from pathlib import Path
import threading
import time


def _arg(args, kwargs, index, name):
  """Returns a wrapped method's argument, passed by position or keyword."""
  return args[index] if len(args) > index else kwargs[name]


def _shapes(prompt, completion):
  """Returns token-array shapes without transferring array contents."""
  return {"prompt": list(prompt.shape), "completion": list(completion.shape)}


def record_dataset(rows):
  directory = os.environ.get("TUNIX_BENCHMARK_DIR")
  if directory:
    value = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(value.encode()).hexdigest()
    with (Path(directory) / "dataset.json").open("x") as file:
      json.dump({"rows": len(rows), "sha256": digest}, file)


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


def _wait_for_result(args, kwargs, result):
  import jax

  del args, kwargs
  jax.block_until_ready(result)


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
  """Makes CLS.METHOD return only after WAIT(args, kwargs, result)."""
  original = getattr(cls, method)

  @functools.wraps(original)
  def wrapped(*args, **kwargs):
    result = original(*args, **kwargs)
    wait(args, kwargs, result)
    return result

  setattr(cls, method, wrapped)


def _timed(fn, clock, wait_fn, on_finish):
  """Wraps sync or async FN; calls on_finish(start, args, kwargs, result, ok).

  Args:
    fn: The function to time.
    clock: Returns the current time.
    wait_fn: Optional completion fence, called with (args, kwargs, result)
      before the call counts as finished.
    on_finish: Receives the call's start time, arguments, result and success.

  Returns:
    The wrapped function.
  """
  if inspect.iscoroutinefunction(fn):

    @functools.wraps(fn)
    async def async_wrapped(*args, **kwargs):
      start, result, ok = clock(), None, False
      try:
        result = await fn(*args, **kwargs)
        if wait_fn is not None:
          wait_fn(args, kwargs, result)
        ok = True
        return result
      finally:
        on_finish(start, args, kwargs, result, ok)

    return async_wrapped

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    start, result, ok = clock(), None, False
    try:
      result = fn(*args, **kwargs)
      if wait_fn is not None:
        wait_fn(args, kwargs, result)
      ok = True
      return result
    finally:
      on_finish(start, args, kwargs, result, ok)

  return wrapped


def trainer_main(*args, **kwargs):
  """Benchmark entry point; fences execute in the process owning TPU arrays."""
  from tunix.experimental.examples.common import run_trainer_node

  install("trainer-worker")
  return run_trainer_node.main(*args, **kwargs)


def rollout_main(*args, **kwargs):
  """Records rollout trajectories in the rollout process."""
  from tunix.experimental.examples.common import run_rollout_node

  install("rollout-worker")
  return run_rollout_node.main(*args, **kwargs)


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
      if not self.file.closed:
        self.file.write(
            json.dumps({"kind": kind, **fields}, allow_nan=False) + "\n"
        )

  def close(self):
    with self.lock:
      self.file.close()

  def finish(self, name, start, end, trajectories=0, ok=True, **fields):
    """Records a span; a successful weight sync after training ends a step."""
    self.emit(
        "span",
        name=name,
        step=self.step,
        start=start,
        end=end,
        seconds=end - start,
        trained_trajectories=trajectories,
        ok=ok,
        **fields,
    )
    if not ok:
      return
    self.pending_trajectories += trajectories
    if name != "weight_sync":
      return
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
    """Records every CLS.METHOD call as span NAME."""

    def on_finish(start, args, kwargs, result, ok):
      del result
      fields = {"shapes": shapes_fn(args, kwargs)} if ok and shapes_fn else {}
      self.finish(
          name,
          start,
          self.clock(),
          trajectories=trajectories_fn(args, kwargs)
          if ok and trajectories_fn
          else 0,
          ok=ok,
          **fields,
      )

    setattr(
        cls,
        method,
        _timed(getattr(cls, method), self.clock, wait_fn, on_finish),
    )


def _step(engine):
  """Returns the policy version (training step) of an episode."""
  extra = engine.env.extra_kwargs
  # Dist records the request's version; agentic records it on the task.
  if "benchmark_policy_version" in extra:
    return extra["benchmark_policy_version"]
  return engine.env.task.get("policy_version", -1)


def _generation_counts(output):
  """Returns the prompt and generated token counts of a RolloutOutput."""
  generated = sum(len(tokens) for tokens in output.tokens)
  if output.prompt_lengths is not None:
    return int(sum(output.prompt_lengths)), generated
  # Both benchmark paths issue one request per model call. In that case the
  # left-padded width is the real prompt length, not cross-request padding.
  return math.prod(output.left_padded_prompt_tokens.shape), generated


def _trajectory_fields(engine):
  trajectory = engine.agent.trajectory
  env_time = engine.env_time
  return {
      "exact_token_continuity": engine.exact_token_continuity,
      "turns": len(trajectory.steps),
      "environment_seconds": (
          env_time["reset_latency"]
          + sum(env_time["step_latency"])
          + env_time["close_latency"]
      ),
      "reward": float(trajectory.reward),
      "status": trajectory.status.name,
  }


def _install_trajectory_hooks(recorder, response_reserve):
  """Records the same episode/model-call boundaries in both implementations.

  Args:
    recorder: Receives the generation and trajectory events.
    response_reserve: Response tokens withheld from generation for environment
      output after the last model call.
  """
  from tunix.rl.agentic.trajectory import trajectory_collect_engine

  engine_cls = trajectory_collect_engine.TrajectoryCollectEngine
  original_timed_call = engine_cls._run_with_timing
  original_collect = engine_cls.collect

  @functools.wraps(original_timed_call)
  async def timed_call(engine, *args, **kwargs):
    try:
      return await original_timed_call(engine, *args, **kwargs)
    except TimeoutError:
      # collect() catches some environment timeouts, including close(). A
      # returned trajectory then does not prove the environment work finished.
      engine._benchmark_environment_timeout = True
      raise

  def generation_recorder(engine):
    def on_finish(start, args, kwargs, output, ok):
      del args, kwargs
      end = recorder.clock()
      prompt, generated = _generation_counts(output) if ok else (0, 0)
      recorder.emit(
          "generation",
          step=_step(engine),
          start=start,
          end=end,
          seconds=end - start,
          ok=ok,
          prompt_tokens=prompt,
          generated_tokens=generated,
      )

    return on_finish

  @functools.wraps(original_collect)
  async def collect(engine, *args, **kwargs):
    start = recorder.clock()
    engine._benchmark_environment_timeout = False
    model_call, budget = engine.model_call, engine.max_response_length
    # Environment output after the last generation can add tokens. Leave room
    # for it within the learner's unchanged completion-padding budget.
    if budget is not None:
      engine.max_response_length = budget - response_reserve
    engine.model_call = _timed(
        model_call, recorder.clock, None, generation_recorder(engine)
    )
    ok = False
    try:
      result = await original_collect(engine, *args, **kwargs)
      ok = not engine._benchmark_environment_timeout
      return result
    finally:
      engine.max_response_length, engine.model_call = budget, model_call
      end = recorder.clock()
      recorder.emit(
          "rollout_trajectory",
          step=_step(engine),
          start=start,
          end=end,
          seconds=end - start,
          ok=ok,
          **(_trajectory_fields(engine) if ok else {}),
      )

  engine_cls._run_with_timing = timed_call
  engine_cls.collect = collect


def _install_dist_policy_version_hook():
  """Exposes the request's policy version to the shared trajectory hooks."""
  from tunix.experimental.rollout.collector import TrajectoryCollectorEngine

  original = TrajectoryCollectorEngine.run_episode

  @functools.wraps(original)
  async def run_episode(collector, *args, **kwargs):
    collector.env.extra_kwargs["benchmark_policy_version"] = (
        collector.request.target_policy_version
    )
    return await original(collector, *args, **kwargs)

  TrajectoryCollectorEngine.run_episode = run_episode


def install(mode):
  """Installs benchmark-only hooks after the entry point's normal imports."""
  directory = os.environ.get("TUNIX_BENCHMARK_DIR")
  if not directory:
    return
  timing_mode = os.environ["TUNIX_BENCHMARK_TIMING"]
  stages = timing_mode == "stages"
  response_reserve = int(os.environ["TUNIX_BENCHMARK_RESPONSE_RESERVE"])
  recorder = Recorder(Path(directory) / f"events-{mode}-{os.getpid()}.jsonl")
  atexit.register(recorder.close)
  recorder.emit("timing_contract", process=mode, timing_mode=timing_mode)
  if mode == "trainer-worker":
    if stages:
      from tunix.experimental.worker.trainer_worker import TrainerWorker

      # A microbatch may be followed by log-probs for the next one. Fence each
      # training RPC so its device work cannot leak into that stage.
      def trainer_ready(args, kwargs, result):
        del kwargs, result
        _wait_for_trainer(args[0]._trainer)

      _after_call(TrainerWorker, "fwd_bwd", trainer_ready)
      _after_call(TrainerWorker, "update", trainer_ready)
      _after_call(
          TrainerWorker,
          "per_token_logps",
          lambda args, kwargs, result: _wait_for_result(
              args, kwargs, result.per_token_logps
          ),
      )
    return
  if mode == "rollout-worker":
    _install_dist_policy_version_hook()
    _install_trajectory_hooks(recorder, response_reserve)
    return
  if mode == "dist":
    from tunix.experimental.orchestrator.distributed_rl_engine import DistributedRLEngine

    def payload(args, kwargs):
      return _arg(args, kwargs, 1, "payload")

    def logps_items(args, kwargs):
      return _arg(args, kwargs, 2, "items")

    recorder.wrap(
        DistributedRLEngine,
        "train_step",
        "training",
        trajectories_fn=lambda a, k: int(payload(a, k).completion_ids.shape[0]),
        shapes_fn=lambda a, k: [
            _shapes(payload(a, k).prompt_ids, payload(a, k).completion_ids)
        ],
    )
    recorder.wrap(
        DistributedRLEngine,
        "per_token_logps",
        "actor_log_probs",
        shapes_fn=lambda a, k: [
            _shapes(
                logps_items(a, k).prompt_tokens,
                logps_items(a, k).completion_tokens,
            )
        ],
    )
    for method, name in (
        ("dispatch_rollouts", "rollout_dispatch"),
        ("poll_rollouts", "rollout_poll"),
        ("sync_weights", "weight_sync"),
        ("save_checkpoint", "checkpoint_save"),
        ("get_metrics", "metrics_fetch"),
    ):
      recorder.wrap(DistributedRLEngine, method, name)
  else:
    from tunix.generate.vllm_sampler import VllmSampler
    from tunix.rl.rl_cluster import RLEngine

    def train_ds(args, kwargs):
      return _arg(args, kwargs, 1, "train_ds")

    # Even pipeline mode needs a completed rollout-weight boundary. Dist's
    # Raiden coordinator already awaits installation on the destination.
    _after_call(VllmSampler, "update_params", _wait_for_sampler)
    recorder.wrap(
        RLEngine,
        "update_actor",
        "training",
        trajectories_fn=lambda a, k: sum(
            int(b.completion_ids.shape[0]) for b in train_ds(a, k)
        ),
        shapes_fn=lambda a, k: [
            _shapes(b.prompt_ids, b.completion_ids) for b in train_ds(a, k)
        ],
        wait_fn=_wait_for_actor if stages else None,
    )
    recorder.wrap(
        RLEngine,
        "get_actor_per_token_logps",
        "actor_log_probs",
        shapes_fn=lambda a, k: [
            _shapes(
                _arg(a, k, 1, "prompt_tokens"),
                _arg(a, k, 2, "completion_tokens"),
            )
        ],
        wait_fn=_wait_for_result if stages else None,
    )
    recorder.wrap(
        RLEngine, "sync_weights", "weight_sync", wait_fn=_wait_for_sync
    )
    _install_trajectory_hooks(recorder, response_reserve)

  from tunix.rl import common as rl_common

  recorder.wrap(
      rl_common,
      "sampler_trainer_agreement",
      "sampler_trainer_agreement",
      wait_fn=_wait_for_result if stages else None,
  )
