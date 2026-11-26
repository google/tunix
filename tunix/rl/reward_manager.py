# Copyright 2025 Google LLC
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

"""Reward output for RL."""

import abc
from collections.abc import Sequence
from dataclasses import asdict
import inspect
import os
from typing import Any, Callable, ClassVar, Dict, List
import weakref

from absl import logging
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import function_registry

import multiprocessing

RewardFn = Callable[..., Any]


def _terminate_pool(pool) -> None:
  try:
    pool.terminate()
  except Exception:  # pylint: disable=broad-except
    pass


def _fork_context():
  """The multiprocessing context for reward workers, or None if unavailable.

  Workers are started through a fork server rather than forked directly
  from the trainer: a direct fork inherits the parent's initialized JAX
  runtime, and the first JAX call in such a child deadlocks (until the
  worker timeout expires and the fn falls back to the parent). Fork-server
  children are fresh interpreters and do not have that problem.

  Every process operation of the pool goes through this one context object
  (its `Pool`, `Process` and `current_process`), so swapping the context is
  enough to change how workers are started.
  """
  try:
    return multiprocessing.get_context("forkserver")
  except ValueError:
    return None


def _sum_trajectory_rewards(
    rewards: np.ndarray,
    env_rewards: np.ndarray | None = None,
) -> np.ndarray:
  """Sums per-trajectory reward-function columns, propagating NaN env rewards."""
  fn_sum = np.nansum(rewards, axis=1)
  return fn_sum if env_rewards is None else env_rewards + fn_sum


# Summary metrics logged as `rewards/<name>`; a reward function with one of
# these names would overwrite them with its own `rewards/<name>` series.
_RESERVED_REWARD_FN_NAMES: frozenset[str] = frozenset(
    {"sum", "mean", "std", "min", "max", "nonfinite_frac"}
)


def _calculate_scalar_reward_log_metrics(
    total_rewards: np.ndarray,
) -> Dict[str, Any]:
  """Builds the step-level `rewards/*` summary of per-trajectory total rewards.

  A trajectory whose total reward is not finite (e.g. a `NaN` environment
  reward) is excluded from training (`advantages`, `payload_builder`), so it is
  excluded here too: its entry is set to `NaN` and every aggregation function
  is `NaN`-aware, which lets the step-level reduction skip it instead of turning
  the whole step's metric into `NaN`. `rewards/nonfinite_frac` reports how many
  trajectories were excluded.

  Args:
    total_rewards: 1D per-trajectory total rewards.

  Returns:
    Mapping from metric name to `(values, agg_fn)`.
  """
  finite = np.isfinite(total_rewards)
  masked = np.where(finite, total_rewards, np.nan)
  return {
      "rewards/sum": (masked, np.nansum),
      "rewards/mean": (masked, np.nanmean),
      "rewards/std": (masked, np.nanstd),
      "rewards/min": (masked, np.nanmin),
      "rewards/max": (masked, np.nanmax),
      "rewards/nonfinite_frac": ((~finite).astype(np.float64), np.mean),
  }


class AbstractRewardManager(abc.ABC):
  """Abstract base class for managing and orchestrating multiple reward function outputs."""

  def __init__(
      self,
      reward_fns: RewardFn | Sequence[RewardFn] | None,
      algo_config: algo_config_lib.AlgorithmConfig,
  ):
    """Initializes the manager with a list of callable reward function objects.

    Args:
        reward_fns: A list of reward functions or models, or None when rewards
          are supplied externally (e.g. environment trajectory rewards).
        algo_config: The algorithm config to use for reward function
          configuration.

    Raises:
        TypeError: If any item in `reward_fns` is not callable.
    """
    if reward_fns is None:
      self.reward_fns = []
    elif callable(reward_fns):
      self.reward_fns = [reward_fns]
    else:
      self.reward_fns = list(reward_fns)
    if not all(callable(fn) for fn in self.reward_fns):
      raise TypeError(
          f"All items in reward_fns must be callable; got {self.reward_fns!r}."
      )
    self.algo_config = algo_config

  @abc.abstractmethod
  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    """Computes the rewards for completions using the provided reward functions.

    Args:
        prompts: A list of input prompts.
        completions: A list of generated text completions.
        **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
        A dictionary of rewards information, including the final rewards for
        advantage computation and intermediate rewards for logging.
    """
    pass

  def close(self) -> None:
    """Releases any resources held by the reward manager."""


@function_registry.register_reward_manager("reward_fn")
class SequenceRewardManager(AbstractRewardManager):
  """Reward manager for sequence-level reward functions (`reward_mode='reward_fn'`)."""

  _require_reward_fns: ClassVar[bool] = True

  def __init__(
      self,
      reward_fns: RewardFn | Sequence[RewardFn] | None,
      algo_config: algo_config_lib.AlgorithmConfig,
      **kwargs,
  ):
    """Initializes the manager with a list of callable reward function objects.

    Args:
      reward_fns: Callable(s) scored per prompt-completion pair.
      algo_config: Algorithm configuration.
      **kwargs: Unused.

    Raises:
      ValueError: If `reward_fns` is empty while required, or a reward function
        is named after a summary metric (e.g. `sum`, `mean`, `min`, `max`),
        whose per-function `rewards/<name>` series would overwrite it.
    """
    super().__init__(reward_fns, algo_config)
    if self._require_reward_fns and not self.reward_fns:
      raise ValueError(
          f"reward_fns cannot be empty for {type(self).__name__}."
          " Provide at least one reward function, or set"
          " algo_config.reward_mode='env' to use environment trajectory"
          " rewards."
      )
    reserved = sorted(
        {fn.__name__ for fn in self.reward_fns} & _RESERVED_REWARD_FN_NAMES
    )
    if reserved:
      raise ValueError(
          f"Reward function names {reserved} are reserved for the"
          " `rewards/*` summary metrics; rename them."
      )
    self._pool = None
    # Reward fns that must be evaluated in the parent process, learned at
    # runtime: fns that are unpicklable, or that spawn subprocesses of their
    # own (which the pool's daemonic workers forbid).
    self._parent_only_fns = set()
    self._parallel_disabled = False
    self._worker_timeout = algo_config.reward_worker_timeout_seconds

  def _parallel_enabled(self) -> bool:
    return (
        self._num_workers() > 1
        and not self._parallel_disabled
        and _fork_context() is not None
    )

  def _num_workers(self) -> int:
    """Resolves `reward_num_workers`; -1 means one worker per CPU."""
    n = self.algo_config.reward_num_workers
    cpus = os.cpu_count() or 1
    if n == -1:
      return cpus
    if n > cpus:
      logging.warning(
          "reward_num_workers=%d exceeds the %d CPUs available; the extra"
          " workers will only contend for cores.",
          n,
          cpus,
      )
    return n

  def _get_pool(self):
    """Lazily creates the reward worker pool."""
    if self._pool is None:
      ctx = _fork_context()
      # Some launchers run the trainer as a daemonic child process, and
      # multiprocessing forbids daemons from having children. The flag is
      # launcher plumbing this code does not rely on: clear it so the pool
      # can fork. The workers are reaped with the trainer on teardown.
      cur_config = getattr(ctx.current_process(), "_config", None)
      if isinstance(cur_config, dict) and cur_config.get("daemon"):
        cur_config["daemon"] = False
      self._pool = ctx.Pool(processes=self._num_workers())
      # Learners hold the manager for their whole lifetime and never call
      # close(); make sure a dropped manager does not leak its workers.
      weakref.finalize(self, _terminate_pool, self._pool)
    return self._pool

  def close(self) -> None:
    """Shuts down the reward worker pool, if one was created."""
    if self._pool is not None:
      self._pool.terminate()
      self._pool = None

  def _call_reward_fn(
      self,
      reward_fn: RewardFn,
      prompts: List[str],
      completions: List[str],
      call_kwargs: Dict[str, Any],
  ) -> Any:
    """Evaluates one reward fn over the batch, using workers when enabled."""
    fn_name = getattr(reward_fn, "__name__", str(reward_fn))
    if not self._parallel_enabled() or fn_name in self._parent_only_fns:
      return reward_fn(prompts=prompts, completions=completions, **call_kwargs)

    try:
      pool = self._get_pool()
    except Exception as e:  # pylint: disable=broad-except
      self._parallel_disabled = True
      logging.warning(
          "Could not create the reward worker pool (%r); using the serial"
          " implementation.",
          e,
      )
      return reward_fn(prompts=prompts, completions=completions, **call_kwargs)

    num_prompts = len(prompts)
    try:
      # Two chunks per worker, not one: reward cost varies a lot across
      # completions (e.g. long answers under a verifier), and the smaller
      # chunks let a worker that finished early pick up more work instead of
      # idling behind the slowest chunk. The pool size itself is exactly
      # `reward_num_workers`.
      n_chunks = min(
          pool._processes * 2, num_prompts  # pylint: disable=protected-access
      )
      bounds = [num_prompts * c // n_chunks for c in range(n_chunks + 1)]
      jobs = []
      for c in range(n_chunks):
        lo, hi = bounds[c], bounds[c + 1]
        chunk_kwargs = {}
        for k, v in call_kwargs.items():
          # A list/tuple/array kwarg with exactly one entry per prompt is
          # treated as a per-example column (e.g. ground-truth answers) and
          # sliced in lockstep with the batch, because that is the contract
          # `__call__` already relies on for such kwargs. Anything else --
          # scalars, configs, sequences of another length -- is passed to
          # every chunk unchanged.
          if isinstance(v, (list, tuple)) and len(v) == num_prompts:
            chunk_kwargs[k] = list(v[lo:hi])
          elif isinstance(v, np.ndarray) and len(v) == num_prompts:
            chunk_kwargs[k] = v[lo:hi]
          else:
            chunk_kwargs[k] = v
        chunk_kwargs["prompts"] = list(prompts[lo:hi])
        chunk_kwargs["completions"] = list(completions[lo:hi])
        jobs.append(pool.apply_async(reward_fn, kwds=chunk_kwargs))
      parts = [j.get(timeout=self._worker_timeout) for j in jobs]
      if any(p is None for p in parts):
        raise RuntimeError(f"{fn_name} returned None in a worker.")
      return [x for part in parts for x in list(part)]
    except Exception as e:  # pylint: disable=broad-except
      # Unpicklable fns and fns that spawn their own subprocesses land here,
      # as does any worker crash. Evaluate this fn in the parent from now on
      # -- its own internal parallelism, if any, works there.
      self._parent_only_fns.add(fn_name)
      logging.warning(
          "Reward fn %s failed in worker processes (%r); evaluating it in the"
          " parent process from now on.",
          fn_name,
          e,
      )
      return reward_fn(prompts=prompts, completions=completions, **call_kwargs)

  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    """Computes sequence-level rewards and log metrics for completions."""
    return self._compute_rewards(prompts, completions, None, **kwargs)

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      env_rewards: np.ndarray | None = None,
      /,
      **reward_kwargs,
  ) -> Dict[str, Any]:
    """Computes rewards and log metrics for completions.

    Args:
        prompts: Input prompts (or 1D array of prompts) for the batch.
        completions: Generated text completions aligned with `prompts`.
        env_rewards: Optional 1D float64 array of per-trajectory environment
          rewards (`reward_mode='env'`), positional-only so dataset columns
          named `env_rewards` in `reward_kwargs` do not collide.
        **reward_kwargs: Extra per-example dataset or metadata columns forwarded
          to `reward_fns`.

    Returns:
        A dictionary with `"rewards"` (1D per-trajectory total rewards) and
        `"log_metrics"` (metric tuples `(values, agg_fn)`).

    Raises:
        ValueError: If `prompts` is empty.
        RuntimeError: If a reward function returns `None` or a list whose length
          does not match `len(prompts)`.
    """

    num_prompts = len(prompts)
    if num_prompts == 0:
      raise ValueError(
          f"{type(self).__name__} received an empty batch; nothing to score."
      )

    algo_config_params = asdict(self.algo_config)
    base_kwargs = reward_kwargs.copy()

    num_reward_fns = len(self.reward_fns)
    rewards = np.zeros((num_prompts, num_reward_fns))

    # Compute all rewards for each prompt-completion pair.
    for i, reward_fn in enumerate(self.reward_fns):
      signature = inspect.signature(reward_fn)
      accepts_var_keyword = any(
          param.kind == inspect.Parameter.VAR_KEYWORD
          for param in signature.parameters.values()
      )
      call_kwargs = (
          base_kwargs.copy()
          if accepts_var_keyword
          else {
              k: v
              for k, v in base_kwargs.items()
              if k in signature.parameters
              and k not in ("self", "prompts", "completions")
          }
      )
      # Fill declared parameters that the caller did not pass from algo_config.
      for name, param in signature.parameters.items():
        if name in ("self", "prompts", "completions") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
          continue
        if name in algo_config_params and name not in call_kwargs:
          call_kwargs[name] = algo_config_params[name]

      r = self._call_reward_fn(reward_fn, prompts, completions, call_kwargs)

      if r is None:
        raise RuntimeError(
            f"Failed to obtain result from {reward_fn.__name__}. Result is"
            " None."
        )
      if isinstance(r, list) and len(r) != len(prompts):
        raise RuntimeError(
            f"Length mismatch after {reward_fn.__name__}: "
            f"len(r)={len(r)}, len(prompts)={num_prompts}. "
            f"Content of r: {r}"
        )

      rewards[:, i] = np.array(r)

    sum_rewards = _sum_trajectory_rewards(rewards, env_rewards)
    # Prepare metrics for logging.
    log_metrics = self._prepare_log_metrics(
        prompts,
        completions,
        rewards,
        sum_rewards=sum_rewards,
        env_rewards=env_rewards,
    )
    rewards_info = {
        "rewards": sum_rewards,
        "log_metrics": log_metrics,
    }

    def _log_one_example(log_metrics: Dict[str, Any]):
      logging.info("======= example rewards =======")

      # add a snippet of the prompt, completion, and reward
      def snippet(s: str, k: int = 50):
        if len(s) <= 2 * k:
          return s
        return s[:k] + "..." + s[-k:]

      for k, v in log_metrics.items():
        logging.info("%s:\t%s", k, snippet(str(v[0][0])))
      logging.info("=======================")

    if os.getenv("TUNIX_DEBUG_REWARDS"):
      _log_one_example(log_metrics)

    return rewards_info

  def _prepare_log_metrics(
      self,
      prompts: List[str],
      completions: List[str],
      rewards: np.ndarray,  # (num_prompts, num_reward_fns)
      *,
      sum_rewards: np.ndarray,
      env_rewards: np.ndarray | None = None,
  ) -> Dict[str, Any]:
    """Builds per-trajectory reward and text log metrics.

    `rewards/{sum,mean,std,min,max}` summarize the per-trajectory total reward
    over the step. Per-source series (`rewards/env`, `rewards/<reward_fn>`)
    skip trajectories with a non-finite total and `NaN` scores, matching the
    trajectories that train.

    Args:
        prompts: Input prompts for the batch.
        completions: Generated text completions for the batch.
        rewards: 2D `(num_prompts, num_reward_fns)` matrix of reward-function
          scores.
        sum_rewards: 1D total reward per trajectory.
        env_rewards: Optional 1D array of environment trajectory rewards; logged
          under `"rewards/env"` only when `self.reward_fns` is non-empty.

    Returns:
        Mapping from metric name to `(values, agg_fn)`.
    """
    metrics_to_log = {}

    # Log prompts and completions.
    metrics_to_log["prompts"] = (prompts, None)
    metrics_to_log["completions"] = (completions, None)

    metrics_to_log.update(_calculate_scalar_reward_log_metrics(sum_rewards))

    finite_total = np.isfinite(sum_rewards)
    if env_rewards is not None and self.reward_fns:
      metrics_to_log["rewards/env"] = (
          np.where(finite_total, env_rewards, np.nan),
          np.nanmean,
      )

    # Log individual rewards for this trajectory
    for i, reward_fn in enumerate(self.reward_fns):
      metric_name = f"rewards/{reward_fn.__name__}"
      metrics_to_log[metric_name] = (
          np.where(finite_total, rewards[:, i], np.nan),
          np.nanmean,
      )

    return metrics_to_log


@function_registry.register_reward_manager("env")
class AgenticSequenceRewardManager(SequenceRewardManager):
  """Reward manager for agentic environment trajectories (`reward_mode='env'`).

  Starts from `trajectory_rewards` produced by the rollout environment and, when
  `reward_fns` are provided, evaluates and sums them on top of
  `trajectory_rewards`. Logs unified `rewards/{sum,mean,min,max}` across active
  reward sources and emits `rewards/env` only when `reward_fns` is non-empty.
  """

  _require_reward_fns: ClassVar[bool] = False

  def __init__(
      self,
      reward_fns: RewardFn | Sequence[RewardFn] | None,
      algo_config: algo_config_lib.AlgorithmConfig,
      **kwargs,
  ):
    """Initializes the agentic reward manager.

    Args:
        reward_fns: Optional callable(s) summed on top of `trajectory_rewards`.
        algo_config: Algorithm configuration.
        **kwargs: Additional keyword arguments forwarded to the parent class.

    Raises:
        TypeError: If any item in `reward_fns` is not callable.
        ValueError: If any reward function is named `'env'` (reserved for
          `rewards/env`).
    """
    super().__init__(reward_fns, algo_config, **kwargs)
    if any(fn.__name__ == "env" for fn in self.reward_fns):
      raise ValueError(
          "Reward function name 'env' is reserved for environment trajectory"
          " rewards (`rewards/env`) in `reward_mode='env'`."
      )

  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    """Computes combined environment and function rewards and log metrics.

    Args:
        prompts: Input prompts (or 1D array of prompts) for the batch.
        completions: Generated text completions aligned with `prompts`.
        **kwargs: Must include `trajectory_rewards` (1D sequence/array of length
          `len(prompts)`); remaining entries are forwarded to `reward_fns`.

    Returns:
        A dictionary with `"rewards"` (propagating `NaN` environment rewards)
        and `"log_metrics"` under `rewards/*` (`"rewards/env"` included only
        when `reward_fns` is non-empty).

    Raises:
        KeyError: If `trajectory_rewards` is missing from `kwargs`.
        ValueError: If `prompts` is empty or `trajectory_rewards` does not have
          1D shape `(len(prompts),)`.
    """
    trajectory_rewards = kwargs.pop("trajectory_rewards")
    trajectory_rewards_array = np.asarray(trajectory_rewards, dtype=np.float64)
    num_prompts = len(prompts)
    if trajectory_rewards_array.shape != (num_prompts,):
      raise ValueError(
          "trajectory_rewards shape mismatch: expected"
          f" {(num_prompts,)}, got {trajectory_rewards_array.shape}."
      )
    return self._compute_rewards(
        prompts,
        completions,
        trajectory_rewards_array,
        **kwargs,
    )
