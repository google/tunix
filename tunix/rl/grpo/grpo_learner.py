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

"""GRPO learner."""

from __future__ import annotations

from concurrent import futures
import dataclasses
import math
from typing import Callable, Dict, Iterable, Iterator, List, Sequence
import warnings

import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.grpo import grpo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

_TrainingInputT = Dict[str, List[str] | ArrayLike]

# prompts, completions, **kargs -> rewards
RewardFn = Callable[..., List[float]]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  pass


@dataclasses.dataclass(slots=True, kw_only=True)
class GrpoConfig:
  """Configuration for GRPO algorithm.

  Attributes:
    training_micro_batch_size: The microbatch size used for training.
    rollout_micro_batch_size: The microbatch size used for model rollouts. If
      -1, it defaults to `training_micro_batch_size`.
    ref_logps_micro_batch_size: The microbatch size used for computing reference
      model log probabilities. If -1, it defaults to
      `training_micro_batch_size`.
    old_logps_micro_batch_size: The microbatch size used for computing old
      policy log probabilities. If -1, it defaults to
      `training_micro_batch_size`.
    num_generations: The number of times the policy generates multiple responses
      for a given prompt within a single training step. This corresponds to 'G'
      in Algorithm 1 in the paper. A higher value means more samples are used to
      compute relative advantages.
    num_iterations: The number of iterations per batch (𝜇 in GRPO algo 1).
    beta: The coefficient for the KL divergence penalty (𝛽) in the GRPO loss
      function. This term prevents policy updates from deviating too far from
      the reference model. A value of 0.0 means no KL penalty is applied.
    epsilon: Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to
      PPO, it ensures stable updates.
    loss_algo: use GRPO or GSPO for loss computation. GRPO loss is per-batch
      normalized instead of per-response normalized as mentioned in the paper.
      For GSPO, we use gspo-token loss which is more flexible.

  References:
  - GRPO: https://arxiv.org/abs/2402.03300
  - GSPO: https://www.arxiv.org/pdf/2507.18071
  """

  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  epsilon: float = 0.2
  loss_algo: str = "grpo"  # grpo or gspo-token

  # Microbatch size configurations
  training_micro_batch_size: int = None  # pytype: disable=annotation-type-mismatch
  rollout_micro_batch_size: int = -1
  ref_logps_micro_batch_size: int = -1
  old_logps_micro_batch_size: int = -1

  def __post_init__(self):
    """Validates the configuration after initialization."""
    # 【合并点1】：结合了两个版本的验证逻辑
    assert self.num_generations > 1, (
        "num_generations must be greater than 1. Received: "
        f"{self.num_generations}"
    )
    
    if self.training_micro_batch_size is None:
      warnings.warn(
          "training_micro_batch_size is None, it should be provided.",
          DeprecationWarning,
      )
      raise ValueError(
          "GrpoConfig instance requires training_micro_batch_size to be set."
      )
    elif self.training_micro_batch_size <= 0:
      raise ValueError("training_micro_batch_size must be positive.")
    
    # Set other parameters to training_micro_batch_size if they are not set
    if self.rollout_micro_batch_size == -1:
      self.rollout_micro_batch_size = self.training_micro_batch_size
    
    if self.ref_logps_micro_batch_size == -1:
      self.ref_logps_micro_batch_size = self.training_micro_batch_size
    
    if self.old_logps_micro_batch_size == -1:
      self.old_logps_micro_batch_size = self.training_micro_batch_size


def _lcm3(a: int, b: int, c: int) -> int:
  return (
      (a * b) // math.gcd(a, b) * c // math.gcd(((a * b) // math.gcd(a, b)), c)
  )


def _chunk_slices_by_size(n: int, micro: int):
  """Returns a list of slices `[slice(...), ...]` for n samples, chunked by micro.

  The last chunk is allowed to be smaller than micro.

  Args:
    n: The total number of samples.
    micro: The maximum size of each chunk.
  """
  i = 0
  out = []
  while i < n:
    out.append(slice(i, min(i + micro, n)))
    i += micro
  return out


class GrpoLearner:
  """GRPO (Group Relative Policy Optimization) learner.

  GRPO is a reinforcement learning algorithm designed to enhance the reasoning
  abilities of large language models, like mathematical problem-solving. It is
  a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
  eliminating the need for a separate value function model. GRPO works by
  generating multiple responses for a given prompt, evaluating these responses
  using a reward model, and then calculating a relative advantage based on the
  group's performance to update the policy.

  References:
  - https://arxiv.org/abs/2402.03300
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      reward_fns: RewardFn | List[RewardFn],
      grpo_config: GrpoConfig,
      metric_fns: (
          Sequence[Callable[..., rl_cluster_lib.MetricsT]] | None
      ) = None,
  ):
    """Initializes the `GrpoTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      grpo_config: An instance of `GrpoConfig` containing all GRPO specific
        parameters.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept `prompts`, `completions`,
        `rewards`, `advantages` and optional keyword arguments, and return a
        dictionary of metric names to tuples of (metric_value, aggregation_fn).
    """
    assert grpo_config.loss_algo in ["grpo", "gspo-token"], (
        "loss_algo should be either grpo or gspo-token. Received: "
        f"{grpo_config.loss_algo}"
    )
    self.grpo_config = grpo_config
    self.rl_cluster = rl_cluster
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )
    self.metric_fns = metric_fns or []  # 【合并点2】：添加metric_fns

    # Workaround for passing in importance_sampling_algo as jax transforms
    # doesn't like partial functions with kwargs.
    loss_fn = lambda model, train_example, beta, epsilon: grpo_loss_fn(
        model,
        train_example,
        beta=beta,
        epsilon=epsilon,
        loss_algo=self.grpo_config.loss_algo,
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "beta": self.grpo_config.beta,
            "epsilon": self.grpo_config.epsilon,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": np.mean})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "rewards/overall",
        lambda: "kl" if self.grpo_config.beta != 0.0 else None,
    ])
    self.rl_cluster.actor_trainer.is_managed_externally = True

    # 【合并点3】：添加global_steps调整
    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.train_steps
        // self.grpo_config.num_iterations
    )

    self.grad_acc_steps = (
        self.rl_cluster.cluster_config.training_config.get_with_default(
            "gradient_accumulation_steps", 1
        )
    )

    self._iter_steps = 0
    self._eval_steps = 0

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        rl_utils.is_sharing_weights(
            self.rl_cluster.actor_trainer.model,
            self.rl_cluster.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
        != self.rl_cluster.cluster_config.role_to_mesh[
            rl_cluster_lib.Role.ROLLOUT
        ]
    )
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self._last_iter_step = self.rl_cluster.actor_trainer.iter_steps

    # 【合并点4】：使用config中的microbatch sizes
    self.rollout_micro_batch_size = grpo_config.rollout_micro_batch_size
    self.ref_logps_micro_batch_size = grpo_config.ref_logps_micro_batch_size
    self.old_logps_micro_batch_size = grpo_config.old_logps_micro_batch_size

  def _rollout_by_micro(self, prompts: list[str], micro: int):
    """Performs rollouts in smaller batches (micro-batches) to manage memory."""
    outs_tokens = []
    outs_text = []
    outs_left_padded = []

    # Iterate through the prompts in micro-batches.
    for slc in _chunk_slices_by_size(len(prompts), micro):
      sub_prompts = prompts[slc]

      # Generate completions for the current sub-batch.
      out = self.rl_cluster.generate(prompts=sub_prompts)

      # Store the results from the generation.
      outs_tokens.append(out.tokens)  # Shape: [batch_size, output_seq_len]
      outs_text.extend(out.text)
      outs_left_padded.append(
          out.left_padded_prompt_tokens
      )  # Shape: [batch_size, input_seq_len]

    # Concatenate the results from all micro-batches.
    completion_ids = jnp.concatenate(outs_tokens, axis=0)
    left_padded = jnp.concatenate(outs_left_padded, axis=0)

    return completion_ids, left_padded, outs_text

  def _ref_logps_by_micro(
      self, prompt_ids: jnp.ndarray, completion_ids: jnp.ndarray, micro: int
  ):
    """Computes reference per-token log probabilities in micro-batches."""
    pad_id = self.rl_cluster.rollout.pad_id()
    eos_id = self.rl_cluster.rollout.eos_id()
    outs = []
    batch_size = prompt_ids.shape[0]
    for slc in _chunk_slices_by_size(batch_size, micro):
      outs.append(
          self.rl_cluster.get_ref_per_token_logps(
              prompt_tokens=prompt_ids[slc],
              completion_tokens=completion_ids[slc],
              pad_id=pad_id,
              eos_id=eos_id,
          )
      )
    return jnp.concatenate(outs, axis=0)

  def _old_logps_by_micro(
      self, prompt_ids: jnp.ndarray, completion_ids: jnp.ndarray, micro: int
  ):
    """Computes old policy per-token log probabilities in micro-batches."""
    outs = []
    batch_size = prompt_ids.shape[0]
    for slc in _chunk_slices_by_size(batch_size, micro):
      outs.append(
          self.rl_cluster.get_old_per_token_logps(
              prompt_tokens=prompt_ids[slc],
              completion_tokens=completion_ids[slc],
          )
      )
    return jnp.concatenate(outs, axis=0)

  def _compute_trajectory_ids(self, example: _TrainingInputT) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch."""
    batch_size = len(example["prompts"]) // self.grpo_config.num_generations
    row_offset = self._iter_steps * batch_size
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.grpo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.grpo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _generate_and_compute_advantage(
      self,
      training_input: _TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Generates text completions and computes the advantages for GRPO training."""
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    # 【合并点5】：使用microbatch rollout
    completion_ids, prompt_ids, completion_text = self._rollout_by_micro(
        training_input["prompts"],
        self.rollout_micro_batch_size * self.grpo_config.num_generations,
    )

    # Assemble masks
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype(
        "int32"
    )
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    # Apply the padding mask to the completion mask.
    completion_mask = completion_mask * completion_padding_mask

    # 【合并点6】：使用microbatch计算ref/old logps
    if self.grpo_config.beta != 0.0:
      ref_per_token_logps = self._ref_logps_by_micro(
          prompt_ids,
          completion_ids,
          self.ref_logps_micro_batch_size * self.grpo_config.num_generations,
      )
    else:
      ref_per_token_logps = None

    if self.grpo_config.num_iterations > 1:
      old_per_token_logps = self._old_logps_by_micro(
          prompt_ids,
          completion_ids,
          self.old_logps_micro_batch_size * self.grpo_config.num_generations,
      )
    else:
      old_per_token_logps = None

    # Compute rewards and advantages
    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_text,
        mode=mode,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )
    advantages = grpo_helpers.compute_advantages(
        rewards, self.grpo_config.num_generations
    )

    # 【合并点7】：使用buffer_metrics记录指标
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )
    
    # 【合并点8】：添加用户定义的metrics
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=completion_text,
          advances=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      **kargs,
  ):
    """Computes the rewards for completions using the provided reward functions."""
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)
      
      # 【合并点9】：使用buffer_metrics记录reward指标
      self.rl_cluster.buffer_metrics(
          {
              f"rewards/{reward_fn.__name__}": (
                  np.mean(r),
                  np.mean,
              ),
          },
          mode=mode,
      )

    rewards = jnp.nansum(rewards, axis=1)
    
    # 【合并点10】：记录更多reward统计信息
    self.rl_cluster.buffer_metrics(
        {
            "rewards/overall": (
                np.mean(rewards),
                np.mean,
            ),
            "rewards/min": (
                np.min(rewards),
                np.min,
            ),
        },
        mode=mode,
    )
    
    # 【合并点11】：记录prompts和completions
    for p, c in zip(prompts, completions):
      self.rl_cluster.buffer_metrics(
          {
              "prompts": (
                  p,
                  None,
              ),
              "completions": (
                  c,
                  None,
              ),
          },
          mode=mode,
      )

    return rewards

  def _prepare_data(
      self,
      iterator: Iterator[_TrainingInputT],
      proceed_num_steps: int,
      sample_repeat: int,
      batch_repeat: int,
      data_queue: queue_lib.AbstractDataQueue[list[TrainExample] | None],
      async_loading: bool = False,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> None:
    """Data preparation pipeline by micro steps: merge → repeat(sample) → single large."""

    def enqueue_examples(examples: list[TrainExample], times: int) -> None:
      """Wrap each TrainExample as [TrainExample] and put it into the queue, repeated `times`."""
      if times <= 0 or not examples:
        return
      for _ in range(times):
        for ex in examples:
          data_queue.put([ex])

    # 【合并点12】：使用LCM作为服务目标批次大小
    service_target_bs = _lcm3(
        self.rollout_micro_batch_size,
        self.ref_logps_micro_batch_size,
        self.old_logps_micro_batch_size,
    )

    buf: list[_TrainingInputT] = []
    buf_sizes: list[int] = []  # Number of samples for each training micro-batch
    buf_b = 0  # Aggregated sample count (before repeating)
    consumed_steps = 0  # Number of consumed training micro-batches

    pending_examples: list[TrainExample] = []

    def aggregate_and_compute_advantages(
        force: bool = False,
    ) -> list[TrainExample]:
      """Merge → repeat(sample) → one large forward pass & advantage → split back."""
      nonlocal buf, buf_sizes, buf_b
      produced: list[TrainExample] = []

      if not buf:
        return produced
      if (not force) and (buf_b < service_target_bs):
        return produced

      # Merge multiple training micro-batches
      merged: _TrainingInputT = {}
      keys = buf[0].keys()
      for k in keys:
        merged[k] = (
            list(buf[0][k])
            if isinstance(buf[0][k], list)
            else np.asarray(buf[0][k])
        )
      for i in range(1, len(buf)):
        for k in keys:
          a, b = merged[k], buf[i][k]
          if isinstance(a, list) and isinstance(b, list):
            a.extend(b)
          else:
            merged[k] = np.concatenate([np.asarray(a), np.asarray(b)], axis=0)

      # Repeat samples (equivalent to repeating each micro, then concat)
      merged_repeated = jax.tree.map(
          lambda x: np.repeat(x, sample_repeat, axis=0),
          merged,
      )
      
      # 【合并点13】：添加trajectory_ids
      if mode == rl_cluster_lib.Mode.TRAIN:
        trajectory_ids = self._compute_trajectory_ids(merged_repeated)
        assert "trajectory_ids" not in merged_repeated
        merged_repeated["trajectory_ids"] = trajectory_ids

      # Single large forward pass + advantage computation
      with jax.profiler.StepTraceAnnotation(
          "sampler",
          step_num=self._iter_steps
          if mode == rl_cluster_lib.Mode.TRAIN
          else self._eval_steps,
      ):
        big_example = self._generate_and_compute_advantage(
            merged_repeated, mode
        )

      # Split back to original training micro size
      offset = 0
      for n in buf_sizes:
        # Calculate slice indices
        start_idx = offset * sample_repeat
        end_idx = (offset + n) * sample_repeat
        token_sl = slice(start_idx, end_idx)

        # Create TrainExample for this micro-batch
        te_small = TrainExample(
            prompt_ids=big_example.prompt_ids[token_sl],
            prompt_mask=big_example.prompt_mask[token_sl],
            completion_ids=big_example.completion_ids[token_sl],
            completion_mask=big_example.completion_mask[token_sl],
            ref_per_token_logps=(
                None
                if big_example.ref_per_token_logps is None
                else big_example.ref_per_token_logps[token_sl]
            ),
            advantages=big_example.advantages[token_sl],
            old_per_token_logps=(
                None
                if big_example.old_per_token_logps is None
                else big_example.old_per_token_logps[token_sl]
            ),
        )

        produced.append(te_small)
        offset += n

      # Clear buffers
      buf.clear()
      buf_sizes.clear()
      buf_b = 0

      return produced

    try:
      while True:
        while (
            mode == rl_cluster_lib.Mode.TRAIN
            and self._iter_steps < self._last_iter_step
        ):  # fast forward the iterator if loading from a previous checkpoint.
          next(iterator)
          self._iter_steps += 1

        # Fetch one training micro-batch
        example = next(iterator)
        cur_batch_size = len(example["prompts"])
        buf.append(example)
        buf_sizes.append(cur_batch_size)
        buf_b += cur_batch_size
        consumed_steps += 1

        # If the LCM threshold is reached, produce one batch
        produced_now = aggregate_and_compute_advantages(force=False)
        if produced_now:
          if async_loading:
            # Async: Enqueue immediately
            enqueue_examples(produced_now, 1)
            if batch_repeat > 1:
              pending_examples.extend(produced_now)
          else:
            # Sync: accumulate; at boundary finalize with batch_repeat times
            pending_examples.extend(produced_now)

        if mode == rl_cluster_lib.Mode.TRAIN:
          self._iter_steps += 1
        else:
          self._eval_steps += 1

        # On proceed boundary: handle tail + enqueue repeats
        if proceed_num_steps > 0 and consumed_steps == proceed_num_steps:
          tail = aggregate_and_compute_advantages(force=True)
          if tail:
            if async_loading:
              enqueue_examples(tail, 1)
              if batch_repeat > 1:
                pending_examples.extend(tail)
            else:
              pending_examples.extend(tail)

          if pending_examples:
            if not async_loading:
              enqueue_examples(pending_examples, batch_repeat)
            else:
              rem = batch_repeat - 1
              if rem > 0:
                enqueue_examples(pending_examples, rem)
            pending_examples.clear()

          consumed_steps = 0
          return
    except StopIteration as e:
      if proceed_num_steps > 0:
        raise e
      else:
        tail = aggregate_and_compute_advantages(force=True)
        if tail:
          pending_examples.extend(tail)
        if pending_examples:
          # 【合并点14】：不再使用RepeatIterable，改为直接enqueue
          enqueue_examples(pending_examples, batch_repeat)
          pending_examples.clear()
        return
    except Exception as e:
      raise e
    finally:
      data_queue.put(None)

  def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """GRPO training loop."""
    train_iterator = iter(train_ds)
    while True:  # loop over M
      try:
        # 【合并点15】：调整queue大小以适应新的数据加载方式
        train_data_queue = queue_lib.SimpleDataQueue(
            maxsize=self.grad_acc_steps * self.grpo_config.num_iterations + 1
        )
        # reserve 1 for None
        eval_data_queue = queue_lib.SimpleDataQueue(maxsize=2)
        initial_steps = self._iter_steps
        future = self.executor.submit(
            self._prepare_data,
            iterator=train_iterator,
            proceed_num_steps=self.grad_acc_steps,
            sample_repeat=self.grpo_config.num_generations,
            batch_repeat=self.grpo_config.num_iterations,
            data_queue=train_data_queue,
            async_loading=self.can_enable_async_rollout,
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        curr_eval_ds = None
        with jax.profiler.StepTraceAnnotation(
            "trainer", step_num=initial_steps
        ):
          while True:
            # 【合并点16】：添加时间测量
            with sft_utils.time_measure(suppress_logging=True) as timer:
              curr_train_ds = train_data_queue.get(block=True)
            
            if curr_train_ds is None:
              break
            
            # 【合并点17】：记录dequeue时间
            if self.can_enable_async_rollout:
              self.rl_cluster.buffer_metrics(
                  {
                      "actor_dequeue_time": (
                          timer(),
                          np.mean,
                      ),
                  },
                  mode=rl_cluster_lib.Mode.TRAIN,
              )
            
            if (
                eval_ds
                and not curr_eval_ds
                and self.rl_cluster.actor_trainer.train_steps
                % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
                == 0
            ):
              self._prepare_data(
                  iterator=iter(eval_ds),
                  proceed_num_steps=-1,
                  sample_repeat=self.grpo_config.num_generations,
                  batch_repeat=1,
                  data_queue=eval_data_queue,
                  async_loading=False,
                  mode=rl_cluster_lib.Mode.EVAL,
              )
              curr_eval_ds = eval_data_queue.get(block=True)
            self.rl_cluster.update_actor(
                curr_train_ds,
                curr_eval_ds,
                skip_jit,
            )  # loop over μ
        # call to throw stop iteration as a singal to break the loop
        future.result()
        # sync the iter steps with internel trainer, this is based on the
        # assumption that the trainer internally doesn't reset the iter steps.
        # there is current a unit test to ensure this assumption.
        self._iter_steps = self.rl_cluster.actor_trainer.iter_steps

        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_steps
          ):
            self.rl_cluster.sync_weights()
        else:
          # 【合并点18】：手动增加global_steps
          self.rl_cluster.global_steps += 1
        
        if (
            self.rl_cluster.actor_trainer.train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    # 【合并点19】：使用rl_cluster.close()而不是actor_trainer.close()
    self.rl_cluster.close()


def grpo_loss_fn(model, train_example, beta, epsilon, loss_algo):
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    beta: The coefficient for the KL divergence penalty. A value of 0.0 means no
      KL penalty is applied.
    epsilon: Epsilon value for clipping.
    loss_algo: The loss algorithm to use. Can be grpo or gspo-token.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  prompt_ids, prompt_mask = (
      train_example.prompt_ids,
      train_example.prompt_mask,
  )
  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )
  input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)
  prompt_completion_mask = jnp.concat([prompt_mask, completion_mask], axis=-1)
  attention_mask = common.make_causal_attn_mask(prompt_completion_mask)
  logits_to_keep = completion_ids.shape[1]
  positions = common.build_positions_from_mask(prompt_completion_mask)

  per_token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  advantages = train_example.advantages

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = per_token_logps - old_per_token_logps
  if loss_algo == "gspo-token":
    seq_importance_ratio = (seq_importance_ratio * completion_mask).sum(
        axis=-1
    ) / jnp.clip(completion_mask.sum(-1), min=1)
    seq_importance_ratio = (
        per_token_logps
        - jax.lax.stop_gradient(per_token_logps)
        + jnp.expand_dims(jax.lax.stop_gradient(seq_importance_ratio), axis=-1)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)

  coef_1 = jnp.exp(seq_importance_ratio)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)

  # TODO(tsbao): We should handle token level advantages.
  per_token_loss = -jnp.minimum(
      coef_1 * jnp.expand_dims(advantages, 1),
      coef_2 * jnp.expand_dims(advantages, 1),
  )

  if loss_algo == "gspo-token":
    loss_denominator = jnp.clip(completion_mask.sum(axis=-1), min=1)
  else:  # grpo
    loss_denominator = jnp.clip(completion_mask.sum(), min=1)

  aux = {"kl": 0.0}
  if beta != 0.0:
    kl = common.compute_kl_divergence(
        per_token_logps, train_example.ref_per_token_logps
    )
    per_token_loss = per_token_loss + beta * kl

    # Log mean KL.
    aux["kl"] = (kl * completion_mask).sum() / loss_denominator.mean()

  if loss_algo == "gspo-token":
    loss = (
        (per_token_loss * completion_mask).sum(axis=-1) / loss_denominator
    ).mean()
  else:  # grpo
    loss = (per_token_loss * completion_mask).sum() / loss_denominator

  return loss, aux