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

import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils
from tunix.rl.grpo import grpo_helpers
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import metrics_logger

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

  def __post_init__(self):
    assert self.num_generations > 1, (
        "num_generations must be greater than 1. Received: "
        f"{self.num_generations}"
    )


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
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": "kl"})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        "rewards/overall",
        lambda: "kl" if self.grpo_config.beta != 0.0 else None,
    ])
    self.rl_cluster.actor_trainer.is_managed_externally = True

    self._metrics_logger = self.rl_cluster.actor_trainer.metrics_logger

    self.grad_acc_steps = (
        self.rl_cluster.cluster_config.training_config.get_with_default(
            "gradient_accumulation_steps", 1
        )
    )

    self._train_steps = 0
    self._eval_steps = 0

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        utils.is_sharing_weights(
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
    self._last_train_step = self.rl_cluster.actor_trainer.train_steps

    self.rollout_micro_batch_size = 2
    self.ref_logps_micro_batch_size = 2
    self.old_logps_micro_batch_size = 2

  def _rollout_by_micro(self, prompts: list[str], micro: int):
    """Performs rollouts in smaller batches (micro-batches) to manage memory."""
    outs_tokens = []
    outs_text = []
    outs_left_padded = []

    # Log the total number of prompts to be processed.
    print(f"Processing {len(prompts)} prompts in this batch.")

    # Iterate through the prompts in micro-batches.
    for slc in _chunk_slices_by_size(len(prompts), micro):
      sub_prompts = prompts[slc]
      print(f"Generating for slice: {slc}")

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
    """Computes reference per-token log probabilities in micro-batches.

    Args:
      prompt_ids: Token IDs of the prompts.
      completion_ids: Token IDs of the completions.
      micro: The maximum size of each micro-batch.

    Returns:
      A JAX array containing the per-token log probabilities from the reference
      model, concatenated from all micro-batches.
    """
    pad_id = self.rl_cluster.rollout.pad_id()
    eos_id = self.rl_cluster.rollout.eos_id()
    outs = []
    B = prompt_ids.shape[0]
    for slc in _chunk_slices_by_size(B, micro):
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
    """Computes old policy per-token log probabilities in micro-batches.

    Args:
      prompt_ids: Token IDs of the prompts.
      completion_ids: Token IDs of the completions.
      micro: The maximum size of each micro-batch.

    Returns:
      A JAX array containing the per-token log probabilities from the old policy
      model, concatenated from all micro-batches.
    """
    outs = []
    B = prompt_ids.shape[0]
    for slc in _chunk_slices_by_size(B, micro):
      outs.append(
          self.rl_cluster.get_old_per_token_logps(
              prompt_tokens=prompt_ids[slc],
              completion_tokens=completion_ids[slc],
          )
      )
    return jnp.concatenate(outs, axis=0)

  def _get_metric_logging_steps(self, mode: metrics_logger.Mode) -> int:
    return (
        self._train_steps
        if mode == metrics_logger.Mode.TRAIN
        else self._eval_steps
    )

  def _generate_and_compute_advantage(
      self,
      training_input: _TrainingInputT,
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> TrainExample:
    """Generates completions and computes advantages for a given batch of prompts."""
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    prompts: List[str] = training_input["prompts"]

    # === 1) Rollout: Process in chunks by rollout_micro_batch_size ===
    print("[new] Beginning rollout...")
    completion_ids, prompt_ids, completion_text = self._rollout_by_micro(
        prompts, self.rollout_micro_batch_size
    )

    # === 2) Assemble masks ===
    print("[new] Assembling masks...")
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype(
        "int32"
    )
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    # Apply the padding mask to the completion mask.
    completion_mask = completion_mask * completion_padding_mask

    # === 3) Calculate ref/old logps in their respective micro-batches ===
    print("[new] Calculating log probabilities...")
    if self.grpo_config.beta != 0.0:
      ref_per_token_logps = self._ref_logps_by_micro(
          prompt_ids, completion_ids, self.ref_logps_micro_batch_size
      )
    else:
      ref_per_token_logps = None

    if self.grpo_config.num_iterations > 1:
      old_per_token_logps = self._old_logps_by_micro(
          prompt_ids, completion_ids, self.old_logps_micro_batch_size
      )
    else:
      old_per_token_logps = None

    # === 4) Compute rewards and advantages (consistent with original logic) ===
    rewards = self._compute_rewards(
        prompts=prompts,
        completions=completion_text,
        mode=mode,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )
    advantages = grpo_helpers.compute_advantages(
        rewards, self.grpo_config.num_generations
    )

    # === 5) Log length metrics (unchanged from original) ===
    agg_completion_mask = completion_mask.sum(axis=-1)
    steps = self._get_metric_logging_steps(mode)
    self._metrics_logger.log(
        "completions/mean_length",
        np.mean(agg_completion_mask),
        mode,
        steps,
    )
    self._metrics_logger.log(
        "completions/max_length",
        np.max(agg_completion_mask),
        mode,
        steps,
    )
    self._metrics_logger.log(
        "completions/min_length",
        np.min(agg_completion_mask.min),
        mode,
        steps,
    )

    # === 6) Return the TrainExample object ===
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
      mode: metrics_logger.Mode,
      **kargs,
  ):
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      **kargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts, num_reward_fns]`) of scalar rewards for
      each prompt-completion pair. The rewards are computed using the provided
      reward functions.
    """
    rewards = jnp.zeros((len(prompts), len(self.reward_fns)))
    steps = self._get_metric_logging_steps(mode)
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kargs)
      r = jnp.array(r)
      rewards = rewards.at[:, i].set(r)

      self._metrics_logger.log(
          f"rewards/{reward_fn.__name__}",
          np.mean(r),
          mode,
          steps,
      )

    rewards = jnp.nansum(rewards, axis=1)

    self._metrics_logger.log(
        "rewards/overall",
        np.mean(rewards),
        mode,
        steps,
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
      mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
  ) -> None:
    """按 micro 流水准备数据：合并→repeat(sample)→一次前向→按原 micro 切回。
    入队策略：
      - 永远只 put 长度为 1 的小 list（[TrainExample]）。
      - 同步：每批先累计到 pending_examples，边界/收尾时一次性 repeat = batch_repeat。
      - 异步：每批先入队 1 次；若 batch_repeat>1，同时缓存到 pending_examples，
              边界/收尾时统一补齐 (batch_repeat - 1) 次。
    """

    # =============== 工具函数 ===============
    def _put_as_singleton(examples: list[TrainExample], times: int) -> None:
      """把每个 TrainExample 单独包成 [TrainExample]，重复 times 次放入队列。"""
      if times <= 0 or not examples:
        return
      for _ in range(times):
        for ex in examples:
          data_queue.put([ex])

    # 用三处 micro 的 LCM 作为一次服务目标批大小（跨阶段对齐）
    service_target_bs = _lcm3(
        self.rollout_micro_batch_size,
        self.ref_logps_micro_batch_size,
        self.old_logps_micro_batch_size,
    )

    buf: list[_TrainingInputT] = []
    buf_sizes: list[int] = []   # 各训练 micro-batch 的样本数
    buf_b = 0                   # 聚合样本总数（repeat 前）
    consumed_steps = 0          # 已消耗的训练 micro-batch 数

    # 同步/异步共同使用：用于“重复补齐”的缓存（异步时也会使用）
    pending_examples: list[TrainExample] = []

    def _flush(force: bool = False) -> list[TrainExample]:
      """聚合→按 sample_repeat 重复→大批生成与打分→切回原 micro→返回小批列表。"""
      nonlocal buf, buf_sizes, buf_b
      produced: list[TrainExample] = []

      if not buf:
        return produced
      if (not force) and (buf_b < service_target_bs):
        return produced

      # 1) 合并多个训练 micro-batch
      merged: dict = {}
      keys = buf[0].keys()
      for k in keys:
        merged[k] = list(buf[0][k]) if isinstance(buf[0][k], list) else np.asarray(buf[0][k])
      for i in range(1, len(buf)):
        for k in keys:
          a, b = merged[k], buf[i][k]
          if isinstance(a, list) and isinstance(b, list):
            a.extend(b)
          else:
            merged[k] = np.concatenate([np.asarray(a), np.asarray(b)], axis=0)

      # 2) repeat 样本（等价于每个 micro 重复后再拼接）
      merged_repeated = jax.tree.map(
          lambda x: np.repeat(x, sample_repeat, axis=0),
          merged,
      )

      # 3) 执行一次大批生成 + 计算优势
      with jax.profiler.StepTraceAnnotation(
          "sampler",
          step_num=self._train_steps if mode == metrics_logger.Mode.TRAIN else self._eval_steps,
      ):
        big_example = self._generate_and_compute_advantage(merged_repeated, mode)

      # 4) 切回原训练 micro 边界（乘以 sample_repeat）
      offset = 0
      for n in buf_sizes:
        token_sl = slice(offset * sample_repeat, (offset + n) * sample_repeat)
        te_small = TrainExample(
            prompt_ids=big_example.prompt_ids[token_sl],
            prompt_mask=big_example.prompt_mask[token_sl],
            completion_ids=big_example.completion_ids[token_sl],
            completion_mask=big_example.completion_mask[token_sl],
            ref_per_token_logps=None
                if big_example.ref_per_token_logps is None
                else big_example.ref_per_token_logps[token_sl],
            advantages=big_example.advantages[token_sl],
            old_per_token_logps=None
                if big_example.old_per_token_logps is None
                else big_example.old_per_token_logps[token_sl],
        )
        produced.append(te_small)
        offset += n

      # 5) 清空缓冲
      buf.clear()
      buf_sizes.clear()
      buf_b = 0

      return produced

    try:
      while True:
        # 拉取一个训练 micro-batch
        example = next(iterator)
        B = len(example["prompts"])
        buf.append(example)
        buf_sizes.append(B)
        buf_b += B
        consumed_steps += 1

        # 达到 LCM 阈值就产出一批
        produced_now = _flush(force=False)
        if produced_now:
          if async_loading:
            # 异步：先入队一次，并缓存以便后续补 (batch_repeat - 1)
            _put_as_singleton(produced_now, 1)
            if batch_repeat > 1:
              pending_examples.extend(produced_now)
          else:
            # 同步：先累计，尾声统一入队 batch_repeat 次
            pending_examples.extend(produced_now)

        # 维护日志步数
        if mode == metrics_logger.Mode.TRAIN:
          self._train_steps += 1
        else:
          self._eval_steps += 1

        # 到达 proceed 边界：处理尾巴 + 入队 repeats
        if proceed_num_steps > 0 and consumed_steps >= proceed_num_steps:
          tail = _flush(force=True)
          if tail:
            if async_loading:
              _put_as_singleton(tail, 1)
              if batch_repeat > 1:
                pending_examples.extend(tail)
            else:
              pending_examples.extend(tail)

          if pending_examples:
            if not async_loading:
              _put_as_singleton(pending_examples, batch_repeat)
            else:
              rem = batch_repeat - 1
              if rem > 0:
                _put_as_singleton(pending_examples, rem)
            pending_examples.clear()

          consumed_steps = 0
          return

    except StopIteration as e:
      # 外部要求固定步数时，StopIteration 交给上层（训练路径通常不会走到这里）
      if proceed_num_steps > 0:
        raise e
      # 否则是整集遍历（如 eval）：做最终 tail + repeats
      tail = _flush(force=True)
      if tail:
        if async_loading:
          _put_as_singleton(tail, 1)
          if batch_repeat > 1:
            pending_examples.extend(tail)
        else:
          pending_examples.extend(tail)

      if pending_examples:
        if not async_loading:
          _put_as_singleton(pending_examples, batch_repeat)
        else:
          rem = batch_repeat - 1
          if rem > 0:
            _put_as_singleton(pending_examples, rem)
        pending_examples.clear()
      return

    finally:
      # 结束标记
      data_queue.put(None)


  def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """GRPO training loop.

    Algorithm as below: extract from https://arxiv.org/abs/2402.03300
    Input initial policy model πθinit; reward models rφ; task prompts D;
    hyperparameters ε, β, μ

    policy model πθ ← πθinit
    for iteration = 1, ..., I do
      reference model πref ← πθ
      for step = 1, ..., M do
        Sample a batch D♭ from D
        Update the old policy model πθold ← πθ
        Sample G outputs {oi}G_i=1 ~ πθold(· | q) for each question q ∈ D♭
        Compute rewards {ri}G_i=1 for each sampled output oi by running rφ
        Compute Âi,t for the t-th token of oi through group relative advantage
        estimation.
        for GRPO iteration = 1, ..., μ do
          Update the policy model πθ by maximizing the GRPO objective (Equation
          21)
      Update rφ through continuous training using a replay mechanism.
    Output πθ

    NOTE:
    1. The outer loop (I) is ignored for now because we never update the
    reference model for now.
    2. Currently sample and train hold the same referece to the model. So we
    also omit the step to update the sampler model.

    Args:
      train_ds: An iterable of training input data, where each element is a
        dictionary containing the key 'prompts'.
      eval_ds: An iterable of evaluation input data, where each element is a
        dictionary containing the key 'prompts'.
      skip_jit: Whether to skip JIT compilation of the training loop.
    """
    print("begin grpo training step")
    train_iterator = iter(train_ds)
    # i = 0
    # while i < 3:
    #   i += 1
    #   item = next(train_iterator)
    #   print(item)

    while True:  # loop over M
      try:
        # reserve 1 for None and the other for repeated interable
        # if batch_repeat > 1
        train_data_queue = queue_lib.SimpleDataQueue(
            maxsize=self.grad_acc_steps + 2
        )
        # reserve 1 for None
        print("the grad_acc_steps is" + str(self.grad_acc_steps))
        eval_data_queue = queue_lib.SimpleDataQueue(maxsize=2)
        initial_train_steps = self._train_steps
        future = self.executor.submit(
            self._prepare_data,
            iterator=train_iterator,
            proceed_num_steps=self.grad_acc_steps,
            sample_repeat=self.grpo_config.num_generations,
            batch_repeat=self.grpo_config.num_iterations,
            data_queue=train_data_queue,
            async_loading=self.can_enable_async_rollout,
            mode=metrics_logger.Mode.TRAIN,
        )
        curr_eval_ds = None
        with jax.profiler.StepTraceAnnotation(
            "trainer", step_num=initial_train_steps
        ):
          while True:
            curr_train_ds = train_data_queue.get(block=True)
            print("here is an example")
            print(curr_train_ds)
            for item in curr_train_ds:
              print("-------------------")
              print(item.advantages)
              print(item.ref_per_token_logps)
              print("-------------------")
            if curr_train_ds is None:
              break
            if (
                eval_ds
                and not curr_eval_ds
                and self.rl_cluster.actor_trainer.train_steps
                % self.rl_cluster.cluster_config.training_config.eval_every_n_steps
                == 0
            ):
              print("begin prepare evaluation data")
              self._prepare_data(
                  iterator=iter(eval_ds),
                  proceed_num_steps=-1,
                  sample_repeat=self.grpo_config.num_generations,
                  batch_repeat=1,
                  data_queue=eval_data_queue,
                  async_loading=False,
                  mode=metrics_logger.Mode.EVAL,
              )
              curr_eval_ds = eval_data_queue.get(block=True)
            self.rl_cluster.update_actor(
                curr_train_ds,
                curr_eval_ds,
                skip_jit,
            )  # loop over μ
        # call to throw stop iteration as a singal to break the loop
        future.result()
        # sync the train steps with internel trainer, this is based on the
        # assumption that the trainer internally doesn't reset the train steps.
        # there is current a unit test to ensure this assumption.
        self._train_steps = self.rl_cluster.actor_trainer.train_steps

        if self.should_sync_weights:
          with jax.profiler.StepTraceAnnotation(
              "sync_sampler_weights", step_num=initial_train_steps
          ):
            self.rl_cluster.sync_weights()
        if (
            self._train_steps
            >= self.rl_cluster.cluster_config.training_config.max_steps
        ):
          break
      except StopIteration:
        break
    self.rl_cluster.actor_trainer.close()


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
