"""GRPO training pipeline for agentic models.

This pipeline orchestrates asynchronous trajectory collection and GRPO training.
"""

import asyncio
from typing import Any, Dict, List

from trajectory_collect_engine import BaseEnv
from trajectory_collect_engine import LLMBaseAgent
from trajectory_collect_engine import TrajectoryCollectEngine
from tunix.rl import common
from tunix.rl.experimental.agentic.adapters.grpo_batch_adapter import GRPOTrajectoryAdapter
from tunix.rl.experimental.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
from tunix.rl.grpo import grpo_helpers


class GRPOTrajectoryTrainer:
  """GRPO trainer with asynchronous rollout and training pipeline."""

  def __init__(
      self, rl_cluster, model_call_fn, reward_fn, config, tokenizer, chat_parser
  ):
    self.rl_cluster = rl_cluster
    self.model_call_fn = model_call_fn
    self.reward_fn = reward_fn
    self.config = config

    # Adapter: converts trajectories into TrainExample batches
    self.adapter = GRPOTrajectoryAdapter(
        tokenizer=tokenizer,
        chat_parser=chat_parser,
        pad_id=rl_cluster.rollout.pad_id(),
        eos_id=rl_cluster.rollout.eos_id(),
    )

    # Save pad/eos for convenience
    self.pad_id = rl_cluster.rollout.pad_id()
    self.eos_id = rl_cluster.rollout.eos_id()

    # Orchestrator: manages parallel trajectory collection
    self.orchestrator = RolloutOrchestrator(
        engine_cls=TrajectoryCollectEngine,
        engine_defaults={
            'model_call': model_call_fn,
            'final_reward_fn': reward_fn,
            'max_steps': config.max_steps_per_episode,
            'gamma': config.gamma,
            'timeout': config.episode_timeout,
            'tokenizer': tokenizer,
            'chat_parser': chat_parser,
        },
        max_concurrency=config.max_concurrent_rollouts,
    )

  async def rollout_producer(
      self,
      prompts: List[str],
      agents: List[LLMBaseAgent],
      envs: List[BaseEnv],
      queue: asyncio.Queue,
  ):
    """Background task: collect trajectories and put them into the queue."""
    pairs = list(zip(agents, envs))

    def group_key(pair_idx: int, env: BaseEnv, traj: Any) -> int:
      # Group by prompt index (num_generations trajectories per prompt)
      return pair_idx // self.config.num_generations

    async for batch in self.orchestrator.run_and_yield_batches(
        pairs=pairs,
        group_size=self.config.num_generations,
        batch_size=len(prompts) * self.config.num_generations,
        group_key=group_key,
        episodes_per_pair=1,
    ):
      all_trajs = [await item.traj.collect(mode='Token') for item in batch]
      grpo_batch = self.adapter.trajectories_to_grpo_batch(
          all_trajs, prompts * self.config.num_generations
      )
      await queue.put(grpo_batch)

    await queue.put(None)  # End-of-data marker

  def compute_train_example(
      self, batch_data: Dict[str, Any]
  ) -> common.TrainExample:
    """Compute GRPO advantages and wrap into TrainExample."""
    rewards = batch_data['rewards']
    advantages = grpo_helpers.compute_advantages(
        rewards, self.config.num_generations
    )

    # KL term
    ref_per_token_logps = None
    if self.config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=batch_data['prompt_ids'],
          completion_tokens=batch_data['completion_ids'],
          pad_id=self.pad_id,
          eos_id=self.eos_id,
          micro_batch_size=len(batch_data['prompt_ids']),
      )

    # Old policy term (if num_iterations > 1)
    old_per_token_logps = None
    if self.config.num_iterations > 1:
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=batch_data['prompt_ids'],
          completion_tokens=batch_data['completion_ids'],
          micro_batch_size=len(batch_data['prompt_ids']),
      )

    return common.TrainExample(
        prompt_ids=batch_data['prompt_ids'],
        prompt_mask=batch_data['prompt_mask'],
        completion_ids=batch_data['completion_ids'],
        completion_mask=batch_data['completion_mask'],
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  async def train_loop(self, prompts, agents, envs):
    """Main training loop: rollout and training run asynchronously."""
    queue = asyncio.Queue(maxsize=8)

    # Start rollout producer as a background task
    producer_task = asyncio.create_task(
        self.rollout_producer(prompts, agents, envs, queue)
    )

    while True:
      batch_data = await queue.get()
      if batch_data is None:
        break

      train_example = self.compute_train_example(batch_data)
      print('train_example: %s', train_example)
      # TODO(haoyugao): connect to train logic

    await producer_task
