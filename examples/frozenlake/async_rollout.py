
import asyncio
from pprint import pprint
from typing import Any, Sequence

from flax import nnx
import jax
import numpy as np
import optax
from orbax import checkpoint as ocp
from transformers import AutoTokenizer


from tunix.generate import tokenizer_adapter
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import tool_agent
from tunix.rl.agentic.environments import tool_environment
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.tools import calculator_tool
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.rollout import base_rollout
from examples.frozenlake.agent import FrozenLakeAgent
from examples.frozenlake.env import FrozenLakeEnv


# %% [markdown]
# ## Configuration
#
# Hyperparameters for generation, training, and the environment.

# %%
# Generation Config
MAX_PROMPT_LENGTH = 2048
TOTAL_GENERATION_STEPS = 2048
TEMPERATURE = 0.7
TOP_P = 1.0
TOP_K = None

MODEL_VERSION = "google/gemma-4-E2B-it"

mesh = jax.sharding.Mesh(
    np.asarray(jax.local_devices()).reshape(1, 8), ("fsdp", "tp")
)

config = model_lib.ModelConfig.gemma4_e2b()

from huggingface_hub import snapshot_download

MODEL_PATH = snapshot_download(repo_id=MODEL_VERSION, max_workers=16)
print(f"{MODEL_PATH=}")
gemma4 = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, mesh)

optimizer = optax.adamw(learning_rate=1e-6)


base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "return_logprobs": True,
    "max_tokens_to_generate": TOTAL_GENERATION_STEPS,
}

vllm_rollout_dict = {
    # vllm-tpu specific configs
    "rollout_vllm_model_version": MODEL_VERSION,
    "rollout_vllm_hbm_utilization": 0.33,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_enable_dp_attention": True,
    "rollout_vllm_async_scheduling": True,
    "rollout_vllm_init_with_random_weights": False,
    "rollout_vllm_max_num_seqs": 16,
    "rollout_vllm_max_num_batched_tokens": 4096,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
    },
}
rollout_engine_config = base_rollout.RolloutConfig(
    **base_rollout_dict, **vllm_rollout_dict
)
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="vllm",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=5,
    ),
    rollout_config=rollout_engine_config,
    # rollout_config=base_rollout.RolloutConfig(
    #     max_tokens_to_generate=TOTAL_GENERATION_STEPS,
    #     max_prompt_length=MAX_PROMPT_LENGTH,
    #     kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    #     top_k=TOP_K,
    # ),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=gemma4,
    reference=gemma4,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

CHAT_PARSER = parser.Gemma4ChatTemplateParser(tokenizer, enable_thinking=False)

# Constants for tools
TOOL_AGENT_CLS = tool_agent.ToolAgent
TOOL_ENV_CLS = tool_environment.ToolEnvironment
TRAJ_ENGINE_CLS = trajectory_collect_engine.TrajectoryCollectEngine
CALCULATOR_TOOL = calculator_tool.CalculatorTool


# %% [markdown]
# ## Define Tasks and Agents
#
# Prepare the math questions and helper functions to create agent-environment
# pairs.

# %%


def inference(prompt: Sequence[str], env: Any = None, **kwargs: Any) -> str:
  """Wrapper for RL cluster generation."""
  chat_lists = CHAT_PARSER.parse(
      messages=prompt,
      add_generation_prompt=True,
      is_first_msg=True,  # no op if system msg is populated in reset
  )
  result = rl_cluster.generate(
      prompts=[chat_lists],
      apply_chat_template=False,
      mode=rl_cluster_lib.Mode.TRAIN,
      max_generation_steps=TOTAL_GENERATION_STEPS,
  )
  return result

import os
TRAIN_DATA_PATH = os.path.join("/home/linchai_google_com/colincai-mc/tunix/examples/frozenlake/data/frozenlake/train.parquet")
TEST_DATA_PATH = os.path.join("/home/linchai_google_com/colincai-mc/tunix/examples/frozenlake/data/frozenlake/test.parquet")

import grain
from google.cloud import storage
import pandas as pd
import fsspec
import datasets as datasets_lib

Dataset = datasets_lib.Dataset
file_open = fsspec.open

def create_datasets(
    train_ds_path: str = TRAIN_DATA_PATH,
    test_ds_path: str = TEST_DATA_PATH,
):
  with file_open(train_ds_path) as train_f, file_open(
      test_ds_path, "rb"
  ) as test_f:
    train_df = pd.read_parquet(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df)
  test_ds = Dataset.from_pandas(test_df)

  def process_item(item):
    item["prompts"] = ""
    return item

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds

from typing import Dict, List
TrainingInputT = Dict[str, List[str]]
def make_pair(
    input: TrainingInputT,
    group_id: int | None = None,
    pair_index: int | None = None,
) -> tuple[tool_agent.ToolAgent, tool_environment.ToolEnvironment]:
  """Creates an agent-environment pair."""
  agent = FrozenLakeAgent()

  env = FrozenLakeEnv(
      entry=input,
      group_id=group_id,
      pair_index=pair_index,
      max_steps=20,
  )
  return agent, env


# %% [markdown]
# ## Main Execution Loop
#
# Run the `RolloutOrchestrator` to collect trajectories asynchronously.

# %%
async def main():
  """Runs the rollout orchestrator."""
  train_ds, _ = create_datasets()
  train_ds = train_ds.shuffle(seed=42)[:8]
  pairs = [make_pair(input, pair_index=i) for i, input in enumerate(train_ds)]

  rollout_sync_lock = utils.RolloutSyncLock()
  orchestrator = rollout_orchestrator.RolloutOrchestrator(
      rollout_sync_lock=rollout_sync_lock,
      engine_cls=TRAJ_ENGINE_CLS,
      engine_kwargs=dict(
          model_call=inference,
          max_response_length=TOTAL_GENERATION_STEPS,
          gamma=1.0,
          timeout=180.0,
          tokenizer=tokenizer_adapter.TokenizerAdapter(tokenizer),
          chat_parser=CHAT_PARSER,
      ),
      max_concurrency=4,
  )

  producer_task = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs,
          group_size=1,
          group_key_fn=lambda i, env, traj: i,
          collect_mode="Conversation",
      )
  )

  # Yield control to allow producer initialization
  await asyncio.sleep(0)

  try:
    async for batch in orchestrator.yield_batches(batch_size=1):
      print("=" * 120)
      print(f"Got batch of size {len(batch)}")
      for item in batch:
        print("Trajectory:")
        pprint(item.traj, width=120)
  finally:
    await producer_task


if __name__ == "__main__":
  asyncio.run(main())