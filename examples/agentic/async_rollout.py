"""Example of an agentic rollout using asynchronous environment steps."""

import optax
from orbax import checkpoint as ocp

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.rl.agentic.agents import tool_agent
from tunix.rl.agentic.environments import tool_environment
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.tools import calculator_tool
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.pipeline import rollout_orchestrator
from flax import nnx
import jax
import numpy as np
from tunix.models.qwen3 import params
from tunix.models.qwen3 import model

TRAIN_FRACTION = 1.0

# === Generation ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0  # implies we don't do nucleus sampling
TOP_K = 50

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9  # Adam beta1
B2 = 0.99  # Adam beta2
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# ====== Checkpoint saving ======
# CKPT_DIR = (
#      "your checkpoint directory here"
# )
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=20
)

mesh = jax.sharding.Mesh(
    np.asarray(jax.local_devices())[:4].reshape(1, 4), ('fsdp', 'tp')
)
MODEL_CP_PATH = '/cns/gg-d/home/qwix-dev/qwen3/torch/0.6b/'

config = model.ModelConfig.qwen3_0p6b()  # pick corresponding config based on model version
qwen3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(qwen3)

# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )
  
# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=None,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)
chat_parser = parser.QwenChatTemplateParser(tokenizer)

# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen3,
    reference=qwen3,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

ToolAgent = tool_agent.ToolAgent
ToolEnvironment = tool_environment.ToolEnvironment
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine
CalculatorTool = calculator_tool.CalculatorTool
is_two_reward = reward.is_two_reward

from typing import Any
def inference(chat_lists, env: Any = None):
  result = rl_cluster.generate(
      prompts=chat_lists,
      apply_chat_template=True,
      mode=rl_cluster_lib.Mode.TRAIN,
  )

  return result.text[0]

def make_pair(question: str):
    agent = ToolAgent(
        system_prompt=(
            "You are qwen tool assistant. You must use the tools below to solve problems. Return the final answer in this manner: The answer is ... "
        ),
        tool_parser_name="qwen",
        tool_map={"calculator": CalculatorTool},
    )


    env = ToolEnvironment(
        task={"question": question},
        tool_map={"calculator": CalculatorTool},
        reward_fn=reward.dummy_reward,
        max_steps=3,
    )
    return agent, env


questions = [
        "845+567=?",
        "2109-783=?",
        "45*62=?",
        "1536/16=?",
        "891+109=?",
        "3000-1578=?",
        "105*11=?",
        "2400/25=?",
        "728/8=?",
        "67*15=?",
        "4321+5678=?",
        "9876-1234=?",
        "125*80=?",
        "1024/32=?",
        "345+155=?",
        "5000-4999=?",
        "303*3=?",
        "600/12=?",
        "555+445=?",
        "900-275=?"
    ]

# ---- Main coroutine with Orchestrator ----

from tunix.rl.agentic import utils as agentic_utils
async def main():
    # agent-env pairs
    pairs = [make_pair(q) for q in questions]
    engine_defaults = dict(
        model_call=inference,
        final_reward_fn=reward.calculate_reward,
        max_steps=10,
        gamma=1.0,
        timeout=10.0,
        tokenizer=tokenizer,
        chat_parser=chat_parser,
    )
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        engine_cls=TrajectoryCollectEngine,
        engine_defaults=engine_defaults,
        max_concurrency=1,
        rollout_sync_lock=agentic_utils.RolloutSyncLock(),
    )

    def pair_generator():
      for pair in pairs:
        yield pair

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=1,
            group_key = lambda i, env, traj: i,
            num_episodes=1,
        )
    )
    await asyncio.sleep(0)

    batches = []
    async for batch in orchestrator.yield_batches(batch_size=1):
      batches.append(batch)
    await producer_task

    all_items = []
    for batch in batches:
      all_items.extend(batch)

    for item in all_items:
      print(f"[pair {item.pair_index}] question={pairs[item.pair_index][1].task['question']}")
      print("Trajectory:")
      from pprint import pprint
      pprint(item.traj, width=120)
      
import asyncio

# Entry point
if __name__ == "__main__":
    asyncio.run(main())      