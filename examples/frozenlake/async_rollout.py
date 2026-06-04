
import asyncio
from pprint import pprint
from typing import Any, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
from transformers import AutoTokenizer


from tunix.cli.utils import data as data_lib
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
from tunix.sft import sharding_utils
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
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = None

# MODEL_VERSION = "google/gemma-4-26B-A4B-it"
MODEL_VERSION = "google/gemma-4-e2b-it"

mesh = jax.sharding.Mesh(
    np.asarray(jax.local_devices()).reshape(1, 4), ("fsdp", "tp")
)

# config = model_lib.ModelConfig.gemma4_26b_a4b()
config = model_lib.ModelConfig.gemma4_e2b()

config.dtype = jnp.bfloat16
config.param_dtype = jnp.bfloat16
config.use_flash_attention = True
config.flash_attention_block_size = 256

from huggingface_hub import snapshot_download

MODEL_PATH = snapshot_download(repo_id=MODEL_VERSION, max_workers=16)
# MODEL_PATH = "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/6e6f6edea8c52db2094dca3086e4b963a0034dfc"
# MODEL_PATH = "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db"
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
    "rollout_vllm_logprobs_mode": "raw_logprobs",
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
        "hf_overrides": {
            "final_logit_softcapping": 30.0,
            "text_config": {
                "final_logit_softcapping": 30.0,
            },
        },
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

original_model_fn = rl_cluster.rollout._sampler._model_runner.model_fn
captured_arguments = None

def patched_model_fn(*args, **kwargs):
  global captured_arguments
  if captured_arguments is None:
    captured_arguments = args
  return original_model_fn(*args, **kwargs)

rl_cluster.rollout._sampler._model_runner.model_fn = patched_model_fn

CHAT_PARSER = parser.Gemma4ChatTemplateParser(tokenizer, enable_thinking=False, strip_past_thinking=False)

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
TRAIN_DATA_PATH = os.path.join("/mnt/disks/linchai-data/gemma4_flax/workspace/tunix_gemma4_2b/tunix/examples/frozenlake/data/frozenlake/train.parquet")
TEST_DATA_PATH = os.path.join("/mnt/disks/linchai-data/gemma4_flax/workspace/tunix_gemma4_2b/tunix/examples/frozenlake/data/frozenlake/test.parquet")

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
      max_steps=8,
  )
  return agent, env


# %% [markdown]
# ## Main Execution Loop
#
# Run the `RolloutOrchestrator` to collect trajectories asynchronously.

def unbatch_inputs(batched_dataset):
  for batch in batched_dataset:
    keys = list(batch.keys())
    if not keys:
      continue
    batch_size = len(batch[keys[0]])
    for i in range(batch_size):
      single_input = {}
      for k in keys:
        val = batch[k]
        if isinstance(val, (np.ndarray, list, tuple, jax.Array)):
          single_input[k] = val[i]
        else:
          single_input[k] = val
      yield single_input

def compare_layers(vllm_model, trainer_model, captured_args):
  print("=" * 40 + " START LAYER-BY-LAYER COMPARISON " + "=" * 40)
  
  state_leaves = captured_args[0]
  kv_caches = rl_cluster.rollout._sampler._model_runner.kv_caches
  input_ids = captured_args[2]
  attn_metadata = captured_args[3]
  inputs_embeds = captured_args[4]
  input_positions = captured_args[5]
  layer_name_to_kvcache_index = dict(captured_args[6])

  gemma4_model = vllm_model.model
  if inputs_embeds is not None:
    x_vllm = inputs_embeds
  else:
    x_vllm = gemma4_model.embed_tokens(input_ids) * gemma4_model.embedding_scale

  per_layer_inputs_vllm = gemma4_model.compute_per_layer_inputs(
      input_ids, x_vllm, is_multimodal=None
  )

  tokens_trainer = sharding_utils.shard_input(
      input_ids[None, :], ("fsdp",)
  )
  positions_trainer = sharding_utils.shard_input(
      input_positions[None, :], ("fsdp",)
  )

  x_trainer = trainer_model.embedder.encode(tokens_trainer)

  if trainer_model.config.per_layer_input_dim > 0:
    per_layer_inputs_trainer = trainer_model.embedder.encode_per_layer_input(
        x_trainer, tokens_trainer
    )
  else:
    per_layer_inputs_trainer = None

  x_vllm_np = np.asarray(x_vllm)
  x_trainer_np = np.asarray(x_trainer[0])
  emb_diff = np.abs(x_vllm_np - x_trainer_np)
  print(
      f"Embedding Out | Mean Abs Diff: {float(emb_diff.mean()):.6e} | Max Abs"
      f" Diff: {float(emb_diff.max()):.6e}"
  )

  @nnx.jit
  def run_trainer_layer(layer, x, pos, cache, mask, pli):
    return layer(x, pos, cache, mask, per_layer_input=pli)

  for i in range(len(gemma4_model.layers)):
    vllm_layer = gemma4_model.layers[i]
    layer_name = f"layer.{i}"
    cache_idx = layer_name_to_kvcache_index.get(layer_name, i)
    kv_cache = kv_caches[cache_idx]
    layer_per_input_vllm = (
        per_layer_inputs_vllm[:, i, :]
        if per_layer_inputs_vllm is not None
        else None
    )

    _, x_vllm, _ = vllm_layer(
        kv_cache,
        positions_trainer,
        x_vllm,
        None,
        per_layer_input=layer_per_input_vllm,
    )

    trainer_layer = trainer_model.layers[i]
    seq_len = tokens_trainer.shape[1]
    attn_mask_trainer = (
        jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
    )

    _, x_trainer, _ = run_trainer_layer(
        trainer_layer,
        x_trainer,
        positions_trainer,
        None,
        attn_mask_trainer,
        per_layer_inputs_trainer[:, :, i, :]
        if per_layer_inputs_trainer is not None
        else None,
    )

    x_vllm_np = np.asarray(x_vllm)
    x_trainer_np = np.asarray(x_trainer[0])
    diff = np.abs(x_vllm_np - x_trainer_np)
    print(
        f"Layer {i:2d} Out   | Mean Abs Diff: {float(diff.mean()):.6e} | Max"
        f" Abs Diff: {float(diff.max()):.6e}"
    )

  x_vllm_normed = gemma4_model.norm(x_vllm)
  vllm_logits = vllm_model.compute_logits(x_vllm_normed)

  trainer_normed = trainer_model.final_norm(x_trainer)
  trainer_logits = (
      trainer_model.embedder.decode(trainer_normed).astype(jnp.float32)
  )
  if trainer_model.config.final_logit_softcap is not None:
    trainer_logits /= trainer_model.config.final_logit_softcap
    trainer_logits = (
        jnp.tanh(trainer_logits) * trainer_model.config.final_logit_softcap
    )

  vllm_logits_np = np.asarray(vllm_logits)
  trainer_logits_np = np.asarray(trainer_logits[0])
  logits_diff = np.abs(vllm_logits_np - trainer_logits_np)
  print(
      f"Final Logits  | Mean Abs Diff: {float(logits_diff.mean()):.6e} | Max"
      f" Abs Diff: {float(logits_diff.max()):.6e}"
  )
  print("=" * 45 + " END LAYER COMPARISON " + "=" * 45)


# %%
async def main():
  """Runs the rollout orchestrator."""
  BATCH_SIZE = 8
  NUM_BATCHES = 1
  MAX_PROMPT_LENGTH = 2048
  train_ds, _ = create_datasets()
  train_ds, _ = data_lib.post_init_dataset(
    train_ds,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=1.0,
    num_epochs=1,
  )
  pairs = [
      make_pair(input, pair_index=i)
      for i, input in enumerate(unbatch_inputs(train_ds))
  ]

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
      max_concurrency=1,
  )

  producer_task = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs,
          group_size=1,
          group_key_fn=lambda i, env, traj: i,
          collect_mode="Token",
      )
  )

  # Yield control to allow producer initialization
  await asyncio.sleep(0)

  # Ensure anchor policy state is initialized for trainer logp recomputation
  rl_cluster.sync_weights()

  all_rollout_logps = []
  all_trainer_logps = []
  all_completion_masks = []

  try:
    async for batch in orchestrator.yield_batches(batch_size=BATCH_SIZE):
      print("=" * 120)
      print(f"Got batch of size {len(batch)}")

      padded_prompts = []
      padded_completions = []
      padded_completion_masks = []
      padded_old_logprobs_list = []

      for item in batch:
        prompt_tokens = item.traj.get("prompt_tokens")
        completion_tokens = item.traj.get("conversation_tokens")
        completion_mask = item.traj.get("conversation_masks")
        old_logprobs = item.traj.get("old_logprobs")
        conversation_text = item.traj.get("conversation_text")

        pad_value = rl_cluster.rollout.pad_id()
        eos_value = rl_cluster.rollout.eos_id()

        # 1. Compare initial turn prompt tokens in rollout vs trainer
        init_messages = []
        if conversation_text:
          for msg in conversation_text:
            if msg["role"] == "assistant":
              break
            init_messages.append(msg)

        python_prompt_tokens, _ = utils.tokenize_and_generate_masks(
            init_messages,
            tokenizer=tokenizer_adapter.TokenizerAdapter(tokenizer),
            parser=CHAT_PARSER,
            contains_first_msg=True,
            contains_generation_msg=True,
        )

        padded_prompt, padded_completion, _ = (
            utils.pad_prompt_and_completion(
                prompt_tokens,
                completion_tokens,
                MAX_PROMPT_LENGTH,
                TOTAL_GENERATION_STEPS,
                pad_value,
            )
        )

        print("\n--- 1. Initial Turn Prompt Tokens Comparison ---")
        unpadded_rollout_prompt = prompt_tokens[prompt_tokens != pad_value]
        print(
            f"Python Initial Prompt Tokens (unpadded len={len(python_prompt_tokens)}):"
            f" {python_prompt_tokens[:20]}..."
        )
        print(
            f"Rollout Worker Prompt Tokens (unpadded len={len(unpadded_rollout_prompt)}):"
            f" {unpadded_rollout_prompt[:20]}..."
        )
        print(
            f"Trainer Padded Prompt (len={len(padded_prompt)}):"
            f" {padded_prompt[:20]}..."
        )
        prompt_match = np.array_equal(
            python_prompt_tokens, unpadded_rollout_prompt
        )
        print(
            "Does Python C-parser match Rollout Worker exactly? "
            f"{'YES' if prompt_match else 'NO (Mismatch!)'}"
        )

        if not prompt_match:
          print(
              "\n" + "-" * 40 + " DEEP PROMPT MISMATCH DIAGNOSTIC " + "-" * 40
          )
          print(f"Python Unpadded Length:  {len(python_prompt_tokens)}")
          print(f"Rollout Unpadded Length: {len(unpadded_rollout_prompt)}")

          min_len = min(len(python_prompt_tokens), len(unpadded_rollout_prompt))
          mismatches = []
          for i in range(min_len):
            p_tok = python_prompt_tokens[i]
            r_tok = unpadded_rollout_prompt[i]
            if p_tok != r_tok:
              mismatches.append((i, p_tok, r_tok))

          print(
              "Total mismatched tokens in overlapping range (0 to"
              f" {min_len}): {len(mismatches)}"
          )
          if mismatches:
            print(
                "\nDetailed Mismatch Breakdown (showing up to first 20"
                " mismatched positions):"
            )
            print(
                f"{'Pos':<5} | {'Py ID':<7} | {'Py Decoded':<25} |"
                f" {'Rollout ID':<10} | {'Rollout Decoded':<25}"
            )
            print("-" * 82)
            for pos, p_tok, r_tok in mismatches[:20]:
              try:
                p_str = repr(tokenizer.decode([p_tok]))
              except Exception:
                p_str = "<error>"
              try:
                r_str = repr(tokenizer.decode([r_tok]))
              except Exception:
                r_str = "<error>"
              print(
                  f"{pos:<5} | {p_tok:<7} | {p_str:<25} | {r_tok:<10} |"
                  f" {r_str:<25}"
              )

          if len(python_prompt_tokens) > min_len:
            print(
                f"\nPython prompt has {len(python_prompt_tokens) - min_len}"
                " extra trailing tokens:"
            )
            extra_py = python_prompt_tokens[min_len : min_len + 10]
            try:
              extra_py_str = repr(tokenizer.decode(extra_py))
            except Exception:
              extra_py_str = "<error>"
            print(f"  IDs: {extra_py} -> Decoded: {extra_py_str}")

          if len(unpadded_rollout_prompt) > min_len:
            print(
                f"\nRollout worker prompt has {len(unpadded_rollout_prompt) - min_len}"
                " extra trailing tokens:"
            )
            extra_r = unpadded_rollout_prompt[min_len : min_len + 10]
            try:
              extra_r_str = repr(tokenizer.decode(extra_r))
            except Exception:
              extra_r_str = "<error>"
            print(f"  IDs: {extra_r} -> Decoded: {extra_r_str}")
          print("-" * 113 + "\n")
        padded_completion_mask = utils.right_pad(
            completion_mask, TOTAL_GENERATION_STEPS, 0
        )[:TOTAL_GENERATION_STEPS]

        if old_logprobs is not None:
          padded_old_logprobs = utils.right_pad(
              old_logprobs,
              length=TOTAL_GENERATION_STEPS,
              pad=0.0,
              dtype=old_logprobs.dtype,
          )[:TOTAL_GENERATION_STEPS]
        else:
          padded_old_logprobs = np.zeros(TOTAL_GENERATION_STEPS, dtype=np.float32)

        padded_prompts.append(padded_prompt)
        padded_completions.append(padded_completion[:TOTAL_GENERATION_STEPS])
        padded_completion_masks.append(padded_completion_mask)
        padded_old_logprobs_list.append(padded_old_logprobs)

      prompt_ids = jnp.asarray(padded_prompts)
      completion_ids = jnp.asarray(padded_completions)
      completion_mask_arr = jnp.asarray(padded_completion_masks)
      rollout_per_token_logps = jnp.asarray(padded_old_logprobs_list)

      attn_completion_mask = (completion_ids != pad_value).astype(jnp.int32)
      trainer_per_token_logps = rl_cluster.get_actor_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
          completion_mask=attn_completion_mask,
          temperature=1.0,
      )

      mask = completion_mask_arr.astype(jnp.bool_)
      mask_f = mask.astype(jnp.float32)
      mask_sum = jnp.maximum(mask_f.sum(), 1.0)
      diff = jnp.abs(rollout_per_token_logps - trainer_per_token_logps)
      diff_mean = float((diff * mask_f).sum() / mask_sum)
      diff_max = float(jnp.where(mask, diff, 0.0).max())

      rp = jnp.exp(rollout_per_token_logps)
      tp = jnp.exp(trainer_per_token_logps)
      prob_diff = jnp.abs(rp - tp)
      prob_diff_mean = float((prob_diff * mask_f).sum() / mask_sum)
      prob_diff_max = float(jnp.where(mask, prob_diff, 0.0).max())

      # Extract first item details for token-level logging
      first_item = batch[0]
      prompt_tokens = first_item.traj.get("prompt_tokens")
      completion_tokens = first_item.traj.get("conversation_tokens")
      completion_mask = first_item.traj.get("conversation_masks")
      old_logprobs = first_item.traj.get("old_logprobs")
      print("Trajectory Conversation:")
      pprint(first_item.traj.get("conversation_text"), width=120)

      all_rollout_logps.append(np.array(rollout_per_token_logps))
      all_trainer_logps.append(np.array(trainer_per_token_logps))
      all_completion_masks.append(np.array(completion_mask_arr))

      rp_flat = rp.reshape(-1)
      tp_flat = tp.reshape(-1)
      mf = mask_f.reshape(-1)
      rp_mean = (rp_flat * mf).sum() / mask_sum
      tp_mean = (tp_flat * mf).sum() / mask_sum
      rp_d = (rp_flat - rp_mean) * mf
      tp_d = (tp_flat - tp_mean) * mf
      cov = (rp_d * tp_d).sum() / mask_sum
      rp_var = (rp_d * rp_d).sum() / mask_sum
      tp_var = (tp_d * tp_d).sum() / mask_sum
      pearson = float(cov / jnp.sqrt(jnp.maximum(rp_var * tp_var, 1e-12)))

      print(
          f"sampler-trainer comparison: logp_diff_mean={diff_mean:.5f}, logp_diff_max={diff_max:.5f}, "
          f"prob_diff_mean={prob_diff_mean:.5f}, prob_diff_max={prob_diff_max:.5f}, pearson={pearson:.5f}"
      )
      
      print("\n" + "=" * 60 + " Token Alignment Debug Info " + "=" * 60)
      print(f"Prompt Tokens Length: {len(prompt_tokens)}")
      print(f"Completion Tokens Length: {len(completion_tokens)}")
      print(f"Completion Mask Length: {len(completion_mask)}")
      if old_logprobs is not None:
        print(f"Old Logprobs Length: {len(old_logprobs)}")

      active_indices = [i for i, m in enumerate(completion_mask) if m > 0]
      print(f"Total Active (Assistant) Tokens in Trajectory: {len(active_indices)}")
      print("Sample Active Tokens (Index, TokenID, Decoded, Rollout Logp, Trainer Logp, Diff):")

      idx_to_print = active_indices[:20] + active_indices[-5:] if len(active_indices) > 25 else active_indices

      rp_np = np.array(rollout_per_token_logps[0])
      tp_np = np.array(trainer_per_token_logps[0])

      for idx in idx_to_print:
        tok_id = completion_tokens[idx]
        try:
          decoded_tok = repr(tokenizer.decode([tok_id]))
        except Exception:
          decoded_tok = "<unknown>"
        r_logp = rp_np[idx]
        t_logp = tp_np[idx]
        diff_val = abs(r_logp - t_logp)
        print(f"  Pos {idx:4d} | ID {tok_id:6d} | {decoded_tok:15s} | Rollout: {r_logp:8.4f} | Trainer: {t_logp:8.4f} | Diff: {diff_val:8.4f}")
      print("=" * 148)

      print("\n" + "=" * 60 + " Deep Token & Logit Debug Info " + "=" * 60)
      p_toks = prompt_tokens[-20:] if len(prompt_tokens) > 20 else prompt_tokens
      c_toks = completion_tokens[:20] if len(completion_tokens) > 20 else completion_tokens
      print("Last 20 Prompt Tokens (ID -> Decoded):")
      for pt in p_toks:
        print(f"  {pt:6d} -> {repr(tokenizer.decode([pt]))}")
      print("First 20 Completion Tokens (ID -> Decoded):")
      for ct in c_toks:
        print(f"  {ct:6d} -> {repr(tokenizer.decode([ct]))}")


      if len(active_indices) > 0:
        rp_active = rp_np[active_indices]
        tp_active = tp_np[active_indices]
        print(f"Rollout Active Logps stats: min={rp_active.min():.4f}, max={rp_active.max():.4f}, mean={rp_active.mean():.4f}")
        print(f"Trainer Active Logps stats: min={tp_active.min():.4f}, max={tp_active.max():.4f}, mean={tp_active.mean():.4f}")

        print("\nTop 20 Largest Logprob Discrepancies (with Context Window):")
        diff_active = np.abs(rp_active - tp_active)
        top_diff_idx = np.argsort(diff_active)[::-1][:20]
        unpadded_p = prompt_tokens[prompt_tokens != pad_value]
        full_seq = np.concatenate([unpadded_p, completion_tokens], axis=0)
        prompt_len = len(unpadded_p)

        for rank, idx_in_active in enumerate(top_diff_idx):
          orig_pos = active_indices[idx_in_active]
          tok_id = completion_tokens[orig_pos]
          try:
            decoded_tok = repr(tokenizer.decode([tok_id]))
          except Exception:
            decoded_tok = "<unknown>"
          r_logp = rp_active[idx_in_active]
          t_logp = tp_active[idx_in_active]
          d_val = diff_active[idx_in_active]

          global_pos = prompt_len + orig_pos
          pre_ctx = full_seq[max(0, global_pos - 15) : global_pos]
          post_ctx = full_seq[
              global_pos + 1 : min(len(full_seq), global_pos + 10)
          ]

          try:
            pre_str = tokenizer.decode(pre_ctx)
          except Exception:
            pre_str = "<error>"
          try:
            post_str = tokenizer.decode(post_ctx)
          except Exception:
            post_str = "<error>"

          print(
              f"  Rank {rank+1:2d} | Pos {orig_pos:4d} | ID {tok_id:6d} |"
              f" {decoded_tok:15s} | Rollout: {r_logp:8.4f} | Trainer:"
              f" {t_logp:8.4f} | Diff: {d_val:8.4f}"
          )
          print(
              f"    Context: ...{repr(pre_str)} ---> [{decoded_tok}] <---"
              f" {repr(post_str)}...\n"
          )
        print("=" * 151 + "\n")
      else:
        print("No active (assistant) tokens in the first trajectory of the batch.\n")

    # After the loop completes, compute global stats over all accumulated batches
    if all_rollout_logps:
      print("\n" + "X" * 50 + " GLOBAL SAMPLER-VS-TRAINER ANALYSIS (ALL BATCHES) " + "X" * 50)
      global_rp_logps = np.concatenate(all_rollout_logps, axis=0)
      global_tp_logps = np.concatenate(all_trainer_logps, axis=0)
      global_masks = np.concatenate(all_completion_masks, axis=0)

      global_mask = global_masks.astype(bool)
      global_mask_f = global_mask.astype(np.float32)
      global_mask_sum = np.maximum(global_mask_f.sum(), 1.0)

      global_diff = np.abs(global_rp_logps - global_tp_logps)
      global_diff_mean = float((global_diff * global_mask_f).sum() / global_mask_sum)
      global_diff_max = float(np.where(global_mask, global_diff, 0.0).max())

      global_rp = np.exp(global_rp_logps)
      global_tp = np.exp(global_tp_logps)
      global_prob_diff = np.abs(global_rp - global_tp)
      global_prob_diff_mean = float((global_prob_diff * global_mask_f).sum() / global_mask_sum)
      global_prob_diff_max = float(np.where(global_mask, global_prob_diff, 0.0).max())

      # Global Pearson correlation computation
      global_rp_flat = global_rp.flatten()
      global_tp_flat = global_tp.flatten()
      global_mf = global_mask_f.flatten()

      g_rp_mean = (global_rp_flat * global_mf).sum() / global_mask_sum
      g_tp_mean = (global_tp_flat * global_mf).sum() / global_mask_sum

      g_rp_d = (global_rp_flat - g_rp_mean) * global_mf
      g_tp_d = (global_tp_flat - g_tp_mean) * global_mf

      global_cov = (g_rp_d * g_tp_d).sum() / global_mask_sum
      global_rp_var = (g_rp_d * g_rp_d).sum() / global_mask_sum
      global_tp_var = (g_tp_d * g_tp_d).sum() / global_mask_sum
      global_pearson = float(global_cov / np.sqrt(np.maximum(global_rp_var * global_tp_var, 1e-12)))

      print(
          f"GLOBAL COMPARISON SUMMARY:\n"
          f"  Total Sequences Analyzed: {global_rp_logps.shape[0]}\n"
          f"  Total Unmasked Tokens:    {int(global_mask_sum)}\n"
          f"  Global logp_diff_mean:    {global_diff_mean:.6f}\n"
          f"  Global logp_diff_max:     {global_diff_max:.6f}\n"
          f"  Global prob_diff_mean:    {global_prob_diff_mean:.6f}\n"
          f"  Global prob_diff_max:     {global_prob_diff_max:.6f}\n"
          f"  Global Pearson Correlation: {global_pearson:.6f}"
      )
      print("X" * 149 + "\n")

    if captured_arguments is not None:
      with rl_cluster._get_mesh_and_logical_axis_rules_cm(rl_cluster_lib.Role.ACTOR):
        compare_layers(
            vllm_model=rl_cluster.rollout._sampler._model_runner.model,
            trainer_model=rl_cluster.actor_trainer.model,
            captured_args=captured_arguments,
        )

  finally:
    await producer_task


if __name__ == "__main__":
  asyncio.run(main())