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
from tunix.models.qwen2 import model as model_lib
from tunix.models.qwen2 import params as params_lib
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
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import task_environment

# %% [markdown]
# ## Configuration

MAX_PROMPT_LENGTH = 1024
TOTAL_GENERATION_STEPS = 8192
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = None

MODEL_VERSION = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

rollout_mesh = jax.sharding.Mesh(
    np.asarray(jax.local_devices()[:4]).reshape(1, 4), ("fsdp", "tp")
)

trainer_mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[4:]).reshape(4, 1), ("fsdp", "tp"))

config = model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()

config.dtype = jnp.float32
config.param_dtype = jnp.float32
config.use_flash_attention = True
config.flash_attention_block_size = 256

from huggingface_hub import snapshot_download

MODEL_PATH = snapshot_download(repo_id=MODEL_VERSION, max_workers=16)
print(f"{MODEL_PATH=}")
qwen2 = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, trainer_mesh)

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
    "rollout_vllm_model_version": MODEL_VERSION,
    "rollout_vllm_hbm_utilization": 0.4,
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
    "rollout_vllm_sampling_kwargs": {
        "skip_special_tokens": False,
    },
}
rollout_engine_config = base_rollout.RolloutConfig(
    **base_rollout_dict, **vllm_rollout_dict
)
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: trainer_mesh,
        rl_cluster_lib.Role.REFERENCE: trainer_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    },
    rollout_engine="vllm",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=5,
    ),
    rollout_config=rollout_engine_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=qwen2,
    reference=qwen2,
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

CHAT_PARSER = parser.DeepseekQwenChatTemplateParser(tokenizer)

TOOL_AGENT_CLS = tool_agent.ToolAgent
TOOL_ENV_CLS = tool_environment.ToolEnvironment
TRAJ_ENGINE_CLS = trajectory_collect_engine.TrajectoryCollectEngine
CALCULATOR_TOOL = calculator_tool.CalculatorTool

def inference(prompt: Sequence[str], env: Any = None, **kwargs: Any) -> str:
  chat_lists = CHAT_PARSER.parse(
      messages=prompt,
      add_generation_prompt=True,
      is_first_msg=True,
  )
  print(f"{chat_lists = }")
  result = rl_cluster.generate(
      prompts=[prompt],
      apply_chat_template=True,
      mode=rl_cluster_lib.Mode.TRAIN,
      max_generation_steps=TOTAL_GENERATION_STEPS,
  )
  return result

import os
TRAIN_DATA_PATH = "gs://tunix/data/DeepScaleR-Preview-Dataset/deepscaler.json"
TEST_DATA_PATH = "gs://tunix/data/HuggingFaceH4/aime_2024/train-00000-of-00001.parquet"

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
  def preprocess_fn(example, index):
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  with file_open(train_ds_path) as train_f, file_open(
      test_ds_path, "rb"
  ) as test_f:
    train_df = pd.read_json(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True).shuffle(123)
  test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True).shuffle(123)

  def process_item(item):
    question = item["question"]
    answer = item["answer"]

    instruction = (
        "Let's think step by step, and put your final answer within \\boxed{}."
    )
    prompt = f"{question} {instruction}"

    return {
        "prompts": prompt,
        "question": question,
        "answer": answer,
    }

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds

from typing import Dict, List
TrainingInputT = Dict[str, List[str]]
def make_pair(
    input: TrainingInputT,
    group_id: int | None = None,
    pair_index: int | None = None,
) -> tuple[model_agent.ModelAgent, task_environment.TaskEnvironment]:
  agent = model_agent.ModelAgent(system_prompt="")

  env = task_environment.TaskEnvironment(
      single_example=input,
      group_id=group_id,
      pair_index=pair_index,
  )
  return agent, env

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
  num_active_tokens = int(np.sum(attn_metadata.seq_lens))

  qwen2_model = vllm_model.model

  vllm_q0_w = getattr(qwen2_model.layers[0].self_attn.q_proj, "weight", None)
  if vllm_q0_w is not None:
    if hasattr(vllm_q0_w, "value"):
      vllm_q0_w = vllm_q0_w.value
    trainer_q0_w = trainer_model.layers[0].attn.q_proj.w.value if hasattr(trainer_model.layers[0].attn.q_proj.w, "value") else trainer_model.layers[0].attn.q_proj.w
    # vLLM is typically shape (N*H, D) -> (1536, 1536). Trainer is (D, N, H) -> (1536, 12, 128)
    vllm_q0_w_np = np.asarray(vllm_q0_w)
    trainer_q0_w_np = np.asarray(trainer_q0_w)
    if vllm_q0_w_np.ndim == 2:
      # Try transposing vLLM weights to match Trainer weights shape logic
      vllm_q0_reshaped = vllm_q0_w_np.reshape(12, 128, 1536).transpose(2, 0, 1)
      diff_w = np.abs(vllm_q0_reshaped - trainer_q0_w_np)
      print(f"Layer 0 Q_Proj Weight | Mean Abs Diff: {float(diff_w.mean()):.6e} | Max Abs: {float(diff_w.max()):.6e}")
  
  if inputs_embeds is not None:
    x_vllm = inputs_embeds
  else:
    x_vllm = qwen2_model.embed_tokens(input_ids)

  fsdp_size = 4
  try:
      from jax.interpreters import pxla
      if pxla.thread_resources.env.physical_mesh and "fsdp" in pxla.thread_resources.env.physical_mesh.shape:
          fsdp_size = pxla.thread_resources.env.physical_mesh.shape["fsdp"]
  except Exception:
      pass

  tokens_trainer = sharding_utils.shard_input(
      jnp.tile(input_ids[None, :], (fsdp_size, 1)), ("fsdp",)
  )
  positions_trainer = sharding_utils.shard_input(
      jnp.tile(input_positions[None, :], (fsdp_size, 1)), ("fsdp",)
  )

  pos_vllm = np.asarray(captured_args[5])[:num_active_tokens]
  pos_trainer = np.asarray(positions_trainer)[0, :num_active_tokens]
  print(f"DEBUG positions vLLM: {pos_vllm[:20]}")
  print(f"DEBUG positions Trainer: {pos_trainer[:20]}")
  print(f"DEBUG positions max diff: {np.abs(pos_vllm - pos_trainer).max()}")

  x_trainer = trainer_model.embedder.encode(tokens_trainer)

  x_vllm_np = np.asarray(x_vllm)[:num_active_tokens]
  x_trainer_np = np.asarray(x_trainer[0])[:num_active_tokens]
  emb_diff = np.abs(x_vllm_np - x_trainer_np)
  print(
      f"Embedding Out | Mean Abs Diff: {float(emb_diff.mean()):.6e} | Max Abs"
      f" Diff: {float(emb_diff.max()):.6e}"
  )

  seq_len = tokens_trainer.shape[1]
  query_start_loc_np = np.asarray(attn_metadata.query_start_loc)
  segment_ids_np = np.zeros(seq_len, dtype=np.int32)
  for seq_idx in range(len(query_start_loc_np) - 1):
    start = query_start_loc_np[seq_idx]
    end = query_start_loc_np[seq_idx + 1]
    segment_ids_np[start:end] = seq_idx

  same_segment = segment_ids_np[:, None] == segment_ids_np[None, :]
  causal = np.tril(np.ones((seq_len, seq_len)))
  attn_mask_trainer = jnp.asarray(causal * same_segment)[None, None, :, :]
  attn_mask_trainer = jnp.tile(attn_mask_trainer, (fsdp_size, 1, 1, 1))

  segment_ids_trainer = sharding_utils.shard_input(
      jnp.tile(segment_ids_np[None, :], (fsdp_size, 1)), ("fsdp",)
  )

  sin, cos = model_lib._generate_pos_embeddings(
      positions_trainer, trainer_model.config.head_dim, trainer_model.config.rope_theta
  )
  sin, cos = sin.astype(x_trainer.dtype), cos.astype(x_trainer.dtype)

  @nnx.jit
  def run_trainer_layer(layer, x, cache, mask, sin_in, cos_in, segment_ids=None):
    return layer(
        x,
        cache,
        attn_mask=mask,
        sin=sin_in,
        cos=cos_in,
        segment_ids=segment_ids,
    )

  for i in range(len(qwen2_model.layers)):
    vllm_layer = qwen2_model.layers[i]
    layer_name = f"layer.{i}"
    cache_idx = layer_name_to_kvcache_index.get(layer_name, i)
    kv_cache = kv_caches[cache_idx]
    x_vllm_input = x_vllm
    x_trainer_input = x_trainer

    kv_cache, x_vllm = vllm_layer(
        kv_cache,
        x_vllm,
        attn_metadata,
    )

    trainer_layer = trainer_model.layers[i]
    _, x_trainer = run_trainer_layer(
        trainer_layer,
        x_trainer,
        None,
        attn_mask_trainer,
        sin,
        cos,
        segment_ids=segment_ids_trainer,
    )

    x_vllm_np = np.asarray(x_vllm)[:num_active_tokens]
    x_trainer_np = np.asarray(x_trainer[0])[:num_active_tokens]
    diff = np.abs(x_vllm_np - x_trainer_np)

    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    tok_idx = max_idx[0]
    dim_idx = max_idx[1]
    tok_id = int(input_ids[tok_idx])
    tok_str = tokenizer.decode([tok_id])

    flat_diff = diff.flatten()
    top_indices = np.argsort(flat_diff)[-3:][::-1]
    top_values = flat_diff[top_indices]
    top_tok_dims = [np.unravel_index(idx, diff.shape) for idx in top_indices]
    top_toks_info = []
    for t_d in top_tok_dims:
      t_idx = t_d[0]
      t_id = int(input_ids[t_idx])
      top_toks_info.append(f"({tokenizer.decode([t_id])!r}, idx={t_idx}, dim={t_d[1]})")

    vllm_val = float(x_vllm_np[tok_idx, dim_idx])
    trainer_val = float(x_trainer_np[tok_idx, dim_idx])
    rel_diff = abs(vllm_val - trainer_val) / max(abs(vllm_val), 1e-5)

    print(
        f"Layer {i:2d} Out   | Mean Abs Diff: {float(diff.mean()):.6e} | Max"
        f" Abs Diff: {float(diff.max()):.6e} at token {tok_str!r} (idx={tok_idx}, dim={dim_idx})"
    )
    print(f"            | Values at max diff: vLLM = {vllm_val:.4f}, Trainer = {trainer_val:.4f}, Rel Diff = {rel_diff:.4e}")
    print(f"            | Top-3 Diffs: {top_values} for tokens {top_toks_info}")

    if i == 0:
      @nnx.jit
      def run_vllm_layer0_steps(vllm_l, x_in, kv_c_in):
        norm_vllm = vllm_l.input_layernorm(x_in)
        q_vllm = vllm_l.self_attn.q_proj(norm_vllm)
        k_vllm = vllm_l.self_attn.k_proj(norm_vllm)
        v_vllm = vllm_l.self_attn.v_proj(norm_vllm)
        _, attn_vllm = vllm_l.self_attn(kv_c_in, norm_vllm, attn_metadata)
        post_attn_norm_vllm = vllm_l.post_attention_layernorm(x_in + attn_vllm)
        ffw_vllm = vllm_l.mlp(post_attn_norm_vllm)
        return norm_vllm, attn_vllm, post_attn_norm_vllm, ffw_vllm, q_vllm, k_vllm, v_vllm

      @nnx.jit
      def run_trainer_layer0_steps(trainer_l, x_in, sin_in, cos_in):
        norm_trainer = trainer_l.input_layernorm(x_in)
        
        q_trainer = trainer_l.attn.q_proj(norm_trainer)
        b, t, n, h = q_trainer.shape
        q_trainer = jnp.reshape(q_trainer, (b, t, n * h)) + trainer_l.attn.q_bias.astype(jnp.float32)
        q_trainer = jnp.reshape(q_trainer, (b, t, n, h))
        
        k_trainer = trainer_l.attn.k_proj(norm_trainer)
        _, s, k, h = k_trainer.shape
        k_trainer = jnp.reshape(k_trainer, (b, s, k * h)) + trainer_l.attn.k_bias.astype(jnp.float32)
        k_trainer = jnp.reshape(k_trainer, (b, s, k, h))

        v_trainer = trainer_l.attn.v_proj(norm_trainer)
        v_trainer = jnp.reshape(v_trainer, (b, s, k * h)) + trainer_l.attn.v_bias.astype(jnp.float32)
        v_trainer = jnp.reshape(v_trainer, (b, s, k, h))
        
        _, attn_trainer = trainer_l.attn(
            norm_trainer,
            None,
            attn_mask=attn_mask_trainer,
            sin=sin_in,
            cos=cos_in,
            segment_ids=segment_ids_trainer,
        )
        post_attn_norm_trainer = trainer_l.post_attention_layernorm(x_in + attn_trainer)
        ffw_trainer = trainer_l.mlp(post_attn_norm_trainer)
        return norm_trainer, attn_trainer, post_attn_norm_trainer, ffw_trainer, q_trainer, k_trainer, v_trainer

      (norm_vllm, attn_vllm, post_attn_norm_vllm, ffw_vllm, q_vllm, k_vllm, v_vllm) = run_vllm_layer0_steps(
          vllm_layer, x_vllm_input, kv_cache
      )
      (norm_trainer, attn_trainer, post_attn_norm_trainer, ffw_trainer, q_trainer, k_trainer, v_trainer) = run_trainer_layer0_steps(
          trainer_layer, x_trainer_input, sin, cos
      )

      diff_norm = np.abs(np.asarray(norm_vllm) - np.asarray(norm_trainer[0]))
      print(f"            | Step 1 (Input Norm) Max Diff: {float(diff_norm.max()):.6e}")
      
      diff_q = np.abs(np.asarray(q_vllm) - np.asarray(q_trainer[0]))
      print(f"            | Q Projection Max Diff: {float(diff_q.max()):.6e}")
      diff_k = np.abs(np.asarray(k_vllm) - np.asarray(k_trainer[0]))
      print(f"            | K Projection Max Diff: {float(diff_k.max()):.6e}")
      diff_v = np.abs(np.asarray(v_vllm) - np.asarray(v_trainer[0]))
      print(f"            | V Projection Max Diff: {float(diff_v.max()):.6e}")

      diff_attn = np.abs(np.asarray(attn_vllm)[:num_active_tokens] - np.asarray(attn_trainer[0])[:num_active_tokens])
      print(f"            | Step 2 (Attn Output) Max Diff: {float(diff_attn.max()):.6e}")

      diff_post_attn_norm = np.abs(np.asarray(post_attn_norm_vllm)[:num_active_tokens] - np.asarray(post_attn_norm_trainer[0])[:num_active_tokens])
      print(f"            | Step 3 (Post Attn Norm) Max Diff: {float(diff_post_attn_norm.max()):.6e}")

      diff_ffw = np.abs(np.asarray(ffw_vllm)[:num_active_tokens] - np.asarray(ffw_trainer[0])[:num_active_tokens])
      print(f"            | Step 4 (MLP Output) Max Diff: {float(diff_ffw.max()):.6e}")

  x_vllm_normed = qwen2_model.norm(x_vllm)
  vllm_logits = vllm_model.compute_logits(x_vllm_normed)

  trainer_normed = trainer_model.final_norm(x_trainer)
  if trainer_model.config.use_tied_embedding:
    trainer_logits = trainer_model.embedder.decode(trainer_normed).astype(jnp.float32)
  else:
    trainer_logits = trainer_model.lm_head(trainer_normed).astype(jnp.float32)

  vllm_logits_np = np.asarray(vllm_logits)[:num_active_tokens]
  trainer_logits_np = np.asarray(trainer_logits[0])[:num_active_tokens]
  logits_diff = np.abs(vllm_logits_np - trainer_logits_np)
  print(
      f"Final Logits  | Mean Abs Diff: {float(logits_diff.mean()):.6e} | Max"
      f" Abs Diff: {float(logits_diff.max()):.6e}"
  )
  print("=" * 45 + " END LAYER COMPARISON " + "=" * 45)


async def main():
  BATCH_SIZE = 1
  NUM_BATCHES = 1
  MAX_PROMPT_LENGTH = 1024
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

  await asyncio.sleep(0)

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
        print(f"{len(prompt_tokens)=}, {pad_value=}")
        unpadded_rollout_prompt = np.array(prompt_tokens)[np.array(prompt_tokens) != pad_value]
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

      trainer_per_token_logps = rl_cluster.get_actor_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
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

      first_item = batch[0]
      prompt_tokens = np.array(first_item.traj.get("prompt_tokens"))
      completion_tokens = np.array(first_item.traj.get("conversation_tokens"))
      completion_mask = np.array(first_item.traj.get("conversation_masks"))
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
      # print("Sample Active Tokens (Index, TokenID, Decoded, Rollout Logp, Trainer Logp, Diff):")

      # idx_to_print = active_indices[:20] + active_indices[-5:] if len(active_indices) > 25 else active_indices

      rp_np = np.array(rollout_per_token_logps[0])
      tp_np = np.array(trainer_per_token_logps[0])

      # for idx in idx_to_print:
      #   tok_id = completion_tokens[idx]
      #   try:
      #     decoded_tok = repr(tokenizer.decode([tok_id]))
      #   except Exception:
      #     decoded_tok = "<unknown>"
      #   r_logp = rp_np[idx]
      #   t_logp = tp_np[idx]
      #   diff_val = abs(r_logp - t_logp)
      #   print(f"  Pos {idx:4d} | ID {tok_id:6d} | {decoded_tok:15s} | Rollout: {r_logp:8.4f} | Trainer: {t_logp:8.4f} | Diff: {diff_val:8.4f}")
      # print("=" * 148)

      # print("\n" + "=" * 60 + " Deep Token & Logit Debug Info " + "=" * 60)
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