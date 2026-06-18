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

"""A script to debug log probabilities between VLLM sampler and reference trainer."""

import contextlib
import logging
import os
import sys
from typing import Any, Dict, Optional

from absl import logging as absl_logging
import datasets as datasets_lib
import flax
from flax import nnx
import fsspec
import jax
from jax import numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
from pprint import pprint
import re
from tqdm.auto import tqdm
import transformers

from tunix.generate import mappings
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import sharding_utils
from tunix.utils import math_utils
from tunix.generate import utils

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer


def get_prompt_logprobs_from_vllm_output(
    token_ids,
    prompt_logprobs,
):
  """Extracts prompt logprobs from vLLM output."""
  if not prompt_logprobs:
    return []

  assert len(prompt_logprobs) == len(token_ids), (
      f'prompt log probs has {len(prompt_logprobs)} number of items !='
      f' {len(token_ids)} token ids'
  )

  extracted = [0.0]  # The logprob of the first prompt token is None, default to 0.0
  for idx, (tok_id, tok_logprobs) in enumerate(zip(token_ids[1:], prompt_logprobs[1:]), start=1):
    if tok_logprobs is not None and tok_id in tok_logprobs:
      extracted.append(tok_logprobs[tok_id].logprob)
    else:
      extracted.append(0.0)
  return extracted

# ====== Logging Configuration ======
# 1. Force absl to use python logging
absl_logging.use_python_logging()

# 2. Configure the root logger
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# 3. Explicitly set levels for relevant loggers
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)

# 4. Set absl verbosity
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

print("Logging configured at INFO level.")
os.environ["JAX_COMPILER_CACHE_TGTS"] = ""

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  from etils import ecolab

  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )

  file_open = gfile.Open

  NOTEBOOK_ENV = "g3"
except Exception:
  NOTEBOOK_ENV = "git"
  cm = contextlib.nullcontext()
  file_open = fsspec.open

MODEL_PATH_PREFIX = "gs://tunix/models"
from huggingface_hub import snapshot_download

model_version = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_path = snapshot_download(repo_id=model_version, max_workers=16)

MODEL_MAPPING = {
    model_version: (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        model_path,
    ),
}

model_config, _ = MODEL_MAPPING[model_version]
model_config.dtype = jnp.float32
model_config.param_dtype = jnp.float32

tokenizer = AutoTokenizer.from_pretrained(model_version)

rollout_device_list = jax._src.mesh_utils.create_device_mesh(
    (2, 2), jax.devices()[:4], allow_split_physical_axes=True
)
rollout_mesh = jax.sharding.Mesh(
    rollout_device_list, axis_names=["fsdp", "tp"]
)

trainer_device_list = jax._src.mesh_utils.create_device_mesh(
    (4, 1), jax.devices()[4:], allow_split_physical_axes=True
)
trainer_mesh = jax.sharding.Mesh(
    trainer_device_list, axis_names=["fsdp", "tp"]
)

print("Loading model from safe tensors...")
with jax.set_mesh(trainer_mesh):
  trainer_model = qwen2_params_lib.create_model_from_safe_tensors(
      file_dir=model_path, config=model_config, mesh=trainer_mesh, dtype=jnp.float32,
  )
  ref_model = trainer_model

os.environ["NEW_MODEL_DESIGN"] = "true"
optimizer = optax.adamw(
    learning_rate=1e-6,
)

base_rollout_dict = {
    "max_prompt_length": 2048,
    "kv_cache_size": 2048 + 128,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "return_logprobs": True,
    "max_tokens_to_generate": 8192,
}

vllm_rollout_dict = {
    # vllm-tpu specific configs
    "rollout_vllm_model_version": model_version,
    "rollout_vllm_hbm_utilization": 0.33,
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": True,
    "rollout_vllm_init_with_random_weights": True,
    "rollout_vllm_max_num_seqs": 1,
    "rollout_vllm_max_num_batched_tokens": 2496,
    "tensor_parallel_size": 2,
    "data_parallel_size": 2,
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

tokenizer = AutoTokenizer.from_pretrained(model_path)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=trainer_model,
    reference=ref_model,
    tokenizer=tokenizer,
    cluster_config=rl_cluster_lib.ClusterConfig(
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
    ),
)

prompt_str = "Given ten 0's and ten 1's, how many 0-1 binary sequences can be formed such that no three or more consecutive 0's are together? For example, 01001001010011101011 is such a sequence, but the sequence 01001000101001110111 does not satisfy this condition. Let's think step by step, and put your final answer within \\boxed{}."
prompts = [[{"role": "user", "content": prompt_str}]]



# Replicate batch_size = 2
batch_size = 1
# prompts = [prompts] * batch_size

rollout_prompts = prompts * batch_size
out_data = rl_cluster.generate(prompts=rollout_prompts, apply_chat_template=True, max_generation_steps=8192)
print("Text from VLLM sampler: ", out_data.text[0] + "...")
print("logprobs from VLLM sampler count: ", len(out_data.prompt_logprobs[0]) if out_data.prompt_logprobs else 0)

# Override vllm_rollout_logps to use the online version for comparison with JAX
vllm_rollout_logps = out_data.logprobs

# # Pad first-turn prompt context
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
# padded_prompt = utils.pad_to_length(
#     np.array(out_data.left_padded_prompt_tokens[0], dtype=np.int32),
#     target_length=2048,
#     pad_value=pad_id,
#     left=True,
# )
# padded_prompt_ids = jnp.asarray([padded_prompt] * batch_size)

# # Pad completion sequence
# max_response_length = 2048
# padded_completion = utils.pad_to_length(
#     np.array(out_data.tokens[0], dtype=np.int32),
#     target_length=max_response_length,
#     pad_value=pad_id,
#     left=False,
# )
# padded_completion_ids = jnp.asarray([padded_completion] * batch_size)

# Shard inputs for FSDP
graphdef, state = nnx.split(trainer_model)

with jax.set_mesh(trainer_mesh):
  # Replicate the single unpadded sequence 4 times to satisfy FSDP mesh constraints
  prompt_arr = np.array([out_data.left_padded_prompt_tokens[0]] * 4, dtype=np.int32)
  completion_arr = np.array([out_data.tokens[0]] * 4, dtype=np.int32)
  
  dest_prompt_tokens = sharding_utils.shard_input(
      jnp.asarray(prompt_arr),
      ("fsdp",),
  )
  dest_completion_tokens = sharding_utils.shard_input(
      jnp.asarray(completion_arr),
      ("fsdp",),
  )

  # Get JAX reference logprobs for completion tokens
  ref_logps = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=dest_prompt_tokens,
      completion_tokens=dest_completion_tokens,
      pad_id=pad_id,
      eos_id=tokenizer.eos_token_id,
      temperature=0.7,
      return_logits=False,
  )

jax.clear_caches()
jax.block_until_ready(ref_logps)
print("\nSuccessfully computed reference logprobs for completion tokens on JAX side!")
print(f"Reference logprobs shape: {ref_logps.shape}")


# padded_sampler_logprobs = []
# for i in range(batch_size):
#   seq_logps = vllm_rollout_logps[i]
#   unpadded_len = len(seq_logps)
#   pad_len = max_response_length - unpadded_len
#   padded = np.concatenate([seq_logps, np.zeros(pad_len, dtype=np.float32)])
#   padded_sampler_logprobs.append(padded)
# padded_sampler_logprobs = jnp.asarray(padded_sampler_logprobs)



# Apply mask and compute difference metrics
# Extract the first sequence for comparison
ref_logps_first = ref_logps[0]
vllm_logps_first = jnp.asarray(vllm_rollout_logps[0])

diff = jnp.abs(ref_logps_first - vllm_logps_first)
print(f"{ref_logps_first = }")
print(f"{vllm_logps_first = }")
print(f"{diff = }")

diff_mean = jnp.sum(diff) / diff.size
diff_std = jnp.sqrt(jnp.sum(jnp.square(diff)) / diff.size)

print(f"\nMean absolute difference on completion tokens (excluding padding): {diff_mean}")
print(f"Standard deviation of absolute difference: {diff_std}")

# Print token-by-token comparison for the first item
print("\n=== Token-by-Token Completion Logprob Comparison (First Item) ===")
first_completion_tokens = out_data.tokens[0]

for idx in range(len(vllm_logps_first)):
  tok_id = int(first_completion_tokens[idx])
  tok_str = tokenizer.decode([tok_id])
  is_compared = "COMPARED"
  ref_val = float(ref_logps_first[idx])
  sampler_val = float(vllm_logps_first[idx])
  print(f"idx={idx:03d} | Token: {tok_str!r:<15} | Ref Logp: {ref_val:.4f} | Sampler Logp: {sampler_val:.4f} | Diff: {ref_val - sampler_val:.4f} | {is_compared}")
