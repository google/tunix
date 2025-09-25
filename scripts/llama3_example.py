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

"""Example of using tunix to load and run Llama3 models."""
import sys
sys.path.insert(0, "/home/grads/tianjiao/tunix")
import os
import tempfile

from flax import nnx
import huggingface_hub
import jax
import transformers
from tunix.generate import sampler
from tunix.models.llama3 import model
from tunix.models.llama3 import params

MODEL_VERSION = "meta-llama/Llama-3.1-8B-Instruct"

# Consider switch to tempfile after figuring out how it works
temp_dir = tempfile.gettempdir()
MODEL_CP_PATH = os.path.join(temp_dir, "models", MODEL_VERSION)


print("Make sure you logged in to the huggingface cli.")

all_files = huggingface_hub.list_repo_files(MODEL_VERSION)
filtered_files = [f for f in all_files if not f.startswith("original/")]

for filename in filtered_files:
  huggingface_hub.hf_hub_download(
      repo_id=MODEL_VERSION, filename=filename, local_dir=MODEL_CP_PATH
  )
print(f"Downloaded {filtered_files} to: {MODEL_CP_PATH}")

mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
config = (
    model.ModelConfig.llama3_1_8b()
)  # pick corresponding config based on model version
llama3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(llama3)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CP_PATH)
tokenizer.pad_token_id = 0


def templatize(prompts):
  """Apply chat template to the prompts using the tokenizer.

  Args:
    prompts: A list of prompts.

  Returns:
    A list of templated prompts.
  """
  outputs = []
  for p in prompts:
    outputs.append(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
  return outputs


inputs = templatize([
    "tell me about world war 2",
    "印度的首都在哪里",
    "tell me your name, respond in Chinese",
])

sampler = sampler.Sampler(
    llama3,
    tokenizer,
    sampler.CacheConfig(
        cache_size=256, num_layers=32, num_kv_heads=8, head_dim=128
    ),
)
out = sampler(inputs, max_generation_steps=128, echo=True, top_p=None)

for t in out.text:
  print(t)
  print("*" * 30)
