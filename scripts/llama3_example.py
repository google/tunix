from flax import nnx
from tunix.models.llama3 import params
from tunix.models.llama3 import model

# MODEL_CP_PATH = '/cns/gg-d/home/qwix-dev/llama3/torch/8b-it'
MODEL_CP_PATH = '/workspace/tunix/rl/grpo/models/meta-llama/Meta-Llama-3-8B-Instruct'

import jax

mesh = jax.make_mesh((1, 8), ('fsdp', 'tp'))

config = model.ModelConfig.llama3_8b()  # pick corresponding config based on model version
llama3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(llama3)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)
tokenizer.pad_token_id = 0

from tunix.generate import sampler

def templatize(prompts):
  out = []
  for p in prompts:
    out.append(
        tokenizer.apply_chat_template(
          [
              {"role": "user", "content": p},
          ],
          tokenize=False,
          add_generation_prompt=True,
          # enable_thinking=True,
      )
    )
  return out

inputs = templatize(
    [
        "tell me about world war 2",
        "讲几句人话",
        "tell me your name, respond in Chinese",
    ]
)

sampler = sampler.Sampler(llama3, tokenizer, sampler.CacheConfig(cache_size=256, num_layers=32, num_kv_heads=8, head_dim=128))
out = sampler(inputs, total_generation_steps=128, echo=True, top_p=0.9)

for t in out.text:
  print(t)
  print('*' * 30)
