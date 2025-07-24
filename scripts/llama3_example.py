import jax

from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tunix.generate import sampler
from tunix.models.llama3 import params
from tunix.models.llama3 import model


MODEL_VERSION="meta-llama/Llama-3.2-1B-Instruct"
MODEL_CP_PATH='/tmp/model/' + MODEL_VERSION


print(f"Make sure you logged in to the huggingface cli.")

snapshot_download(
    repo_id=MODEL_VERSION,
    local_dir=MODEL_CP_PATH,
    local_dir_use_symlinks=False  # optional: avoids symlinks for portability
)

mesh = jax.make_mesh((1, 8), ('fsdp', 'tp'))
config = model.ModelConfig.llama3_1b()  # pick corresponding config based on model version
llama3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(llama3)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH)
tokenizer.pad_token_id = 0

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
