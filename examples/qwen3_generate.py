"""Generate text using Qwen3-0.6B with Tunix."""

import os

from transformers import AutoTokenizer, AutoConfig
from tunix.models.qwen3 import params
from tunix.generate.sampler import Sampler, CacheConfig

REPO_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    # Load tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = params.from_pretrained(REPO_ID)
    hf_cfg = AutoConfig.from_pretrained(REPO_ID)

    cache_config = CacheConfig(
                cache_size=256,
                num_layers=hf_cfg.num_hidden_layers,
                num_kv_heads=getattr(
                    hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads
                ),
                head_dim=getattr(
                    hf_cfg, "head_dim", hf_cfg.hidden_size // hf_cfg.num_attention_heads
                ),
            )
    
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
                enable_thinking=True,
            )
            )
        return out
            
    inputs = templatize(
        [
            "which is larger 9.9 or 9.11?",
            "讲几句人话",
            "tell me your name, respond in Chinese",
        ]
    )

    sampler = Sampler(model, tokenizer, cache_config, use_jit=False)
    out = sampler(inputs, total_generation_steps=128, echo=True)
    print(out.text)

if __name__ == "__main__":
    main()
