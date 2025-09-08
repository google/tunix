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

"""Default configs for LLaVA model."""

from tunix.models.llava.model import LlavaConfig

# Example config for a LLaVA model using a ViT-B/16 vision encoder and a Llama-like text decoder.
LLAVA_CONFIG = LlavaConfig(
    vision_hidden_dim=768,        # Output dim of ViT-B/16
    vocab_size=32000,             # Match your tokenizer
    text_hidden_dim=4096,         # Hidden dim of your LLM
    num_text_layers=32,           # Number of transformer layers
    num_text_heads=32,            # Number of attention heads
    max_text_length=2048,         # Max context length
    image_token_position=0,       # Prepend image embedding
)