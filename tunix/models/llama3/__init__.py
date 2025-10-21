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

"""Llama3 API."""

from tunix.models.llama3 import mapping_sglang_jax
from tunix.models.llama3 import mapping_vllm_jax
from tunix.models.llama3 import model
from tunix.models.llama3 import params

BACKEND_MAPPINGS = {
    'vllm_jax': mapping_vllm_jax.VLLM_JAX_MAPPING,
    'sglang_jax': mapping_sglang_jax.SGLANG_JAX_MAPPING,
}


__all__ = ['BACKEND_MAPPINGS', 'model', 'params']
