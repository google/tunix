# Copyright 2025 Google LLC
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
"""AutoModel class for Tunix."""

from typing import Any, Tuple
from flax import nnx
import jax
from tunix.models import loader
from tunix.models import naming


class AutoModel:
  """Factory class for instantiating Tunix models from pretrained checkpoints,

  similar to the Hugging Face AutoModel API.
  """

  @classmethod
  def from_pretrained(
      cls,
      model_id: str,
      mesh: jax.sharding.Mesh,
      model_source: loader.ModelSource = loader.ModelSource.HUGGINGFACE,
      model_download_path: str | None = None,
      **kwargs,
  ) -> Tuple[nnx.Module, str | None]:
    """Loads a pretrained model from a given identifier.

    This method mimics the Hugging Face `from_pretrained` interface,
    providing a unified way to load models from various sources such as
    Hugging Face Hub, Kaggle, or GCS. The mainstream case it downloads the model
    and creates the model from safe tensors. However, for special cases, such as
    Gemma models from certain sources, different logic is used to create the
    model.

    Args:
        model_id: The full model id, e.g., "meta-llama/Llama-3.1-8B" for
          HuggingFace/Kaggle or a GCS path for GCS sources.
        mesh: The JAX sharding Mesh object.
        model_source: The source of the model (e.g., Kaggle, HuggingFace, GCS).
          Default is HuggingFace.
        model_download_path: The local directory where the model should be
          downloaded. If None, and downloading is required, a temporary
          directory might be used or downloading might be skipped if the
          model_id is a local path.
        **kwargs: Additional keyword arguments passed to the underlying model
          creation functions. - For Kaggle Gemma: `intermediate_ckpt_dir`,
          `rng_seed`.

    Returns:
        The loaded nnx.Module model.
        The model_path: The path where the model was downloaded.
    """

    model: nnx.Module = None
    model_params: Any = None
    model_name = naming.get_model_name_from_model_id(model_id)

    model_path = loader.download_model(
        model_id, model_download_path, model_source
    )

    # Case 1: Special handling cases for Gemma models
    if model_source == loader.ModelSource.GCS and (
        model_name.startswith(('gemma3', 'gemma-3'))
    ):
      model, model_params = loader.create_gemma3_model_from_checkpoint(
          ckpt_path=model_id, model_name=model_name, mesh=mesh
      )
    elif model_source == loader.ModelSource.KAGGLE and model_name.startswith(
        'gemma'
    ):
      # Download model from Kaggle requires NNX conversion and can takes long.
      # It is recommended to save the NNX converted model for later runs.
      # kwargs.get('intermediate_ckpt_dir') is used to save the intermediate
      # checkpoint. If it is not provided, the model will be converted but not
      # saved.
      # TODO(sizhi): Remove gemma conversion logic once load safetensors for
      # gemma is ready.
      intermediate_ckpt_dir = kwargs.get('intermediate_ckpt_dir')
      rng_seed = kwargs.get('rng_seed', 0)
      model, model_params = loader.create_gemma_model_with_nnx_conversion(
          model_name=model_name,
          ckpt_path=model_path,
          intermediate_ckpt_dir=intermediate_ckpt_dir,
          rng_seed=rng_seed,
          mesh=mesh,
      )

    # TODO(b/451662153): Are there certain combinations that are not supported?
    # e.g., llama from gcs? if so, why and which ones?

    # Case 2: Common path for all models -- create model from safe tensors
    if not model_params:
      # pick corresponding config based on model version
      model_params = loader.call_model_config(model_name)

      with mesh:
        model = loader.create_model_from_safe_tensors(
            model_name,
            model_download_path,
            model_params,
            mesh,
        )

    return model, model_path
