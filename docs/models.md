<!-- DO NOT REMOVE! Placeholder for TOC. -->

# Models

## Models supported

Tunix supports the following models:

| Model | Sizes |
|:---|:---|
| Gemma | 2B, 7B, 9B |
| Gemma 2 | 2B, 9B |
| Gemma 3 | 270M, 1B, 4B, 12B, 27B |
| Llama 3 | 70B, 405B |
| Llama 3.1 | 8B, 70B, 405B |
| Llama 3.2 | 1B, 3B |
| Qwen 2.5 | 0.5B, 1.5B, 3B, 7B |
| Qwen 3 | 0.6B, 1.7B, 4B, 8B, 14B, 30B |

### Model Sources

#### Huggingface & Kaggle
The model configurations and checkpoints should be accessible from Huggingface and Kaggle.
For example, following snippets shows how to load the Qwen 2B model from Huggingface:

```python
ignore_patterns = [
    "*.pth",  # Ignore PyTorch .pth weight files
]
MODEL_PATH = snapshot_download(repo_id="google/gemma-2-2b-it", ignore_patterns=ignore_patterns)
```

#### GCS
You can also store model checkpoints to GCS. So if you have GCS bucket resources
 and have uploaded the model checkpoints there, you can access them as well.

```python
MODEL_PATH = "gs://<your-bucket-dev>/your-model-checkpoints"
```

Once you have an accessible model path from one of the above approach, you are able to load it through Tunix model loading API as following:

```python
config = model_lib.ModelConfig.gemma2_2b()
mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
with mesh:
  gemma = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, mesh)
```

## Fully optimized models
Model optimization is critical for efficient model execution. This includes optimal shardings on TPUs, optimization with Pallas kernels, etc. Tunix provides a lightweight suite of models which is only optimized to some extent. Integration of Tunix and [Maxtext](https://github.com/AI-Hypercomputer/maxtext) enables users to run the RL workloads with fully optimized models. Refer to the [single-host](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/rl.md) and [multi-host](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/rl_on_multi_host.md) tutorial on how to run an optimized model RL workload with Maxtext and Tunix.

## Adding a new model
You can add new models to Tunix codebase by following the Tunix convention.
### Model Family
If the new model falls into one of the existing model families (e.g. Gemma, Llama, etc.) then adding a new model doesn't need to create new files. You just need to add the model specs to the corresponding model family. Take a look at the [Llama examples](https://github.com/google/tunix/blob/main/models/llama3/model.py;l=98-135).
If the new model is from a new model family that Tunix hasn't supported yet. You will need to follow the design and APIs as the existing model families to create the model implementation.

### Naming
Adding the new model needs to following the naming convention that Tunix supports so that `AutoModel`(as described below) could work correctly. We use the pattern of `<model_family><major_version>p<minor_version>_<model_size>`to name a model. For example, the `Llama3.2 1b` model is named as `llama3p2_1b` while a `Qwen2.5 1.5b` model is named as `qwen2p5_1p5b`.

## AutoModel

`AutoModel` provides a unified interface for instantiating Tunix models from
pretrained checkpoints, similar to the Hugging Face `AutoModel` API. It allows
you to load a model simply by providing its `model_id`, handling the download
and initialization for you.

### Basic Usage

To load a model, use the `AutoModel.from_pretrained` method with the model
identifier and your JAX sharding mesh. By default this will download the model
from HuggingFace.

```python
from tunix.models.automodel import AutoModel
import jax

# 1. Define your mesh
mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)

# 2. Load the model
# By default, this downloads from Hugging Face.
model, model_path = AutoModel.from_pretrained(
  model_id="google/gemma-2-2b-it",
  mesh=mesh
)

print(f"Model loaded from: {model_path}")
```

### Specifying Model Source

You can load models from different sources (e.g., Kaggle, GCS, etc.) using the
`model_source` argument.

#### From HuggingFace:

This is the default choice (`ModelSource.HUGGINGFACE`) as shown in the
example above.

#### From Kaggle:

For Kaggle, you must provide the `model_id` which is the Hugging Face identifier
(to determine the model configuration) and the `model_path` which is the Kaggle
Hub model identifier (used to download the model from Kaggle).

```python
model, model_path = AutoModel.from_pretrained(
    model_id="google/gemma2-2b-it",
    mesh=mesh,
    model_source=ModelSource.KAGGLE,
    model_path="google/gemma-2/flax/gemma2-2b-it",
)
```

For example the `model_path` for the `google/gemma-2/flax/gemma2-2b-it` is extracted on Kaggle as shown below

![Kaggle extracting Model ID](images/model_id_kaggle.png){: width="75%"}

#### From GCS:

For GCS, you must provide the `model_id` which is the Hugging Face identifier
(to determine the model configuration) and the `model_path` (the actual GCS
location).

```python
model, model_path = AutoModel.from_pretrained(
    model_id="google/gemma-2-2b-it",
    mesh=mesh,
    model_source=ModelSource.GCS,
    model_path="gs://my-bucket/gemma-2-2b-it"
)
```

### Model Download Path

Optionally, you can also provide the `model_download_path` argument, which
specifies where the model is to be downloaded to. Depending on the
`model_source` the effect of specifying this variable is different:

*   **Hugging Face**: Files are downloaded directly to this directory.
*   **Kaggle**: Sets the `KAGGLEHUB_CACHE` environment variable to this path.
*   **GCS**: No-op.
*   **Internal**: Files are copied to this directory. If omitted, the model is loaded directly from the `model_path`. This mode (Internal) is not supported in OSS version.

## Naming Conventions

This section outlines the naming conventions used within Tunix for model
identification and configuration. These conventions ensure consistency when
loading models from various sources like Hugging Face or Kaggle.

The `ModelNaming` dataclass handles the parsing and standardization of model names.

*   **`model_id`**: The full model name identifier (case sensitive), as it appears
    on Hugging Face, including the parent directory. For example,
    `meta-llama/Llama-3.1-8B` is extracted as shown below:
      ![Hugging Face extracting Model ID](images/model_id_huggingface.png){: width="75%"}


*   **`model_name`**: The unique full name identifier of the model. This
    corresponds to the full name and should match exactly with the model name
    used in Hugging Face or Kaggle. It is typically all lowercase and formatted
    as `<model-family>-<model-version>`.
    *   *Example*: `gemma-2b`, `llama-3.1-8b`, `gemma2-2b-it`.

*   **`model_family`**: The standardized model family. Unnecessary hyphens are
    removed, and versions are standardized (e.g., replacing dot with `p`).
    *   *Example*: `gemma`, `gemma2`, `qwen2p5`.
    *   *Conversion*: `gemma-2` -> `gemma2`, `qwen2.5` -> `qwen2p5`.

*   **`model_version`**: The standardized version of the model family (lowercase,
    hyphens to underscores, dots to `p`). This is usually the second portion of
    the `model_name` and includes size information or tuning variants (e.g., "it"
    for instruction tuned).
    *   *Example*: `2b_it`.
    *   *Conversion*: `2b-it` -> `2b_it`

*   **`model_config_category`**: The Python class name of the `ModelConfig` class. This groups models that share the same configuration structure.
    *   *Example*: Both `gemma` and `gemma2` models fall under the `gemma` category, with the `ModelConfig` class defined in `models/gemma/model.py`.

*   **`model_config_id`**: The standardized configuration ID used within the `ModelConfig` class. It is composed of the `model_family` and `model_version`.
    *   *Example*: `gemma_2b_it` or `qwen2p5_0p5b`.

You can initialize `ModelNaming` by providing either the `model_id` or the
`model_name`. If `model_id` is provided, the `model_name` is inferred as the
last segment of the `model_id`. If `model_name` is provided, it is used
directly. All other naming attributes are then automatically derived and
validated.
