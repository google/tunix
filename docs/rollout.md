

# Rollout

In Tunix RL pipelines (e.g. GRPO), **rollout** is the step where the current
policy (the actor) generates completions for a batch of prompts. Those sampled
completions are then scored by reward functions, and the results are used to
compute RL advantages/updates.

At a high level, rollout is responsible for:

- Turning prompt strings into model inputs (tokenization + padding/truncation).
- Generating **N** completions per prompt with configurable sampling (e.g. temperature / top-p / top-k).
- Returning text, tokens, and (when available) token-level log-probabilities needed by the algorithm.

Tunix supports multiple rollout engines (selected by `rollout_engine`):

- `vanilla`: Tunix-native generation. This option provides the basic inference engine without advanced features.
- `vllm`: vLLM-backed generation. The vLLM engine is backed by `vllm` and Google supported `tpu-inference` backend.
- `sglang_jax`: SG-Lang JAX rollout. This is another advanced inference backend from the sglang OSS community.

`vllm` and `sglang` provide better performance with large batch size and agentic RL.

#### Sampling Knobs

These are used by all the rollout engines:

-   `max_tokens_to_generate`: max new tokens.
-   `max_prompt_length`: prompts are padded/truncated to this length.
-   `temperature`, `top_p`, `top_k`: sampling knobs.

The rollout interface and config live in [Base Rollout](https://github.com/google/tunix/blob/main/rl/rollout/base_rollout.py).

This doc focuses on the basic rollout. For tool calling enabled rollout or multi
turn rollout, please refer to [Agentic RL](agentic_rl.md).

## vLLM
This section explains how Tunix integrates **vLLM** as the rollout (sampling)
engine during RL (e.g. GRPO), and how to configure and run it.

### How the integration works

At a high level:

-   Tunix trains an **actor** model (Flax NNX) and periodically needs samples
    (completions) for prompts.
-   When `rollout_engine="vllm"`, Tunix creates a vLLM-based rollout worker:
    -   Implementation:
        [vllm_rollout.py](https://github.com/google/tunix/blob/main/rl/rollout/vllm_rollout.py)
    -   vLLM sampler wrapper:
        [vllm_sampler.py](https://github.com/google/tunix/blob/main/generate/vllm_sampler.py)
-   The vLLM engine is initialized (optionally with **dummy/random weights**)
    and then Tunix **synchronizes weights in-memory** from the trainer to vLLM.
    -   Today, vLLM rollout in Tunix supports **in-memory weight sync** (not
        loading rollout weights from a checkpoint path). This ensures the
        rollout model's weights remain synchronized with the actor model during
        training. This approach leverages Tunix's existing weight sync API, with
        the trade-off of a slightly longer startup time for the initial weight
        transfer. See `VllmSampler.load_checkpoint`.

### Installation

#### TPU (JAX backend)

1. Direct install from pypi
```
VLLM_TARGET_DEVICE="tpu" pip install vllm
pip install tpu-inference
```

2. Install and run the docker image
```
docker pull vllm/vllm-tpu:nightly
docker run -it local_vllm vllm/vllm-tpu:nightly /bin/bash
```

#### GPU

If you are using GPUs, install a vLLM build compatible with your environment.
Tunix only requires that `import vllm` works and that your vLLM build supports
the backend you intend to run.

### Choosing vLLM as the Rollout

In the code, rollout engine selection happens in
[rl_cluster.py](https://github.com/google/tunix/blob/main/rl/rl_cluster.py).
Setting `cluster_config.rollout_engine="vllm"` enables the vllm rollout/sampler.

### Configuration knobs

Tunix uses `tunix.rl.rollout.base_rollout.RolloutConfig` for rollout settings.
The fields below are the vLLM-relevant ones.

#### vLLM-specific fields

In addition to the common sampling parameters mentioned above, the following
settings are specific to vLLM:

-   `rollout_vllm_model_version` (required)

    -   HuggingFace model id or a local path (depending on your vLLM build).
    -   Note: Tunix will still sync weights from the trainer; this value is
        primarily used to initialize the vLLM engine.

-   `rollout_vllm_init_with_random_weights`

    -   If `True`, Tunix asks vLLM to use a dummy/random weight init (faster
        engine bootstrap) and then relies on in-memory weight sync.

-   `rollout_vllm_hbm_utilization`

    -   How much accelerator memory (HBM) vLLM is allowed to use. There is no
        official guideline on how to set these values. For colocated case, users
        need to estimate the rollout model weights and the KV cache budget and
        coordinate with the other models. For disaggregated setup, users can set
        it to a number close to 1 to make full utilization of the HBM.

-   `rollout_vllm_swap_space_size_gb`

    -   CPU swap space (GiB) for KV cache spilling.

-   `rollout_vllm_server_mode`

    -   `False`: batch inference mode (`vllm.LLM(...)`).
    -   `True`: in-process engine + driver loop (`VLLMInProcessDriver`). Useful
        for higher-throughput request scheduling leveraging the vLLM continuous
        batching capabilities.

-   `rollout_vllm_async_scheduling`

    -   Enables vLLM async scheduling.

-   `rollout_vllm_tpu_backend_type`

    -   Sets `TPU_BACKEND_TYPE` for vLLM TPU backend selection (e.g. `"jax"`,
        `"torchax"`).

-   `tensor_parallel_size`, `data_parallel_size`

    -   If unset (`-1`), Tunix derives them from the rollout mesh.
    -   If `data_parallel_size > 1`, Tunix sets `NEW_MODEL_DESIGN=1` for vLLM.

-   `rollout_vllm_hf_config_path`, `rollout_vllm_additional_config`

    -   For MaxText/custom model support in vLLM; passed through to vLLM engine
        args.

-   `rollout_mapping_config`

    -   Controls how Tunix trainer weights are mapped into vLLM/HF parameter
        names.
    -   Tunix builds a `MappingConfig` via
        `tunix.generate.mappings.MappingConfig.build(..., backend="vllm_jax")`.
    -   If mappings are missing, Tunix may fall back to *direct structural sync*
        (currently only supported for MaxText-style configs; see error message
        in `VllmSampler.update_params`).

#### LoRA

There is a `rollout_vllm_lora_config` field, but note that LoRA support for
Tunix + vLLM is WIP. Check [vLLM Sampler](https://github.com/google/tunix/blob/main/generate/vllm_sampler.py)
for the latest status.

### Example: using vLLM rollout in a Python entrypoint

The most direct way to use vLLM rollout today is via a Python script that
constructs a `RolloutConfig` with the vLLM fields set.

Pseudocode (simplified):

```python
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

rollout_config = base_rollout.RolloutConfig(
    max_tokens_to_generate=768,
    max_prompt_length=256,
    temperature=0.9,
    top_p=1.0,
    top_k=50,
    tensor_parallel_size=8,
    data_parallel_size=1,
    rollout_vllm_model_version="meta-llama/Llama-3.2-1B-Instruct",
    rollout_vllm_hbm_utilization=0.2,
    rollout_vllm_tpu_backend_type="jax",
    rollout_vllm_server_mode=False,
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh=role_to_mesh,
    rollout_engine="vllm",
    offload_to_cpu=False,
    training_config=training_config,
    rollout_config=rollout_config,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
```

For a concrete end-to-end example, see [grpo_demo_llama3_qwen2.py](https://github.com/google/tunix/blob/main/scripts/grpo_demo_llama3_qwen2.py)
(it supports selecting `--rollout-engine vllm`).

### Using vLLM rollout via the CLI

The CLI support for vLLM rollout engine is WIP.

### Troubleshooting

#### vLLM fails to initialize / `model` is empty

- Ensure `rollout_vllm_model_version` is set to a valid HF repo id or local path.

#### Out-of-memory (HBM)

- Lower `rollout_vllm_hbm_utilization`.
- Reduce `max_prompt_length` and/or `max_tokens_to_generate`.
- Increase `rollout_vllm_swap_space_size_gb` (trade-off: may increase latency if swapping occurs).

#### Data parallel issues

- If you set `data_parallel_size > 1`, Tunix sets `NEW_MODEL_DESIGN=1` for vLLM.
- Ensure your rollout mesh size matches `tensor_parallel_size * data_parallel_size`.

#### Weight sync / mapping errors

- If you see errors about missing key mappings, provide a `rollout_mapping_config` or use a model implementation that exposes mapping helpers.
- For MaxText-style models, provide `rollout_vllm_additional_config` including a `maxtext_config` entry (required for direct sync).


## SGLang

This section explains how Tunix integrates **SGLang-Jax** as the rollout
(sampling) engine during RL (e.g. GRPO), and how to configure and run it.

### How the integration works

At a high level:

-   When `rollout_engine="sglang_jax"`, Tunix creates an SGLang-Jax rollout worker:
    -   Rollout worker: `tunix/rl/rollout/sglang_jax_rollout.py`
    -   Sampler wrapper: `tunix/generate/sglang_jax_sampler.py`
-   Tunix initializes an in-process SGLang-Jax `Engine` (`sgl_jax.srt.entrypoints.engine.Engine`).
-   Like the vLLM integration, **SGLang-Jax rollout currently relies on in-memory weight sync**
    from the trainer to the rollout engine.
    -   `SglangJaxSampler.load_checkpoint(...)` only supports passing a PyTree of weights; loading
        rollout weights from a checkpoint path is not implemented yet.
-   Parallelism:
    -   SGLang-Jax rollout currently derives `tp_size` as the total device count of the rollout
        mesh (it does not support data-parallel for rollout yet).

### Installation

SGLang-Jax is not installed by default with Tunix. The recommended setup is to
install Tunix first, then install SGLang-Jax from source:

```sh
# Install Tunix (see README for options)

# Then install SGLang-Jax
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax/python
pip install -e .
```

If you see import errors for `sgl_jax`, double-check that you installed the
`sglang-jax/python` package in the same environment as Tunix.

### Choosing SGLang-Jax as the Rollout

Set `cluster_config.rollout_engine="sglang_jax"`.

Rollout engine selection happens in `tunix/rl/rl_cluster.py`.

### Configuration knobs

Tunix uses `tunix.rl.rollout.base_rollout.RolloutConfig` for rollout settings.
In addition to the common sampling parameters, the following fields are specific
to SGLang-Jax:

-   `rollout_sglang_jax_model_version`

    -   Model id or local path used by SGLang-Jax as `model_path`.
    -   Note: Tunix still syncs weights in-memory; this value is primarily used
        to bootstrap the engine.

-   `rollout_sglang_jax_context_length`

    -   Passed to SGLang-Jax as `context_length`.
    -   Recommendation: set this explicitly to your model context length to
        avoid surprises.

-   `rollout_sglang_jax_mem_fraction_static`

    -   Fraction of accelerator memory reserved for static allocations
        (weights + runtime buffers) in SGLang-Jax.

-   `rollout_sglang_jax_init_with_random_weights`

    -   If `True`, Tunix asks SGLang-Jax to use dummy/random weights during
        engine initialization (`load_format="dummy"`), then relies on in-memory
        weight sync.

-   `rollout_sglang_jax_disable_radix_cache`

    -   Disables SGLang's radix cache.
    -   Recommended for RL-style training where the rollout weights are updated
        frequently (in-memory weight sync). Cached prefix states may no longer
        match the new weights, so disabling the cache avoids stale reuse.

-   `rollout_sglang_jax_enable_deterministic_sampling`

    -   Enables deterministic sampling mode in SGLang-Jax.

-   `rollout_sglang_jax_precompile_bs_paddings`, `rollout_sglang_jax_precompile_token_paddings`

    -   Optional “bucket sizes” to precompile common batch sizes / token lengths.
    -   Useful to reduce JIT/compile overhead when rollout shapes vary.

-   `rollout_sglang_jax_chunked_prefill_size`

    -   Enables chunked prefill when set to a positive value.
    -   Set to `-1` to disables chunked prefill.

-   `rollout_sglang_jax_page_size`

    -   Number of tokens per KV-cache page.

#### Weight mapping

SGLang-Jax rollout uses the same `rollout_mapping_config` field as vLLM.
Internally, Tunix builds a `MappingConfig` via
`tunix.generate.mappings.MappingConfig.build(..., backend="sglang_jax")`.

If mappings are missing, weight sync will fail. The model implementations in
Tunix ship SGLang-Jax mappings.

### Example: using SGLang-Jax rollout in a Python entrypoint

Pseudocode (simplified):

```python
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

rollout_config = base_rollout.RolloutConfig(
    max_tokens_to_generate=768,
    max_prompt_length=256,
    temperature=0.9,
    top_p=1.0,
    top_k=50,
    rollout_sglang_jax_model_version="meta-llama/Llama-3.2-1B-Instruct",
    rollout_sglang_jax_context_length=4096,
    rollout_sglang_jax_mem_fraction_static=0.2,
    rollout_sglang_jax_init_with_random_weights=True,
    rollout_sglang_jax_disable_radix_cache=True,
    rollout_sglang_jax_enable_deterministic_sampling=False,
    rollout_sglang_jax_precompile_bs_paddings=[8],
    rollout_sglang_jax_precompile_token_paddings=[2048],
    rollout_sglang_jax_chunked_prefill_size=2048,
    rollout_sglang_jax_page_size=64,
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh=role_to_mesh,
    rollout_engine="sglang_jax",
    offload_to_cpu=False,
    training_config=training_config,
    rollout_config=rollout_config,
)
```

For a concrete example in this repo, `scripts/grpo_demo_llama3_qwen2.py`
supports `--rollout-engine sglang_jax`.

### Troubleshooting

#### ImportError: `sgl_jax` not found

- Ensure SGLang-Jax is installed (`pip install -e .` from `sglang-jax/python`).
- Ensure you installed it into the same Python environment used to run Tunix.

#### `max_generation_steps` exceeds `context_length`

- SGLang-Jax requires `max_tokens_to_generate <= rollout_sglang_jax_context_length`.
  Increase `rollout_sglang_jax_context_length` or lower `max_tokens_to_generate`.

#### Out-of-memory (HBM)

- Lower `rollout_sglang_jax_mem_fraction_static`.
- Reduce `max_prompt_length` and/or `max_tokens_to_generate`.
- Consider enabling chunked prefill via `rollout_sglang_jax_chunked_prefill_size`.

#### Weight sync / mapping errors

- Provide a correct `rollout_mapping_config` or use a model that ships SGLang-Jax mappings.

#### Logprobs are missing

- The current SGLang-Jax sampler wrapper does not populate token-level logprobs.
  If your algorithm needs logprobs, compute them via the trainer model (or add
  logprob plumbing to the sampler).


## Vanilla

This section explains how Tunix integrates its **vanilla** (Tunix-native)
rollout engine, and how to configure and run it.

### How the integration works

At a high level:

-   When `rollout_engine="vanilla"`, Tunix uses an in-process JAX/Flax NNX model
    to generate samples.
-   Implementation:
    -   Rollout worker: `tunix/rl/rollout/vanilla_rollout.py`
    -   Sampler: `tunix/generate/sampler.py`
-   Sampling uses a compiled prefill + decode loop. The first
    rollout for a new (prompt length, batch size, generation length) shape
    triggers compilation.
-   Vanilla rollout uses an explicit KV cache whose size is configured via
    `RolloutConfig.kv_cache_size`.
-   Weight updates are applied in-process via `VanillaRollout.update_params(...)`
    (no separate inference server).

### Installation

No extra installation is required beyond installing Tunix and its JAX/Flax
dependencies.

### Choosing vanilla as the Rollout

Set `cluster_config.rollout_engine="vanilla"`.

Rollout engine selection happens in `tunix/rl/rl_cluster.py`.

### Configuration knobs

Tunix uses `tunix.rl.rollout.base_rollout.RolloutConfig` for rollout settings.

#### Vanilla-specific fields

In addition to the common sampling parameters mentioned above, vanilla rollout
uses these fields:

-   `kv_cache_size`

    -   Total KV cache capacity (in tokens) used by the vanilla sampler.
    -   Must satisfy:

        `kv_cache_size >= max_prompt_length + max_tokens_to_generate`

      Otherwise, vanilla rollout raises a `ValueError`.

-   `eos_tokens`

    -   Optional list of token ids that will stop generation.
    -   If unset, the tokenizer's `eos_id` is used.

#### Notes on sampling mode

Vanilla rollout uses:

-   Top-p sampling when `top_p` is set (including the default `top_p=1.0`).
-   Greedy decoding when `top_p=None`.

### Example: using vanilla rollout in a Python entrypoint

Pseudocode (simplified):

```python
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

max_prompt_length = 256
max_tokens_to_generate = 768

rollout_config = base_rollout.RolloutConfig(
    max_tokens_to_generate=max_tokens_to_generate,
    max_prompt_length=max_prompt_length,
    kv_cache_size=max_prompt_length + max_tokens_to_generate + 256,
    temperature=0.9,
    top_p=1.0,
    top_k=50,
    # eos_tokens=[...],  # optional
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh=role_to_mesh,
    rollout_engine="vanilla",
    offload_to_cpu=False,
    training_config=training_config,
    rollout_config=rollout_config,
)
```

### Troubleshooting

#### `Total sampling steps ... must be less than the cache size ...`

- Increase `kv_cache_size`, or reduce `max_prompt_length` / `max_tokens_to_generate`.

#### Unexpected recompilations / slow first step

- Keep `max_prompt_length` fixed across runs.
- Ensure your prompts never exceed `max_prompt_length`; otherwise the sampler
  will round up the prompt length (next power-of-2) and may trigger a recompile.

#### Out-of-memory (HBM)

- Reduce `kv_cache_size` (KV cache scales with batch size and `kv_cache_size`).
- Reduce `max_prompt_length` and/or `max_tokens_to_generate`.


