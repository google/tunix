<!-- DO NOT REMOVE! Placeholder for TOC. -->

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

`vllm` provides better performance with large batch size and agentic RL.

## Sampling Knobs

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

-   `rollout_vllm_server_mode`

    -   `False`: batch inference mode (`vllm.LLM(...)`).
    -   `True`: in-process engine + driver loop (`VLLMInProcessDriver`). Useful
        for higher-throughput request scheduling leveraging the vLLM continuous
        batching capabilities.

-   `rollout_vllm_server_mode_submission_threshold`

    -   Only applies when `rollout_vllm_server_mode=True`.
    -   `0`: drain the submission queue immediately when requests arrive.
    -   `N > 0`: hold queued requests in the in-process driver until at least
        `N` requests have accumulated, then release them to the vLLM engine in
        one drain cycle.

-   `rollout_vllm_server_mode_submission_timeout_s`

    -   Only applies when `rollout_vllm_server_mode=True` and used together with
        `rollout_vllm_server_mode_submission_threshold > 0`.
    -   Flush timeout (seconds) that bounds how long queued requests wait when
        **fewer** than `submission_threshold` accumulate. The clock starts when
        the **first** request of the current window arrives; once it elapses the
        partial batch is drained even though the threshold was not reached. The
        clock resets after each drain.
    -   `0` (default): no timeout — below-threshold requests wait until the
        threshold is met (previous behavior).

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
from tunix.rl import rl_cluster as rl_engine_lib
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
    rollout_vllm_server_mode_submission_threshold=0,
    rollout_vllm_server_mode_submission_timeout_s=0.0,
)

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh=role_to_mesh,
    rollout_engine="vllm",
    offload_to_cpu=False,
    training_config=training_config,
    rollout_config=rollout_config,
)

rl_engine = rl_engine_lib.RLEngine(
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

#### Data parallel issues

- If you set `data_parallel_size > 1`, Tunix sets `NEW_MODEL_DESIGN=1` for vLLM.
- Ensure your rollout mesh size matches `tensor_parallel_size * data_parallel_size`.

#### Weight sync / mapping errors

- If you see errors about missing key mappings, provide a `rollout_mapping_config` or use a model implementation that exposes mapping helpers.
- For MaxText-style models, provide `rollout_vllm_additional_config` including a `maxtext_config` entry (required for direct sync).


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
from tunix.rl import rl_cluster as rl_engine_lib
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

cluster_config = rl_engine_lib.ClusterConfig(
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

## Mock

This section explains how to use the `MockRollout` engine, which is useful for
testing and performance benchmarking the RL pipeline infrastructure (especially
the trainer side) without requiring heavy model weights or accelerator hardware.

### How the integration works

At a high level:

- When `rollout_engine=mock_rollout.MockRollout` (or a `functools.partial` wrapping it), Tunix uses `MockRollout` instead of a real inference engine.
- **Text Generation**: It generates random sequences of words from a dummy list.
- **Latency Simulation**: It sleeps for a random duration between `min_generation_time`
    and `max_generation_time` to simulate inference delay.
- **Tensors**: It returns arrays of zeros for logits and log probabilities as NumPy
    arrays. This keeps the data on the **host** (CPU) memory and avoids device
    memory allocation, making the mock extremely lightweight.
- **RNG Seeding**: If a seed is provided in `rollout_config`, it is used to
    initialize the RNG in `__init__`, ensuring that successive calls to `generate`
    advance the state but remain deterministic as a sequence.

### Choosing mock as the Rollout

Pass `mock_rollout.MockRollout` or `functools.partial(mock_rollout.MockRollout, **kwargs)` to `rollout_engine`.

Rollout engine selection happens in `tunix/rl/rl_cluster.py`.

### Configuration knobs

In addition to common sampling parameters in `RolloutConfig`, you can pass these directly as `kwargs` to `MockRollout` (or via `functools.partial`):

- `min_generation_time`: Minimum sleep time in seconds.
- `max_generation_time`: Maximum sleep time in seconds.
- `length_distribution`: The distribution type for mock generated sequence lengths. Supported modes:
    - `"uniform"`: Random length. Defaults to full range `[1, max_tokens]`. **Best for**: Broad testing of load and handling varying lengths without specific distribution assumptions.
    - `"normal"`: Bell curve. Defaults to `mean = max_tokens / 2`. **Best for**: Testing scenarios where lengths are expected to cluster around a typical response length.
    - `"skewed"`: Right-skewed. Defaults to `mean = max_tokens / 4`. **Best for**: Simulating realistic LLM behavior where most responses are short but a few are very long.
    - `"fixed"`: Exactly `mean` tokens. **Best for**: Deterministic testing or benchmarking specific fixed workloads.
- `length_mean`: Optional float to override the default mean for the distribution.
- `length_std`: Optional float to override the default standard deviation for the distribution.

Note: While `MockRollout` itself can operate without a full `RolloutConfig`, the
`RLEngine` requires a `rollout_config` to be present in `ClusterConfig`.
Essential fields like `max_tokens_to_generate` and `max_prompt_length` from
`RolloutConfig` are used by `MockRollout` to determine the size of generated
outputs.

### Example: using mock rollout in a Python entrypoint

```python
import functools
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import mock_rollout

# Optional: define mock-specific kwargs via partial
mock_engine_cls = functools.partial(
    mock_rollout.MockRollout,
    min_generation_time=2,
    max_generation_time=20,
    length_distribution="normal",
)

rollout_config = base_rollout.RolloutConfig(
    max_tokens_to_generate=768, # Specify to override default
    max_prompt_length=256,  # Specify to override default
    # Other common rollout config fields can be set here,
    # but are not strictly used by MockRollout beyond length constraints.
)

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh=role_to_mesh,
    rollout_engine=mock_engine_cls,
    training_config=training_config,
    rollout_config=rollout_config,
)

rl_engine = rl_engine_lib.RLEngine(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
```
