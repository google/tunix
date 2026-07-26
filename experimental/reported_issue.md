## 1. Issue: XLA ExecuteSharded Segfault during vLLM TPU Streaming Load on Submesh

**Date**: 2026-07-26
**Severity**: High

**Symptom**:  
When executing `safetensors.load` on a large TPU cluster (e.g. 256-chip Pathways proxy), if vLLM is confined to a physical submesh (e.g., 64 chips) and forced to load real weights from disk through its `tpu_streaming_loader`, JAX XLA throws a fatal C++ `Segfault` due to an invalid `absl::StatusOr` access during `jax::PyLoadedExecutable::ExecuteSharded`. The job instantly dies when the loading bar is at 0%.

**Setup & Hyperparameters**:
- **Model**: Qwen3-32B (61GB parameter footprint)
- **Engine**: vLLM `TpuPlatform` via Pathways Proxy (`JAX_PLATFORMS=proxy,cpu`, `JAX_BACKEND_TARGET=grpc://127.0.0.1:29000`)
- **Mesh Configuration**: 256-chip cluster (`tpuv5:4x8x8`). vLLM is physically confined to `devices[:64]`, i.e., `fsdp=8, tp=8` (a 4x8x2 block).
- **Fidelity Constraints**: `JAX_RANDOM_WEIGHTS=1` is EXPLICITLY OMITTED here to load true weights off disk for Logp diffing.

**Root Cause / Bottleneck Analysis**:
1. DeepSWE RL baselines normally circumvent this entirely by configuring vLLM with `init_with_random_weights=True` (which skips `safetensors.` entirely). The actual inference weights are later populated in-memory from the Actor.
2. In the `logp-diff-probe`, passing `load_format="auto"` natively routes to vLLM TPU optimization logic (`tpu_streaming_loader.py` doing chunked asynchronous `device_put`).
3. This custom loader's XLA PJIT pipeline fails catastrophically when attempting to shard and execute tensor population mappings over a partial sub-mesh (64 chips mapped from a 256-chip proxy head node).
4. XLA silently aborts the internal transfer but incorrectly returns `absl::OkStatus()` instead of propagating an exception payload. Hence, `ValueOrThrowWrapper` fatally Segfaults when accessing the empty result payload meant for Python execution status tracking.

**Consequence / Conflict**:
Trying to load large real weights from disk directly into a vLLM subset mesh via a Pathways Proxy head node throws unreachable Segfaults, permanently blocking the probe execution. 

**Diagnostic Proof**:
```text
INFO 07-26 16:27:53 [weight_utils.py:922] Filesystem type for checkpoints: EXT4. Checkpoint size: 61.02 GiB. Available RAM: 734.91 GiB.
Loading safetensors checkpoint shards:   0% Completed | 0/17 [00:00<?, ?it/s]
[external/com_google_absl/absl/status/statusor.cc : 77] RAW: An OK status is not a valid constructor argument to StatusOr<T>
!!!!!!! Segfault encountered !!!!!!!
  File "<unknown>", line 0, in absl::lts_20260107::status_internal::StatusRep::Unref() const
  File "<unknown>", line 0, in absl::lts_20260107::internal_statusor::Helper::HandleInvalidStatusCtorArg(absl::lts_20260107::Status*)
  File "<unknown>", line 0, in xla::ifrt::PjRtLoadedExecutable::Execute(absl::lts_20260107::Span<tsl::RCReference<xla::ifrt::Array> >, xla::ifrt::ExecuteOptions const&, std::optional<xla::ifrt::RCReferenceWrapper<xla::ifrt::DeviceList> >)
  File "<unknown>", line 0, in jax::PyLoadedExecutable::ExecuteSharded(std::vector<jax::PyArray, std::allocator<jax::PyArray> >, bool)
  File "<unknown>", line 0, in xla::ValueOrThrowWrapper<absl::lts_20260107::StatusOr<jax::PyExecuteResults> (std::vector<jax::PyArray, std::allocator<jax::PyArray> >, bool), jax::PyLoadedExecutable>::operator()(jax::PyLoadedExecutable&, std::vector<jax::PyArray, std::allocator<jax::PyArray> >, bool) const
...
  File "<unknown>", line 0, in jax::(anonymous namespace)::PjitFunction::Call(nanobind::handle, _object* const*, unsigned long, _object*)
  File "<unknown>", line 0, in PjitFunction_tp_vectorcall
```

**Suggested Fixes**:
Modify the `LLMEngine` / `VllmConfig` kwargs to explicitly bypass `tpu_streaming_loader` and fall back to standard Hugging Face CPU payload preparation by setting `"load_format": "hf"`. Because the Pathways head in this cluster has over 300GB of RAM (safely cushioning the 61GB parameters), CPU-backed parsing correctly scatters arrays without crashing XLA.

**Over to the reviewing agent:**
When doing RL deep dives on large proxy topologies (200GB+ params) that exceed the Proxy Head's RAM limits, falling back to HuggingFace CPU fallback won't work anymore (OOM). We will need the TPU streaming loader. Could you check JAX issues via the code search to ascertain why `PjRtLoadedExecutable::Execute` crashes when `VLLM` restricts `jax.device_put` to a 64-chip `rollout_mesh` on a globally larger 256-chip execution grid? Is a subset enforcer flag like `FLAGS_pathways_enforce_subset_devices_form_subslice=true` required server-side?
