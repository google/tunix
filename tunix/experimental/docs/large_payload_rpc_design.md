# Design Doc: Overcoming the 4 GB gRPC Payload Limit for Packed Training Batches

<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->

**Author:** `tsbao`
**Status:** Draft (Phase 1 Implemented)
**Components:** [`remote_execution.py`](https://github.com/google/tunix/tree/main/tunix/experimental/worker/remote_execution.py), [`batch_assembly.py`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/batch_assembly.py), [`rl_program.py`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/rl_program.py), [`datatypes.py`](https://github.com/google/tunix/tree/main/tunix/experimental/common/datatypes.py)

---

## 1. Problem Statement & Sizing Analysis

During distributed RL training in [`rl_program.py`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/rl_program.py), the Orchestrator packs rollout trajectories into [`RLTrainerPayload`](https://github.com/google/tunix/tree/main/tunix/experimental/common/datatypes.py) microbatches using [`SequencePackedBatchAssembler`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/batch_assembly.py) or [`PaddedBatchAssembler`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/batch_assembly.py) and dispatches them to `REFERENCE` and `ACTOR` workers via [`DistributedRLEngine.per_token_logps`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/distributed_rl_engine.py) and [`DistributedRLEngine.train_step`](https://github.com/google/tunix/tree/main/tunix/experimental/orchestrator/distributed_rl_engine.py).

### 1.1 Why Payloads Exceed 128 MiB and 4 GiB

1. **Global Mesh Packing (`pack_size = trainer_fsdp * trainer_dp`)**:
   In `create_batch_assembler`, a single microbatch packs `B = trainer_fsdp * trainer_dp` rows, each of length `T = max_seq_token_per_tpu` (or `max_prompt_length + max_response_length`).
2. **Per-Token Array Footprint in `RLTrainerPayload`**:
   - **Standard 2D fields** (`completion_ids`, `completion_mask`, `advantages`, `segment_ids`, `segment_positions`, `old_per_token_logps`, `ref_per_token_logps`, `sampler_is_weights`): **7-9 arrays** x 4 bytes (`int32`/`float32`) = **28-36 bytes/token**.
   - **MoE Router Replay (`routed_experts`)**: Shape `[B, T, num_layers, top_k]` in `int16` (2 bytes). For a 94-layer, top-8 MoE model (e.g. Qwen3-235B / DeepSeek-V3 class), `routed_experts` requires `94 * 8 * 2 = 1,504 bytes/token` (**~50x larger** than all other fields combined).

| Workload Scenario | `B` (`pack_size`) | `T` (`max_packed_len`) | Total Tokens (`B * T`) | Dense Fields Only (32 B/tok) | With MoE `routed_experts` (`L=94, K=8`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Small TPU Slice (Dense)** | 16 | 8,192 | 131K | ~4.2 MB | ~201 MB |
| **Medium Pod (MoE, 32K ctx)** | 64 | 32,768 | 2.1M | ~67 MB | **~3.22 GB** |
| **Large Pod (MoE, 64K ctx)** | 128 | 65,536 | 8.4M | ~268 MB | **~12.88 GB** |
| **Mega Pod (Dense, 128K ctx)** | 512 | 131,072 | 67.1M | **~2.15 GB** | **~103.0 GB** |

### 1.2 Hard Limits in Unary gRPC Transport

1. **Configured Limit (`128 MiB`)**: `_MAX_MESSAGE_BYTES = 128 * 1024 * 1024` rejects any single unary message `> 128 MiB` with `RESOURCE_EXHAUSTED`.
2. **gRPC C-Core Option Limit (`2 GiB`)**: `grpc.max_send_message_length` and `grpc.max_receive_message_length` are signed 32-bit integers (`INT32_MAX = 2^31 - 1` bytes).
3. **gRPC Wire Framing Limit (`4 GiB`)**: The HTTP/2 gRPC length-prefixed frame header uses a 4-byte unsigned integer (`uint32 = 4 GiB` hard ceiling).
4. **3x-4x Host RAM Amplification**: Calling `cloudpickle.dumps(obj)` without `buffer_callback` allocates a single contiguous multi-GB `bytes` buffer in Python even under Pickle Protocol 5, plus a second copy in gRPC C-core slice buffers—causing host OOMs well before 4 GB.

---

## 2. Solution Architecture Overview

We adopt a **three-tiered architecture** that removes the hard 4 GB transport limit immediately (Phase 1) while providing a clear path to shrink wire bytes in subsequent phases:

| Tier | Mechanism | Solves `> 4 GB` Hard Limit? | Eliminates `3x` RAM Spike? | Reduces Network Bytes? | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 1 (Phase 1)** | **Pickle Protocol 5 (`buffer_callback`) + gRPC Chunked Streaming** | **Yes (Unlimited)** | **Yes (Zero-copy views)** | No (unless paired with compression) | **Implemented** |
| **Tier 2 (Phase 2)** | **Out-of-Band Buffer Compression (LZ4/Zstd) + Trainer-Side Packing / Dtype Compacting** | Reduces payload by `2x-10x` | Yes | **Yes (`2x-10x` less traffic)** | Future Work |
| **Tier 3 (Phase 3)** | **Out-of-Band `TrajectoryStore` Pass-by-Reference** | **Yes** | **Yes** | **Yes (Orchestrator sends IDs only)** | Future Work |

---

## 3. Tier 1 Detailed Design: Pickle Protocol 5 + Chunked gRPC Streaming (Phase 1)

### 3.1 Zero-Copy Out-of-Band Serialization (`protocol=5`)

Although `cloudpickle.DEFAULT_PROTOCOL` is already `pickle.HIGHEST_PROTOCOL` (Protocol 5), calling `cloudpickle.dumps(obj)` with `buffer_callback=None` forces `Pickler` to serialize all NumPy arrays **in-band** into a single contiguous `io.BytesIO` buffer.

By passing `buffer_callback=raw_buffers.append` to `cloudpickle.dumps(obj, protocol=5, buffer_callback=raw_buffers.append)`:
1. **Sender (`_iter_serialized_chunks(obj, chunk_size=_STREAM_CHUNK_BYTES)`)**:
   - `header_bytes` contains only structural metadata and `NEXT_BUFFER` opcodes (typically `< 64 KiB`).
   - `raw_buffers` collects zero-copy `pickle.PickleBuffer` views over every NumPy array in `RLTrainerPayload` (`completion_ids`, `routed_experts`, etc.).
   - Yields **Frame 0 (Manifest)**: `cloudpickle.dumps((len(header_bytes), tuple(buffer_lengths)))`.
   - Yields fixed-size slices (`_STREAM_CHUNK_BYTES = 8 MiB`) of `memoryview(header_bytes)` followed by each `pb.raw()` buffer view without ever allocating a single contiguous multi-GB `bytes` buffer.

2. **Receiver (`_deserialize_from_chunks` / `_deserialize_from_async_chunks`)**:
   - Reads Frame 0 to obtain `header_len` and `buffer_lengths`.
   - Pre-allocates exact-sized `bytearray(header_len)` and `[bytearray(n) for n in buffer_lengths]`.
   - Copies incoming `8 MiB` stream frames directly into `memoryview` slices of the pre-allocated `bytearray`s (no `b"".join()` reallocation).
   - Reconstructs the object via `cloudpickle.loads(header_buf, buffers=buffers)` so NumPy arrays directly wrap the pre-allocated `bytearray` memory.

### 3.2 Streaming gRPC Handlers in `GrpcRemoteExecutionServer` and `GrpcRemoteActorHandle`

`GrpcRemoteExecutionServer` and `GrpcRemoteActorHandle` use chunked streaming RPCs directly across all three service endpoints (replacing the legacy unary handlers and single-blob `serialize`/`deserialize` methods):

- `/tunix.ExecutionService/Execute` (`stream_stream`): Streams `ExecutionRequest` chunks from client to server and streams `ExecutionResponse` chunks back (`submit` / `asubmit`).
- `/tunix.ExecutionService/DispatchTask` (`stream_unary`): Streams `ExecutionRequest` chunks and returns the task `request_id` (`dispatch_task`).
- `/tunix.ExecutionService/PollResponses` (`unary_stream`): Sends `timeout_s` and streams `ExecutionResponse` chunks back, or yields 0 frames on timeout (`poll_responses`).

---

## 4. Future Phases (Tier 2 & Tier 3)

### 4.1 Phase 2: Buffer Compression, Dtype Compacting, and Trainer-Side Packing

- **Transparent Out-of-Band Buffer Compression**: Compress large `PickleBuffer`s (`> 1 MiB`) using `lz4` or `zstd` inside `_iter_serialized_chunks`, compressing `-1` padding in `routed_experts` and zero-padded tails by `5x-15x`.
- **Compact Dtypes in `RLTrainerPayload`**: Use `bool`/`uint8` for `prompt_mask` and `completion_mask` (4x smaller than `float32`) and `int16`/`uint16` for `segment_ids` and `segment_positions` (2x smaller than `int32`).
- **Trainer-Side Packing**: Perform bin assignment on the Orchestrator while deferring dense 2D/3D array materialization (`pack_chunk` / `_pack_chunk`) to the Trainer worker.

### 4.2 Phase 3: Decentralized Pass-by-Reference via `TrajectoryStore`

- Wire `StandardRLProgram._trajectory_store` so Rollout and Critique workers write heavy trajectory tensors directly to `TrajectoryStore` keyed by `traj_id`, and the Orchestrator passes only `traj_id` references and scalar advantages to `TrainerWorker`.
