# Raiden FFI Porting Notes and Rollout Output Investigation

## Executive Summary

This debugging session had two intertwined goals:

1. Port trainer-side Raiden FFI support into Tunix and make it user-configurable.
2. Make the full trainer-to-rollout weight sync path work on Pathways proxy runtimes, including the destination side.

The final live validation on the v5e cluster succeeded:

- rollout registered successfully
- trainer-side FFI D2H completed successfully
- rollout-side Pathways destination transport initialized successfully
- source and destination metadata snapshots were collected
- weight synchronization completed
- the orchestrator finished `Step 0` and exited with `EXIT_CODE=0`

The first rollout debug sample looked like gibberish, but that is explainable from the run timeline:

- rollout logged `Pathways dummy weight loading (jax)` during startup
- the debug `Rollout response snapshot` was captured before the first weight sync
- the first successful sync happened later in the run

So the gibberish sample was produced by the pre-sync rollout state, not by the post-sync Raiden FFI path.

## Why the Rollout Output Looked Like Gibberish

### Observed raw output

The debug rollout snapshot contained text like this:

```text
_userdataשינוי(pdfꦟцентрMiddleцентр NEXT csrf impulses outrage terme outrage()){מורся Ancienttheon Ancientのか Ancient来回-tags rated offen⬤ quotationynes⽥myśl_close山西 אזרחיkeley珊󠄁いただきました Emperor大卫旷 śwież논病理 Launch.Parameter []

чет �rtl�也知道ToStrئةacteriaน้ำadows.sendKeys-passocoder_Rectedula傥叙述ryptawns
```

This is not coherent reasoning and not a valid GSM8K answer.

### Why this happened

The most likely explanation is a combination of three facts from the successful run:

1. The rollout side started from dummy weights.
   - Rollout logs showed: `Pathways dummy weight loading (jax) took 11.73s`.
   - That means the initial rollout policy was not yet the real trainer policy.

2. The gibberish sample was collected before the first successful weight sync.
   - Rollout requests were created at `02:19:10` and `02:19:14`.
   - The raw rollout response snapshot was logged at `02:19:17`.
   - `>>> Step 0 starting` happened at `02:19:20`.
   - `Synchronizing weights` happened later at `02:19:36`.
   - `Weight synchronization complete` happened at `02:20:48`.

3. Sampling was not deterministic/greedy.
   - The rollout runtime logged that vLLM overrode defaults from `generation_config.json` with:
     - `temperature: 0.6`
     - `top_k: 20`
     - `top_p: 0.95`
   - That makes nonsense outputs more likely if the starting weights are dummy or otherwise poor.

### Important consequence

The debug snapshot tells us that the pre-sync rollout policy was bad. It does **not** tell us that the post-sync FFI path corrupted weights, because the sample was taken before the sync completed.

### What to do if you want to inspect post-sync output quality

To verify whether the synced rollout model talks normally after weight sync, do one of these:

1. Run for more than one training step so there is at least one rollout after the first successful sync.
2. Add a forced post-sync probe rollout after `Weight synchronization complete`.
3. Use a more stable decode setup for debugging, for example lower temperature or deterministic decoding.
4. Use an instruction-tuned model if the intent is to inspect reasoning quality rather than just transport correctness.

## Change-by-Change Explanation

This section explains the changes I made, why each one exists, and what would break without it.

## 1. `tunix/experimental/weight_sync/raiden_synchronizer.py`

This file absorbed most of the real transport logic changes.

### 1.1 Replace `host_stage` with explicit `use_ffi`

#### What changed

- Removed the old `host_stage` behavior.
- Added `use_ffi: Optional[bool] = None`.
- Defaulted `use_ffi` to `True` when `JAX_PLATFORMS` contains `proxy`.
- Added a `use_ffi` property.

#### Why

The old code path assumed proxy-backed arrays needed to be copied to host CPU memory before native binding. That is not the same thing as Pathways FFI mode. FFI needs a distinct code path, not just host staging.

#### Without this change

The trainer-side Pathways proxy runtime would try to follow the wrong path and fail later in transport setup.

#### Failure symptoms without it

- trainer-side arrays would not go through the FFI path
- rollout-side and trainer-side behavior would be conflated
- later errors would include missing shard addresses or incompatible proxy/native binding behavior

### 1.2 Add `_ensure_ffi_compute_on_compat()`

#### What changed

Added a compatibility shim that patches `jax._src.compute_on.compute_on` to `compute_on2` when the installed JAX exposes the old decorator signature.

#### Why

The prebuilt TPU-sync wheel was built against a newer JAX `compute_on` API that accepts `out_memory_spaces`. The original runtime had a mismatch.

#### Without this change

Trainer-side FFI setup would fail before any successful D2H.

#### Actual error without it

```text
TypeError: compute_on() got an unexpected keyword argument 'out_memory_spaces'
```

### 1.3 Add `unpack_ip()` and FFI metadata address handling

#### What changed

Added code to decode the FFI metadata rows into shard addresses and control-plane listener addresses.

#### Why

The FFI transport returns packed metadata for each participating rank. The coordinator cannot register or transfer work units unless it knows the data-plane shard addresses and control-plane listener addresses.

#### Without this change

The coordinator would not have usable addresses for source or destination work units.

#### Failure symptoms without it

- empty or missing shard metadata
- registration failures before transfer starts

### 1.4 Make `_bindable()` and `_filter_bindable()` optionally allow proxy-backed arrays

#### What changed

- `_bindable()` now accepts `allow_proxy=False`
- `_filter_bindable()` forwards `allow_proxy`
- proxy device platform is accepted when `allow_proxy=True`

#### Why
nPathways FFI must bind proxy-backed arrays directly. The old native filter rejected them.

#### Without this change

The FFI path would see proxy arrays as unbindable and drop them.

#### Failure symptoms without it

- empty array set in FFI mode
- no useful transport metadata
- registration or sync failure due to missing source or destination tensors

### 1.5 Add `_init_ffi_transport(execute_d2h: bool)`

#### What changed

Added one shared helper that initializes Pathways FFI transport for both:

- source mode: `init_weight_synchronizer_and_d2h`
- destination mode: `init_weight_synchronizer`

It also:

- computes slice byte sizes
- builds `shard_idx`
- calculates `devices_per_host`
- gathers transport metadata across processes
- stores decoded shard/control addresses

#### Why

We needed both source-side FFI and destination-side FFI support under Pathways, while still preserving native McJax behavior when not in FFI mode.

#### Without this change

The source and destination Pathways flows could not share consistent transport initialization logic.

#### Failure symptoms without it

- trainer-side FFI D2H unsupported or incomplete
- rollout-side Pathways H2D unsupported or hanging
- missing metadata for coordinator registration

### 1.6 Add rollout-side Pathways destination init branch

#### What changed

When `use_ffi=True` and `auto_h2d=True`, `bind()` now eagerly initializes the destination FFI transport and records its shard/control addresses.

#### Why

The rollout warmup path on Pathways needs transport initialization before the coordinator later requests destination metadata or H2D.

#### Without this change

Rollout would warm up without usable destination transport addresses.

#### Actual failure without it

Earlier we saw:

```text
ValueError: work unit WorkUnitId(job_name='rollout', ...) registered without any data-plane address; the synchronizer must be constructed before registration so its assigned ports are known
```

### 1.7 Add `_ffi_h2d()` and call it from `h2d()`

#### What changed

Added destination-side FFI H2D execution using `_raiden_ffi.multi_h2d(...)`.

#### Why

Pathways destination rollout workers cannot rely on the native McJax `WeightSynchronizer.h2d()` path when operating through proxy-backed arrays. They need the FFI H2D path.

#### Without this change

The destination side would initialize but fail to ingest synced weights onto device memory correctly.

#### Failure symptoms without it

- destination transport may initialize, but weights would not land on device properly
- sync round would stall or silently fail later

### 1.8 Inline the destination `init_weight_synchronizer` shard-map wrapper

#### What changed

Instead of calling the wheel helper directly for destination init, I inlined a wrapper that uses `self.arrays[0].sharding.spec` for the anchor input instead of a too-long all-axis spec.

#### Why

The wheel helper assumes an input partition spec shaped by all mesh axes. That is not valid for lower-rank rollout tensors on the 7-axis Pathways mesh.

#### Without this change

The rollout warmup would crash inside destination FFI init.

#### Actual error without it

```text
ValueError: shard_map applied to the function 'wrapped' was given an in_specs entry which is too long to be compatible with the corresponding input value
```

Specifically, the helper tried to apply a 7-axis input spec to a rank-2 tensor.

### 1.9 Expand `active` and `work_unit_metadata()` for FFI mode

#### What changed

- `active` now returns `True` if native sync exists or FFI shard IPs exist
- `work_unit_metadata()` emits shard and control addresses from FFI state when `use_ffi=True`

#### Why

The coordinator must see FFI destination and source transports as active and registrable.

#### Without this change

FFI transports could succeed locally but still appear inactive or address-less to the coordinator.

#### Failure symptoms without it

- missing or empty metadata
- registration failures
- incorrect state during round coordination

## 2. `tunix/experimental/weight_sync/raiden_weight_sync_delegate.py`

This file controls rollout-side weight sync delegation.

### 2.1 Stop forcing rollout to the native non-FFI path

#### What changed

Originally I temporarily forced `use_ffi=False` for rollout to prove that trainer-only FFI was the immediate source of one failure. After confirming that native Pathways rollout bind hung, I removed that override and let the shared synchronizer choose the Pathways FFI destination path.

#### Why

The real requirement became: both source and destination need Pathways-aware code, while McJax still needs to work in non-proxy environments.

#### Without this final change

Rollout would hang in native Pathways warmup.

#### Failure symptom without it

Rollout stalled at:

```text
Eagerly warming up Raiden weight sync...
```

and never registered to discovery.

### 2.2 Add rollout bind diagnostics

#### What changed

Added logging around bind/warmup so we could see:

- whether rollout was using FFI
- how many arrays were staged
- whether destination transport became active

#### Why

These logs were necessary to distinguish:

- warmup hang before bind
- zero bindable arrays
- failure inside FFI init
- successful registration metadata creation

#### Without this change

We would not have been able to localize the rollout hang precisely.

## 3. `tunix/experimental/examples/math_gsm8k_dist/run_trainer_node.py`

### 3.1 Add `--weight_sync_use_ffi`

#### What changed

Added a trainer CLI flag with values like:

- `auto`
- `true`
- `false`

and wired it into trainer synchronizer construction.

#### Why

The original behavior auto-enabled FFI when `JAX_PLATFORMS=proxy`. You asked for explicit user control.

#### Without this change

FFI behavior would remain implicit and tied to environment detection only.

#### Failure mode without it

Not a crash, but a configuration limitation: no way to force-enable or force-disable trainer-side FFI independently.

## 4. `tunix/experimental/train/peft_trainer_v2.py`

### 4.1 Build the trainer synchronizer with `use_ffi`

#### What changed

The default trainer worker factory now creates `RaidenSynchronizer("trainer", use_ffi=is_proxy)`.

#### Why

This is the main entry point for trainer-side FFI routing.

#### Without this change

Trainer would never enter the Pathways FFI source path under the intended configuration.

### 4.2 Add `set_target_state()` and use rollout-shaped conversion in `prepare_weight_sync()`

#### What changed

Added `_target_state` injection and changed `prepare_weight_sync()` to convert trainer state into rollout-shaped state before binding, when mappings and rollout target state are available.

#### Why

The source manifest must match the rollout model’s expected parameter tree, names, and shapes, not the raw trainer tree.

#### Without this change

The coordinator would compare mismatched manifests and abort before downtime.

#### Actual failure without it

The sync round failed during manifest preflight with messages like:

```text
manifest preflight failed before any destination was quiesced; no rollback needed; final state preparing
```

and earlier mismatch logs showed source and destination metadata did not align.

## 5. `tunix/experimental/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py`

### 5.1 Configure trainer target state from rollout worker

#### What changed

Added `_configure_trainer_target_state(...)` that:

- fetches rollout target state through RPC
- sends it to the trainer before orchestration begins

#### Why

The trainer cannot do rollout-shaped conversion unless it knows the rollout target-state skeleton.

#### Without this change

The trainer would bind its own raw state instead of rollout-aligned state.

#### Failure without it

This fed directly into the manifest preflight failure described above.

### 5.2 Make debug logging actually honor `--debug`

#### What changed

Updated logging setup so debug mode really enables the new rollout and metadata snapshots.

#### Why

We needed live source/destination metadata dumps and rollout output snapshots to diagnose the real failures.

#### Without this change

Important debug logs would stay hidden even with `--debug` requested.

## 6. `tunix/experimental/worker/trainer_worker.py`

### 6.1 Add `set_target_state()` RPC

#### What changed

Added an RPC-facing `set_target_state()` method that forwards target-state injection into the trainer implementation.

#### Why

The orchestrator needed a supported control-plane path to push the rollout target state into the trainer process.

#### Without this change

There would be no trainer-side endpoint for the new target-state wiring.

## 7. `tunix/experimental/rollout/manager.py`

### 7.1 Add `get_target_state()`

#### What changed

Added a rollout-manager method that returns the sampler target-state skeleton.

#### Why

This is the source-side of the new trainer-target-state RPC plumbing.

#### Without this change

The orchestrator could not fetch rollout target state for trainer-side conversion.

## 8. `tunix/experimental/worker/rollout_worker.py`

### 8.1 Add `get_target_state()` RPC

#### What changed

Added an RPC-facing rollout worker method that exposes rollout target state.

#### Why

The orchestrator talks to the rollout worker, not directly to the manager.

#### Without this change

Target-state fetch would fail at the RPC boundary.

### 8.2 Make `initialize()` idempotent when already `READY`

#### What changed

Adjusted rollout worker initialization to tolerate the already-ready case.

#### Why

Fetching target state can initialize the rollout early. Later normal bring-up re-entered initialization and failed.

#### Without this change

The rollout worker would fail during bring-up after the new target-state RPC path was introduced.

#### Actual error without it

```text
Invalid transition from READY to INITIALIZING
```

## 9. `tunix/experimental/orchestrator/distributed_rl_engine.py`

### 9.1 Add rollout response snapshot logging

#### What changed

Added one-time debug logging for raw rollout responses and trajectory conversion.

#### Why

This is how we captured the gibberish output and proved the parser was seeing bad raw text rather than inventing it.

#### Without this change

We would not know whether the rollout problem was in text generation or in post-processing.

### 9.2 Fix debug gate to use stdlib logging instead of `absl.logging.getLogger()`

#### What changed

Replaced the invalid `absl.logging.getLogger()` usage with stdlib logging checks.

#### Why

`absl.logging` does not expose the same API as stdlib `logging`.

#### Without this change

The polling loop crashed during debug logging.

#### Actual error without it

```text
Error in polling_stage: module 'absl.logging' has no attribute 'getLogger'
```

## 10. `tunix/experimental/weight_sync/weight_sync_coordinator.py`

### 10.1 Add source/destination metadata snapshots and mismatch logging

#### What changed

Added one-time debug dumps for:

- source metadata
- destination metadata
- per-entry mismatch diagnostics

#### Why

These logs were necessary to separate three different classes of failure:

- source/destination manifest mismatch
- missing destination data-plane addresses
- later transport/runtime issues after metadata already matched

#### Without this change

We would only see the high-level round failure wrapper and not the real reason.

## 11. `tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh`

### 11.1 Expose trainer-side FFI control and Pathways image overrides

#### What changed

Added launcher support for:

- `WEIGHT_SYNC_USE_FFI`
- `PATHWAYS_SERVER_IMAGE`
- `PATHWAYS_PROXY_IMAGE`
- explicit cluster target selection

#### Why

This let us run the exact trainer-side FFI configuration and the Raiden-capable Pathways images required for the test.

#### Without this change

We would either use the wrong Pathways images or have no easy way to drive the trainer FFI setting from the launch command.

#### Actual failure without the image override

Using the generic Pathways images produced:

```text
No FFI handler registered for init_weight_synchronizer_and_d2h on a platform Host
```

## 12. `tunix/experimental/rollout/inprocess_vllm_sampler_adapter.py`

### 12.1 Make repeated bind effectively idempotent

#### What changed

Adjusted the adapter so `bind_weight_sync()` returns early if the Raiden delegate is already bound.

#### Why

The same rollout process can be asked to bind multiple times across phases. The arrays themselves do not change between those calls.

#### Without this change

Repeated bind attempts would fail even though the bound transport was still valid.

#### Actual error without it

```text
RuntimeError: InprocessVllmSamplerAdapter [lancewang-roll] weight sync delegate is already bound
```

## Test Changes

The test updates were not cosmetic. Each one locked in a real bug fix.

### `tests/experimental/weight_sync/raiden_synchronizer_test.py`

Added coverage for:

- proxy arrays being accepted in FFI mode
- auto-defaulting to FFI under `JAX_PLATFORMS=proxy`
- source-side FFI routing through D2H init
- destination-side FFI routing through bind-time init and H2D
- compute_on compatibility shim behavior

### `tests/experimental/weight_sync/raiden_weight_sync_delegate_test.py`

Added/updated coverage for:

- rollout delegate config shape
- bind once semantics
- compatibility with the new synchronizer surface (`use_ffi`, `active`, `arrays`)

### `tests/experimental/rollout/inprocess_vllm_sampler_adapter_test.py`

Added an idempotence regression test for repeated bind.

### `tests/experimental/weight_sync/raiden_integration_test.py`

Split the test into:

- native source + native destination
- trainer FFI source + native destination

This validated the intended mixed-mode source/destination configuration.

### `tests/experimental/train/peft_trainer_v2_weight_sync_test.py`

Updated the fake synchronizer API from `host_stage` to `use_ffi` and verified trainer-side factory injection still overrides the default.

### `tests/experimental/examples/math_gsm8k_dist/run_trainer_node_test.py`

Added CLI coverage for `--weight_sync_use_ffi` defaulting and explicit override.

## Chronology of the Major Failures and Their Fixes

1. Repeated rollout bind failed.
   - Error: already bound
   - Fix: make repeated bind idempotent

2. Trainer FFI wheel mismatched JAX API.
   - Error: `compute_on(... out_memory_spaces=...)`
   - Fix: add compatibility shim and upgrade JAX runtime

3. Generic Pathways images lacked the required FFI handler.
   - Error: no FFI handler registered
   - Fix: use Raiden-capable Pathways images

4. Source manifest did not match destination manifest.
   - Error: manifest preflight failure
   - Fix: fetch rollout target state and convert trainer state to rollout shape before sync

5. New target-state RPC caused rollout lifecycle re-entry.
   - Error: invalid transition READY -> INITIALIZING
   - Fix: make rollout initialize idempotent

6. Debug polling path crashed.
   - Error: `absl.logging.getLogger`
   - Fix: use stdlib logging API

7. Rollout work unit had no data-plane address.
   - Error: work unit registered without any data-plane address
   - Fix: initialize destination transport before metadata registration

8. Native Pathways destination bind hung during warmup.
   - Symptom: rollout stuck at `Eagerly warming up Raiden weight sync...`
   - Fix: add Pathways destination FFI init/H2D path

9. Destination FFI wheel helper used an invalid shard-map input spec.
   - Error: `in_specs entry which is too long`
   - Fix: inline the init wrapper with the real anchor array sharding spec

10. End-to-end sync succeeded.
   - Evidence: source metadata snapshot, destination metadata snapshot, destination checksums, `Weight synchronization complete`, `<<< Step 0 finished`, `EXIT_CODE=0`

## Remaining Known Caveat

There is still a rollout-side XLA warning during destination FFI init:

```text
Invalid backend config found for instruction ... init_weight_synchronizer ... unexpected character '='; expected ':'
```

This warning was non-fatal in the successful run, because rollout still:

- completed transport init
- registered to discovery
- participated in the sync round
- emitted destination checksums
- allowed the orchestrator to complete successfully

So this is cleanup work, not a correctness blocker for the successful validation we just ran.

## Current Conclusion

- The transport changes are working end to end on the v5e Pathways cluster.
- The gibberish rollout sample is best explained by the fact that it was captured before the first real trainer-to-rollout sync, while rollout was still using dummy-loaded weights and non-greedy decoding.
- If the next goal is to judge answer quality after sync, we should run a probe that happens after `Weight synchronization complete`, not before it.
