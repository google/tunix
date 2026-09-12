# Gemma 4 E2B-IT default FrozenLake: offline implementation handoff

## Read this boundary first

This is an **in-progress implementation**, not a runnable three-arm release.
Native/TIS carriers are implemented and CPU consumer-tested. The Zero script
can resolve its plan, but deliberately refuses execution: the Gemma canonical
serving/trainer adapter and independent B/D collectors are not implemented.
Do not remove that refusal merely because the CPU cache VJP tests pass.

On 2026-09-12 the user approved committing and pushing this in-progress
handoff to `yuxzhang/canon-zero-tim` for another agent to take over development.
This approval does not authorize TPU launch, Kubernetes operations or image
publication. The older Qwen/E4B launch approvals do not apply.

Source worktree: `/home/yuxuan/code_rl_repro/worktrees/gemma4_e2b_default_three_arm_0911`.
Branch: `local/gemma4-e2b-default-three-arm-0911`.
Historical development base: `79fe3572220415c03c46661c6a09d41b1523383f`.
The delivery stack must be rebased onto the latest shared branch and tested
again. Use the clean delivered commit, not that historical base, for new work.
Authoritative phase ledger, outside Git:
`/home/yuxuan/code_rl_repro/tasks/zero_tim_gemma_frozenlake_eval/`.
Read this revision's `canon-zero-tim/AGENTS.md` and canonical branch/flag skills.
Do not continue the old E4B P45/M15 launch order; that worktree is preserved.

## Receiving agent: development ownership and phase order

The receiving agent owns the remaining implementation as well as the future
experiment handoff. This is not a TPU-operator-only assignment. The external
phase ledger above is local to the originating machine; this file carries the
portable current scope and gates if that directory is unavailable remotely.

1. Resolve the delivered full commit and clean worktree; read its package
   AGENTS and canonical skills. Run the CPU/image gate below in a fresh output
   directory before changing code. Existing r11/r12 receipts describe the old
   baseline; they do not certify a rebased tree without fresh validation.
2. E0/E1 have CPU construction evidence. E2 is the only active phase: complete
   the six CODE/INTEGRATION items listed below, starting with runtime/weight/
   memory admission and the installed Gemma model binding. Preserve all stock
   Native/TIS behavior and the Zero execution refusal until its path exists.
3. E3 has only a CPU cache/reference candidate. Wire it to real engine RPA,
   prove independent forward/gradient negatives, and keep full-shape/TPU
   validation explicitly pending. Do not inherit Qwen/E4B certificates.
4. E4 completes only when a runnable three-arm implementation and its local
   and pinned-image evidence are packaged. Keep durable state, plan and log
   in one task directory on the receiving machine, linking this handoff.
5. E5 requires separate launch approval and follows the target ladder below.
   Zero qualification requires three updates and fresh-process resume after
   the first-update gate, before the fresh five-update comparison. The
   default save interval does not itself produce this recovery checkpoint;
   implement an explicit qualification path without changing the comparison.

The first safe command after checkout is the CPU-only `run_cpu_gate.py`
command below. A plan printed by the Zero script is not an executable adapter.

## Locked comparison

Use the existing `examples/frozenlake/configs/gemma4_e2b.yaml` defaults,
with all roles on the same DP1 x TP4 four-device mesh. No P45/M15 selectors.

| Arm | Programs | Actual loss-old source | Sampler correction |
|---|---|---|---|
| Native | stock vLLM + stock Tunix | rollout A | absent |
| TIS | same stock programs | frozen trainer C | detached token min(exp(clip(C-A,-20,20)),2) |
| Zero | pending Gemma canonical engine/trainer | A after strict proof | absent |

Native is the historical mismatch control, not trainer-old/no-IS Standard.
The YAML's original sampler-is default is TIS; Native explicitly removes it.
All arms retain B64, G8, mini64/micro2, prompt/response2048/2048, eight
environment turns, seed42, temperature.7/top-p1/top-k0, GSPO-token/RLOO,
epsilon.003/.005, beta0, AdamW1e-6/.9/.95/weight-decay0/clip100, BF16 compute
and FP32 actor storage. Existing CLI resolution gives warmup.5, decay9 and
32 gradient accumulation groups; compute-logps microbatch is absent/None.
The selected horizon is **five updates**, not the old Python recipe's450.

The full train split must match the actual default generator: 10000 rows,
seed42, grids2..9, p in [.6,.85]. Train/test files are hashed; test is not
consumed by this five-update carrier. Reusing a materialized M15 dataset is
rejected, even if its filename is train.parquet. All arms use identical input
manifest and initial weights, but live trajectories may differ after updates.

E2B geometry: 35 layers, hidden1536, vocabulary262144, query heads8/KV heads1,
local head dim256/global512, local window512, PLE256. Producers13/14 feed the
20 shared-KV consumer layers. Consumers have doubled FFN width and must not
write their own KV. CPU tests reduce widths/vocabulary, never layer count.

## Runtime identity

Local Docker image configuration ID (NOT registry manifest digest):
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
Versions: vllm-tpu0.25.0, tpu_inference0.25.0, Flax0.12.4,
JAX/JAXLIB0.10.2. **Not Flax0.12.8.**
Installed `tpu_inference/models/jax/gemma4.py` SHA256:
`5b93b3b3dd5d42374c0115a4523661ed584191fb0cb9fc9639ae2248b3896fbd`.
An alternative image is a new dependency admission, not an alias.

The E2B HF revision and actual payload SHA are still operator inputs, not
the E4B revision from historical records. Materialize a self-contained local
snapshot: external HF cache symlinks are rejected. The input manifest binds
every weight shard and tokenizer/chat side file. Declaring a revision is not
proof that loaded serving/trainer weight values are identical; that remains
a target gate. No model download was performed in this work period.

## Code and current evidence

- `examples/frozenlake/gemma4_tim/recipe.py`: strict default resolver; no
  CANON/FL/T foreign environment keys, including empty/zero values.
- `run.py`, `worker.py`, `pipeline.py`: clean-source/input checks and actual
  stock GRPO execution. Worker completion and all process writers precede
  terminal status, manifest creation, Python verification and sha256sum -c.
- `admission.py`: full B64/G8 rollout-only gate, 512 unique trajectories,
  no optimizer advancement; raw tokens/actions/A/prompts persisted per group.
- `observer.py`: actual loss-source assertion, A/C arrays, masks, trajectory
  identities, policy version and unique coverage. One device fetch per group,
  not per token. B/D remain explicitly NOT_OBSERVED.
- `cache_vjp.py`: candidate sliding/global, producer/read-only cache
  pullback. Both attention and cache cotangents included; kernel primal
  unchanged. Not wired to the installed RPA or any production caller.
- `reference.py`: ordinary real-Tunix oracle, not canonical C or engine B.
- `tests/rl/gemma4_tim_*test.py`: construction and negative controls.

Only shared runtime edit: optional default-absent `processed_batch_observer`
in GRPOLearner. No existing loss, optimizer, Qwen profile, model or installer
numerics changed. Existing P33 tests now use their real registered GSM8K
workload spec, complete16-group finite-gradient receipts and a fake global
step counter. The logging test unregisters only its own scalar callbacks.
These are test compatibility/isolation fixes, not production/Qwen changes.

Latest complete receipt: `e2b_default_cpu_r11` is GREEN for offline
construction: Gemma37/37, existing GRPO49/49, canonical3/3, E4B3/3,
host recipe10/10, artifacts5/5, flags410/410 and diff check. Its SHA256SUMS
digest is `996b9ed3514838116ccd93b6fbc6df72ae9d5bba7b7518d17d927efeca1c2b98`;
all11 manifest members verify. T0 was18/18 in r7. All r1..r10 receipts remain
unchanged and SHA-verified; their RED statuses were not rewritten.

The installed-method PLE probe found469 differing elements, max
4.76837158203125e-7. The CPU gate passes because this is a live diagnostic of
stock program disagreement, not because PLE or full-engine parity passed.

## Safe local commands (no TPU)

Run from the source root:

```bash
bash examples/frozenlake/run_gemma4_e2b_native.sh
bash examples/frozenlake/run_gemma4_e2b_tis.sh
bash examples/frozenlake/run_gemma4_e2b_zero.sh
python3 canon-zero-tim/tests/gemma4_e2b/run_cpu_gate.py --image --output /absolute/fresh/evidence-directory
```

The first three commands print plans only. Duplicate arm/unknown overrides
fail closed. The image gate is network-disabled/read-only/CPU-only, without
TPU device mounts. Optional `--bench /absolute/path/to/three_lane_system/bench/run_bench.py`
also runs the external T0 gate self-tests. Do not use an existing evidence dir.

## Remaining implementation before a three-arm TPU handoff

1. Finish target admission instrumentation: external image-ID/environment
   attestation, actual E2B live-weight equality and HBM/optimizer inventory.
   The current stock-admission GREEN proves only full rollout completion,
   not those missing receipts. The Python entrypoint checks versions and
   Gemma source hash; it cannot by itself prove the enclosing container ID.
2. Bind the actual installed engine model and E2B weight tree to a
   differentiable Gemma adapter; preserve Qwen dispatch/defaults.
3. Canonicalize serving and trainer projection/norm/PLE/head/logsoftmax
   boundaries with an explicit shape ledger. Native PLE scales weights
   before matmul while the engine scales outputs afterwards; do not alter
   the stock control to hide this difference.
4. Bind the cache VJP to actual RPA metadata, kernel order and return order,
   including replicated KV at TP4 and checked VMA. Active shared physical
   page aliases are currently unsupported and rejected in the CPU candidate.
5. Independent full-history engine B using actual processed sampling
   transforms; actual value-and-grad D from the training loss. A/C alone
   does not admit Zero. Prove collector on/off neutrality.
6. Full-depth target gradient oracle/health and a real first AdamW
   transaction/sync/checkpoint-resume proof. No Qwen gradient norm or E4B
   parameter-count acceptance may be inherited.

These are still CODE/INTEGRATION tasks, not just experiments for a TPU
operator to run. The CPU suite does not certify these missing paths.
The dense backward replica is an arithmetic candidate, not a performance-
admitted production kernel; full-shape compilation and cache-gradient HBM
remain unknown. Global/local head dimensions need separate kernel coverage.

## Future one-host experiment order (not launch approval)

On an exclusively reserved v5p host with the admitted image and a reviewed
clean implementation commit:

1. Stock admission: full default rollout, exact loaded values/tokenizer/data,
   four devices/mesh order, HBM at load/optimizer/cache/rollout, 512/512 rows,
   zero updates. Preserve failure artifacts. Do not shrink the workload to
   label a fit failure green.
2. Native five-update carrier; all five512-row receipts, actual optimizer/
   sync0..5, finite gradients and checkpoint inventory. A/C mismatch measured,
   not required to be zero. Missing B/D recorded as missing.
3. TIS same five-update workload and inputs, real trainer-old source and
   detached token correction verified. No canonical numerical flags.
4. After the missing Zero integration is completed: A-B, B-C and C-D exact
   action coverage, full35-layer/TP4 gradient gate, then one update and
   next-policy strict checks. Continue qualification to three updates and
   fresh-process checkpoint/resume before the matched run; failures stop at
   their first boundary. A successful first update alone is not qualification.
5. Fresh Zero five-update comparison, with identical initial input binding.

For performance prefer the same exclusive host for the three measurements;
the second host can qualify/debug. Two different hosts are not a free matched
timing pair. Report compile/startup separately from the five-update wall time,
trajectory/action-token counts, individual update times and observer time.
Do not subtract cumulative observer time from overlapped wall time or call a
five-update cold-start result steady-state throughput/convergence.

### Candidate Native/TIS commands inside an admitted runtime

These commands do not select or launch a TPU host. Variables must be supplied
by the operator, with absolute paths and full reviewed source/model SHAs.
Use a full clean checkout, or mount its Git common directory too; an isolated
worktree's .git pointer must resolve inside the container.

```bash
python -m examples.frozenlake.gemma4_tim.artifacts seal-inputs \
  --snapshot "$GEMMA_SNAPSHOT" --data "$GEMMA_DATA" \
  --revision "$GEMMA_MODEL_REVISION" --output "$GEMMA_INPUT_MANIFEST"

bash examples/frozenlake/run_gemma4_e2b_native.sh --execute --stage stock-admission \
  --source-sha "$GEMMA_SOURCE_SHA" --snapshot "$GEMMA_SNAPSHOT" --data "$GEMMA_DATA" \
  --input-manifest "$GEMMA_INPUT_MANIFEST" --input-sha "$GEMMA_INPUT_SHA" \
  --output "$GEMMA_ADMISSION_OUTPUT"

bash examples/frozenlake/run_gemma4_e2b_native.sh --execute --stage train \
  --source-sha "$GEMMA_SOURCE_SHA" --snapshot "$GEMMA_SNAPSHOT" --data "$GEMMA_DATA" \
  --input-manifest "$GEMMA_INPUT_MANIFEST" --input-sha "$GEMMA_INPUT_SHA" \
  --output "$GEMMA_NATIVE_OUTPUT"

bash examples/frozenlake/run_gemma4_e2b_tis.sh --execute --stage train \
  --source-sha "$GEMMA_SOURCE_SHA" --snapshot "$GEMMA_SNAPSHOT" --data "$GEMMA_DATA" \
  --input-manifest "$GEMMA_INPUT_MANIFEST" --input-sha "$GEMMA_INPUT_SHA" \
  --output "$GEMMA_TIS_OUTPUT"
```

Set GEMMA_INPUT_SHA from the seal-inputs output. All output directories must
be fresh and outside the clean source checkout. There is deliberately no
Zero --execute command until implementation exists. Do not pass a Qwen flag
bundle, install unpinned dependencies, or run the original xtrace example
with credentials in arguments. Verify every SHA256SUMS before analysis;
classification GREEN has an explicit limited claim, not implicit TARGET PASS.
