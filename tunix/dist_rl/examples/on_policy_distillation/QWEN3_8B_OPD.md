# Qwen3 8B from Qwen3 32B Sampled Token OPD Recipe

This recipe trains a fixed Qwen3-8B student on its own GSM8K trajectories using a frozen Qwen3-32B teacher. It includes immutable model preparation, a supervised teacher process, a complete training profile, baseline/final generation evaluation, checkpoint-only evaluation, and experiment records. The existing 0.6B two-update smoke recipe remains available separately.

The default objective is pure sampled-token OPD: task advantage scale 0, teacher coefficient 1, and reference beta 0. Student sampling uses the complete temperature-one distribution. The teacher scores the exact generated token IDs; no top-k candidate-distribution API is used.

## Fixed Models and Data

| Role | Hugging Face model | Immutable revision |
| --- | --- | --- |
| Student | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Teacher | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |

The model IDs and revisions are constants in [models.py](models.py). Environment variables cannot replace them. Preparation downloads only the pinned safetensors, configuration, and tokenizer artifacts, with remote model code disabled. It verifies the shard index, required tensors, tokenizer semantics, and manifest consistency without loading model tensors onto an accelerator. The manifest records small-artifact hashes and shard sizes; it does not claim cryptographic attestation of every remote weight byte.

The actual tokenizer/configuration files at these revisions have been checked with the production tokenizer compatibility code. Their shared tokenizer fingerprint is:

```text
6b06d0afe078c85bc53ce525ebc9cee9ae68e8ef8c4022bc8e04364c4aaaf051
```

Training uses `gsm8k:1.0.0`, split `train`, shuffled with seed 42. The 300-update budget consumes 4,800 distinct prompts from the 7,473 training examples. The iterator does not wrap under this budget. Evaluation uses the entire official `test` split, 1,319 examples in fixed order. Test answers are held on the control plane and never sent in rollout prompts or environment metadata.

Both phases use the existing raw VTC prompt from `tunix.utils.gsm8k_vtc`: the prompt opens a reasoning block and requests a boxed numeric answer inside answer tags. This is a task-specific prompt and grading protocol, not a claim of comparability with every published GSM8K leaderboard.

## Training and Evaluation Configuration

The complete source configuration is [qwen3_8b_from_qwen3_32b.json](configs/qwen3_8b_from_qwen3_32b.json). All training, evaluation, hardware, and teacher-service settings in that profile are consumed by the entry points below.

| Setting | Default |
| --- | --- |
| Optimizer updates | 300 |
| Prompt groups per update | 16 |
| Responses per prompt | 4 |
| Trajectories per update | 64 |
| Training microbatch | 1 trajectory; 64 accumulation microbatches |
| Training prompt/response budgets | 1,024 / 4,096 tokens |
| Student sampling | Temperature 1, top-p 1, top-k disabled, no penalties or support filtering |
| Objective | OPD coefficient 1, task advantage scale 0, reference beta 0 |
| Optimizer | AdamW, constant LR `1e-6`, weight decay `0.1`, betas `0.9/0.98`, epsilon `1e-8`, gradient norm cap 1 |
| Parameters | Full-parameter training, bfloat16 model/load, decoder rematerialization |
| Policy freshness | Zero staleness, no in-flight updates, Raiden weight synchronization |
| Checkpoints | Every 25 updates, retain 3; the final update is a checkpoint boundary |
| Evaluation | Greedy pass@1, one completion per question, response limit 4,096 |
| Evaluation schedule | Fresh baseline before training; final synchronized student after training; optional restored-checkpoint evaluation |

The engine derives stable per-request seeds from the recorded base seed, prompt ID, and group index. Sibling responses therefore do not accidentally reuse the same seed. Seeds do not guarantee bitwise reproducibility across accelerator kernels, software versions, or scheduling changes.

The main evaluation metric is `answer_accuracy`: exact canonical numeric equality for the extracted boxed answer. Decimal and supported simple-fraction representations are compared exactly; malformed grouping, multiple answers, units, and arbitrary expressions are rejected. Reports also include strict boxed-string accuracy, VTC format accuracy, mean VTC reward, response lengths, truncation counts, and elapsed time. Truncated responses remain in the denominator. Missing, failed, duplicate, or stale responses fail the evaluation instead of shrinking the denominator.

## Runtime Requirements

Run commands from the repository root with the project dependencies and native extensions installed, using Python 3.12 or later.

- **Student host:** a single eight-chip TPU host is the initial topology target. Chips `0,1,2,3` train with TP=4; chips `4,5,6,7` run in-process vLLM-JAX with TP=4. The existing native Tunix and Raiden dependencies must be installed.
- **Teacher host:** a compatible SGLang GPU environment. The default is TP=2, bfloat16, context length 8,192, memory fraction 0.8. Two 80-GB-class GPUs are a starting allocation for the 32B teacher; actual memory/throughput must be measured in the deployment environment.
- **Storage:** both pinned BF16 weight sets require roughly 82 GB before cache overhead, checkpoints, optimizer state, logs, and evaluation outputs. Use persistent storage with adequate additional capacity.
- **Networking:** the student control plane must reach the teacher HTTP service. Keep the service inside the appropriate network/data-access boundary; it receives full student trajectories.

This is an execution configuration, not a measured memory-fit or learning-quality guarantee. The 8B/32B accelerator run has not been executed in the CPU-only validation environment. Do not infer its convergence or throughput from the small-model correctness tests.

## 1 Inspect the Profile Without Downloading Models

```bash
python -m tunix.dist_rl.examples.on_policy_distillation.recipe show
```

This validates and prints the profile without network requests, downloads, workers, or model-weight loading. An alternate profile can be passed with `--config`; model identities remain fixed, and invalid sampling, overlapping chip sets, dataset leakage, or incompatible budgets are rejected. A modified profile must use a new run directory.

## 2 Prepare Immutable Model Snapshots

```bash
python -m tunix.dist_rl.examples.on_policy_distillation.models prepare \
  --output-dir /shared/opd/models
```

The command writes `/shared/opd/models/models.json` and uses an HF cache beneath that directory. Local trainer and tokenizer paths point directly at the immutable snapshot directories; safetensor symlinks are supported. Training disables the base launcher's automatic model download, so missing prepared artifacts fail before worker startup rather than fetching a mutable default.

Use shared storage accessible at the same paths on both hosts, or run preparation independently on each host and use that host's manifest. The student host needs its own valid local manifest for tokenizer/identity validation. Teacher preflight compares revision-qualified service identity rather than assuming that remote and local absolute paths are identical.

Preparation requires access to the official Hugging Face models. It supports `--local-files-only` for a fully populated cache. Do not edit or repoint a manifest to another checkpoint.

## 3 Start the Frozen Teacher

On the teacher host, in the compatible SGLang environment:

```bash
python -m tunix.dist_rl.examples.on_policy_distillation.recipe teacher \
  --models-manifest /shared/opd/models/models.json \
  --cuda-visible-devices 0,1
```

The command uses the profile's bind host, port 30001, tensor parallelism, memory fraction, and context length. It starts SGLang in the foreground with explicit local model/tokenizer paths, safetensors loading, a revision-qualified served model name, and the fixed weight version. It waits for health/model metadata and checks the scoring context budget. Startup failure or interruption terminates only this launcher's own process group.

Use `teacher --dry-run` to inspect the exact command without starting SGLang. If the installed SGLang lacks the required identity metadata/arguments, startup fails explicitly; no model-identity check is silently disabled. Metadata checks verify the service's declarations, not remote weight attestation. Keep the teacher weights fixed for the entire run.

## 4 Train and Evaluate the Student

On the student TPU host:

```bash
bash tunix/dist_rl/examples/recipes/run_gsm8k_opd_qwen3_8b_from_qwen3_32b.sh \
  --models-manifest /shared/opd/models/models.json \
  --run-dir /persistent/opd/qwen3-8b-from-32b-run1 \
  --teacher-url http://teacher-host:30001
```

The run directory must be new or empty. The entry point validates local artifacts and the remote teacher identity/context, records the resolved experiment configuration, and delegates to the existing distributed GSM8K launcher with explicit settings. It uses the invoking Python environment and does not select a different student or inherit the smoke recipe's two-step budget.

Execution order is:

1. Bring up the native trainer and rollout workers.
2. Restore and validate OPD checkpoint identity if explicitly resuming.
3. On a fresh run, synchronize the initial student and evaluate all test questions as `baseline`.
4. Train on the fixed train split and synchronize updates through Raiden.
5. Evaluate the final synchronized rollout policy on the same held-out questions as `final`.
6. Shut down the worker processes through the existing launcher lifecycle.

Evaluation invokes `DistributedRLEngine.generate` at training barriers. It does not pretend that an Orbax trainer checkpoint is an HF serving checkpoint. The legacy trainer `EVAL_EVERY_N_STEPS` setting is not used as a substitute for this generation evaluation; this profile evaluates before and after training, not periodically.

Add `--dry-run` after model preparation to print the fully resolved launcher command and explicit environment overrides. This does not contact the teacher or create a run directory.

## 5 Resume an Interrupted Training Run

```bash
bash tunix/dist_rl/examples/recipes/run_gsm8k_opd_qwen3_8b_from_qwen3_32b.sh \
  --models-manifest /shared/opd/models/models.json \
  --run-dir /persistent/opd/qwen3-8b-from-32b-run1 \
  --teacher-url http://teacher-host:30001 \
  --resume
```

Resume requires the recorded profile/model manifest and an existing nonzero checkpoint. It fails if the model identities, profile digest, OPD objective, seed, or tokenizer semantics change, or if the engine cannot restore a valid checkpoint. It never silently starts fresh. The initial baseline is not regenerated from a restored student. Reports are not overwritten.

On graceful trainer shutdown, an externally managed checkpoint is finalized only when its weights match a complete program metadata boundary. The same OPD identity and data cursor are preserved even when the regular save interval skipped that step. If weights have advanced beyond the last known boundary, shutdown retains the previous safe checkpoint instead of publishing an unlabelled newer one. Work since the last persisted boundary may be replayed after interruption.

Use the same `--config` and dataset cache when resuming a customized profile. A different teacher endpoint may serve the same declared immutable model; preflight still validates that declaration. Changing the objective or model requires a new run directory.

## 6 Evaluate the Latest Saved Student Without Training

```bash
python -m tunix.dist_rl.examples.on_policy_distillation.recipe eval \
  --models-manifest /shared/opd/models/models.json \
  --run-dir /persistent/opd/qwen3-8b-from-32b-run1
```

This launches the student workers, restores the latest nonzero Orbax checkpoint through the real training engine, validates the OPD manifest, resynchronizes rollout weights, and writes `evaluation/checkpoint.json`. It performs no optimizer updates or teacher-scoring requests. The original teacher URL is read from the recorded run; the teacher service does not need to be online. The prepared local tokenizer/model manifest is still required.

The command fails if the checkpoint is missing, identity validation fails, or `checkpoint.json` already exists. Preserve previous reports before intentionally starting another evaluation. Baseline/final reports describe the actual synchronized live policy; checkpoint evaluation separately records the metadata of the checkpoint that was restored.

## Artifacts and Comparison

| Path under the run directory | Contents |
| --- | --- |
| `run_manifest.json` | Complete profile, profile SHA256, pinned model records, resolved launcher environment, and control-plane package versions |
| `evaluation.json` | The separate held-out evaluation configuration consumed by the control plane |
| `teacher_preflight.json` | The teacher metadata accepted for the latest training launch |
| `checkpoints/` | Native trainer checkpoints with the OPD objective manifest |
| `evaluation/baseline.json` | Initial synchronized student, full held-out test split |
| `evaluation/final.json` | Final synchronized student, same evaluation protocol |
| `evaluation/checkpoint.json` | Optional latest-checkpoint-only evaluation |
| `logs/`, `metrics/`, `trajectories/` | Worker/control-plane logs, training metrics, and trajectory records |

Compare baseline and final `metrics.answer_accuracy`, `format_accuracy`, `mean_completion_tokens`, and `truncated_count`. Confirm identical `dataset.sha256`, `prompt_template_sha256`, and evaluation settings. Reports include model provenance, policy version, and individual completions; they may contain sensitive task data and need appropriate storage/access controls.

No benchmark score is pre-populated. A successful command exit, falling sampled OPD signal, or a small-model KL decrease is not evidence of improved held-out accuracy. Report the actual baseline/final metrics and runtime costs from the accelerator run.

## Validation

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/dist_rl/examples/on_policy_distillation/ \
  tests/dist_rl/examples/math_gsm8k_dist/opd_workflow_test.py \
  tests/dist_rl/orchestrator/rollout_seed_test.py
```

The tests cover model pinning and tokenizer/shard checks, local HTTP teacher metadata, owned-process cleanup, configuration propagation, fresh/resumed/checkpoint-only evaluation lifecycles, exact numeric grading, label isolation, stale/error responses, and deterministic sibling seed derivation. They use small fixtures and test doubles for remote workers; they do not download model weights or establish GPU/TPU performance. The fixed real tokenizer files have also passed the production compatibility check independently.

The final combined CPU regression on October 29, 2026 passed **944 tests and 394 subtests, with 6 skipped**. An additional **20 existing checkpoint/restore tests** passed in a separate process using the test suite's simulated CPU mesh. Real CPU Orbax tests cover committed-boundary metadata preservation, skipped save intervals, model/optimizer restoration, and retaining the last safe checkpoint after uncommitted updates.
