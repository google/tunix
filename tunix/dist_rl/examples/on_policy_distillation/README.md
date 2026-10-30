# On-policy distillation on GSM8K

This example adds a fixed SGLang teacher to the existing distributed GSM8K
training loop. The student generates a response, the teacher scores those exact
token IDs, and the orchestrator adds a detached token-level distillation signal
to the existing GRPO advantages before batch assembly.

For each assistant token sampled by the student:

```text
sampled_reverse_kl_t = log_student_rollout_t - log_teacher_t
advantage_t = task_reward_coefficient * task_advantage_t
              - coefficient * sampled_reverse_kl_t
```

`task_advantage_t` is the normalized GRPO advantage. Its coefficient is applied
after normalization; multiplying raw rewards would otherwise be canceled by
GRPO normalization. Set `OPD_TASK_REWARD_COEFFICIENT=0` for pure OPD, or `1` for
GRPO plus OPD. Raw GSM8K rewards remain useful as task metrics in either mode.
The sampled reverse-KL signal can be negative at an individual token. It is
not a full-vocabulary KL calculation. Teacher scores, recorded student scores
and the resulting advantages are fixed training data; gradients update only
the student policy. Reference KL retains its separate `BETA` coefficient.

## Start a fixed teacher

Use a SGLang installation and accelerator that can serve the teacher. It can
run on a separate GPU host; it does not need to share the student's TPU slice.
For an initial mechanism check, the student's initial checkpoint can also be
the fixed teacher. A transfer-quality experiment needs a teacher with the
desired behavior and a compatible tokenizer.

On the teacher host, for example:

```bash
export TEACHER_MODEL=/path/to/fixed-teacher-snapshot
python -m sglang.launch_server \
  --model-path "$TEACHER_MODEL" --host 0.0.0.0 --port 30001
curl --fail http://localhost:30001/health_generate
```

The scorer uses SGLang's `/generate` endpoint with the student `input_ids`,
`max_new_tokens=0` and input log-probabilities. It does not ask the teacher to
generate another answer. A base URL or a URL ending in `/generate` is accepted.
An ordinary text-only chat API that cannot score supplied token IDs is
insufficient.

## Run the two-step student recipe

On a configured TPU host with the existing Tunix, vLLM-JAX and Raiden worker
dependencies:

```bash
export OPD_TEACHER_URL=http://teacher-host:30001
export OPD_TEACHER_TOKENIZER_PATH=/path/to/the-served-teacher-snapshot
export OPD_TEACHER_REVISION=immutable-teacher-snapshot-digest
bash tunix/dist_rl/examples/recipes/run_gsm8k_opd_qwen3_0p6b.sh
```

The tokenizer path must be readable on the student control-plane host. It may
also be a Hugging Face tokenizer ID. Only tokenizer files are loaded there;
the control plane does not load teacher model weights. The supplied teacher
tokenizer must correspond to the checkpoint actually served at the URL. The
client compares it with the student tokenizer and rejects incompatible token
IDs, added-token semantics, special tokens or backend encoding/decoding rules.
Both tokenizers need a verifiable fast backend (`use_fast=True`). This local
comparison does not prove the identity of remote teacher weights. Prefer
immutable snapshots and declare the served revision with
`OPD_TEACHER_REVISION`. Checkpoints record that declaration, the tokenizer
fingerprint, OPD coefficients and effective sampling contract in `opd_manifest`.
Resume rejects changes to these fields, enabling OPD on a legacy checkpoint
with an unknown objective, or disabling OPD on an OPD checkpoint. A positive
coefficient without a declared revision/fingerprint permits a fresh run, but
its checkpoint cannot be resumed. A URL is not a teacher identity; the caller
must ensure the endpoint actually serves the declared immutable weights.

The recipe delegates to `ci_smoke_gsm8k_qwen3_0p6b.sh`: native Tunix trainer,
Qwen3-0.6B, in-process vLLM rollout, two prompt groups per update with two
responses per group, two updates, Raiden synchronization, and a small GSM8K
subset. Existing model, chip-layout and checkpoint environment overrides remain
available. The smoke
recipe's two steps are a correctness check, not a learning-quality benchmark.
No TPU training result has been established for this new OPD recipe.

To combine the task advantage with distillation:

```bash
OPD_TASK_REWARD_COEFFICIENT=1 OPD_COEFFICIENT=0.5 \
  bash tunix/dist_rl/examples/recipes/run_gsm8k_opd_qwen3_0p6b.sh
```

## Configuration and supported scope

| Environment variable | CLI argument | Default |
| --- | --- | --- |
| `OPD_TEACHER_URL` | `--opd_teacher_url` | Empty disables OPD |
| `OPD_TEACHER_TOKENIZER_PATH` | `--opd_teacher_tokenizer_path` | Explicitly required with OPD |
| `OPD_TEACHER_REVISION` | `--opd_teacher_revision` | Empty permits a fresh run, but prevents positive-coefficient OPD resume |
| `OPD_COEFFICIENT` | `--opd_coefficient` | `1.0` |
| `OPD_TASK_REWARD_COEFFICIENT` | `--opd_task_reward_coefficient` | `0.0` |
| `OPD_TEACHER_TIMEOUT_SECONDS` | `--opd_teacher_timeout_seconds` | `60.0` |
| `OPD_MAX_CONCURRENCY` | `--opd_max_concurrency` | `8` |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `--temperature` / `--top_p` / `--top_k` | `1.0` / `1.0` / `-1` |

The initial OPD path requires text-only student-generated trajectories,
`USE_ROLLOUT_LOGPS=true`, temperature 1, top-p 1, and disabled top-k (`-1` or
`0` in this entry point). It retains GRPO's requirement for at least two valid
responses per prompt. `MAX_STALENESS=0` and
`IN_FLIGHT_WEIGHT_UPDATES=false` prevent accepting older policy versions or
changing worker weights during sampling; positive-coefficient OPD also requires
real weight synchronization (`WEIGHT_SYNC_MODE=raiden` in this launcher).
These settings do not eliminate numerical differences between inference and
training engines. The first recipe uses padded batches. CPU integration tests
also exercise the sequence-packed assembler.

Prompt and completion lengths must fit the assembler's budgets. OPD rejects
post-rollout truncation before contacting the teacher, since a teacher score
computed on one context must not supervise another. Sampler output must be
unpadded completion tokens, without echoing the prompt.

OPD forces `require_unfiltered=True` through the rollout request. The sampler
checks its actual engine/worker parameters, including penalties, min-p,
minimum-token EOS suppression, constrained decoding and best-of/beam search.
The in-process vLLM path starts from neutral sampling parameters instead of
inheriting distribution-changing defaults from model generation config.
Ordinary sampling retains its existing defaults. Native vanilla decoding uses
static worker sampling settings, so these must also match the OPD contract.

`BETA=0` skips the reference worker in the default smoke layout. If a separate
reference worker is configured, `BETA` can independently regularize the
student while OPD remains controlled by `OPD_COEFFICIENT`.

Missing, malformed or misaligned teacher scores fail the request rather than
becoming zero supervision. Teacher requests have bounded concurrency and an
end-to-end deadline that includes waiting for a slot and reading the complete
response body. Multi-teacher routing, top-k distillation, multimodal OPD and
off-policy schedules are outside this first recipe. With `OPD_TEACHER_URL`
unset, the existing GRPO path does not construct a teacher scorer or load a
teacher tokenizer.

Before increasing the training run, check that the second rollout uses updated
student weights, frozen teacher identity is unchanged, padding and non-assistant
tokens are excluded, and OPD produces a nonzero update when teacher and student
differ. Compare original student, pure OPD, GRPO and GRPO plus OPD on a fixed
held-out set; report task scores, response lengths, valid-token counts, teacher
latency and sampler/trainer score differences together.

## Implementation and validation

OPD applies sampled-token advantage shaping before batch assembly.
`distillation/teacher.py` owns the teacher protocol and HTTP scoring;
`distillation/on_policy.py` owns the immutable advantage transform.
`StandardRLProgram.critique_stage` awaits scoring before enqueueing a group for
batch assembly. Per-prompt overrides that change the sampling probability
space, including token bans, are rejected before rollout dispatch.

The implementation enforces the following correctness invariants:

| Area | Enforced behavior |
| --- | --- |
| Teacher alignment | Score the original prompt plus completion IDs; verify every returned ID and row count. Completion position `j` uses input-score row `prompt_length + j`, including the first response token and EOS. No text round-trip or extra causal shift is applied. |
| KL direction | Use `log_student_rollout - log_teacher`, then subtract its coefficient from the advantage. An independent dense `KL(student || teacher)` gradient oracle checks the actual GRPO loss. |
| Loss versus attention | Prompt, tool observations, failed trajectories and padding receive no OPD loss. Tool tokens and real EOS/pad-valued tokens remain attention context. Explicit validity reaches actor rescoring, reference scoring and the policy loss; packed rows retain segment metadata. |
| Advantage | Normalize task rewards through the existing GRPO adapter first, scale the resulting task advantage, then add the detached teacher signal. No extra whitening is applied. Pure OPD ignores task rewards for training while retaining them as metrics; a degenerate GRPO group remains masked. |
| Unused scores | Masked completion scores may be non-finite and are ignored; their token IDs must still align. Entirely masked rows skip teacher I/O. Active completion scores must be finite. |

At the rollout boundary, requested student logprobs must exist and match the
generated token vector; missing scores cannot become synthetic zeros. Genuine
zero logprobs remain valid. vLLM candidate dictionaries are indexed by the
actual sampled token ID, independent of dictionary order. Packed reference
scoring retains segment IDs and reset positions through chunking. Reference KL
is a separate loss term and does not change the OPD teacher or the
advantage-shaping direction.

The orchestrator emits `opd/reverse_kl`, `opd/teacher_logprob`,
`opd/student_logprob`, `opd/token_count` and `opd/scoring_time_seconds`.
Logprob means are weighted by active token counts across groups, so a short
response does not carry the same weight as a long one. Original task rewards
remain in the existing reward metrics. A zero OPD coefficient bypasses teacher
requests; with the same task-advantage coefficient, this agrees with training
without the OPD signal at the advantage level. The explicit attention-validity
mask still applies to padded OPD batches. Pure OPD with both coefficients zero
has no effective policy-gradient update.

Run the CPU correctness checks from the repository root in an installed Tunix
development environment:

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/dist_rl/distillation/ \
  tests/dist_rl/orchestrator/opd_program_test.py \
  tests/dist_rl/orchestrator/opd_checkpoint_identity_test.py \
  tests/dist_rl/common/datatypes_test.py \
  tests/dist_rl/worker/trainer_worker_test.py \
  tests/dist_rl/worker/inference_worker_test.py \
  tests/dist_rl/rollout/collector_test.py \
  tests/dist_rl/rollout/inprocess_vllm_sampler_test.py \
  tests/dist_rl/rollout/vllm_output_logprobs_test.py \
  tests/dist_rl/rollout/opd_sampling_contract_test.py \
  tests/generate/utils_test.py \
  tests/dist_rl/examples/math_gsm8k_dist/run_gsm8k_dist_grpo_test.py \
  tests/dist_rl/examples/math_gsm8k_dist/opd_identity_cli_test.py \
  tests/dist_rl/examples/math_gsm8k_dist/launcher_test.py
```

These tests include a local HTTP teacher, tokenizer and token-alignment errors,
timeouts and bounded concurrency, an independent full-distribution reverse-KL
gradient oracle through the existing GRPO loss, and real padded/packed batch
assembly. Worker tests additionally cover actor/reference validity masks and
microbatch slicing. They do not establish TPU compatibility, convergence or
throughput. See the [design document](../../../../docs/opd_design.md) for the
runtime contract, acceptance gates and later extensions.

The additional `opd_gradient_test.py` compares nonuniform three-action policy
gradients against an independent dense reverse-KL objective through both real
assemblers. `opd_training_test.py` runs two actual native optimizer updates
through the full program and a local HTTP teacher, checks each update against
an independent NumPy oracle, and tests EOS/tool/padding masks and synced rollout
weights. Its transport and teacher are local test implementations. Detailed
results and remaining hardware acceptance gates are recorded in the
[validation report](../../../../docs/opd_validation.md).
