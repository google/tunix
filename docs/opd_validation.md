# OPD Correctness Validation Record

Date: October 29, 2026. Branch: `haoyu-next`. This record documents validation of Tunix's sampled-token on-policy distillation (OPD) implementation.

Validation currently covers mathematical gradients, the HTTP scoring protocol, batch assembly, and two rounds of actual parameter updates on a local CPU. The real SGLang inference engine, the vLLM/Raiden/TPU combination, and learning quality on pretrained models require separate acceptance tests.

The initial implementation regression completed with **775 passed, 286 subtests passed, and 6 skipped**, with process exit code 0. The extended regression including the complete pinned recipe passed **944 tests and 394 subtests, with 6 skipped**; another **20 existing checkpoint/restore tests** passed separately. The combined coverage includes the core checks below and existing regressions for the program, algorithm adapters, assemblers, workers, collector, native sampler, sampling utilities, and CLI/launcher.

## 1. Objective Under Test

The training signal is:

```text
A_t = task_scale × A_GRPO,t
      − opd_coefficient × (logp_rollout_student,t − logp_teacher,t)
```

The task advantage is computed by the existing GRPO group estimator, then scaled and combined with OPD. The behavior student scores, teacher scores, and resulting advantages are fixed training data. Reference KL uses an independent beta. The initial implementation requires compatible fast tokenizers, actual rollout logprobs, full-vocabulary sampling at temperature 1, G >= 2, zero staleness, and actual weight synchronization for positive-coefficient OPD.

## 2. Independent Full-Distribution Gradient Check

Test: [opd_gradient_test.py](../tests/dist_rl/distillation/opd_gradient_test.py).

The test uses three contexts, three actions, and three distinct nonuniform student distributions. Integer frequencies enumerate the student's sampling probabilities exactly. This avoids Monte Carlo error and prevents a uniform binary distribution from hiding probability-weighting errors.

The production path is OPD transform → native padded/packed assembler → `algo_core.grpo_loss_fn` → NNX autodiff. The independent objective is computed directly from complete probability tables, without calling the OPD transform or GRPO loss:

```text
J(θ) = mean_context[
    opd_coefficient × KL(p_θ || teacher)
    − task_scale × sum_action(p_θ × fixed_task_advantage)
]
```

This comparison checks the one-step gradient at the behavior policy. It does not claim that the clipped surrogate remains equivalent to full KL after multiple PPO updates.

| OPD coefficient | Task coefficient | Padded maximum absolute gradient error | Packed maximum absolute gradient error |
| --- | --- | --- | --- |
| 1.0 | 0.0 | `2.98e-8` | `2.98e-8` |
| 0.0 | 0.7 | `2.24e-8` | `2.24e-8` |
| 0.4 | 0.7 | `2.98e-8` | `2.98e-8` |
| 0.0 | 0.0 | `0` | `0` |

After one pure-OPD SGD step using the actual gradient and a learning rate of 0.1, the independently computed dense reverse KL decreases from `0.702256918` to `0.688158751`. Padded and packed results agree. Token ID 0 appears as an actual prompt/action token; explicit attention validity preserves it.

## 3. Local Training Loop with Actual Parameter Updates

Test: [opd_training_test.py](../tests/dist_rl/distillation/opd_training_test.py).

The loop includes all stages of `StandardRLProgram.run_async`, actual HTTP requests from the production `SGLangTeacherScorer`, the real padded assembler, `TrainerWorker`, `PeftTrainerV2`, GRPO loss, NNX parameters, and Optax SGD. A controlled sampler draws from the synchronized student's complete probability distribution using a fixed random seed and records fresh behavior logprobs each round.

The teacher is an independent NumPy conditional distribution exposed by a localhost HTTP server that returns the SGLang scoring format. It scores using the preceding token and actual position, so correct array lengths alone cannot satisfy the test. Every updated parameter is compared with an independent NumPy sampled-policy-gradient oracle. The test also checks tool masks, actual pad/EOS-valued context tokens, an unchanged teacher, and use of updated weights in the next rollout. The zero-coefficient pure-distillation control must issue no teacher requests and leave parameters exactly unchanged.

Observed results: mean dense reverse KL decreases from `0.149617002` to `0.147754008`. After the first synchronization, the maximum logprob change in the rollout distribution table is `0.021109263`. Rollout versions are `0, 1`, and the final synchronized version is `2`. All parameters after each round match the independent sampled-gradient oracle within `atol=rtol=2e-6`.

Sampling with the fixed seed covers 1 sampled EOS token (loss mask = 1, attention validity = 1), 8 tool tokens with ID 0 (loss mask = 0, attention validity = 1), response lengths `[1, 3, 4]`, and 10 padding positions. The test checks the raw trajectories and assembled batches individually. A negative control preserves correct teacher token IDs but deliberately shifts the scores cyclically by one position. The independent parameter oracle detects a maximum update error of `0.058922343`, confirming that validation checks more than shapes and token IDs.

This test uses a local engine with queues and explicit NNX state copies instead of RPC/Raiden. Rollouts use a small probability-table model without starting vLLM. The localhost teacher is not an actual SGLang engine. Checkpointing in this test records metadata only. The test validates production orchestration, loss, and optimizer integration; it does not replace acceptance tests for those external runtimes.

## 4. Scoring and Data Boundaries

The existing [teacher tests](../tests/dist_rl/distillation/teacher_test.py), [program tests](../tests/dist_rl/orchestrator/opd_program_test.py), and worker/rollout regressions jointly cover:

- Original prompt and completion token IDs pass through without decoding or retokenization. Every teacher response ID and row count must align exactly. The first response token uses input-score position `prompt_length`, without an additional shift.
- Teacher scores contribute to advantages only at valid assistant tokens. Tool observations and template suffixes have zero loss weight but remain in the attention context for later tokens.
- Explicit attention validity reaches actor scoring, reference scoring, and the actual policy loss. Packed reference scoring preserves segment IDs and local positions without leaking context across sequences.
- Padding is not inferred from token values. Actual tokens remain valid when EOS and padding share an ID. Active trajectories that exceed the assembler budget fail before contacting the teacher.
- vLLM candidate dictionaries are indexed by the actual generated token ID. Requested scores that are missing, misaligned, or nonfinite cause failure; a valid zero logprob remains acceptable.
- Missing fields, incorrect token IDs, incompatible vocabularies or fast tokenizer backends, HTTP errors, complete-request deadlines, and concurrency limits.
- Failed or entirely masked trajectories skip teacher requests. OPD does not reactivate GRPO groups with too few valid responses. Score metrics are weighted by valid token count.

Two additional runtime contract issues were addressed:

- **Implicit sampling defaults:** checking only the requested temperature/top-p/top-k does not establish full-distribution sampling; model generation configuration can still introduce min-p or penalties. OPD now forces `require_unfiltered=True`. In-process vLLM starts from neutral parameters, and integrated samplers validate their effective parameters. Native vanilla sampling checks the worker configuration actually used by JIT decoding.
- **Silent objective changes on resume:** checkpoints now store an OPD manifest that checks the declared teacher revision, tokenizer fingerprint, coefficients, and resolved program sampling contract. An unknown or mismatched objective fails before rollout dispatch or teacher requests. Existing non-OPD checkpoint behavior remains compatible. A declared revision does not prove which weights a remote service currently serves; deployment must keep the actual teacher identity fixed.

## 5. Reproduction

Install the repository dependencies and native extensions, using Python >= 3.12. These three test files provide the core mathematical and local training-loop evidence:

```bash
JAX_PLATFORMS=cpu python -m pytest -q -s \
  tests/dist_rl/distillation/on_policy_test.py \
  tests/dist_rl/distillation/opd_gradient_test.py \
  tests/dist_rl/distillation/opd_training_test.py
```

The HTTP tests require permission to bind ephemeral localhost ports. The validation environment uses Python 3.12.2, JAX 0.11.0, Flax 0.12.8, and one CPU device. More extensive commands for protocol, worker, sampler, and example-entry-point regressions are in the [OPD README](../tunix/dist_rl/examples/on_policy_distillation/README.md).

The real vanilla CPU decoder was also exercised. Effective-parameter checks in the async vLLM adapter were tested through its production adapter code with lightweight engine/parameter test doubles; they do not exercise the installed vLLM library. New regressions in `tests/generate/vllm_sampler_test.py` use real vLLM `SamplingParams` to check model defaults and configuration overrides, but that file was not run because vLLM is not installed in this environment. The actual vLLM library path remains subject to runtime acceptance testing.

## 6. Complete Pinned Recipe Validation

The [full Qwen3-8B from Qwen3-32B recipe](../tunix/dist_rl/examples/on_policy_distillation/QWEN3_8B_OPD.md) adds immutable model preparation, a supervised teacher service, a separate full-test evaluation configuration, fresh/resume/checkpoint-only control flow, and experiment records. It uses the existing trainer and live synchronized rollout workers for baseline/final evaluation rather than assuming Orbax checkpoints can be served as HF models.

The extended combined CPU regression passed **944 tests and 394 subtests, with 6 skipped**. Another **20 existing checkpoint/restore tests** passed in a separate process. The new real-CPU Orbax cases verify model, optimizer, OPD identity, and cursor restoration across interval-skipped saves and safe shutdown. A trainer that advances beyond its last committed program boundary cannot publish an unlabelled newer checkpoint at shutdown.

CPU tests cover the complete configuration-to-launcher contract, actual engine request construction, label isolation, numeric answer grading, failure/staleness handling, model artifacts and service metadata, and distinct deterministic seeds for sibling responses. The fixed real Qwen3-8B and Qwen3-32B tokenizer/configuration files were also downloaded without weights and passed the production semantic-compatibility check. Their fingerprint is `6b06d0afe078c85bc53ce525ebc9cee9ae68e8ef8c4022bc8e04364c4aaaf051`.

No actual 8B/32B weights were loaded in this CPU validation. Accelerator memory fit, throughput, checkpoint recovery on the target runtime, and held-out quality remain unmeasured until the full recipe is executed.

## 7. Remaining Acceptance Tests with Real Models

1. Start a real SGLang teacher at a fixed revision with a tokenizer compatible with the student. Check a set of saved trajectories covering the first response token, EOS, and long contexts.
2. Run the [two-round Qwen3 OPD recipe](../tunix/dist_rl/examples/recipes/run_gsm8k_opd_qwen3_0p6b.sh). Verify actual weight updates and second-round sampling on vLLM/Raiden/TPU, and measure sampler/trainer score differences at identical weights.
3. Interrupt and restore an actual on-disk checkpoint while keeping the teacher identity, tokenizer, and objective configuration unchanged.
4. Execute the [full pinned recipe](../tunix/dist_rl/examples/on_policy_distillation/QWEN3_8B_OPD.md), compare its actual baseline/final reports, and then add GRPO or mixed-objective ablations as separate runs. Report quality, response length, valid tokens, teacher latency, and training cost. Neither a two-round smoke test nor decreasing toy-model KL establishes capability transfer to a pretrained model.
