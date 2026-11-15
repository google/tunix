# Director-only mathematical orchestration

This recipe adapts R²Orch to Qwen3-4B and Tunix `dist_rl`. It trains a Director
to construct a small mathematical collaboration graph while Workers and the
reference policy remain frozen. It is **not a complete reproduction of R²Orch**.
The [design](../../docs/r2orch_design.md) describes the extension sequence and
acceptance criteria.

## Implemented scope

- Original AIME train/development/test splits, pinned to upstream commit
  `12ceb7563aa50439687f21ba5da60ad74a75eb89`; source IDs, prompts, answers and
  curriculum metadata are retained. The importer also preserves MATH-Hard OOD.
- Production profiles execute the pinned original `GraphDirector`, `GraphCanvas`
  and `MultiAgentRuntime`: mandatory configuration, constrained relation choices,
  incremental Worker feedback, artifact repair and unified final submission.
  The compact graph editor remains available as the local CPU fixture.
- Independent Director and frozen Worker model instances. The frozen base also
  supplies reference log probabilities; it is never published by the optimizer.
- Per-task reward centering, source-compatible active-token weighted whitening,
  low-signal selection, fixed old/reference log probabilities, truncated sampler
  importance weights, asymmetric PPO clipping and k3 reference KL.
- Two policy minibatches per collection round through native `TrainerWorker`
  and `PeftTrainer`. Checkpoints preserve optimizer and collection step separately.
- Training traces, held-out accuracy and pass@k, generated tokens, call count,
  graph termination, original-source provenance and deterministic resume.

The CPU fixture executes real forward/backward and optimizer updates on a tiny
categorical graph editor. It demonstrates numerical and lifecycle correctness,
not mathematical capability or pretrained-model performance.

## Prepare the original data

Obtain the original repository separately; this command reads committed data
bytes and does not execute the source repository's installation or scripts.

```bash
git clone https://github.com/beita6969/R2Orch.git /tmp/r2orch-reference
git -C /tmp/r2orch-reference checkout 12ceb7563aa50439687f21ba5da60ad74a75eb89
python -m examples.r2orch_math.prepare_data \
  --source /tmp/r2orch-reference --output /tmp/r2orch-data
```

The output contains `train.jsonl` (947 AIME questions), `dev.jsonl` (30),
`test.jsonl` (30 AIME 2026 questions), `ood.jsonl` (128 MATH-Hard questions),
and a manifest with source and output SHA-256 hashes. Training never reads the
test split. Exact normalized duplicates are rejected. The original repository
also provides its broader near-duplicate audit.

## Local numerical training

```bash
python -m examples.r2orch_math.train --backend cpu \
  --output /tmp/r2orch-cpu
python -m pytest -q tests/dist_rl/swarm/director_credit_test.py \
  tests/dist_rl/swarm/r2orch_gradient_test.py \
  tests/dist_rl/swarm/r2orch_workflow_test.py \
  tests/dist_rl/swarm/r2orch_recipe_test.py
```

`report.json` reports parameter changes and frozen-model integrity. The fixture
uses synthetic arithmetic tasks, separate from the original training split.

The optional original-source checks execute the exact reference scorer and audit
imported file hashes:

```bash
TUNIX_R2ORCH_SOURCE=/tmp/r2orch-reference \
  python -m pytest -q tests/dist_rl/swarm/r2orch_reference_test.py
```

## Define a different swarm

`custom_swarm.py` demonstrates a user-written async workflow with two parallel
frozen solvers and a trainable reviewer. `SwarmSpec` binds roles to policies;
`SwarmSession.generate` records exact calls. The workflow receives only the
public problem, while grading occurs afterward in the collector.

```bash
python -m examples.r2orch_math.custom_swarm
```

Replace its sampler factory with real-model `RecordingRuntime` instances and
feed the returned episodes to `SwarmRLProgram` with the appropriate credit
builder. The public API does not impose the R²Orch graph or a fixed topology.

## Qwen3-4B training and evaluation

Download the pinned model snapshot using the same model revision as the validated
Dr. MAS recipe. `--model-path` must name the snapshot revision directory.

```bash
hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
python -m examples.r2orch_math.train --backend qwen \
  --model-path /path/to/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --reference-source /tmp/r2orch-reference \
  --tool-image sha256:IMAGE_ID_FROM_INSPECT \
  --train-data /tmp/r2orch-data/train.jsonl \
  --eval-data /tmp/r2orch-data/dev.jsonl \
  --output /path/to/r2orch-run
```

The Qwen path requires the pinned upstream checkout for the original executor
and AIME benchmark scorer, plus the isolated tool image described below.
It does not substitute the CPU fixture's simplified workflow or grader.
The production profile uses native Tunix sampling, one mesh, float32 training,
reference attention, a frozen base, and rank-64 Director LoRA updates with
alpha 64. Adapter dropout is zero, matching the original paper launcher's
`--lora-dropout-override 0` even though its base TOML lists 0.05. The seven target
projections also match that TOML. Nonzero adapter dropout is rejected.
It is an initial bring-up profile, not a validated TPU throughput configuration.
The device count, TP size and static batch sizes must satisfy the existing
`dist_rl` mesh divisibility rules. Memory must accommodate the Director, its
optimizer and serving snapshot, and an independent frozen base/serving snapshot.

### First TPU acceptance run

Start with the Director-only Qwen3-4B profile before adding the alternating
Proposer and skill lifecycle. This is a new accelerator acceptance run; the
existing Dr. MAS TPU results do not validate this sampler or graph executor.

Use a separate smoke configuration and output directory:

- Keep `group_size: 5`, `tasks_per_step: 4`, float32 and reference attention.
- Set `eval_attempts: 1`, `checkpoint_every_steps: 1` and
  `val_every_steps: 2`; select a fixed four-question subset of the original
  development split. The default otherwise evaluates 30 questions four times
  before training begins.
- Run `--num-steps 2`, then the same command with `--resume --num-steps 8`.
  Keep `optimizer.total_optimizer_steps: 400` so stopping and resuming do not
  change the learning-rate schedule. Eight collection rounds permit up to
  sixteen optimizer updates, beyond the ten-update warmup; check the actual
  optimizer count because zero-signal rounds can skip updates.
- Verify original graph submission and all three container tools, ordinary and
  binary-policy logprob alignment, finite gradients, changing Director LoRA
  weights, unchanged frozen/base weights, and restored optimizer/policy state.
  The Qwen report's `frozen_policy_version: 0` is metadata: the automatic
  parameter-digest comparison currently runs only in the CPU fixture. A TPU
  acceptance run must separately verify the frozen parameter invariant.

The training controller needs a working container runtime and the explicitly
installed tool image; a TPU Kubernetes job does not imply Docker availability.
The native sampler uses the same mesh as training, not Dr. MAS's split-vLLM
deployment. With `tp_size: 8`, compute FSDP from the actual JAX device count and
check divisibility of both batch sizes. Measure first-compilation latency
separately before interpreting the 300-second graph timeout as task failure.

After smoke acceptance, start a new formal run with all 947 training questions,
all 30 development questions and four evaluation attempts. Configuration and
development-data identity are checkpoint invariants, so the smoke checkpoint
cannot silently switch to the full evaluation configuration. Inspect the first
25–50 rounds before completing the 200-round budget. Compare against the same
frozen Worker in single-agent, fixed-graph and untrained-Director settings with
matched compute budgets. Keep the test split locked until checkpoint selection.

AdamW uses the source CLI defaults: weight decay 0.01, global gradient norm 1.0,
ten optimizer-step warmup updates from zero and cosine decay. The explicit
schedule horizon is 400 Director updates for 200 collection rounds; the RSI
Proposer has its own 200-update horizon. Stopping early preserves these horizons
on resume. Tests compare clipped optimizer trajectories with Torch AdamW and
the pinned source scheduler.

`configs/qwen3_5_9b.json` selects the original model family. Download
`Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, then
pass `--config examples/r2orch_math/configs/qwen3_5_9b.json` and its matching
snapshot directory. The native model preset includes the 9B hybrid-attention
configuration. Tests exercise adapter-only updates, frozen-base integrity and
serving synchronization on a tiny Qwen model; no pretrained 9B run or benchmark
result is claimed.

The 9B presets reserve a 32,768-token serving window, matching the original
vLLM limit: Director prompts may use 31,768 tokens plus 1,000 generated tokens;
Worker prompts may use 28,672 plus 4,096. These sequences also fit the source
learner's 35,000-token acceptance limit. The 4B bring-up presets retain the
smaller 8,192-token prompt budget as an explicit resource adaptation.

The production profiles set `executor: source` and require all three original
AIME tools. Build their image from an approved, digest-pinned Python 3.11+ base,
then obtain its immutable local image ID:

```bash
docker build -f examples/r2orch_math/Dockerfile.tools \
  --build-arg PYTHON_BASE='python:3.11-slim@sha256:YOUR_VERIFIED_BASE_DIGEST' \
  -t tunix-r2orch-tools:local .
docker image inspect tunix-r2orch-tools:local --format '{{.Id}}'
```

Add `--tool-image sha256:IMAGE_ID_FROM_INSPECT` to every training, resume and
evaluation command above. A registry `repository@sha256:DIGEST` also works.
Tool execution uses the pinned source's `python_exec`, `symbolic_compute` and
`finite_search` implementations inside a fresh container with no network,
read-only filesystem and resource limits. Generated code never executes on the
host. The adapter does not pull images automatically. Setting
`require_source_tools: false` is an explicit tool-free ablation.

The source controller runs in a worker thread; its model calls return to the
sampler's owning event loop. Cancellation drains pending calls and joins that
controller before returning. Native accelerator kernels remain cooperative;
the interface does not promise to preempt an already running device operation.

Resume from a committed boundary using the same output/checkpoint directory:

```bash
python -m examples.r2orch_math.train --backend qwen --resume \
  --model-path /path/to/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --reference-source /tmp/r2orch-reference \
  --tool-image sha256:IMAGE_ID_FROM_INSPECT \
  --train-data /tmp/r2orch-data/train.jsonl \
  --eval-data /tmp/r2orch-data/dev.jsonl \
  --output /path/to/r2orch-run
```

After selecting the checkpoint on development results, use `--mode eval
--resume --eval-data /tmp/r2orch-data/test.jsonl` to evaluate the locked test.
Evaluation mode performs no optimization. The current launcher deliberately
rejects MATH-Hard until a math-equivalence scorer is attached.

## Differences from the complete original recipe

| Area | Current implementation | Original target |
| --- | --- | --- |
| Model/update | Pinned Qwen3-4B or Qwen3.5-9B; rank-64 LoRA, dropout 0 | Qwen3.5-9B; Director and Proposer LoRA, paper launcher overrides dropout to 0 |
| Task selection | Seeded Director-only runner; separate self-play driver | Learned Proposer with ADS/TSDS candidate selection |
| Skills | Optional Director-only context; visible/hidden self-play conditions | Retrieved skills, visible/hidden assignments, distillation and withdrawal |
| Actions | Original parser and constrained off/on policy in source profiles; simplified JSON in CPU fixture | Original action parsing and additional relation-choice policy calls |
| Execution | Pinned original Canvas/runtime in source profiles | Original typed artifacts, repair, output-role changes and unified submission protocol |
| Worker tools | Original three AIME tools inside an explicitly supplied container image | Original AIME tools and tools for five other domains |
| Serving | Native Tunix sampler | vLLM with two LoRA adapters and frozen base |
| Evaluation | AIME scorer and splits | Six IID and six OOD tasks with domain-specific tools/scorers |

`source_adapter.py` loads individual modules from the pinned Git archive, using
an isolated package namespace and committed bytes even if the checkout is dirty.
`source_workflow.py` executes those original graph/canvas components with Tunix
sampling transports. Optional tests run the actual source controller to check
mandatory configuration, binary relation token provenance, final submission and
cancellation. Container argument/lifecycle tests are separate from an actual
container run; no pretrained-model benchmark parity is claimed.

```bash
TUNIX_R2ORCH_SOURCE=/tmp/r2orch-reference \
  python -m pytest -q tests/dist_rl/swarm/r2orch_source_workflow_test.py \
  tests/dist_rl/swarm/r2orch_source_tools_test.py
```

Set `TUNIX_R2ORCH_TOOL_IMAGE` to the immutable installed image ID in the same
command to additionally execute the three real container tools. Without it,
that integration check is skipped explicitly.

These differences can materially affect accuracy and compute. The source's
headline results must not be attributed to this adaptation. Evaluate fixed
graphs, an untrained Director and a trained Director with the same frozen Worker
checkpoint and total token/call budgets before making a performance claim.

## References

- [R²Orch paper](https://arxiv.org/abs/2610.22740).
- [Original implementation](https://github.com/beita6969/R2Orch/tree/12ceb7563aa50439687f21ba5da60ad74a75eb89).
- [Original configuration](https://github.com/beita6969/R2Orch/blob/12ceb7563aa50439687f21ba5da60ad74a75eb89/configs/r2orch.toml).
- [Source and data provenance](THIRD_PARTY.md).

## Alternating RSI training

`selfplay.py` runs two independent native learners: the Proposer selects task
pairs from the original ADS/TSDS lists, then the Director trains on five sibling
workflows per selected task. Workers, the reviewer and reference policies stay
frozen. Three siblings hide the candidate skill; two see it. Raw correctness
feeds recursive learnability and the Proposer pair objective; Director credit
uses the rollout reward/cost and its own token masks.
After collection, the Proposer updates first, then the Director. The persisted
global role-update counter controls the source KL rotation; a skipped role
does not advance it. The math-only production profile uses four candidate lists
to satisfy the source minimum of four eligible pairs. The full source uses two
lists per dataset across six datasets; this dataset/batch change is explicit.

The production reviewer runs REVISE followed by DISTILL through the original
PATS prompt packing, evidence aliases, rule checks, deduplication and capacity
checks. Revisions create a new skill version; deleted or rewritten versions are
archived. Statistical retirement is separate and remains eligible for audit and
reinstatement. Evaluation always removes skills.

Run the real two-learner CPU fixture:

```bash
python -m examples.r2orch_math.selfplay --backend cpu \
  --output /tmp/r2orch-selfplay-cpu
```

The fixture performs two outer cycles, four Director updates and two Proposer
updates. It validates optimizer integration and recovery; its tiny categorical
models do not measure Qwen reasoning quality.

After importing the original data as above, run the Qwen3-4B adaptation:

```bash
python -m examples.r2orch_math.selfplay --backend qwen \
  --config examples/r2orch_math/configs/selfplay_qwen3_4b.json \
  --model-path /path/to/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --reference-source /tmp/r2orch-reference \
  --tool-image sha256:<local-image-id> \
  --embedding-model /path/to/local/e5-base-v2 \
  --train-data /tmp/r2orch-data/train.jsonl \
  --eval-data /tmp/r2orch-data/dev.jsonl \
  --output /path/to/r2orch-selfplay
```

For the original model family, use `selfplay_qwen3_5_9b.json` and snapshot
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Both launch profiles use independent
rank-64/alpha-64 Director and Proposer adapters with dropout zero, matching the
original paper launcher's override. The reviewer uses a separate frozen native
sampler with greedy decoding at temperature zero, matching the source. The
source E5 model and original training-pool metadata are required; changed
curriculum rows are rejected before model loading. Eight candidate completions
require an FSDP size dividing eight, in addition to the existing mesh rules.
Both production profiles use the pinned original Director/canvas executor and
require the isolated AIME tool image described above. The compact native graph
remains the CPU fixture, and is not substituted for the original executor.

Add `--resume` to restart the same run. Each committed cycle points to immutable
Director and Proposer parameter/optimizer checkpoints, curriculum RNG and
reservations, skill versions/evidence, prior estimates and publication versions.
A failure between learner updates or checkpoint writes replays the whole cycle
from the last committed pointer. No partially updated pair is published.

After selecting a development checkpoint, use the same command with
`--mode eval --resume --eval-data /tmp/r2orch-data/test.jsonl`. This restores both
learners and reports zero-skill test performance without optimization. Training
resume rejects a changed development split. Keep the final test locked until
checkpoint selection is complete.

Local tests cover nonzero updates to both learners, immutable frozen weights,
bitwise interrupted/resumed CPU trajectories, failure before atomic commit,
source candidate prompts/token labels, original PATS revisions/distillation and
no train/evaluation question overlap. Pretrained Qwen training, target-device
throughput and original-paper benchmark parity still require measured runs.
