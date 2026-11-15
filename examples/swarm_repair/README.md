# Collective-reward training on modular repair

This is a runnable **validation environment for Avalon/RLCR mechanisms**. A team shares
one trainable policy, communicates through mailboxes and a workspace, and commits
an answer before a physical wall-clock deadline. Team reward produces one group
advantage shared by all completed action tokens. Updates use native Tunix CISPO,
with detached importance weights capped at 5 and no reference KL.

The task, model scale, action space and execution environment differ from
Avalon-M1. This example does not reproduce its reported performance or training
corpus. The source references are [How to Train a Swarm](https://www.gr.inc/blog/how-to-train-a-swarm)
and the [Avalon-M1 paper](https://www.gr.inc/avalonm1.pdf).

The reusable runtime and credit hooks are independent of the repair task. This
directory's DSL is an integration test workload, not the primary Avalon benchmark
or evidence of a general coding improvement.

## Public repair environment

Two files define a small numerical pipeline. Each file contains a JSON module
with `op` and `value`. Operations are `identity`, `scale`, `shift`, `square`, and
`absolute`; values are bounded integers. Natural-language requirements describe
the intended module behavior, alongside three public input/output examples. The
target module JSON is private to the grader. The grader applies the submitted
pipeline to sixteen hidden, seeded inputs disjoint from the public examples.
Reward is one only if all outputs are correct.

This deliberately small configuration-repair task requires no host execution of
generated Python or shell commands. It is suitable for validating collaboration,
credit, deadlines and native training. It is not a general repository-repair or
SWE-bench benchmark. Harder public environments can replace this DSL behind the
same action/trace and grading interfaces.

The generator and seed are versioned in each manifest. Training uses affine
pipelines, development uses shift-then-scale pipelines, and the locked test uses
quadratic and absolute-value pipelines. The task families are disjoint, and
hidden input vectors are never included in model observations.

## Learned actions and execution

The shared model emits one JSON action per call:

| Action | Effect |
| --- | --- |
| `spawn` | Start a new peer responsible for a module file |
| `read` | Read a workspace file into the agent's observation |
| `write` | Apply a validated JSON-module patch |
| `send` | Send a bounded message to another agent |
| `submit` | Commit an immutable workspace snapshot; optionally keep working |

`CollectiveRuntime` launches real concurrent async agent tasks. Calls, tokens,
agents, messages, workspace size and elapsed time are bounded. `submit` implements
anytime settlement: the last committed snapshot is graded even if other agents
are cancelled at the deadline. Incomplete sampling calls are excluded from the
loss; their partial tokens are not fabricated. An attempt without any complete
action causes its entire comparison group to be retried, at most three times.
Failure after that aborts collection explicitly.

The generic runtime also exposes `finish(answer)`: a CEO final answer or a
terminal environment event cancels the remaining team immediately. `commit`
preserves an anytime answer without terminating peers.

The current native sampler requires cooperative cancellation. Python deadlines
cannot preempt a synchronous accelerator kernel already executing. Reported
elapsed time therefore includes such overrun; deadline behavior must be accepted
on the target serving backend before making physical-latency claims. The CPU
fixture tests cancellation with genuinely awaitable sampling calls.

## Bind an external environment

`custom_collective.py` provides an environment-independent adapter. Implement
`observe(agent_id, role)`, async `step(agent_id, tool, arguments)`, and async
`settle(final_text)`. Tool steps return observations, reward events and an
optional terminal flag. Settlement returns only the additional terminal or
anytime reward; previously emitted reward events are not counted twice. The
environment owns isolation and cancellable execution of its tools. No arbitrary
host code is executed by this adapter.

The shared generation callback receives `(agent_id, role, prompt)` and returns
an exact `CallTrace`, normally from a `RecordingRuntime` per agent. Roles are
free-form, and tools belong to the environment. The CEO can hire specialists,
exchange messages, maintain shared notes and end the team. The adapter returns
an episode, complete reward events and per-call diagnostics. Collect K attempts
of each task, retain their diagnostics by episode ID, and supply
`rlcr.build_rlcr_credit` as the existing `SwarmRLProgram` credit builder.

```python
from functools import partial
from tunix.dist_rl.swarm import rlcr

credit_builder = partial(
    rlcr.build_rlcr_credit,
    diagnostics=diagnostics_by_episode,
    config=rlcr.RLCRConfig.avalon(
        underhire_penalty=environment_underhire_weight,
        signed_zero_mask=environment_has_signed_rewards,
    ),
)
```

The source-described preset enables parsing-error and execution-error penalties
of 0.1 and 0.05 per affected active token, summed into collective return;
agent-level tandem-repeat masking; and misplaced-tool-turn masking. The harness
must flag affected turns and identify each call's agent. Underhire weight is
required because the paper specifies an environment-dependent value. Repetition
means a unit of at most 32 tokens repeated at least 20 times over at least 300
tokens. A signed-zero mask uses zero net environment return; a custom episode
eligibility predicate can use a different explicitly declared convention.

Credit drops constant groups, normalizes shaped collective returns and broadcasts
one advantage to every eligible agent token. This implementation uses sample
standard deviation plus 1e-6; the paper does not specify its correction or
epsilon. Masks are applied after comparison moments are computed, retaining
masked attempts in that comparison population. That ordering is an explicit
implementation choice, not a verified source detail. Excluded completion tokens
remain attention context. Prompt/tool/received-message tokens receive no loss.

The local repair CLI defaults to unshaped binary reward and no optional masks.
It does not reproduce Avalon's full organisation harness, context compaction,
private training environments, data/SFT pipeline, GKPO branch, curriculum
schedules, or model scale. The generic adapter's direct mailboxes and notes are
also a subset of the paper's channels, archive and permission-management tools.

## Local training and validation

```bash
python -m examples.swarm_repair.train --backend cpu \
  --output /tmp/swarm-repair-cpu
python -m pytest -q tests/dist_rl/swarm/repair_environment_test.py \
  tests/dist_rl/swarm/repair_workflow_test.py \
  tests/dist_rl/swarm/repair_rlcr_test.py \
  tests/dist_rl/swarm/repair_training_test.py
```

The CPU fixture is a categorical policy over structured actions, rendered using
the public task context. It performs real native forward/backward and optimizer
updates with exact sampled-action log probabilities. It is not a pretrained
language model or a substitute for a Qwen performance experiment.

Tests cover shared-policy gradients, within-team overlapping calls, valid and
invalid DSL patches, private test isolation, anytime settlement, cancellation,
constant-reward group filtering and exact parameter equality after resume.

## Qwen3-4B bring-up

```bash
hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
python -m examples.swarm_repair.train --backend qwen \
  --model-path /path/to/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --output /path/to/swarm-repair-qwen
```

The profile uses K=16 attempts per task, the float32/reference-attention path, native sampling,
one device mesh and full-parameter updates. It is a bring-up configuration, not a
validated TPU throughput profile. TP/static batch sizes must divide the selected
mesh. Tune physical deadlines only after measuring warm model-serving latency;
compilation and cold startup can otherwise consume the entire first deadline.

## Comparisons and artifacts

Every evaluation measures four topology modes at the same policy checkpoint:

- `single`: one agent, no spawning or communication.
- `independent`: isolated workspaces and no communication; submissions are chosen
  by deterministic majority vote, never by hidden-test oracle selection.
- `fixed`: two preassigned communicating peers, with dynamic spawning disabled.
- `swarm`: one initial lead and learned peer spawning/communication.

All modes have the same team call, token and wall-clock caps. Actual consumption
is recorded separately. `initial_untrained_eval` and `final_eval` compare the
policy before and after learning. To train independent baseline policies, use
`--training-mode single`, `independent` or `fixed` in separate fresh runs with the
same seed and training budget; the default is `swarm`. Evaluation topology
ablations alone are not independently trained baselines.

`report.json` contains accuracy, pass@k, token/call costs, mean/p95 elapsed time,
observed sampling concurrency, failures, deadline rate, dataset provenance and
optimizer history. `train_traces.jsonl` stores exact calls, committed answers and
runtime events. CPU reports additionally include parameter change and snapshots.
The model's arbitrary sleep/delay fixture is not used to claim accelerator speedup.

After choosing the checkpoint on development data, evaluate the locked test:

```bash
python -m examples.swarm_repair.train --backend qwen --resume --mode eval \
  --eval-split test \
  --model-path /path/to/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --output /path/to/swarm-repair-qwen
```

`initial_eval_manifest` identifies the development split used for the saved
untrained baseline; final test metrics use the separately recorded test manifest.

## Checkpoint and failure semantics

Each durable native checkpoint includes model/optimizer state and complete round
metadata. A checksummed `round_state.json` is atomically replaced only after
checkpoint durability. It can advance the data/round cursor through zero-signal
rounds without advancing Adam or duplicating immutable checkpoint IDs.

On restart, the highest valid complete boundary wins: a newer durable native
checkpoint can roll forward after a manifest-write interruption; a newer cursor
at the same optimizer step preserves zero-update rounds. Run identity and
configuration/data hashes must agree. Tests inject a failure between native
checkpoint durability and manifest publication, and verify equality with an
uninterrupted run. The local output directory and native checkpoint directory
must both be retained when using a remote checkpoint store.
