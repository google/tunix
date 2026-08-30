# P58.20 — Qwen3-4B one-host full-stack Zero-TIM admission

Status: target RED; superseded by P58.21/P58.22 diagnosis and repair.

## Deliverable

Build and verify an additive Qwen3-4B-Instruct-2507 DP1xTP4 canonical model
variant, then run one direct-attached v5p-4 DeepSWE carrier through real R2E
rollout, strict A/B/C pre-alignment, and repeat backward without committing an
optimizer transaction.  This is the mandatory gate before any TP8 promotion.

## Bound source and prior evidence

- Clean implementation base:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.
- Corrected prior development artifact:
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s19fsampling_20260828t234224z`.
- That artifact proved real Qwen3-4B rollout, two complete R2E trajectories,
  durable journaling, explicit sampling, and a working strict stop.  It did
  not prove Zero-TIM: A-B and B-C were finite RED and backward was unreachable.
- Root construction gap: the prior TP4 carrier installed only the generic TPU
  runner because `qwen4b` model contracts were TP8-only.

## Shape and program ledger

| Quantity | P58.20 contract |
|---|---|
| devices / topology | four direct v5p devices / DP1xTP4 colocated |
| model hidden / intermediate | 2,560 / 9,728 |
| TP4 projection local widths | Q=1,024; K/V=256; O input=1,024; MLP=2,432 |
| tied-head local vocab | 37,984; padded width derived by the fixed-head TP4 contract |
| caller-global M | only registered request buckets and learner M=4,096 |
| shard-local M | equal to caller M at DP1; never confused with kernel M |
| canonical-kernel M | 256 |
| semantic valid rows | trajectory/action mask metadata only |
| scheduler capacity | max-num-seqs 2; max-batched-tokens 256 per DP rank |
| sampling | explicit temperature/top-k/top-p `0.7/0/1.0` |
| optimizer | TPU resident, state unchanged, commits=0 |

## Single selector and isolation

`CANON_P58_Q4_TP4_ZERO_ADMISSION=1` is the sole P58.20 workload selector.  It
is default-off and legal only with the existing one-host Zero-HP
backward-no-commit identity.  It selects a new `qwen4b_tp4` model manifest and
all seven engine replacements.  The existing `qwen4b` TP8 model variant,
P58.19 selector, production P58 arm, P59/checked-VMA, native observer, prefix
cache, and warning-only alignment are forbidden.  Neighboring Qwen1.7B,
Qwen8B, Qwen32B, native, and TP8 profiles must reject or ignore the selector.

## Gates

### L0 — ledger and registry

Register the selector with lifecycle/sunset; while this phase was active,
phase, preserve P58.19 history, and record the five-way shape ledger.

Exit gate: deterministic flag audit and task-file consistency pass.

### L1 — TP4 model construction

Add `qwen4b_tp4` projection and fixed tied-head geometry.  Assemble Qwen3,
Qwen2 helper, linear, embed, attention, RPA, and runner replacements from one
37-file manifest.  Reject wrong TP, wrong hidden width, TP8 manifest reuse,
partial overlay, and mixed treatment flags.

Exit gate: host unit/negative tests plus exact-image install and import probes
pass with `model=qwen4b_tp4 files=37 targets=7`.

### L2 — deterministic replay

Replay a saved real multi-turn trajectory without a sandbox, holding token IDs
and weights fixed.  Compare decode A, engine prefill B, and trainer C across an
initial action and an environment-to-action seam.  No XProf or bulk observer
is admitted in this numerical gate.

Exit gate: all valid replay rows have zero differing elements and bytes at
A-B, B-C, and A-C.

### L3 — real one-host R2E and backward-no-commit

Run one signed Pillow task with two generations, explicit neutral top-k/top-p,
strict alignment, full trajectories, and zero optimizer commit.  If rewards
are flat, fixed diagnostic advantages may exercise backward but are not a
quality claim.

Exit gate: real trajectories are structurally complete; A=B=C byte-exact;
backward gradients are finite, nonzero, and repeat-exact; model, reference,
optimizer, accumulator, and step fingerprints are unchanged; artifacts and
SHA-256 verify.

## Decision table

| First failing boundary | Next action |
|---|---|
| B-C RED | one-host embedding/block/final-norm/head/log-softmax localization |
| B-C exact, A-B RED | one-host decode/cache/environment-injection fingerprinting |
| A=B=C, backward RED | isolate one-host VJP/reducer without TP8 or commits |
| all gates PASS | open a separate TP8 promotion phase; do not claim TP8 yet |

## Claim ceiling and rollback

A PASS proves only Qwen3-4B-Instruct-2507 DP1xTP4 one-host integration and
numerical exactness for the signed carrier.  It does not prove TP8, DP8,
Pathways, disaggregated roles, P59, optimizer commit, or convergence.  The new
model directory, selector, profile, runner branch, and tests are additive and
default-off; removing the P58.20 concern restores all existing profiles.
