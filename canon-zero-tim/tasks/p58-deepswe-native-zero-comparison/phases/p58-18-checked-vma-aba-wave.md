# P58.18 — Checked-VMA exact-geometry matched triplicate

Status: implementation and complete pinned-image construction validation pass;
source publication approved for this delivery; matching source image not
published; target not run.

## Trigger

P58z07 and p58z08 both returned finite decode-vs-prefill A-B RED with exact
B-C before backward. The existing P58.17 `off` carrier is decisive only if its
baseline ON behavior reproduces in the same source/image/geometry. This phase
therefore replaces an isolated OFF attempt with one matched OFF control and
two independently named ON replicates.

## Frozen carrier

Each JobSet is Qwen3-4B-Instruct-2507 on 128 TPU chips: rollout DP8xTP8 plus
trainer DP8xTP8. It keeps the reviewed 1,012-task whitelist, B8xG16, 16K
response, 50 turns, seed 42, concurrency 128, fixed lm-head, continue-decode
8, prefix cache off, strict pre-alignment, and full durable trajectory logs.
It executes exactly one Step-0 precheck and exits code 42 before fixed-head
VJP, P59/P66 backward, AdamW, checkpoint, or optimizer commit.

The logical arm order is `ON-A/OFF/ON-B`. Both ON arms derive checked VMA,
P66 alias, and P67 scoping as `1/1/1`; OFF derives `0/0/0`. All three disable
the first-update gate and P63 clip because no backward is permitted. An absent
selector preserves the production Zero-HP tuple `1/1/1/1/1` unchanged.

The user plans concurrent submission. In that mode the names retain logical
ABA ordering, but the evidence is a concurrent matched OFF control with two ON
replicates, not a temporal drift sandwich. Cross-run token identity is not a
hard gate; each arm must independently have exact B-C and valid finite/exact
A-B evidence.

## Resource and launch boundary

The three independent JobSets request 384 TPU chips in aggregate, three
anti-affined CPU head nodes, and up to 384 concurrent R2E sandboxes. At the
signed two-CPU/four-GiB sandbox request this is 768 CPU and 1,536 GiB aggregate
sandbox request, excluding the three Pathways heads. Construction PASS does
not prove Kueue will admit them concurrently. Server dry-run, aggregate CPU
capacity evidence, image publication, and apply each require separate user
approval.

## Exit gate

1. Renderer, profile, authoritative env reload, Python contract, postflight,
   per-arm classifier, and flag registry admit only absent/off/on as specified.
2. Render and re-parse three digest-pinned YAMLs with unique JobSet names,
   persistent roots, arm labels, and identical frozen recipe signatures.
3. ON-A and ON-B treatment signatures are identical; OFF differs only by the
   registered checked-VMA selector.
4. Every returned arm has 128 durable rows, finite positive action count,
   exact B-C, one valid A-B classification, controlled exit, and zero
   VJP/backward/optimizer evidence.
5. `ON RED / OFF exact / ON RED` supports checked VMA as the reproduced causal
   discriminator. `RED/RED/RED` rejects it as sufficient. Nonreplicating ON
   controls are inconclusive and must not be averaged away.

No commit, push, image publication, Kubernetes mutation, or TPU launch is
authorized by this phase file.

## Local checkpoint

- Focused renderer 27/27, profile 9/9, per-arm classifier 7/7, and wave
  render/verify/classifier 4/4 pass.
- Deterministic flag audit passes declared/actual/unique `393/393/393`.
- The complete pinned dependency-image CPU gate terminates with
  `P58_EXACT_IMAGE_CPU_PASS ... checked_vma_diagnostic=1
  checked_vma_aba=1 ... regressions=1`.
- This proves construction and adjacent regressions only. No real Pathways,
  DP8xTP8 serving, Kueue, R2E capacity, or TPU numerical result exists.
