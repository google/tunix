# P58.11 — Qwen3-4B Zero-HP checked-VMA admission

## Status

`ACTIVE / SOURCE PUBLISHED / TARGET NOT RUN`

## Objective

Admit the shared P66 checked-VMA P59 backward repair, the first committed
update gate, and the P63 overflow-safe global-norm implementation into exactly
the P58 Qwen3-4B-Instruct-2507 strict Zero-HP full recipe. Preserve the frozen
1,012-task B8 x G16, 16K/50-turn, rollout DP8 x TP8 plus trainer DP8 x TP8,
TPU-resident AdamW, and 1,000-commit scientific workload.

Native raw, Native+IS, ordinary non-HP Zero, P44/P46, and every diagnostic
profile remain unchanged and must reject the complete production bundle.

## Shape ledger

| Boundary | Registered value |
|---|---:|
| prompt groups | 8 |
| generations per prompt | 16 |
| caller-global trajectories | 128 |
| DP-shard-local trajectories | 16 |
| outer trajectory chunks | 8 (`128 / 16`) |
| rank-major gradient groups / accumulator microsteps | 16 |
| trajectories per rank-major group | 8 (one per DP rank) |
| caller-global canonical M | 2,048 |
| shard-local / canonical-kernel M | 256 |
| semantic valid trajectories | 0–128 after compact filtering |
| scheduler capacity | 16 sequences and 256 batched tokens per DP rank |

The first-update accumulator denominator is 16. The eight outer trajectory
chunks and sixteen rank-major gradient groups are distinct dimensions and may
not be collapsed.

The campaign horizon is 1,000 **commits**, not 1,000 rollout attempts. An
all-compact attempt writes a zero-commit journal row and its P59/checked-VMA
receipts, but no P63 commit receipt or global-step completion. Postflight uses
the ordered journal to remove those attempts from committed-step timing.

## Registered production bundle

The existing `--arm zero --high-performance` selector remains the single
operator-facing source of truth. The exact P58 Zero-HP profile derives:

```text
CANON_P59_CHECKED_VMA=1
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_P63_OVERFLOW_SAFE_CLIP=1
```

`00_env.sh` validates the complete P58 identity and derives the historical
`CANON_P66_P59_CHECK_VMA=1` spelling only as an internal compatibility alias.
Partial bundles and explicit conflicting aliases fail closed.

## Gates

1. **Contract gate:** profile, renderer, resolved environment, Python
   workload contract, and flag registry agree on the exact Zero/full geometry.
2. **Backward gate:** checked VMA is observed only in the P59 trainer
   backward; ordinary serving and every Native arm remain uncontaminated.
3. **First-update gate:** before AdamW, update 0 requires denominator and
   microsteps 16, all-finite, nonzero, stable L2 in `(0, 1e6]`; after AdamW it
   requires a finite coherent `0 -> 1` transaction before outer sync.
4. **Clip gate:** max norm remains 1.0. Stock-finite gradients retain stock
   Optax output; only an all-finite naive-norm overflow uses the stable L2;
   NaN/Inf never fall back.
5. **Isolation gate:** Native raw, Native+IS, non-HP Zero, P44/P46, diagnostic
   workloads, and partial flag tuples fail their positive-path assertions.
6. **Construction gate:** focused host tests, adjacent P59/P63/P66 gates, flag
   audit, syntax/diff hygiene, and complete pinned-image P58 aggregation pass.
7. **Target gate:** after separate launch approval, one fresh Attempt-0
   128-chip JobSet passes sandbox capacity, strict A=B=C, the first update, and
   exactly 1,000 committed updates with complete postflight artifacts.

## Decision table

| Observation | Verdict | Action |
|---|---|---|
| any strict A/B/C difference | Zero-TIM red | stop before the next transaction and preserve evidence |
| non-finite gradient or stable norm above `1e6` | backward red | stop before AdamW; P63 may not excuse it |
| first optimizer receipt invalid | optimizer red | stop before outer weight sync/checkpoint |
| P63 fallback used on an all-finite tree | admitted numerical event | continue only if every gate passes; disclose count and norms |
| all 128 rows compact-filtered after real admission | no-signal batch | zero commit, advance batch index only, consume next batch |
| all 128 sandboxes never start | infrastructure blocked | persist journal and stop without consuming another prompt batch |
| first update passes | inline admission PASS | continue the same full JobSet; do not stop at one or three updates |

## Claim ceiling

Host and exact-image evidence prove construction only. The shared P66
one-host oracle supports the checked-VMA mechanism but does not certify P58
DP8 x TP8. Until the target completes, the status is `TARGET NOT RUN`; no
zero-TIM, optimizer-convergence, or 128-chip performance claim is permitted.

## Rollback

The change is additive and default off. Remove the three exact P58 Zero-HP
profile opt-ins and their P58-only admission/classifier branches while leaving
the shared P59/P63/P66 implementation and all historical evidence intact.

## Local construction checkpoint

The implementation was constructed on
`644beb38cee2388862941019269ad264a581064f`, then fast-forwarded without
overlap over the operator's V1-only evidence tip
`4003f61cabb6f2d5e43d4c217cebb4dca2c3d217` before publication. It passes the
focused profile, environment, classifier, first-update, stable-clip, and real
CPU commit regressions. The commit regression executes 128 synthetic
trajectory rows as
16 rank-major groups through the P58 `_run_p28_g6_update` path, observes a
denominator of 16, emits both first-update receipts, performs one finite
parameter-changing optimizer transaction, and exports the P63 W&B metrics.
The classifier regression also inserts a legal all-compact attempt before the
first commit and proves attempts/commits/timing remain correctly reconciled.

Adjacent host gates pass: P34 static 10 suites, P59 37 tests, V1 Phase4 76
tests, P57 146 tests, and the flag registry 383/383. The dependency-bearing
pinned-image gate exits zero with:

```text
P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1 checked_vma=1 first_update=1 stable_clip=1 ... regressions=1
```

The image logs `/dev/vfio` as unavailable, so this remains CPU/construction
evidence. No direct TPU, Pathways, R2E, Kubernetes, optimizer-convergence, or
128-chip target execution occurred. Image publication and target launch remain
separately approval-bound.
