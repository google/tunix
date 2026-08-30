# Phase E0w — exact-TiTO one-host APC carrier

## Purpose

Before any fresh DP8xTP8 launch, exercise the E0v exact-token program on the
idle four-chip v5p host as a matched APC-off/APC-on pair. This is a carrier
gate only. A one-host result cannot inherit the historical prefix-1226
localization and cannot certify the multihost TP8 target.

Any later target launch remains exactly the fresh three-round E0v debug pair.
This phase does not authorize a production M15 full run.

## Signed one-host identity

- Qwen3-8B, DP1xTP4, one physical four-chip v5p host;
- `CANON_P38_ONEHOST_REHEARSAL=1` and no target-debug selector/profile;
- M15/main token program with `CANON_M15_TOKEN_CONTINUITY=exact`;
- APC-off control followed by APC-on treatment from one source/diff/image;
- three ordered rounds per arm, strict A/B/C, zero backward, zero optimizer
  commits, no evaluation, no checkpoint, no GCS, and no Kubernetes;
- B must emit a full-reset/all-cached-token-zero receipt in every round;
- every round must contain at least one exact-equal TiTO receipt.

The target DP8xTP8 identity continues to reject
`CANON_P38_ONEHOST_REHEARSAL=1`. The one-host exception is selected by the
existing rehearsal flag and admits only DP1xTP4 with APC exactly `0` or `1`;
no new flag or production default is added.

## Implementation

- `run_m15_e0v_tito_onehost_arm.sh` runs one immutable arm, records the exact
  source diff and pinned image ID, and refuses a busy lane or evidence path
  collision.
- `classify_m15_e0v_onehost_arm.py` requires three strict records, B-C zero,
  per-round B reset receipts, and positive APC-on cache hits. Treatment A-B
  may complete as exact or red; neither is laundered into infrastructure
  `INCONCLUSIVE`.
- `classify_m15_apc_debug_tito.py --scope onehost` requires per-round
  exact-equal token receipts.
- `run_m15_e0v_tito_onehost_pair.sh` runs control first, stops if it is
  invalid, then treatment, seals each arm, and invokes
  `classify_m15_e0v_onehost_pair.py`.
- The pair classifier verifies both arm manifests, raw-log bindings,
  source/diff/image equality, and that the contracts differ only at APC.

## Gate ladder

1. canonical E0w host aggregate;
2. official pinned exact-image aggregate on immutable image
   `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
3. one-host pair on an idle v5p lane;
4. only after a separate approval, prepare and launch the fresh three-round
   DP8xTP8 debug pair.

The user separately approved both pinned exact-image and one-host execution.
Both gates have now passed. Step 4 remains separately approval-gated.

## Decision table

| Result | Classification | Next action |
|---|---|---|
| control A-B nonzero | carrier invalid | hard stop; do not run treatment |
| either B-C nonzero | shared-path red | hard stop; not APC-specific |
| missing TiTO or B-reset receipt | invalid carrier | preserve evidence and fix carrier only |
| treatment exact in all rounds | `ONEHOST_PAIR_EXACT` | one-host gate green; still run fresh DP8xTP8 debug |
| treatment A-B red, B-C zero | `ONEHOST_RED_REPRODUCED` | preserve local reproducer; FIRST_RED still not localized |
| Docker/TPU infrastructure failure | `INCONCLUSIVE` | preserve run directory; no numerical verdict |

## Current status

- Focused CPU: token identity 7/7, TiTO postflight 7/7, arm classifier 5/5,
  pair classifier 5/5, runner contract 3/3.
- Canonical host aggregate: PASS. Terminal
  `M15_E0W_HOST_PASS task_discovery=225 return=1 round0_recovery=8
  tito_postflight=7 onehost_arm=5 onehost_pair=5 onehost_runner=3
  token_continuity=7 v1_cpu=92 p3_prefix_cache=31 persistence=1 flags=409
  manifest=dae6dfa8 syntax=1 diff_check=1 exact_image=0 onehost_v5p=0
  target_rerun=0 gcs=0 kubernetes=0 tpu=0`.
- Latest post-ledger host log:
  `/tmp/m15-e0w-host-gate-20260830-r3.log`, SHA256
  `4a159cbed02337b4c878de582e06f319d991e1609f04da2b9de3f8d3c8af9762`.
- Official pinned exact-image: PASS. Latest complete raw log
  `/tmp/m15-e0w-exact-image-20260830-r4.log`, SHA256
  `65a1c193887601db845aada30532bddce1b6b69b5d79c266b1357f4d0414c105`,
  terminal `V1_HP_EXACT_IMAGE_PASS` with TiTO/arm/pair/runner `7/5/5/3`,
  durability and round provenance `1/1`, and `manifests=3`.
- Post-rebase publication admission: canonical host PASS at
  `/home/yuxuan/code_rl_repro/m15-e0w-host-gate-postrebase-r1.log`, SHA256
  `fcf24a07c0ab7f6199fa0555ac583968568f58e9ed51bd59144e94eaf8140f05`;
  complete pinned-image PASS at
  `/home/yuxuan/code_rl_repro/m15-e0w-exact-image-postrebase-r1.log`, SHA256
  `7f4b2dc4703ce4713b5ed2a6802f279481f27a7cd60eec301678a3b114984b49`.
  Its terminal retains `m15_tito_impl=1 m15_tito_default=off`, E0w
  `7/5/5/3`, durability/provenance `1/1`, and `manifests=3`.
- One-host attempts `e0w1`, `e0w2`, and `e0w3` are immutable
  `INCONCLUSIVE` carrier failures at P57 CLI/env identity, narrow
  profile-less M15 admission, and replicated trainer mesh admission,
  respectively. The fixes only change one-host identity/admission and runner
  CLI geometry.
- One-host `e0w4`: `ONEHOST_PAIR_EXACT`. Both arms completed three rounds;
  A-B and B-C byte counters are `[0,0,0]`; APC-on reached 91.5% cache hits;
  TiTO receipts are `[4,7,6]` and B full-reset receipts are `[1,1,1]` per
  arm. Root `SHA256SUMS` SHA256 is
  `b52fe75af56f5c66c2ba352d25163a18ab854c823b17c71d91a555fc02155589`.
- The `e0w4` TPU result is bound to its pre-rebase source/diff. It is preserved
  evidence, not a TPU certification of the rebased publication source.
- DP8xTP8 target: NOT RUN.
- Numerical repair: NOT AUTHORIZED.
- Infrastructure: the root filesystem reports 100% use with about 488 MiB
  available. No evidence was deleted; check capacity before another local
  gate.
