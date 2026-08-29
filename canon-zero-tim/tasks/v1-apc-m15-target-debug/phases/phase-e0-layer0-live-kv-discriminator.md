# Phase E0 — Layer-0 live-KV discriminator

## Purpose

D3e established one canonical first-action boundary on Attempt 17:

```text
last exact = layer 0 k_post_rope
first red  = layer 0 rpa_output
shape      = [2048,1,15,8]
source     = row 217, completion position 0, source position 1225
A-B        = 207 bytes / 95 elements
B-C        = 0
```

E0 tests exactly one bit-relevant degree of freedom inside that interval: does
the uniquely bound A request already have different stored layer-0 KV content
than an exact-prefix B full-reset rescore? E0 is diagnostic observation, not a
numerical repair and not an authorization to edit RPA, attention, RoPE, page
mapping, or KV values.

## Identity problem and solution

At the 1226-token red prefix, eight concurrent A requests legitimately share
the same token prefix. Selecting the first matching request would repeat the
pre-D3c identity bug. The E0 observer therefore:

1. captures all eight matching A aliases at the exact 1226-token boundary;
2. reads only layer 0 and masks the fingerprint to the 77 valid logical pages;
3. captures a same-prefix B record only through the existing full-reset
   rescore path;
4. joins all aliases to the red capsule row; and
5. uses later replay-ledger token-history receipts to require one future-prefix
   match and explicit conflicts for every alternative before choosing the
   mechanism verdict.

The static page bound is 96. The output/read bounds remain 128 MiB / 640 MiB.
The fingerprints are integer aggregates plus fixed samples. Equality is not a
collision-free proof of complete KV byte equality, and that ceiling must stay
in every claim.

## Implementation contract

- Patch 35 is append-only and consumes three default-absent flags:
  `CANON_P38_KV_OBSERVER_LAYER`,
  `CANON_P38_KV_OBSERVER_TARGET_PREFIX_SHA256`, and
  `CANON_P38_KV_OBSERVER_TARGET_PREFIX_TOKENS`.
- When those flags are absent, the historical all-layer observer remains
  unchanged and retains its 32-page maximum.
- The M15 renderer admits `--observer kv` only as one round with
  `round-alignment-v1`; seam/tail observers remain absent.
- The two arms differ only at the signed APC treatment.
- Control does not require a red join; it must still produce eight valid A/B
  fingerprint pairs and authoritative A-B=0, B-C=0.
- Treatment requires a red join and unique future-prefix request binding.
- B remains `reset_prefix_cache=True`; no cache reuse or approximate judge is
  introduced.
- Production M15 remains APC-off. Backward and optimizer commit remain zero.

## Preparation command

Run only from a clean `local/*` worktree whose HEAD equals the full published
E0 source SHA. `RUN_ID` must be a fresh 1-16 character lowercase DNS label
component:

```bash
SOURCE_COMMIT=<full-published-E0-SHA>
RUN_ID=<fresh-1-to-16-char-label>
OUT=/tmp/m15-e0-kv-${RUN_ID}
test ! -e "$OUT"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_attempt18_e0_kv_pair.sh \
  "$SOURCE_COMMIT" "$RUN_ID" "$OUT"
```

Success must end with both markers:

```text
[M15.E0.KV] RENDER_PASS ... rounds=1 layer=0 aliases=8 pages=96 ...
[M15.E0.KV] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0
```

The output contains two immutable YAMLs, `D3E_ADMISSION.json`,
`KV_CLASSIFIER_RUNTIME.json`, `RUN_CONTRACT.json`, and `SHA256SUMS`. The
runtime receipt records whether the classifier used host Python or the local
Docker dependency fallback. The Docker route accepts only the registered exact
image ID already present in the daemon and runs with `--pull=never` and
`--network=none`; it cannot contact a registry. It is only a focused dependency
route, not the official pinned exact-image aggregate. The wrapper has no
Kubernetes or GCS operation. On failure it prints `scratch_preserved=<path>`
and does not delete that diagnostic directory.

## Certification and approval order

1. host tests and source-scope audit;
2. user-approved official pinned exact-image aggregate on the published SHA;
3. separate user approval to launch the rendered off/on pair;
4. remote terminal durability and compact read-only return;
5. mechanism decision review.

Pinned-image PASS does not authorize step 3. The target pair must never be
launched from a dirty or unpublished tree, and rendered YAML must never be
edited.

After a separately approved target has finished, a bucket-capable agent runs:

```bash
RETURN=/mnt/disks/tunix-data/m15-e0-kv-return-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt18_e0_kv_gcs_return.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data
```

That wrapper reads only compact terminal/alignment/classifier artifacts. It
does not write GCS, query Kubernetes, or launch TPU. It returns
`E0_KV_RETURN.json`, two small classifier JSONs, and a self-verifying
`SHA256SUMS`; raw archives, replay ledgers, tokens, pages, credentials, and
remote roots remain outside Git and chat.

## Decision table

| Result | Meaning and next action |
|---|---|
| control A-B nonzero | `CONTROL_RED_STOP`; shared carrier/observer invalid, preserve and stop |
| any B-C nonzero | non-APC red; preserve and stop |
| treatment A-B zero | `TARGET_NON_REPRODUCTION`; no repair claim, review whether the known red was reproduced before another launch |
| treatment red + `LIVE_KV_FINGERPRINT_DIFFERS` | the uniquely bound live A cache fingerprint is already red before RPA; next hypothesis is cache production/storage/page ownership, but fingerprint evidence alone does not authorize a value fix |
| treatment red + `LIVE_KV_FINGERPRINT_EQUAL` | observed stored KV fingerprints match B; next hypothesis is page-table/read/RPA execution context, not stale KV content; equality remains non-cryptographic |
| missing alias, red join, future proof, COMPLETE, manifest, or classifier | `INCONCLUSIVE`; preserve raw failure and do not repair |

Phase E numerical repair remains closed after E0. Any proposed repair still
requires a user discussion and a new phase contract.

## Current status

- Implementation: the default-off observer/renderer/classifier and additive
  prepare-wrapper launch-readiness hardening are published.
- Host validation: PASS (task 173/173, E0 admission/runtime 9/9, KV 7/7,
  carrier 19/19, resolved-env 11/11, V1 CPU 91/91, P3 12/12, persistence and
  flags 398/398). Host Python executed the real classifier. A mocked
  forced-Docker route proved immutable-ID/`pull=never`/`network=none` command
  construction; missing and wrong images failed before run. Real Docker was
  not executed. Optional broad P33 host aggregate is dependency-INCONCLUSIVE;
  exact-image remains required.
- Official pinned exact-image for this E0 source: NOT RUN.
- Fresh DP8xTP8 E0 pair: NOT RUN.
- Publication: user-approved; the full published source SHA is returned by the
  delivery operation rather than self-recorded inside its own commit.
- Numerical repair: NOT IMPLEMENTED / NOT AUTHORIZED.
