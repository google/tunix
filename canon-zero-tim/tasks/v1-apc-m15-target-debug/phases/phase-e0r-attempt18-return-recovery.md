# Phase E0r — Attempt-18 official-return recovery

## Purpose

Recover and admit the already-executed Attempt-18 E0 evidence without another
TPU/Kubernetes run. This phase repairs evidence intake only. It does not change
A/B/C, APC behavior, model arithmetic, or production defaults.

## Incoming state

- Latest evidence commit reviewed:
  `971bb2281417ecb6e33cfa6bb68a422f7fd24f00`.
- Runtime source: `12207e3281db13461350fe7ef68dbaadfe713a58`.
- Reported control: A-B=0, B-C=0, `N_action=123010`.
- Reported treatment: A-B=1499 bytes / 88 elements, B-C=0,
  `N_action=117834`, 92.8% cache hits.
- The first committed two-member package was not official. Commit 971bb228
  replaced it with a manifest-valid four-file package, but that package is
  also rejected: it names the wrong classifier and wrong SHA, repeats one
  digest across unrelated inputs and both arm manifests, omits fields emitted
  by the runtime classifier, uses impossible absolute provenance paths, and
  has no preserved recovery raw-log receipt. The replacement did not convert
  the reported target metrics into an admitted E0 verdict.

## Implementation gate

The recovery wrapper must:

1. require a clean exact-SHA `local/*` worktree and canonical preflight;
2. verify the preserved original `e01` render contract and its manifest;
3. accept only runtime source `12207e3281db13461350fe7ef68dbaadfe713a58`;
4. run the official read-only return into a fresh output path without a pipe;
5. preserve raw log, output, and official scratch on failure;
6. require exactly the summary, two classifier JSONs, and `SHA256SUMS`;
7. require canonical JSON, exact pinned runtime classifier path/SHA, its exact
   four-line claim ceiling, and 64-character SHA256 cross-links;
8. require 16 complete observer records with distinct identity/provenance,
   eight complete A/B comparisons, exact 1226-token geometry, basename-only
   capsule/replay paths, and one explicit match plus seven future-prefix
   conflicts;
9. verify manifest-bound runtime source, B full reset, all B cached tokens
   zero, zero backward, and zero optimizer commit from the remote raw log;
10. reject collapsed off/on root manifests or unrelated equal digests;
11. require a raw log at the CLI and all official/recovery terminal markers;
12. emit `numerical_repair_authorized=0` and perform no GCS write,
    Kubernetes operation, or TPU action.

## Validation gate

- focused return-intake/recovery tests: 14/14 PASS, including a complete fake
  read-only transport round trip, the locked 971bb228 rejection, collapsed
  provenance negatives, and absolute-path rejection;
- existing E0 admission/runtime tests: 9/9 PASS;
- Python and Bash syntax: PASS;
- task discovery 187/187, V1 CPU 91/91, P3 prefix-cache 31/31, P38
  persistence, and flag registry 398/398: PASS;
- host raw log:
  `/tmp/m15-e0r-provenance-hardening-971bb228-retry2-20260829.log`, SHA256
  `f11ab8b9bf137f7f7ca39a801fe06b6da6298b7b558fe817ea2f503f7f74a4e4`;
- official pinned exact-image for this additive tree: NOT RUN;
- real read-only GCS recovery: NOT RUN;
- TPU/Kubernetes: NOT RUN.

The present provenance-hardening changes are uncommitted. A previous
publication approval does not authorize this commit/push, the pinned
exact-image, or the GCS-read gate; each remains a separate user action.

## Claim gate

The 971bb228 return has failed this gate and is permanently preserved as a
negative regression. No mechanism verdict is admitted until a fresh additive
recovery passes exact-image, official, intake, and recovery markers with a
preserved raw-log SHA. Fingerprint equality would remain a diagnostic
aggregate/fixed-sample result, not proof of all KV bytes.

## Decision table

| Result | Phase result |
|---|---|
| render/remote member/manifest/source/terminal/B-reset proof absent | `INCONCLUSIVE`; preserve and stop |
| classifier identity/complete-field/distinct-provenance check fails | `OFFICIAL_RETURN_PROVENANCE_FAIL`; preserve and stop |
| control A-B red or either B-C red | carrier/shared-path hard stop |
| treatment exact | `TARGET_NON_REPRODUCTION`; not fixed |
| `LIVE_KV_FINGERPRINT_DIFFERS` | discuss cache content/page ownership discriminator |
| `LIVE_KV_FINGERPRINT_EQUAL` | discuss exact block-table/metadata/gather discriminator before internal RPA math |

Phase E numerical repair stays closed. The immediate next gates are a
separately approved commit/push, official pinned exact-image with `m15_e0=30`,
and a separately approved read-only GCS recovery from the preserved render.
No TPU/Kubernetes run is currently requested.
