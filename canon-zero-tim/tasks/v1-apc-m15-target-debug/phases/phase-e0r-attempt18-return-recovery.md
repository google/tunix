# Phase E0r — Attempt-18 official-return recovery

## Purpose

Recover and admit the already-executed Attempt-18 E0 evidence without another
TPU/Kubernetes run. This phase repairs evidence intake only. It does not change
A/B/C, APC behavior, model arithmetic, or production defaults.

## Incoming state

- Incoming evidence commit:
  `ff33dcd200a4577927ac4917839a0b86bac42e7a`.
- Runtime source: `12207e3281db13461350fe7ef68dbaadfe713a58`.
- Reported control: A-B=0, B-C=0, `N_action=123010`.
- Reported treatment: A-B=1499 bytes / 88 elements, B-C=0,
  `N_action=117834`, 92.8% cache hits.
- The committed two-member package is not the official return: classifier
  files and terminal log are absent, digests are invalid, request binding is
  truncated, and the mechanism claim exceeds the classifier ceiling.

## Implementation gate

The recovery wrapper must:

1. require a clean exact-SHA `local/*` worktree and canonical preflight;
2. verify the preserved original `e01` render contract and its manifest;
3. accept only runtime source `12207e3281db13461350fe7ef68dbaadfe713a58`;
4. run the official read-only return into a fresh output path without a pipe;
5. preserve raw log, output, and official scratch on failure;
6. require exactly the summary, two classifier JSONs, and `SHA256SUMS`;
7. require canonical JSON and 64-character SHA256 cross-links;
8. require 16 observer records, eight A/B pairs, exact 1226-token geometry,
   and one explicit match plus seven future-prefix conflicts;
9. verify manifest-bound runtime source, B full reset, all B cached tokens
   zero, zero backward, and zero optimizer commit from the remote raw log;
10. emit `numerical_repair_authorized=0` and perform no GCS write,
    Kubernetes operation, or TPU action.

## Validation gate

- focused return-intake/recovery tests: 10/10 PASS, including a complete fake
  read-only transport round trip;
- existing E0 admission/runtime tests: 9/9 PASS;
- Python and Bash syntax: PASS;
- task discovery 183/183, V1 CPU 91/91, P3 prefix-cache 31/31, P38
  persistence, and flag registry 398/398: PASS;
- host raw log: `/tmp/m15-e0r-host-gate-ff33dcd2-20260829.log`, SHA256
  `7758ee965a06edddd5fed1c37f6253e6e5629d30a791521ed887bb34cb2e687c`;
- official pinned exact-image for this additive tree: NOT RUN;
- real read-only GCS recovery: NOT RUN;
- TPU/Kubernetes: NOT RUN.

The user explicitly approved commit/push of the HOST-PASS recovery tree on
2026-08-29. This publication approval does not authorize or waive the pinned
exact-image and GCS-read gates.

## Claim gate

No mechanism verdict is admitted until all official, intake, and recovery
terminal markers pass and the exact four-file directory verifies unchanged.
Fingerprint equality is a diagnostic aggregate/fixed-sample result, not proof
of all KV bytes.

## Decision table

| Result | Phase result |
|---|---|
| render/remote member/manifest/source/terminal/B-reset proof absent | `INCONCLUSIVE`; preserve and stop |
| control A-B red or either B-C red | carrier/shared-path hard stop |
| treatment exact | `TARGET_NON_REPRODUCTION`; not fixed |
| `LIVE_KV_FINGERPRINT_DIFFERS` | discuss cache content/page ownership discriminator |
| `LIVE_KV_FINGERPRINT_EQUAL` | discuss exact block-table/metadata/gather discriminator before internal RPA math |

Phase E numerical repair stays closed for every E0r result until the user
reviews and explicitly opens a bounded next implementation.
