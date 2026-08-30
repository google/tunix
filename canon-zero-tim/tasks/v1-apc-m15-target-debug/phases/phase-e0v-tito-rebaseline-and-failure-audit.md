# Phase E0v — preserve the failed join and re-baseline under exact TiTO

## Verified input

Attempt 20 treatment round 0 reached its durable classifier-input checkpoint,
and E0u retrieved the archive, receipt, and manifest. The archived classifier
failed closed before a live-KV mechanism classification because no observer
token vector exactly prefixed a red capsule history. The committed incident
reports a first divergence at token index 913 in a 1226-token observer vector.
The current compact bundle proves the traceback, but it does not contain a
machine-replayable bounded join receipt or the already validated round input.

Attempt 20 predates native M15 exact token continuity. That chronology supports
pre-TiTO re-tokenization as the carrier explanation, but it does not prove an
APC numerical root cause or that TiTO makes A-B exact.

## Deliverables

1. A classifier failure still writes a self-hashed compact return containing
   source-bound object hashes, A-B/B-C counters, eight unbound A/B KV
   comparisons, and a bounded token-prefix matrix. It must retain
   `classification=NONE`, unavailable B-reset receipts, no completed round,
   and no repair authorization.
2. One local-only wrapper re-audits the already preserved E0u scratch. It must
   not invoke GCS, Kubernetes, or TPU and must never return raw NPZ/token rows.
3. A new matched-pair prepare wrapper renders three-round `observer=layer`
   APC-off/on jobs with exact TiTO in both arms. Historical `kv`/`kv3` and the
   old 1226-token selector must reject TiTO rather than inherit an obsolete
   program identity.
4. Runtime postflight requires at least one exact-equal bounded TiTO receipt in
   each diagnostic round, no different/malformed receipt, ordered 1/3..3/3
   round completion, zero backward, and zero optimizer commits.

## Gates

- failure-audit focused tests include source drift, ambiguous future binding,
  token-history mismatch, bounded request hashing, and local-only scratch
  routing;
- flag delivery is verified renderer -> debug profile -> real `00_env.sh` ->
  resolved `env.sh` -> `m15_token_continuity_mode()`;
- opposite-arm positive control: both APC-off and APC-on render exact TiTO;
- adjacent negatives: `kv`, `kv3`, `full`, one round, wrong arm, verify mode,
  another topology, or a P57 training identity fail closed;
- postflight classifier tests exact 3-round PASS, one different receipt,
  missing per-round receipt, wrong arm marker, and skipped round;
- canonical host aggregate, then separately approved official pinned image;
- target launch remains a later, separate approval.

## Decision table

| Observation | Claim | Next action |
|---|---|---|
| preserved scratch returns `TOKEN_HISTORY_JOIN_MISMATCH` | historical Attempt-20 carrier identity is incompatible; no live-KV verdict | retain audit; do not patch classifier or numerical code |
| scratch is absent | local recovery unavailable | report `classification=NONE`; request direction before any read-only GCS retry |
| TiTO layer control A-B is red | carrier/shared serving path invalid | hard stop and preserve all rounds |
| either arm B-C is red | not APC-specific | hard stop |
| TiTO treatment A-B is exact | new-program target non-reproduction | do not say APC fixed; plan repeat/dirty-page certification |
| TiTO treatment is red | fresh APC mismatch under the new program | use its new source identity to localize; never reuse old prefix 1226 |
| any TiTO receipt differs or a round has none | token contract invalid | fail target before interpreting A/B/C |

## Claim ceiling

Canonical host validation PASS:

```text
M15_E0V_HOST_PASS task_discovery=210 return=1 round0_recovery=8 tito_postflight=5 token_continuity=6 v1_cpu=92 p3_prefix_cache=31 persistence=1 flags=409 manifest=dae6dfa8 syntax=1 diff_check=1 exact_image=0 target_rerun=0 gcs=0 kubernetes=0 tpu=0
```

Raw log: `/tmp/m15-e0v-host-gate-20260830-r1.log`, SHA256
`75f7263daea0768ef74dc7c27cbabeae35c438045a4d0ec1d7be7928ef697e69`.
Preserved-scratch execution, official pinned image, and target remain NOT RUN.

Until a fresh matched DP8xTP8 pair completes, the ceiling is:

```text
ATTEMPT20_E0U_CLASSIFIER_JOIN_FAILED
FAILURE_AUDIT_IMPLEMENTED
EXACT_TITO_DEBUG_CARRIER_IMPLEMENTED
CANONICAL_HOST_PASS
EXACT_IMAGE_NOT_RUN
TARGET_NOT_RERUN
HISTORICAL_FIRST_RED_NOT_INHERITED
NO_TARGET_PASS
NO_NUMERICAL_REPAIR_AUTHORIZATION
```
