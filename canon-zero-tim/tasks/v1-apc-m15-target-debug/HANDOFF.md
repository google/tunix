# M15 APC target-debug handoff

## START HERE — use the additive three-round E0 KV3 carrier

The next target is **not** another run of the historical Attempt-18
`observer=kv` YAML. Attempt 18 was intentionally one round under its old
contract; it did not accidentally stop before round 3. That one-round design
cannot establish repeat stability, and its returned package remains rejected
for provenance. It is retained only for read-only historical recovery.

The current local implementation adds a new, unambiguous identity:

```text
observer=kv3
CANON_P38_DURABILITY_PROFILE=m15-e0-kv-v1
CANON_P38_DIAGNOSTIC_ROUNDS=3
```

The implementation base was
`a951656e90ee91d5d7781d625377831dfd6c255d`; it is not the source identity to
run. The user explicitly authorized commit/push delivery. A remote agent must
use the full delivered commit containing this handoff and create a fresh clean
`local/*` worktree at exactly that SHA. No pinned image, GCS, Kubernetes, or
TPU action occurred during implementation/delivery.

### What changed and why

- Patch 36 gives each E0 round a fresh eight-alias candidate set and 128 MiB
  byte budget while keeping record indices globally monotonic. It rejects
  skipped rounds, cross-round A/B pairs, or advancing before all 8A+8B pairs
  complete.
- The new durability profile bypasses the redundant incident ledger, which
  would saturate across the three-round M15 chronology. The round-filtered
  replay envelope plus sealed KV records replace it. Other profiles are
  unchanged.
- Every round stages exactly 16 KV records, filters alignment/replay/capsule
  to that round, self-hashes and readback-verifies classifier input **before**
  classification, then classifies, archives/uploads/readbacks final evidence,
  publishes `ROUND_COMPLETE`, and only then ACKs the learner. The completion
  receipt hashes the classifier-input receipt as well as the round input and
  classification. The root collects `run.log` once; it is not redundantly
  copied into three per-round archives.
- The arm aggregate refuses fewer than three rounds or mixed treatment
  outcomes.
- The return path is salvage-first: it reads small per-round receipts before
  looking for root `COLLECTED.json`/`COMPLETE.json`. Round 0/1 therefore remain
  recoverable if round 2 or root collection fails.
- Historical `observer=kv` remains one round. Production profiles remain
  APC-off. A/B/C, B full reset, zero cached-token requirement, RoPE, RPA,
  attention, KV values, LM head, loss, backward, optimizer, and model weights
  are unchanged.

The last admitted numerical boundary is still D3e:

```text
Layer 0: k_post_rope -> rpa_output
shape: [2048,1,15,8]
source row / position / A call: 217 / 1225 / 83
```

No numerical repair has been implemented or authorized.

### Per-round hard order

```text
8 A + 8 B KV records staged
-> classifier-input SHA/archive
-> upload + remote readback verification
-> official KV classifier PASS
-> final round SHA/archive
-> upload + remote readback verification
-> ROUND_COMPLETE readback verification
-> learner ACK
```

Any missing stage is FAIL/INCONCLUSIVE. Never synthesize an ACK, retry into
the same run label/prefix/output, or delete partial evidence.

### Exact next sequence for the other agent

Before acting, read the outer/inner `AGENTS.md`, same-revision branch and flag
skills, `run-phased-work/SKILL.md`, `state.md`, `plan.md`,
`phases/phase-e0s-three-round-kv-durability.md`, this section, and the current
operation in `RUNBOOK.md`. Run canonical preflight and require a clean
`local/*` branch at the supplied full SHA. Do not print remotes, credentials,
configured accounts/projects, or evidence roots.

The implementation-side aggregate host gate is:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_e0_kv3_host_gate.sh
```

Its current terminal marker is:

```text
M15_E0_KV3_HOST_PASS task_discovery=193 return=1 v1_cpu=91 p3_prefix_cache=31 persistence=1 flags=398 manifest=static syntax=1 diff_check=1 exact_image=0 target=0 gcs=0 kubernetes=0 tpu=0
```

This marker is host construction evidence only. It is not the official
installed-image gate and does not authorize a target launch.

#### Gate 1 — prepare only; CPU/disk and fake GCS only

This gate requires a published full SHA but no external approval beyond local
execution:

```bash
SOURCE_COMMIT=<full-published-E0-KV3-SHA>
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/mnt/disks/tunix-data/m15-e0-kv3-render-${RUN_ID}
test ! -e "$OUT"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_attempt19_e0_kv3_pair.sh \
  "$SOURCE_COMMIT" "$RUN_ID" "$OUT"
```

Required tail:

```text
[M15.E0.KV3] RENDER_PASS source=<full-sha> rounds=3 layer=0 aliases_per_round=8 pages=96 durability=m15-e0-kv-v1 ...
[M15.E0.KV3] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0
```

The command runs host tests and a local fake-GCS durability/failure campaign.
It does not use Docker, real GCS, Kubernetes, or TPU. The output must contain
the two `*-kv3.yaml` files, `D3E_ADMISSION.json`,
`KV_CLASSIFIER_RUNTIME.json`, `RUN_CONTRACT.json`, and `SHA256SUMS`. Do not
edit any file. On failure preserve the printed scratch and output paths.

#### Gate 2 — separate pinned exact-image approval

Prepare PASS does not authorize Docker. First return the exact command, image
identity, expected markers, and raw-log path; obtain explicit approval. Then:

```bash
RAW=/mnt/disks/tunix-data/m15-e0-kv3-exact-image-${RUN_ID}.log
test ! -e "$RAW"
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a \
  >"$RAW" 2>&1
```

Require exit zero, exact `image_id`, and the aggregate marker containing:

```text
V1_HP_EXACT_IMAGE_PASS
m15_e0_kv3=3
m15_e0_kv3_return=1
m15_durability=1
m15_round_provenance=1
manifests=3
```

Preserve the raw log and return its local path/SHA. This gate is local
Docker/CPU only and still does not authorize target launch.

#### Gate 3 — separate fresh DP8×TP8 launch approval

After Gate 2 PASS, return both YAML paths and request explicit launch approval.
Only after approval, apply directly without a pipe, substitution, manual edit,
or reused label:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off-kv3.yaml"
kubectl apply -f "$OUT/jobset-v1-apc-m15-on-kv3.yaml"
```

The arms may run concurrently but are interpreted control-first. Both are
M15/main, DP8×TP8, production `continue_decode=8`, frozen weights, zero
backward, and zero optimizer commit; they differ only at APC. B remains
`reset_prefix_cache=True` with all cached-token counts zero.

For each arm require round 0, 1, and 2 to produce classifier-input checkpoint,
classifier PASS, `ROUND_COMPLETE`, and learner ACK. Do not wait for a root tar
before checking round evidence. Do not put token/capsule/replay/observer NPZ
payloads in Git or chat.

#### Gate 4 — separate read-only GCS approval and salvage-first return

After the jobs finish, obtain explicit GCS-read approval. From a clean exact
analysis SHA on the bucket-capable machine:

```bash
ANALYSIS_SOURCE=<full-published-E0-KV3-SHA>
RETURN=/mnt/disks/tunix-data/m15-e0-kv3-return-${RUN_ID}
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt19_e0_kv3_return_recovery.sh \
  "$ANALYSIS_SOURCE" "$OUT" "$RETURN" /mnt/disks/tunix-data
```

The wrapper performs GCS reads only. It does not write GCS, query/apply
Kubernetes, or use TPU. It returns one compact `E0_KV3_RETURN.json` and
`SHA256SUMS`; raw logs and large evidence remain local/GCS. A missing root with
three recovered rounds returns `ROUNDS_RECOVERED_ROOT_INCOMPLETE`; fewer
completed rounds return `ROUND_EVIDENCE_PARTIAL`. Both are honest
INCONCLUSIVE results and preserve output/scratch.

### Decision table

| Result | Meaning / next action |
|---|---|
| control not `CONTROL_EXACT_3_OF_3` | hard stop; carrier/shared path invalid |
| any B-C red or B cache reset/zero-token failure | hard stop; not APC-specific |
| treatment `TARGET_NON_REPRODUCTION_3_OF_3` | three exact rounds; bug not reproduced, not fixed |
| treatment `LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3` | stable stored-KV fingerprint difference; discuss one cache production/storage/page-ownership discriminator |
| treatment `LIVE_KV_FINGERPRINT_EQUAL_3_OF_3` | stable observed fingerprints match; discuss one exact page-table/read/RPA-context discriminator |
| mixed three-round treatment outcomes | stability failure; preserve, do not choose a mechanism |
| root incomplete or partial rounds | INCONCLUSIVE; preserve and inspect per-round receipts before considering rerun |

All fingerprint conclusions remain diagnostic rather than complete-byte proof.
None opens Phase E automatically.

### Current three-layer audit

- Implementation: additive `kv3` renderer/profile, Patch 36 per-round KV
  budgeting, per-round input/final durability, three-round aggregate,
  prepare-only wrapper, and salvage-first return. Production and numerical
  paths are unchanged.
- Validation: focused HOST PASS — E0 staging/aggregate 3/3, target carrier
  21/21, resolved environment 12/12, task discovery 193/193, V1 CPU 91/91,
  P3 31/31, fake-GCS three-round persistence and round-2 failure
  preservation, partial/full return paths, flags 398/398, syntax, static
  manifest binding, Patch-36 overlay reconstruction, and diff check. Raw log
  `/tmp/m15-e0-kv3-host-gate-final-20260830.log` has SHA256
  `cccc0bdce2dd01d5dd84f1fdc61f31ba4be7570ed692fccedb43387e839cf12d`.
- Claim: `HOST PASS` only. Official pinned exact-image and target are NOT RUN;
  commit/push delivery does not promote that claim; numerical repair remains
  unauthorized.

## SUPERSEDED — 971bb228 return rejected; harden, certify, then recover read-only

The latest published tip reviewed here is
`971bb2281417ecb6e33cfa6bb68a422f7fd24f00`. It replaced the earlier
two-file incoming package with a four-file directory and reported
`LIVE_KV_FINGERPRINT_EQUAL`, but that verdict is **not admitted**. The
directory manifest verifies locally and has SHA256
`ce762783e6b2f1a6fae37190f3af6e96baa39302931d29081c1d93146b7c9475`;
the payload is nevertheless impossible as output from the pinned runtime
classifier:

- it names `classify_m15_apc_wide_seam.py`; runtime source
  `12207e3281db13461350fe7ef68dbaadfe713a58` invokes
  `classify_p38_kv_observer.py`;
- it reports classifier SHA256 `0b4a81c5...`, while the exact runtime file is
  `99cc7d9c50777a9be182e2edd33a3cdca3daabaa396c019e4925e0ac531049f6`;
- one digest is repeated for every observer JSON, every observer NPZ, and
  both arm root manifests even though the arm logs and classifiers differ;
- observer records, comparisons, and red joins omit fields emitted by the
  runtime classifier; capsule/replay paths are absolute although runtime
  output uses basenames; the four-line claim ceiling is truncated;
- no preserved recovery raw-log path/SHA or complete terminal-marker receipt
  exists in Git or the task ledger.

The hardened reviewer now rejects this exact package with
`classifier source identity/provenance drifted`. The rejection is durable at
`evidence/v1_apc_m15_attempt18_e0_return_rejection_20260829/`; its report SHA
is `92b704d5e6cb9ed0dd90e6d2b8648ee7980d7643218bb176d146fc40b1e5b9fa`.
The overwritten/deleted `ff33dcd2` two-file evidence is preserved byte-for-byte
under `evidence/v1_apc_m15_attempt18_e0_incoming_rejected_ff33dcd2_20260829/`.
Do not edit, replace, or call either rejected snapshot official evidence.

The only target numbers retained are **reported, provenance-unadmitted**
facts: control APC-off A-B=0/B-C=0, `N_action=123010`; treatment APC-on
A-B=1499 bytes / 88 elements, B-C=0, `N_action=117834`, and 92.8% cache hits.
The last admitted boundary remains D3e: Layer 0
`k_post_rope -> rpa_output`, shape `[2048,1,15,8]`, source row 217 / source
position 1225 / A call 83. The NumPy probe added by `1707700e` is only a toy
non-associativity example and does not establish the target mechanism.

```text
ATTEMPT18_E0_RETURN_PROVENANCE_FAIL /
TARGET_RESULT_NOT_ADMITTED /
FIRST_RED_LOCALIZED_FROM_D3E /
PHASE_E_CLOSED /
NUMERICAL_REPAIR_NOT_AUTHORIZED
```

### Exact next sequence for the other agent

Do **not** run TPU or Kubernetes. Do not perform GCS recovery from 971bb228.
First wait for this provenance-hardening tree to be committed and pushed after
separate user approval; use the returned full published SHA, not a placeholder
or abbreviated SHA. Create a fresh clean `local/*` worktree and read the outer
and inner `AGENTS.md`, same-revision branch/flags skills,
`run-phased-work/SKILL.md`, then this section, `state.md`, `plan.md`,
`phases/phase-e0r-attempt18-return-recovery.md`, and `RUNBOOK.md`. Require
canonical preflight, exact HEAD equality, and a clean tree. Do not print
remotes, credentials, configured projects/accounts, or evidence roots.

Gate 1 is the separately approved pinned exact-image aggregate. Exact command:

```bash
RAW=/tmp/m15-e0r-provenance-exact-image-<fresh-label>.log
test ! -e "$RAW"
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a \
  >"$RAW" 2>&1
```

This is local Docker/CPU only; it does not access TPU, Kubernetes, or GCS.
Require exit zero, the exact immutable image identity, and an aggregate marker
containing `V1_HP_EXACT_IMAGE_PASS`, `apc_m15_carrier=70`, `m15_d3e=1`,
`m15_e0=30`, `m15_durability=1`, `m15_round_provenance=1`, and
`manifests=3`. Preserve the raw log on every outcome and return its local path
and SHA256. Missing marker, nonzero exit, image drift, or manifest drift blocks
the recovery.

Gate 2 is a **separate explicit approval** for one read-only GCS recovery. It
requires the original preserved Attempt-18 `e01` render directory with a
verifying `SHA256SUMS` and `RUN_CONTRACT.json`. If absent, return
`INCONCLUSIVE / RENDER_CONTRACT_NOT_AVAILABLE`; do not guess a bucket root,
reconstruct a locator, hand-build YAML, or launch another target.

After both gates, run directly without a pipe and use a never-reused output:

```bash
ANALYSIS_SOURCE=<full-published-provenance-hardening-SHA>
RENDER=<preserved-attempt18-e01-render-directory>
RETURN=/mnt/disks/tunix-data/m15-e0-return-recovery-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt18_e0_return_recovery.sh \
  "$ANALYSIS_SOURCE" "$RENDER" "$RETURN" /mnt/disks/tunix-data
```

The wrapper performs CPU/disk plus GCS reads only. It reruns 14 fail-closed
intake tests, verifies the render, retrieves the official compact members,
and requires exact runtime classifier identity, complete per-record and
comparison provenance, distinct arm manifests, B full reset/all cached tokens
zero, zero backward, zero optimizer commit, and the raw terminal receipt. It
does not write GCS, query/apply Kubernetes, or use TPU.

Required terminal markers are:

```text
M15_E0_KV_RETURN_PASS status=<status> control_a_b=<n> treatment_a_b=<n> b_c=0
[M15.E0.KV.RETURN] COMPLETE status=<status> manifest_sha256=<sha> ...
[M15.E0.KV.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
M15_E0_RETURN_INTAKE_PASS status=<status> ... claim=diagnostic-fingerprint-only numerical_repair_authorized=0
[M15.E0.RECOVERY] COMPLETE status=<status> runtime_source=12207e3281db13461350fe7ef68dbaadfe713a58 ...
[M15.E0.RECOVERY] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Return the unchanged four-file output, its manifest SHA256, sanitized markers,
and the recovery raw-log local path/SHA256. Do not return raw-log contents,
remote roots/URLs, tokens, capsules, replay ledger, observer NPZs, or archives.
On any failure, preserve output, scratch, and raw log; never retry into the
same path. Import any successful or failed compact return into a **new additive
evidence directory**. Never overwrite the rejected 971bb228 directory.

### Recovery decision table

| Result | Meaning and next action |
|---|---|
| original render or raw terminal receipt missing | `INCONCLUSIVE`; preserve, no target rerun |
| classifier identity/field/path/hash/manifest invariant fails | `OFFICIAL_RETURN_PROVENANCE_FAIL`; preserve, Phase E closed |
| control A-B red or any B-C red | hard stop; carrier/shared path invalid |
| treatment exact | `TARGET_NON_REPRODUCTION`; bug not fixed |
| admitted `LIVE_KV_FINGERPRINT_DIFFERS` | discuss one cache content/page-ownership probe; no repair yet |
| admitted `LIVE_KV_FINGERPRINT_EQUAL` | only diagnostic fingerprints match; discuss exact block-table/metadata/gather discriminator before internal RPA math |

### Current three-layer audit

- Implementation: reviewer now pins runtime source/classifier identity, full
  runtime-emitted fields, per-record provenance, basename-only paths, distinct
  arm/root digests, exact claim ceiling, and mandatory CLI raw log. It changes
  no A/B/C, APC read, RoPE/RPA/attention/KV arithmetic, production default,
  backward, or optimizer behavior.
- Validation: task discovery 187/187, intake/recovery 14/14, E0 admission 9/9,
  V1 CPU 91/91, P3 prefix-cache 31/31, P38 persistence, flags 398/398,
  Python/Bash syntax, and `git diff --check` PASS. Marker:
  `M15_E0R_PROVENANCE_HARDENING_HOST_PASS task_discovery=187
  return_intake=14 e0_admission=9 v1_cpu=91 p3_prefix_cache=31
  persistence=1 flags=398 syntax=1 diff_check=1 exact_image=0 gcs=0
  kubernetes=0 tpu=0`. Raw log:
  `/tmp/m15-e0r-provenance-hardening-971bb228-retry2-20260829.log`, SHA256
  `f11ab8b9bf137f7f7ca39a801fe06b6da6298b7b558fe817ea2f503f7f74a4e4`.
- Claim: HOST PASS only. Official pinned exact-image and real GCS recovery are
  NOT RUN; target verdict is NOT ADMITTED; numerical repair is unauthorized.

## SUPERSEDED — pre-Attempt-18 E0 launch instructions

The current numerical result is **not** “APC fixed.” Attempt 17 remains red:
A-B=207 bytes / 95 elements, B-C=0 over 119,150 action tokens. D3e has now
localized the canonical first action to Layer 0 `k_post_rope -> rpa_output`,
shape `[2048,1,15,8]`, source row 217 / completion position 0 / source
position 1225. The next useful experiment is one default-off live-KV
fingerprint discriminator inside that interval.

The E0 implementation was published at
`1c7391da5336033abd0727e610f7bad4c5c4e2be`. A later published fallback at
`12207e3281db13461350fe7ef68dbaadfe713a58` used a mutable image name and was
not launch-ready. The current additive follow-up replaces that fallback with a
strict already-local immutable-image route, aligns the run-label contract,
preserves failed scratch, and self-hashes the classifier-runtime receipt. The
user approved this additive follow-up for publication; its exact source identity
is the full fact-branch commit containing this section, returned by the delivery
operation rather than self-recorded here. A remote executor must receive that
**exact full published SHA**. It must not render or launch from either older
SHA, an abbreviated SHA, or a dirty tree.

### What E0 measures

At the 1226-token first-red prefix there are eight legitimate A requests with
the same prefix. The E0 observer captures all eight aliases at exactly 77
logical pages, reads only Layer 0 with a static 96-page bound, then uses later
replay-ledger token-history receipts to require one matching source request and
explicit conflicts for the other seven. B remains the independent
`reset_prefix_cache=True` rescore.

This closes the request-identity hole before choosing between:

- stored live KV already differs before RPA; or
- stored live KV fingerprints match and the red is in page selection/read/RPA
  execution context.

The result is an integer diagnostic fingerprint, not a collision-free dump of
all KV bytes. It does not authorize a numerical repair.

### Current three-layer status

- Implementation: default-absent Layer-0/target-prefix selector in append-only
  Patch 35; M15 `--observer kv` renderer; request-aware replay-ledger binding;
  prepare wrapper; compact read-only GCS return wrapper. Production profiles,
  APC defaults, A/B/C, RoPE, attention/RPA math, KV values, backward, loss, and
  optimizer are unchanged.
- Validation: task-local discovery 173/173, KV classifier 7/7, M15 carrier
  19/19, real resolved-env 11/11, E0 admission/runtime 9/9, V1 CPU 91/91,
  P3 12/12, P38 persistence, flags 398/398, overlay patch/manifest,
  and Python/Bash syntax PASS. The real host-Python route and mocked forced
  Docker route PASS; missing and wrong images fail before `docker run`. Real
  Docker has not run. The optional broad P33 host aggregate is
  INCONCLUSIVE because this host lacks `datasets` and `metrax`; the official
  pinned-image gate is still required. E0 pinned exact-image and DP8xTP8 target
  are NOT RUN.
- Claim: `FIRST_RED_LOCALIZED / E0_IMPLEMENTED_PUBLISHED /
  E0_LAUNCH_READINESS_FOLLOWUP_PUBLISHED / HOST_PASS / REAL_DOCKER_NOT_RUN /
  EXACT_IMAGE_NOT_RUN / TARGET_NOT_RUN / NUMERICAL_REPAIR_NOT_AUTHORIZED`.

### Cold-start instructions for the other agent

Create a new clean `local/*` worktree from the exact full published E0 SHA.
Do not use or modify
`/home/yuxuan/code_rl_repro/worktrees/p57_zero_noeval_0828`.

Before any action, read in order:

1. `/home/yuxuan/code_rl_repro/AGENTS.md`;
2. `canon-zero-tim/AGENTS.md`;
3. `canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/SKILL.md`;
4. `canon-zero-tim/.claude/skills/manage-canon-flags/SKILL.md`;
5. `/home/yuxuan/.codex/skills/run-phased-work/SKILL.md`;
6. APC/M15 entries in `THREADS.md`, `FLAGS.md`, and `EVIDENCE.md`;
7. this section, `state.md`, `plan.md`, and
   `phases/phase-e0-layer0-live-kv-discriminator.md`;
8. the current operation in `RUNBOOK.md` and latest E0 checkpoint in `log.md`.

Run canonical preflight and require `CANON_PREFLIGHT PASS`, a `local/*`
branch, exact HEAD equality, and a clean tree. Do not print remotes,
credentials, configured accounts/projects, or the registered evidence root.

### Step 1 — render only; no external or remote access

This command verifies the committed D3e manifest, rechecks focused host gates,
and writes two immutable JobSet YAMLs plus a self-hashed contract. It has no
TPU, Kubernetes, or GCS operation:

```bash
SOURCE_COMMIT=<full-published-E0-follow-up-SHA>
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/tmp/m15-e0-kv-${RUN_ID}
test ! -e "$OUT"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_attempt18_e0_kv_pair.sh \
  "$SOURCE_COMMIT" "$RUN_ID" "$OUT"
```

Required terminal tail:

```text
[M15.E0.KV] RENDER_PASS source=<full-sha> rounds=1 layer=0 aliases=8 pages=96 ...
[M15.E0.KV] TARGET_NOT_RUN pinned_exact_image=required launch_approval=required gcs=0 kubernetes=0 tpu=0
```

The output must include self-hashed `KV_CLASSIFIER_RUNTIME.json`. If host Python
lacks NumPy, the focused classifier may use only the registered pinned image
ID already present locally; it verifies exact identity, sets `--pull=never`
and `--network=none`, and records the route. It cannot pull an image or access
a registry and is not the official aggregate in Step 2. Any missing marker,
dirty/source failure, D3e manifest drift, runtime-receipt failure, extra arm,
non-APC pair difference, or environment-resolution failure is a hard stop.
The wrapper prints `scratch_preserved=<path>` on failure. Do not delete it or
edit the YAML.

### Step 2 — separate pinned exact-image approval

Rendering does not authorize this gate. First return the exact command, image
identity, marker, and raw-log path to the user and obtain explicit approval.
The registered immutable image identity remains:

```text
sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

After approval, run the official aggregate directly without a pipe:

```bash
RAW=/tmp/m15-e0-exact-image-<source8>-<fresh-label>.log
test ! -e "$RAW"
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a \
  >"$RAW" 2>&1
```

Require exit 0 and the complete terminal marker containing:

```text
V1_HP_EXACT_IMAGE_PASS
apc_m15_carrier=70
m15_d3e=1
m15_e0=16
m15_durability=1
m15_round_provenance=1
manifests=3
```

Missing markers, install/manifest drift, nonzero exit, or an unverified image
identity is FAIL/INCONCLUSIVE. Preserve the raw log. This local Docker/CPU gate
does not use TPU, Kubernetes, or GCS and still does not authorize launch.

### Step 3 — separate target-launch approval

Only after Step 2 is green, return both YAML paths and request a new explicit
launch approval. After approval, apply the two checked-in render outputs
directly, with no pipe, dry-run substitution, hand edit, or reused label:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off-kv.yaml"
kubectl apply -f "$OUT/jobset-v1-apc-m15-on-kv.yaml"
```

The pair is one frozen round, DP8xTP8, M15/main, production
`continue_decode=8`, zero backward, and zero optimizer commit. Control and
treatment differ only at APC. B full reset and zero cached-token evidence are
immutable. The JobSet persistence worker owns remote write/readback and
terminal completion; do not manually copy large archives.

### Step 4 — compact read-only return on the bucket-capable machine

After both JobSets have terminal evidence, obtain separate GCS-read approval.
Then run directly:

```bash
RETURN=/mnt/disks/tunix-data/m15-e0-kv-return-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt18_e0_kv_gcs_return.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data
```

Required terminal tail:

```text
M15_E0_KV_RETURN_PASS status=<status> control_a_b=<n> treatment_a_b=<n> b_c=0
[M15.E0.KV.RETURN] COMPLETE status=<status> manifest_sha256=<sha> ...
[M15.E0.KV.RETURN] READ_ONLY gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Return only those markers, `E0_KV_RETURN.json`, the two small classifier JSONs,
`SHA256SUMS`, and the SHA256 of `SHA256SUMS`. Do not return the serving archive,
replay ledger, raw token/capsule/page payload, remote root, URL, or credential.
On failure, return the sanitized error and printed `scratch_preserved` path;
do not delete it or retry into the same output directory.

### E0 decision table

| Returned status | Required interpretation |
|---|---|
| `CONTROL_RED_STOP` | carrier/observer invalid; stop before any mechanism claim |
| any B-C red or CANON_ALIGN failure | non-APC/shared red; preserve and stop |
| `TARGET_NON_REPRODUCTION` | fresh treatment exact; bug not reproduced, not fixed |
| `LIVE_KV_FINGERPRINT_DIFFERS` | uniquely bound live A Layer-0 KV fingerprint is already red before RPA; next discussion is cache production/storage/page ownership, still no numerical repair authorization |
| `LIVE_KV_FINGERPRINT_EQUAL` | observed stored KV fingerprint matches B; next discussion is page table/read/RPA execution context, not a stale-content claim |
| missing aliases, future binding, red join, manifest, COMPLETE, or terminal marker | `INCONCLUSIVE`; preserve and stop |

No E0 result alone opens numerical edits. Update the phase ledger and discuss a
single-variable next probe or repair with the user first.

## Historical — D3e host/exact-image gates and canonical-action reclassification

Do **not** rerun the old D3d command and do **not** launch a TPU pair. The
verified D3d return is already committed at
`b74c4ba38f293606000398c29818cea0c8ca5c8b`. It proves that source row 217 /
completion position 0 uniquely binds to one A request. The remaining
candidate-set verdict comes from classifier decision accounting across later
red actions, not from unresolved identity at the first action.

The current D3e implementation is analysis-only. It declares
completion-position-zero as the decision scope when the existing
`--require-first-action` contract is active, while preserving every later
joinable signature and all unobserved continue-decode red points in separate
fields. It changes no model, cache, attention, RoPE, KV, A/B/C, backward,
optimizer, or production-profile behavior.

### Current immutable facts

- Attempt-17 runtime source:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.
- D3d analysis source:
  `ec46033673442949ff956092b8f4ea3074285a13`.
- Verified D3d evidence commit / current published tip:
  `b74c4ba38f293606000398c29818cea0c8ca5c8b`.
- D3d manifest SHA256:
  `c3dd6ab4e8ee191e1012b011a6e8ff8d845e528aa85f59936c06315b10cbbb31`.
- APC-off rounds 0/1/2 are sealed exact.
- APC-on Round 0 is sealed with A-B=207 bytes / 95 elements, B-C=0, and
  119,150 action tokens. Round 1 failed Stage-10 assembly with exit 2; Round 2
  is absent. Root `COLLECTED`/`COMPLETE` is absent, so the pair remains
  analysis-grade partial evidence.
- D3d uniquely binds source row 217 / completion position 0 / source position
  1225 to A request `79-b8334848`. Selected proof prefix 1300 reaches beyond
  required horizon 1227 and explicitly eliminates seven alternatives.
- The unique first-action candidate is Layer 0
  `k_post_rope -> rpa_output`. Fingerprint geometry is
  `[2048,1,15,8]` for the layer record and `[2048,8]` for final norm.
- Across all seven joinable red points, signatures remain Layer-0
  `rpa_output` and `final_norm`; 88/95 red points are unobserved because they
  run under continue-decode. Those facts must remain visible and are not
  inherited by the first-action boundary.
- Production M15 APC remains off. Phase E remains closed. No numerical repair
  exists.

### Why D3e exists

The classifier already selected completion-position-zero anchors for its
public result, but computed `unique_signature` over every joinable red point.
That mixed the one uniquely bound first-action candidate with six later red
points. D3e makes the decision scope explicit and keeps global signatures as
diagnostics. It remains fail closed for a mixed/exact first-action candidate,
B numeric variants, same-request conflicts, B-C red, missing first-action
coverage, source drift, or manifest drift.

The exact phase contract is in
`phases/phase-d3e-canonical-first-action-scope.md`.

### Current gate and approval order

1. **Host CPU:** PASS. Task-local discovery, focused classifier/reviewer,
   durability/P38 persistence, flags, syntax, scope, secret, and diff gates
   are green.
2. **Pinned exact-image:** PASS on the official aggregate with exit 0 and the
   immutable image identity recorded below. The terminal includes
   `apc_m15_carrier=68 m15_d3e=1 m15_durability=1
   m15_round_provenance=1` and `manifests=3`. This used local Docker/CPU only;
   it did not use TPU, Kubernetes, or GCS.
3. **Commit/push:** the user explicitly approved publication after the host
   and pinned exact-image gates passed. The bucket-capable agent must receive
   the resulting full D3e commit SHA; it must not use the D3d base SHA.
4. **Read-only GCS D3e:** after publication, separate GCS-read approval. A
   bucket-capable agent uses the one command below. It performs no GCS write,
   Kubernetes query, or TPU launch.
5. **TPU:** not currently requested or admitted. Consider a fresh matched
   DP8xTP8 pair only if D3e still preserves the decision-scope candidate set or
   the returned shape/coordinate ledger fails review. Pinned-image green and
   GCS-read green do not authorize a launch.

The read-only local image inspection and completed aggregate resolved the exact identity
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
The separately approved aggregate was run directly with no pipe and its raw
log was preserved at the following path:

```bash
test ! -e /tmp/m15-d3e-exact-image-b74c4ba3-20260829.log
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a \
  >/tmp/m15-d3e-exact-image-b74c4ba3-20260829.log 2>&1
```

Observed result:

```text
exit=0
V1_HP_EXACT_IMAGE image_ref=sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a image_id=sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=68 m15_d3e=1 m15_durability=1 m15_round_provenance=1 ... manifests=3
raw_log=/tmp/m15-d3e-exact-image-b74c4ba3-20260829.log
raw_log_sha256=59efa6ddc6e0399050cbbbbc5b463fc6b94486d96834f1e8b50f4fd9d3b22d97
```

After publication, the next gate is a separate GCS-read approval. Exact-image
PASS is not target PASS and does not authorize TPU/Kubernetes or Phase E.

### Cold-start contract for the bucket-capable D3e executor

The user must provide the full published D3e analysis SHA. Create a fresh
clean `local/*` worktree from exactly that SHA; never use or modify
`/home/yuxuan/code_rl_repro/worktrees/p57_zero_noeval_0828`.

Read, in order:

1. `/home/yuxuan/code_rl_repro/AGENTS.md`;
2. `canon-zero-tim/AGENTS.md`;
3. `canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/SKILL.md`;
4. `/home/yuxuan/.codex/skills/run-phased-work/SKILL.md`;
5. `canon-zero-tim/THREADS.md`;
6. APC/M15 entries in `FLAGS.md` and `EVIDENCE.md`;
7. this section, then `state.md`, `plan.md`, and
   `phases/phase-d3e-canonical-first-action-scope.md`;
8. the current operation in `RUNBOOK.md` and the latest D3e checkpoint in
   `log.md`.

Run canonical preflight and require `CANON_PREFLIGHT PASS`. Confirm the branch
is `local/*`, HEAD equals the exact published D3e SHA, and the tree is clean.
Do not print a remote URL, configured account, project, credential, or bucket
root.

After explicit GCS-read approval, use a fresh local return directory and run
directly without a pipe:

```bash
RETURN=/mnt/disks/tunix-data/m15-d3e-canonical-action-return-<fresh-label>
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d3e_canonical_action.sh \
  "$RETURN" /mnt/disks/tunix-data
```

Expected terminal tail:

```text
M15_D3E_CANONICAL_ACTION_REVIEW_PASS status=<status> decision_scope=completion-position-zero ... numerical_repair_authorized=0
[M15.D3E.OFFLINE] COMPLETE status=<FIRST_RED_LOCALIZED|FIRST_RED_CANDIDATE_SET_PRESERVED> ...
[M15.D3E.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

On success, return only the terminal marker lines, the three JSON files,
`SHA256SUMS`, and the SHA256 of `SHA256SUMS`. Never return the downloaded tar,
token/capsule payload, replay ledger, physical page list, remote root, URL, or
credentials. On any failure, return the exit code, complete sanitized stderr,
and printed `scratch_preserved` path; do not delete it or retry into the same
output directory.

Interpretation:

| Result | Required action |
|---|---|
| `FIRST_RED_LOCALIZED` with unique binding, `k_post_rope -> rpa_output`, fingerprint geometry, cache-page receipt, request/call/token coordinate, and both source anchors | Return the package and stop. This opens user review of the localization ledger, not Phase E automatically. |
| `FIRST_RED_CANDIDATE_SET_PRESERVED` | Return the package and stop. Prepare direct producer/request provenance plus original checkpoint-shape metadata, host and exact-image gates, then request approval for a new matched target pair. |
| Any identity, manifest, B, geometry, source-anchor, or page-receipt failure | Preserve and stop. Do not launch TPU. |
| Auth/network/tool failure | `INCONCLUSIVE`; preserve and repair infrastructure only. |

The intermediate commit `653a10d5ce23c3c426dfd0f69c480610289fd6fa`
changes full-training carriers and the aggregate exact-image harness. It does
not change the immutable Attempt-17 interpretation, but its scope must be
included in exact-image review before any future render SHA is frozen.

## Superseded — pre-D3d executor instructions retained for provenance

### Cold-start contract for the bucket-capable executor

Your task is **not** to launch a new target pair. Your only current operation is
the Phase D3d read-only GCS + CPU reclassification described below.

Two commits have different roles and must not be confused:

- Attempt-17 runtime source:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`;
- Phase D3d analysis source: a new full published SHA supplied by the user that
  contains this HANDOFF, the future-prefix classifier change, and
  `run_m15_attempt17_d36_offline_binding.sh`.

`6e4e7f587941ee7e0c83753bc321a995912c8021` contains the Attempt-17 evidence
return but not the Phase D3d implementation. Do not use it as the analysis
source. If the user has not supplied the new full analysis SHA, stop and ask
for it.

First read `/home/yuxuan/code_rl_repro/AGENTS.md`. Then create the clean
worktree. Do not use or modify
`/home/yuxuan/code_rl_repro/worktrees/p57_zero_noeval_0828`.

Create the new worktree from the exact published analysis SHA:

```bash
ANALYSIS_SHA=REPLACE_WITH_FULL_USER_SUPPLIED_PHASE_D3D_SHA
UNIQUE_LABEL=replace-with-a-fresh-unique-label
PRIMARY=/home/yuxuan/code_rl_repro/sequence_packing/tunix
WT=/home/yuxuan/code_rl_repro/worktrees/m15_d36_offline_binding_${UNIQUE_LABEL}
test ! -e "$WT"
git -C "$PRIMARY" fetch --quiet origin yuxzhang/canon-zero-tim
test "$(git -C "$PRIMARY" rev-parse origin/yuxzhang/canon-zero-tim)" = "$ANALYSIS_SHA"
git -C "$PRIMARY" worktree add -b local/m15-d36-offline-${UNIQUE_LABEL} \
  "$WT" "$ANALYSIS_SHA"
cd "$WT"
test "$(git rev-parse HEAD)" = "$ANALYSIS_SHA"
```

After entering the new worktree, read in this order before preflight or any
remote access:

1. `canon-zero-tim/AGENTS.md`;
2. `canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/SKILL.md`;
3. `/home/yuxuan/.codex/skills/run-phased-work/SKILL.md`;
4. `canon-zero-tim/THREADS.md`;
5. the APC/M15 entries in `canon-zero-tim/FLAGS.md`;
6. the M15 APC entries in `canon-zero-tim/EVIDENCE.md`;
7. this HANDOFF section;
8. `state.md` and `plan.md`;
9. `phases/phase-d3d-attempt17-offline-request-binding.md`;
10. the current operation in `RUNBOOK.md`;
11. the latest Phase D3d checkpoint in `log.md`.

Then run canonical preflight:

```bash
python3 canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py \
  --repo . --require-clean
```

Required preflight terminal marker:

```text
CANON_PREFLIGHT PASS
```

The user must separately approve read access to the registered Attempt-17 GCS
evidence. Check only that `gcloud` or `gsutil` is available; do not print
credentials, configured accounts, project values, remote roots, or environment
secrets. No Kubernetes or TPU permission is required or implied.

After explicit GCS-read approval, use a fresh local output directory and run
the wrapper directly, without a pipe:

```bash
RETURN=/mnt/disks/tunix-data/m15-d36-offline-binding-return-${UNIQUE_LABEL}
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d36_offline_binding.sh \
  "$RETURN" /mnt/disks/tunix-data
```

The complete success marker sequence includes:

```text
CANON_PREFLIGHT PASS
M15_D36_RENDER_IDENTITY_PASS source=16c224aa rounds=3 observer=full seam_layer=0
[M15.MULTIROUND] COMPLETE status=<refreshed-remote-status> ...
M15_D36_BUNDLE_IDENTITY_PASS treatment_round=0 sealed=1
M15_D36_OFFLINE_REVIEW_COMPLETE status=<status> gate=<gate> ...
[M15.D36.OFFLINE] COMPLETE status=<FIRST_RED_LOCALIZED|FIRST_RED_CANDIDATE_SET_PRESERVED> ...
[M15.D36.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0
```

Missing markers or a nonzero exit are not a numerical verdict. Return the
exit code, complete stderr, and the printed `scratch_preserved` path. Do not
delete the preserved failure directory, retry into the same output directory,
or bypass an identity/manifest failure. Authentication, permission, or network
failures are `INCONCLUSIVE`.

On success, verify the return and send back only:

```bash
(cd "$RETURN" && sha256sum -c SHA256SUMS)
sha256sum "$RETURN/SHA256SUMS"
```

- the complete terminal marker lines;
- `D36_OFFLINE_REVIEW.json`;
- `D36_RECLASSIFICATION.json`;
- `REMOTE_MULTIROUND_SUMMARY.json`;
- `SHA256SUMS` and its SHA256.

Do not return the downloaded tar, capsule, replay ledger, token hashes or token
payloads, bucket root, remote URL, credentials, or secret-bearing environment
values. Do not commit or push the returned package unless the user separately
approves that exact publication.

Interpretation is fail closed:

| Returned status | Required decision |
|---|---|
| `FIRST_RED_LOCALIZED` | Confirm one `UNIQUE_FUTURE_PREFIX_BINDING`, selected proof horizon at least the required elimination horizon, last exact, first red, shape ledger, request/call/token/cache/page coordinates, and both source anchors. Then report the result and stop. Pinned exact-image and Phase E are separate approvals. |
| `FIRST_RED_CANDIDATE_SET_PRESERVED` | Report that existing d36 evidence cannot uniquely bind the request. Do not rerun TPU immediately. The next code phase is one observational producer-row/request provenance field, followed by host and exact-image gates and a separately approved matched DP8xTP8 pair. |
| Any identity, manifest, B invariant, classifier, or return verification failure | Preserve everything and stop. Do not downgrade or relaunch. |

In every case: production M15 APC remains off; B remains an independent
full-reset judge; Phase E remains closed until the user reviews a complete
`FIRST_RED_LOCALIZED` return.

Attempt 17 (`d36`) used runtime source
`16c224aa80eb6b3a544be19f693c0542ab4b0dcb`. The committed operator return
proves three sealed control rounds and one sealed treatment round. It does not
contain root completion markers, terminal JobSet conditions, raw logs, or the
original render contract, so the pair remains analysis-grade partial evidence.

The self-hashed operator return is sealed in:
`evidence/v1_apc_m15_attempt17_d36_operator_return_20260829/` (all 84 manifest members verify via `SHA256SUMS`; the manifest excludes itself).

### What Attempt 17 (d36) proves

- **Phase D3c Request-Aware Candidate Disambiguation Verified on Cluster**:
  - The classifier identity bug from Attempt 16 (where distinct concurrent requests sharing a prefix collided on `request_id`) is completely resolved.
  - APC-on Round 0 assembled 2,218 record pairs / 658,468 unique keys and executed the official classifier without crashing, successfully emitting `FIRST_RED_CANDIDATE_SET` across 2 candidate anchors with full replay-ledger receipts.
- **Stage 15 Preclassify Input Checkpoint Durability Verified**:
  - Both arms uploaded and verified `STAGE_15_checkpoint-input_PASS.json` and `CLASSIFIER_INPUT_RECEIPT.json` to GCS before entering classification.
- **APC-On Target Red Fork Confirmed**:
  - Re-confirmed exact serving-only APC cache fork:
    - $A-B = 207$ differing bytes / 95 elements
    - $B-C = 0$ differing bytes (policy forward pass exact)
    - $N_{\text{action}} = 119,150$ actions
- **APC-Off 3/3 Complete Durable Rounds**:
  - Control arm completed all 3 diagnostic rounds (`PIPELINE_COMPLETE` on rounds 0, 1, 2).
  - Round 0: 2,633 record pairs / 1.76 GB, $A-B=0$, $B-C=0$, $N_{\text{action}}=117,236$, `M15_OBSERVER_CONTROL_EXACT`.
  - Round 1: 2,223 record pairs, $A-B=0$, $B-C=0$, `M15_OBSERVER_CONTROL_EXACT`.
  - Round 2: 2,599 record pairs, $A-B=0$, $B-C=0$, `M15_OBSERVER_CONTROL_EXACT`.
- **Fail-Fast Treatment Lifecycle**:
  - APC-on completed, sealed, and ACKed Round 0. Round 1 then failed at
    Stage 10 assembly with exit code 2. The return lacks raw stderr, so its
    cause is unknown; Round 2 is absent.

### Round Inventory Table

| Arm | Round | Record Pairs | Differing Bytes ($A-B$) | Differing Bytes ($B-C$) | Classification Verdict | Stage Pipeline Status |
|---|---:|---:|---:|---:|---|---|
| APC off | 0 | 2,633 | 0 | 0 | `M15_OBSERVER_CONTROL_EXACT` | `PIPELINE_COMPLETE` (Stages 10..70 PASS) |
| APC off | 1 | 2,223 | 0 | 0 | `M15_OBSERVER_CONTROL_EXACT` | `PIPELINE_COMPLETE` (Stages 10..70 PASS) |
| APC off | 2 | 2,599 | 0 | 0 | `M15_OBSERVER_CONTROL_EXACT` | `PIPELINE_COMPLETE` (Stages 10..70 PASS) |
| APC on | 0 | 2,218 | 207 (95 el) | 0 | `M15_INTERNAL_FIRST_RED_CANDIDATE_SET` | `PIPELINE_COMPLETE` (Stages 10..70 PASS) |

Current claim ceiling:

```text
REQUEST_AWARE_CLASSIFIER_CLUSTER_PASS /
PRECLASSIFY_INPUT_DURABILITY_CLUSTER_PASS /
CONTROL_3_ROUNDS_EXACT_PASS /
ATTEMPT17_TARGET_RED_PRESERVED /
FIRST_RED_CANDIDATE_SET_CAPTURED /
PARTIAL_ROUNDS_RECOVERED_OPERATOR_RECEIPTS_INCOMPLETE /
FIRST_RED_NOT_YET_LOCALIZED /
APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
PHASE_E_CLOSED
```

### Phase D3d next action — CPU/GCS read only, no TPU

The durable treatment bundle contains the selected seam candidates, mismatch
capsule, and replay ledger. Phase D3d tests whether later token-prefix receipts
bind source row 217 to one request. The implementation and host tests are in
`phases/phase-d3d-attempt17-offline-request-binding.md`.

After the Phase D3d analysis change is published with explicit user approval,
a bucket-capable agent uses a clean `local/*` worktree and one command:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt17_d36_offline_binding.sh \
  /mnt/disks/tunix-data/m15-d36-offline-binding-return
```

This command reconstructs and verifies the d36 render identity, reads the
sealed treatment Round-0 compact bundle, verifies both manifests, and runs the
official classifier. It does not query Kubernetes, launch TPU work, or write
GCS. GCS read access still requires a separate user approval.

Accepted terminal outcomes are:

```text
[M15.D36.OFFLINE] COMPLETE status=FIRST_RED_LOCALIZED ...
```

or:

```text
[M15.D36.OFFLINE] COMPLETE status=FIRST_RED_CANDIDATE_SET_PRESERVED ...
```

The first outcome must still contain and survive review of last exact, first
red, shape, request/call/token/cache/page coordinates, and source anchors. The
second means one observational source-row/request provenance field is needed
before a separately approved target rerun. Neither outcome by itself claims a
numerical fix.

## Historical — Attempt 16 exposed a classifier identity bug after capturing a real APC red

Phase D3c was developed from
`fbc4fa03cdb35ac519d183b03ecd25ede485a5e3`. Delivery may rebase it onto a
later non-overlapping operator tip; use `git rev-parse HEAD` as the published
source identity. A pinned image or TPU/Kubernetes launch still requires a
separate user approval.

### What Attempt 16 actually proves

- The Attempt-16 incident manifest verifies all eight members it lists.
- Patch 33 worked: treatment Round 0 assembled 70 verified shards / 2,187
  record pairs, then entered the official classifier.
- APC-on reached 92.5% cache hits and reproduced the real serving-only red:
  A-B=1,711 bytes / 786 elements while B-C=0.
- The classifier failed because its key was
  `(diagnostic_round, token_prefix_sha256, arm)`. Distinct concurrent requests
  may have the same token prefix, but the old alias resolver also compared
  `request_id`, so valid requests were misreported as conflicting aliases.
- In the exception `(0, b'fde77c...')`, `0` is diagnostic round 0, not token
  position 0. The checked-in incident prose saying position 0 is corrected
  here without rewriting immutable evidence.
- APC-off had three numerically exact rounds. The returned logs prove complete
  seal/upload/ACK for rounds 0 and 1; round 2 reached `ROUND_SEAL_REQUESTED`
  only. Do not claim 3/3 terminal durability from this return.
- The Git package is an incident subset, not a complete treatment round. Since
  Attempt 16 predates the input checkpoint below, its assembled classifier
  inputs are not reconstructible from the checked-in subset alone unless the
  original pod-local round directory still exists.

### Phase D3c implementation

1. Resolve duplicate records only inside one exact serving observation
   `(request_id, call_index, position, token/target identity)`.
2. Keep distinct same-prefix requests as candidates; group them only when the
   measured tensor payload is bitwise identical.
3. Require full-reset B to have one numerical variant. Multiple B variants and
   conflicting duplicates inside one request remain hard failures.
4. Evaluate every A numerical variant, including exact-through-observer
   candidates. One shared first-red signature with no exact candidate may be
   localized; mixed/exact candidates emit `FIRST_RED_CANDIDATE_SET`, with no
   fake selected layer or source interval.
5. Count unique red coordinates separately from candidate anchors and package
   every selected candidate plus its replay-ledger receipt.
6. After assemble and before classify, upload a self-hashed
   `classifier-input` checkpoint containing the round receipt, pre-alignment,
   replay envelope, and red capsule when present. Observer tensors remain in
   the verified shards. A future classifier failure can therefore be analyzed
   without repeating rollout.

No APC, RoPE, attention/RPA, KV, LM-head, loss, backward, optimizer, A/B/C, or
production APC-default logic changed.

### Local gates

- task-local discovery: PASS;
- request-aware classifier/packager: 18/18 PASS;
- durability/input checkpoint: 11/11 PASS;
- P38 fake-GCS persistence integration: `PERSISTENCE_TEST_PASS`;
- Python compilation, Bash syntax, and `git diff --check`: PASS.

### Next approval gates

1. Phase E remains closed because the numerical root is not yet localized or
   repaired.
2. Pinned exact-image is a separate approval and must exercise the new
   classifier-input checkpoint plus existing aggregate gate.
3. If Attempt-16 pod-local state is gone, a fresh matched DP8xTP8 pair is still
   required after publication. Both arms may run concurrently. A candidate-set
   result is useful evidence but does not authorize a numerical repair until a
   stable request/source-row join yields `FIRST_RED_LOCALIZED`.

Current claim ceiling:

```text
REQUEST_AWARE_CLASSIFIER_LOCAL_PASS /
PRECLASSIFY_INPUT_DURABILITY_LOCAL_PASS / NUMERICAL_PATH_UNCHANGED /
ATTEMPT16_TARGET_RED_PRESERVED / FIRST_RED_NOT_YET_LOCALIZED /
APC_NUMERICAL_FIX_NOT_IMPLEMENTED / EXACT_IMAGE_NOT_RUN /
TARGET_NOT_RERUN / PHASE_E_CLOSED
```

## d33 flat-shard content audit verified Round 0 only; fix first seal/ACK before rerun

Attempt 14 (`d33`) now has three immutable small returns:

- `evidence/v1_apc_m15_attempt14_d33_operator_return_20260828/` records the
  original `NO_DURABLE_ROUND_OPERATOR_RECEIPTS_INCOMPLETE` result;
- `evidence/v1_apc_m15_attempt14_d33_inventory_return_20260828/` resolves the
  query ambiguity and proves both recursive listings succeeded (265 off / 223 on objects);
- `evidence/v1_apc_m15_attempt14_d33_flat_shard_audit_20260828/` completes the
  receipt-bound flat-shard content audit for all 162 shards.

The flat-shard content audit verified:

| Arm | Shards | Record pairs | Payload bytes | Diagnostic rounds | Receipt/manifest audit |
|---|---:|---:|---:|---|---|
| APC off | 88 (`000000..000087`) | 2,780 | 1,792,189,157 | Round 0: 88 (100%) | 88/88 completion + manifest OK |
| APC on | 74 (`000000..000073`) | 2,302 | 472,614,342 | Round 0: 74 (100%) | 74/74 completion + manifest OK |

Every listed shard directory contains `SHARD_COMPLETE.json`, `SHA256SUMS`, and
`SHARD_ARCHIVE.tar`.  The small audit independently binds each completion
receipt to its manifest and confirms that the producer receipt carries a
well-formed archive digest.  It does **not** download or independently re-hash
the archive payload; the old wording that archive contents/digests were
independently verified is withdrawn.  The round metadata itself is decisive:
**100% of receipts belong to diagnostic round 0**. Neither arm crossed the
first round 0 seal to emit round 1 or round 2.

The machine decision is:

```text
AUDIT_M15_ATTEMPT14_D33_FLAT_SHARDS decision=D33_FLAT_SHARDS_ROUND0_ONLY rounds=[0] off_shards=88 on_shards=74
```

The strict status is:

```text
FLAT_SHARD_AUDIT_PASS / D33_FLAT_SHARDS_ROUND0_ONLY /
ROUND0_RECEIPTS_AND_MANIFESTS_VERIFIED /
ARCHIVE_PAYLOAD_NOT_INDEPENDENTLY_REHASHED / ROUND1_2_NOT_REACHED /
OFFICIAL_CLASSIFIER_MISSING / FIRST_RED_NOT_LOCALIZED /
PHASE_E_CLOSED / NUMERICAL_REPAIR_NOT_AUTHORIZED
```

### Current work order — review locally green seal/ACK hardening before publication

Per the decision table:
- `D33_FLAT_SHARDS_ROUND0_ONLY`: "valid content exists only for round 0 -> inspect/fix the first seal/ACK path before any rerun".
- `phases/phase-d3-seal-ack-hardening.md` owns the additive repair: stage
  receipts, an atomic `round-N.failure.json`, learner fail-fast handling, a
  three-round positive control, a forced-persistence negative control, and a
  stage-aware small-return audit.
- Host gates are green: 137/137 M15 tests, the P38 persistence suite, 394/394
  flag audit, syntax/compile checks, and `git diff --check` all pass. The fake
  GCS end-to-end return distinguishes explicit failure from interrupted
  progress without accepting either as numerical evidence.
- This is not target admission. Exact-image, commit/push, and a fresh matched
  pair each require separate user approval.
- Phase E remains closed; production APC stays off; B remains an independent full-reset computation.

## Historical — offline-review d32, then render d33; do not launch implicitly

The seven-file Attempt-13 (`d32`) inventory is transport-complete and proves
that both recursive listings succeeded and both registered roots contain no
`live/` or `wide/rounds/` objects. It also exposes an unresolved count drift:

| Arm | Physical shard completion `record_pairs` | Immutable receipt/classifier `seam_records` | Delta |
|---|---:|---:|---:|
| APC off | 2,445 | 2,474 | -29 |
| APC on | 2,188 | 2,087 | +101 |

These field names are not assumed to be the same metric. The checked-in
inventory was produced by changing the expected values after observing GCS, so
its old generic `PASS` is accepted only as an object-transport fact. It is not
an official classifier replay and does not authorize an RPA repair.

`prepare_m15_multiround_pair.sh` now reruns a checked-in offline validator
before rendering. The validator verifies every member of the immutable
seven-file return, re-derives both object geometries and count deltas, and emits
`D32_LIVE_ABSENT_WITH_COUNT_DRIFT`. The renderer embeds that review in its
self-hashed run contract. A reviewer may prepare d33; only the user may approve
the two 64-TPU launches.

### Remote executor command for d33 review and preparation

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
git pull --ff-only origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
test ! -e "$OUT"

bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  "$SOURCE_SHA" "$RUN_ID" "$OUT" full 0
(cd "$OUT" && sha256sum -c SHA256SUMS)
```

The renderer must produce exactly:
- `jobset-v1-apc-m15-off-full.yaml`
- `jobset-v1-apc-m15-on-full.yaml`
- `D32_REVIEW.json` with `live_absence_status=CONFIRMED`,
  `count_contract_status=DRIFT`, `d33_preparation_eligible=true`, and both
  authorization booleans false;
- `RUN_CONTRACT.json` with `diagnostic_rounds=3`, `observer=full`,
  `seam_layer=0`, zero backward/commit, and the exact D32 review SHA;
- `SHA256SUMS` covering the two YAMLs, review, and run contract.

After separate launch approval, both standalone `kubectl apply` commands may
be issued concurrently. Preparation is not launch authority.

---

## Historical — Attempt 13 (`d32`) object inventory (transport-complete)

Attempt 13 (`d32`) was a **single diagnostic round** produced by the older
flat-shard runtime. The registered roots contain 77 contiguous control shards
whose completion receipts sum to 2,445 record pairs and 70 contiguous treatment
shards whose receipts sum to 2,188. The historical classifier receipts instead
report 2,474 and 2,087 seam records; that difference remains explicit.

The self-hashed read-only inventory completed with:
```text
M15_ATTEMPT13_REVIEW_PASS decision=D32_LIVE_ABSENT_WITH_COUNT_DRIFT count_contract_status=DRIFT d33_preparation_eligible=1 d33_launch_authorized=0 numerical_repair_authorized=0
```
Evidence is sealed in `evidence/v1_apc_m15_attempt13_d32_inventory_20260828/`.
Because no `live/` directory exists, historical flat replay is unviable. d33 is
the next evidence-producing experiment after separate review and launch
approval; it is not a numerical repair.

### Mechanical interpretation reference

| Decision | Meaning | Next action |
|---|---|---|
| `D32_LIVE_ABSENT_WITH_COUNT_DRIFT` | both recursive queries succeeded and neither root listed a `live/` object, while shard and classifier counts disagree | d33 preparation is eligible; preserve the drift and require separate launch approval |
| `D32_LIVE_PRESENT_REPLAY_SHOULD_CONTINUE` | the registered roots contain at least one `live/` object | run the existing flat replay only after inspecting this return; do not launch d33 first |
| `D32_INVENTORY_AUDIT_RED` or non-zero exit | a query, identity, shard geometry, completion receipt, or count failed | fix only the read-only inventory path; absence is unproven and d33 remains blocked |

d33 is one matched APC-off/APC-on pair, each containing three evaluation-only
rounds with frozen weights, zero backward, and zero optimizer commits.  The
full observer is pinned to Layer 0 because Attempt 12 placed the analysis-grade
coarse interval between Layer-0 input and output.  A previously rendered
`/tmp` directory is not a durable source artifact and is not launch authority;
render again from the reviewed, published full SHA when that experiment is
approved.

This is not “run longer and hope the final upload works.”  At the end of each
round the learner blocks until the live worker has:

```text
sealed bounded shards -> uploaded them -> downloaded and verified them
-> classified the sealed union -> written WIDE_ROUND_COMPLETE -> ACKed the learner
```

Only then may the next evaluation begin.  Therefore a death after round 1
cannot erase round 0.  `COLLECTED.json` and `COMPLETE.json` are still required
for a full signed run, but the new small-return script can recover every sealed
round even when that final root close is missing.

After the reviewed source is separately committed and pushed, the remote
executor must fetch it and use exactly:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
git pull --ff-only origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
RUN_ID=<fresh-1-to-16-char-lowercase-dns-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
test ! -e "$OUT"

bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_multiround_pair.sh \
  "$SOURCE_SHA" "$RUN_ID" "$OUT" full 0
(cd "$OUT" && sha256sum -c SHA256SUMS)
```

The renderer must produce exactly:

- `jobset-v1-apc-m15-off-full.yaml`;
- `jobset-v1-apc-m15-on-full.yaml`;
- `D32_REVIEW.json` preserving the count drift and both false authorization
  fields;
- `RUN_CONTRACT.json` with `diagnostic_rounds=3`, `observer=full`,
  `seam_layer=0`, zero backward/commit, and the D32 review SHA.

After separate launch approval, both standalone `kubectl apply` commands may
be issued concurrently.  Do not pipeline either command and do not reuse a run
label.  When both JobSets terminate, the same executor must have read-only
Kubernetes access plus bucket access and run exactly one return wrapper:

```bash
RETURN=/tmp/v1-apc-m15-${RUN_ID}-small-return
test ! -e "$RETURN"
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_multiround_operator_return.sh \
  "$OUT" "$RETURN" /mnt/disks/tunix-data default
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

Do not separately run `run_m15_multiround_gcs_return.sh`; the operator wrapper
calls it internally.  It performs only `kubectl get` and GCS reads.  It does not
delete a JobSet, mutate GCS, download `run.log`, or return a token-bearing tar.
It reads the root manifest and object metadata to bind each remote `run.log` by
sanitized identity, SHA-256, and byte size.

Return the complete small `$RETURN` directory unchanged, the wrapper's final
`[M15.OPERATOR.RETURN] COMPLETE ...` line, and independent
`sha256sum -c SHA256SUMS` output.  Do not manually transcribe statuses or JSON.
The directory itself must contain:

- `MULTIROUND_SUMMARY.json`;
- `off.round-000000..000002.classification.json` for sealed off rounds;
- `on.round-000000..000002.classification.json` for sealed on rounds;
- `off/on.round-XXXXXX.stage-<ordinal>-<stage>-<status>.json` for every small
  stage receipt found remotely; these contain no token payload;
- `JOBSET_STATUS.json`, containing the sanitized terminal condition for both
  exact JobSet names;
- `RAW_LOG_RECEIPTS.json`, containing each immutable `run.log` identity,
  SHA-256, and byte size without the log payload or GCS root;
- `OPERATOR_RETURN_SUMMARY.json` and `OPERATOR_PACKAGING.txt`;
- `PACKAGING.txt` and one final `SHA256SUMS` covering every returned file.

Interpretation is mechanical:

- `COMPLETE`: all six rounds sealed and both roots terminal;
- `ROUNDS_RECOVERED_ROOT_INCOMPLETE`: all six classifiers survived, but the
  overall run is analysis-grade because root finalization died;
- `PARTIAL_ROUNDS_RECOVERED`: at least one round survived; use it, but do not
  call the paired target run complete;
- `ROUND_STAGE_FAILURE_IDENTIFIED`: no round sealed, but a remote FAIL receipt
  names the exact publisher stage and exit code; repair that stage first;
- `ROUND_STAGE_PROGRESS_ONLY`: no round sealed, but ordered stage receipts show
  the last completed or active stage; inspect the terminal worker log before
  relaunch;
- `NO_DURABLE_ROUND`: neither a sealed round nor any stage receipt exists;
- any off-arm red, B-C red, source/round/hash mismatch: hard stop.

`OPERATOR_RETURN_SUMMARY.json.status` equals the numerical core status only
when both JobSets are terminal (`Completed` or controlled-exit `Failed`) and
both raw-log receipts are present.  Otherwise it appends
`_OPERATOR_RECEIPTS_INCOMPLETE` while preserving any sealed numerical rounds.
The operator status never upgrades the numerical status in
`MULTIROUND_SUMMARY.json`.

The script deliberately queries `wide/rounds/000000..000002`; it does not rely
on the root aliases that an early exit may omit.  This is the required answer
to the previous “run finished but wanted data did not return” failure mode.

## Historical — Attempt 12 audit before the Layer-0 full observer

Attempt 12 (`d20-395c0e0d`) is currently **analysis-grade**, not signed target
evidence.  The checked-in five-file return is internally intact (`4/4` entries
listed by its `SHA256SUMS` verify), and its summaries report:

```text
off: A-B=0 bytes, B-C=0 bytes
on:  A-B=477 bytes / 227 elements, B-C=0 bytes
coarse interval: Layer 0 layer_input fingerprint exact -> layer_output fingerprint red
```

That package does **not** bind those summaries to the remote bounded shards.
It omits the remote `PREFLIGHT.json`, `COLLECTED.json`, `COMPLETE.json`, root
`SHA256SUMS`, compact-bundle verification, raw-log identity, and Kubernetes
terminal receipt.  The returned on-arm classification is also a minimized
copy: it omits the canonical classifier's `anchors`,
`first_difference_signatures`, `mixed_first_difference_signatures`,
`replay_ledger_receipts`, and `expected_layer` fields.  Therefore its current
gate is only `COARSE_FIRST_RED_INTERVAL`; it is not the final
`FIRST_RED_LOCALIZED` gate and it does not authorize a numerical repair.

### Why the previous return did not contain the complete evidence chain

Do not describe this as proof that the runtime failed to upload its data.  The
large seam JSON/NPZ payloads were intentionally left in GCS because the two
arms report approximately 6.6 GiB and 5.5 GiB of observer data.  The missing
step was the **post-run return audit**: the executor committed a manually
minimized receipt plus two classifier summaries and hashed only those four
small files.  It did not run the checked-in wide-seam GCS audit and return the
audit package that proves the classifiers are bound to the remote root or the
compact bundle.

The old Handoff made this mistake easier: its top `START HERE` section still
described Phase D2 publication, a later generic section referred to the older
replay audit, and the Attempt-12 entry jumped directly to the next launch.
There was no fail-closed Attempt-12 return checklist adjacent to that entry.
This section supersedes those stale operational instructions.  It does not
assert that the remote terminal objects are present or absent; the audit below
answers that question mechanically.

### Bucket-capable executor: perform this read-only audit now

Do not launch TPU/Kubernetes and do not retain, return, or commit the
token-bearing bundle.  The checked-in audit may fetch that compact tar into a
temporary directory solely to verify its internal manifest, then deletes the
scratch directory.  Use a clean checkout containing the published Attempt-12
receipt and run:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
test -z "$(git status --porcelain)"
git fetch origin yuxzhang/canon-zero-tim
git pull --ff-only origin yuxzhang/canon-zero-tim

RECEIPT=canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt12_paired_d20_20260827/receipt.json
RETURN=/tmp/v1-apc-m15-attempt12-d20-gcs-audit
test -f "$RECEIPT"
test ! -e "$RETURN"

bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_wide_seam_gcs_salvage.sh \
  "$RECEIPT" "$RETURN" /mnt/disks/tunix-data
(cd "$RETURN" && sha256sum -c SHA256SUMS)
```

Despite the historical `salvage` name, this wrapper accepts the two roots from
the supplied receipt and is the correct read-only verifier for Attempt 12.  It
checks both classifier aliases, all three terminal markers, the root manifest,
the compact bundle's internal manifest, and source identity.  The
token-bearing tar is verified in scratch space and excluded from the return.

### Exact return contract

Return the complete small `$RETURN` directory without editing or reformatting
any JSON.  It must contain:

1. `SALVAGE_SUMMARY.json`;
2. `PACKAGING.txt`;
3. `SHA256SUMS`;
4. `off.classification.json` and `on.classification.json` when present.

Also return, without copying token arrays:

5. the terminal line printed by the wrapper, including `status`, summary SHA,
   manifest SHA, and return path;
6. independent `sha256sum -c SHA256SUMS` output;
7. the exact full source SHA, both JobSet names, Attempt number, and the
   Kubernetes terminal status for each arm;
8. for each arm, either the immutable raw-log object identity plus SHA/size or
   a self-hashed text excerpt containing every line with `CANON_ALIGN_PRE`,
   `P3_APC_CONFIG`, `Prefix cache hit rate`, `CONTROLLED_EXIT`, `FATAL`, or
   `Traceback`.

The audit summary must mechanically report the presence/hash/source fields for
`PREFLIGHT.json`, `COLLECTED.json`, and `COMPLETE.json`, whether a root
manifest exists, whether the classifier is manifest-bound, and whether the
compact bundle's internal manifest passes.  A prose statement or a new
four-file summary is not an acceptable substitute.

Acceptance is exactly:

```text
status=LAYER_SELECTED
next_action=render full observer only at layer 0
off classifier=M15_OBSERVER_CONTROL_EXACT
on classifier=M15_LAYER_FIRST_RED_LOCALIZED
source conflicts=[]
both arms evidence_bound=true
```

`INCOMPLETE`, `SOURCE_MISMATCH`, missing terminal markers, missing manifest
binding, a failed hash, off-arm red, or B-C red is a hard stop.  Preserve the
return and repair/recover only the missing evidence; do not launch `d21` and do
not change model numerics.

Only after this audit passes may a separately approved paired Layer-0 full
observer run be rendered from the exact Attempt-12 source.  That run must use
all 15 checkpoints listed in the Phase-D document, rerun both APC-off and
APC-on arms, and reach `M15_INTERNAL_FIRST_RED_LOCALIZED` /
`FIRST_RED_LOCALIZED` before Phase E may propose a repair.  More diagnostic
rounds are not a substitute for the full Layer-0 checkpoint walk.

## Background — Phase D2 durability contract

The published source implements an evidence-transport repair, not an APC
numerical fix. The intended runtime contract is:

```text
observer JSON+NPZ complete
  -> bounded shard (<=32 pairs, <=256 MiB)
  -> upload archive+SHA
  -> remote read-back verify
  -> SHARD_COMPLETE
  -> classifier reads sealed shard union only
  -> WIDE_ROUND_COMPLETE
  -> COLLECTED
  -> postflight COMPLETE
```

`m15-wide-v1` also bypasses the redundant legacy incident ledger. The replay
envelope, request journal, seam/tail pairs, pre-alignment record and capsule
remain authoritative. No RoPE, attention, KV, LM-head, loss, backward,
optimizer, B-arm reset, or production APC behavior changes.

The Phase D2 source has passed host and pinned exact-image gates. The following
host gates must be rerun after any further edit:

```bash
cd /mnt/disks/tunix-data/worktrees/m15_wide_observer_0826

python3 -m unittest discover \
  -s canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts \
  -p 'test_*.py'
bash canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
python3 canon-zero-tim/tests/manage_canon_flags/test_audit_flag_registry.py
bash -n canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh
git diff --check
```

The fake-GCS persistence test is the required forced-death gate. It must report
`m15_shards=bounded-survive-abrupt-exit`, and both `COLLECTED.json` and
`COMPLETE.json` must be absent in that simulated interrupted run. The source
mismatch negative must report `source_mismatch=rejected`.

Pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
terminated with `V1_HP_EXACT_IMAGE_PASS ... apc_m15_carrier=66
m15_durability=1 ...`. The image gate uses a read-only Git common-directory
mount so the existing live `git rev-parse HEAD` source check runs inside the
container; no receipt-only provenance shortcut was admitted.

Claim ceiling now:

```text
ANALYSIS_GRADE_COARSE_LAYER_0 / REMOTE_EVIDENCE_BINDING_PENDING /
NUMERICAL_FIX_NOT_AUTHORIZED
```

See [Phase D2](phases/phase-d2-durable-wide-shards.md).

## Historical — Attempt-9 read-only GCS inventory (complete)

The first salvage pass is complete and self-verifies 2/2 files under
`evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/`. It established two
separate defects:

1. both registered Attempt-0 roots contain `PREFLIGHT.json`, but the six
   expected post-preflight objects (`COLLECTED.json`, `COMPLETE.json`, root
   `SHA256SUMS`, both classifier aliases, and the compact bundle) are absent;
2. both runtime preflight markers name source
   `3f159250c4781b3faafde238f768457a0478446b`, while the later prose receipt
   names the nonexistent full SHA
   `3f159250917fa9ee6062fbe7554f67644fcffec9`.

Therefore the receipt's claimed `0/1329` byte verdict and 2,313 tensor records
are not signed or reproducible evidence. Do not infer a layer from them. The
salvage wrapper checked seven exact object names; it did **not** enumerate
other objects that might survive under those roots. The later full inventory
did enumerate them and found only `PREFLIGHT.json` in each arm, so Attempt 9
is irrecoverable. The command below is retained only as historical provenance.

From a bucket-capable checkout of the latest published operator branch, run
exactly. This command downloads no object payload and mutates no GCS state:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
git fetch origin yuxzhang/canon-zero-tim
BASE_SHA="$(git rev-parse origin/yuxzhang/canon-zero-tim)"
WORKTREE=/mnt/disks/tunix-data/worktrees/m15_attempt9_full_inventory_20260827
test ! -e "$WORKTREE"
git worktree add --detach "$WORKTREE" "$BASE_SHA"
cd "$WORKTREE"

RECEIPT=canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt9_paired_d15_20260826/receipt.json
RETURN=canon-zero-tim/tasks/v1-apc-m15-target-debug/evidence/v1_apc_m15_attempt9_gcs_full_inventory_20260827
test -f "$RECEIPT"
test ! -e "$RETURN"
command -v gcloud >/dev/null

python3 - "$RECEIPT" "$RETURN" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

receipt_path = Path(sys.argv[1])
output = Path(sys.argv[2])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
output.mkdir(parents=True)
records = {}
for arm, key in (("off", "control_arm_off"), ("on", "treatment_arm_on")):
  root = receipt[key]["gcs_source_uri"].rstrip("/")
  completed = subprocess.run(
      ["gcloud", "storage", "ls", "--recursive", root + "/**"],
      check=False,
      capture_output=True,
      text=True,
  )
  if completed.returncode:
    raise SystemExit(
        f"GCS inventory failed for {arm}: rc={completed.returncode} "
        f"stderr={completed.stderr[-500:]}"
    )
  prefix = root + "/"
  relative = []
  for raw in completed.stdout.splitlines():
    value = raw.strip()
    if not value or value.endswith(":"):
      continue
    if not value.startswith(prefix):
      raise SystemExit(f"unexpected inventory entry outside {arm} root")
    relative.append(value[len(prefix):])
  records[arm] = sorted(set(relative))

summary = {
    "schema": "m15-apc-attempt9-full-object-inventory-v1",
    "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "receipt_source_commit": receipt["source_commit"],
    "runtime_source_commit": "3f159250c4781b3faafde238f768457a0478446b",
    "source_identity_matches": False,
    "payloads_downloaded": False,
    "remote_state_mutated": False,
    "objects": records,
    "object_counts": {arm: len(values) for arm, values in records.items()},
}
inventory = output / "OBJECT_INVENTORY.json"
inventory.write_text(
    json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
packaging = output / "PACKAGING.txt"
packaging.write_text(
    "M15 Attempt-9 full object-name inventory\n"
    "payloads_downloaded=0\n"
    "remote_state_mutated=0\n"
    "source_identity_matches=0\n",
    encoding="utf-8",
)
manifest = output / "SHA256SUMS"
manifest.write_text(
    "".join(
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("OBJECT_INVENTORY.json", "PACKAGING.txt")
    ),
    encoding="ascii",
)
print(json.dumps(summary["object_counts"], sort_keys=True))
PY

(cd "$RETURN" && sha256sum -c SHA256SUMS)
python3 - "$RETURN/OBJECT_INVENTORY.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(value["objects"], sort_keys=True, indent=2))
PY
```

Return exactly these three small files, the two printed object counts, the
independent `sha256sum -c` output, and `git status --short`:

```text
OBJECT_INVENTORY.json
PACKAGING.txt
SHA256SUMS
```

Do not return full bucket roots, credentials, environment dumps, raw logs,
NPZs, compact bundles, or token contents. Do not download anything yet. Do not
commit or push unless the user separately authorizes that exact evidence-only
action.

Interpretation is mechanical:

| Full inventory | What the execution agent does |
|---|---|
| Any object other than `PREFLIGHT.json` exists | stop and return the inventory; the analysis owner prepares a narrowly scoped downloader/classifier for those exact names |
| Each arm contains only `PREFLIGHT.json` | stop and return the inventory; classify Attempt 9 as irrecoverable from its registered GCS roots |
| Listing fails or contains an out-of-root entry | hard stop; return stderr and do not retry with broader permissions |

Regardless of inventory outcome, the later receipt's source SHA remains
invalid and cannot authenticate the run. Attempt 11/d17 also remains
inconclusive: it collected roughly 2,100 observer records per arm in the pod,
but the legacy incident ledger exceeded 2 GiB before classifier/bundle
persistence. No current result selects a layer or authorizes a numerical fix.

## Historical decision — do not rerun before Phase D2 certification

Before any new target run, implement and certify all four durability changes:

1. wide mode must bypass the redundant legacy P38 incident ledger rather than
   raising its byte bound;
2. bounded observer shards must upload incrementally while the worker is
   alive;
3. the classifier must run from persisted shards and write `COLLECTED`, then a
   self-hashed manifest, then `COMPLETE` from the surviving worker;
4. runtime source identity must come from the executing checkout and agree
   with the rendered source SHA.

Rehearse forced failure after one shard and require that the shard, source
marker, and an `INCONCLUSIVE` terminal receipt survive. Keep one diagnostic
round only. After host/exact-image packaging gates pass, a new DP8xTP8 off/on
pair still requires separate user approval.

The one-host ladder is exhausted: real scheduler publication, 32-request
composition, `continue_decode=8`, and full M15 chronology all stayed exact on
DP1xTP4. The root remains a scale/topology seam. Do not run another one-host
replay and do not guess a RoPE/page repair.

This source prepares a known-red target localization run. It changes no model
arithmetic and keeps production APC off. The first target run attaches one
identical observer to an APC-off control and APC-on treatment:

- all 36 layer input/output fingerprints;
- final norm and terminal tail;
- positions 960..4096, covering the historical 1226 and Attempt-6 prompt
  boundaries;
- exact request/call/token-prefix/page receipts;
- automatic M15-aware classification and compact selected-record bundle.

After the user explicitly approves commit/push and the exact source SHA is
available, render only with:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<40-character-published-sha>
RUN_ID=<fresh-unique-label>
OUT=/tmp/v1-apc-m15-wide-${RUN_ID}
test ! -e "$OUT"
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --observer layer \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Expected manifests:

```text
jobset-v1-apc-m15-off-layer.yaml
jobset-v1-apc-m15-on-layer.yaml
```

If both allocations are available, the user may submit the two standalone
commands without waiting between them. Do not append pipes, `tee`, `&&`, or a
monitor:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off-layer.yaml"
```

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on-layer.yaml"
```

Interpret control first. The off arm must remain A-B=0 and B-C=0. The on arm
must keep B-C=0; if red, it must emit one of:

```text
M15_LAYER_FIRST_RED_LOCALIZED
M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED
```

If the first result selects layer `L`, do not guess or edit YAML. A separately
approved follow-up is rendered by:

```bash
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "${RUN_ID}-full-l${L}" \
  --observer full \
  --seam-layer "$L" \
  --output-dir "/tmp/v1-apc-m15-${RUN_ID}-full-l${L}"
```

Return exactly:

1. full source SHA, both JobSet names, attempts, Kubernetes terminal states,
   and both GCS Attempt-0 URIs;
2. the complete `CANON_ALIGN_PRE` line for each arm;
3. both `p38_seam.classification.json` files and their SHA-256;
4. the `CANON_APC_M15_SEAM_BUNDLE` path/size/SHA marker for each arm;
5. on red: `classification`, `gate`, `selected_layer`, `last_exact_boundary`,
   `first_red_boundary`, `coverage`, and `source_interval` from the JSON;
6. any nonzero return code plus complete stderr/raw-log tail.

The compact bundle contains real token/capsule material. Under the dedicated
`m15-wide-v1` contract it is uploaded only to the task's already authorized
P38 evidence prefix, after classification from sealed shards. Do not copy it
to any other location or return it through chat/Git.

Current claim ceiling is `WIDE OBSERVER READY / TARGET NOT RUN / ROOT CAUSE
NOT LOCALIZED`. See [Phase D](phases/phase-d-wide-target-observer.md).

## Historical Phase C replay input and Attempt-6 evidence

Attempt 6 paired execution (`d12-9f91d930`, source commit `9f91d93001dd5b44659f062626eb93fc65e6fcb4`) ran on 64 TPUs (DP8xTP8) for both control and treatment arms, persisted complete raw payloads to GCS Attempt-0 roots, and successfully passed the GCS replay audit `run_m15_replay_gcs_audit.sh`:

- **Control Arm (`canon-v1-apc-m15-off-d12-9f91d930`)**:
  - Rollout: 2,560 requests completed, 0.0% prefix cache hit rate.
  - JAX Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=117415 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - GCS Audit Verdict: `CONTROL_GREEN` (`receipt_sha256=c9550f73...`, `manifest_sha256=b91cd34c...`).
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d12-9f91d930`)**:
  - Rollout: 2,560 requests completed, **92.9%** prefix cache hit rate.
  - JAX Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=119565 bounds=[('S_decode_vs_S_prefill', 1770), ('S_prefill_vs_T_old', 0)]` (**Captured exact mismatch of 1,770 bytes / 748 elements**).
  - Canonical first mismatch: row 201, completion position 0, logical prefix 1066; its request starts at call 187 and the bounded interval ends at call 188.
  - Earliest request belonging to any red row: row 245 at call 164.
  - First fully captured tensor incident: row 245, request `400-bc7daec5`, serving call 565, DP rank 0, slot 29, `num_computed_tokens=1248`, 296 exact joins. This is not the onset.
  - Mismatch Capsule: 15,148 bytes (`sha256:9e79a18d...`).
  - Producer Unit: 762 KB, 256 rows (`m15_producer_unit.npz`).
  - Replay Envelope: 103.7 MB, 3,027 calls (`m15_replay_envelope.jsonl`).
  - GCS Audit Verdict: `FRESH_TARGET_RED_FROZEN` (`receipt_sha256=557801a3...`, `manifest_sha256=93f56a0a...`).
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

### Phase C Execution Summary (Replay Input Plan Prepared)

`run_m15_replay_gcs_prepare.sh` was executed on `canon-v1-apc-m15-on-d12-9f91d930/attempt-0` and terminated with:
```text
[M15.APC.REPLAY.PREPARE] COMPLETE status=M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED analysis_sha256=a3c381f8d5e8143ac266a96fb082679e86d85a96eb749255696aaebe649ceff0 manifest_sha256=ed0c67413e51acd639e79dbb95df8698ed8e0386ea606dfcfcb0b1a4fb3e2355 prefix_sha256=b8c00fc704cdd698318a2088c70b9593737a996da8eda1e55d98986d5a8f30a7 prefix_bytes=3938394 red_rows=201,245 replay_prefix_end_call=188 destination=gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-v1-apc-m15-on-d12-9f91d930/attempt-0/derived/m15-replay-input-plan-v1
```

Evidence sealed in `evidence/v1_apc_m15_attempt6_paired_d12_20260825/` and `evidence/v1_apc_m15_replay_input_plan_d12_20260826/`.

Current claim ceiling:
```text
M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED
```

Next phase: **Phase D (Deterministic replay and tensor-level tap)**.

## Historical Phase-D replay proposal — superseded by the target observer

The replay plan is prepared (`replay-prefix-plan.jsonl`, 188 calls, 3.94 MB).
The next action is to execute the deterministic replay harness using the saved 188 prefix calls to:
1. Prime the cache from Call 1 through Call 188.
2. Verify reproduction of the exact 1,770 byte A-B mismatch on Row 201/245.
3. Tap the attention / RoPE / block-table layers to pinpoint the exact numerical root cause.

## Scope and current ceiling

This task is the independent APC/prefix-cache numerical lane. It does not
change production APC defaults and it does not authorize a TPU launch,
commit, or push.

Current immutable facts:

- one-host Qwen3-8B DP1xTP4 Phase3 G-A through G-D is exact;
- M15 `m15i` on DP8xTP8 is A-B red by 1389 bytes / 760 elements and B-C exact;
- the first historical red is row 192, completion position 0, logical prefix
  1226;
- the historical archive has hashes but not reversible tokens/request order/
  cache lineage, so `m15i` itself is not an exact replay carrier;
- no numerical source has been repaired and all production full recipes remain
  APC-off.
- Attempt 0 (`canon-v1-apc-m15-off-d3-eb58954f`) is `INCONCLUSIVE`: it never
  reached alignment or created a serving capture. Its command selected
  `m15/main` on the CLI but omitted `CANON_P57_WORKLOAD_CANDIDATE` and
  `CANON_P57_DATA_SPLIT`, which the FrozenLake entrypoint requires to match.
- The bootstrap repair carries exact `m15/main` identity through the renderer,
  profile, and Step-00 resolver and preserves the package-safe module
  entrypoint. It changes no APC, model, alignment, backward, or optimizer math.
- Attempt 1 (`canon-v1-apc-m15-off-d4-283cb67e`) is also prelearner-only: it
  passed all overlays and GCS preflight, then legacy P38 DP16 assertions
  rejected the M15 carrier's `(mini_batch_size, sampler_is)=(32, none)` and
  would next have rejected its DP8 workload/unit identity. No A/B/C verdict or
  replay payload was produced.
- The bounded follow-up keeps legacy P38 exactly at `frozenlake`, DP16,
  8 x 4-prompt producer units and token IS, while only
  `CANON_APC_M15_TARGET_DEBUG=off|on` admits `frozenlake-dp8-tp8`, DP8,
  1 x 32-prompt unit and no IS. Cross-mode and partial geometry negatives are
  executable host tests. It changes no numerical code.
- Attempt 2 (`canon-v1-apc-m15-off-d7-41a2043c`) finally reached the real
  DP8xTP8 rollout and completed more than 1,800 serving calls plus all four
  standard capture strata. It did not reach A/B/C classification: the
  incident ledger saturated at call 326 (268,192,266 bytes) and the drain tail
  later entered the production `continue_decode` path, which the old
  single-path observer rejected.
- Removing `CANON_CONTINUE_DECODE=8` is explicitly rejected because `m15i`
  used it. Attempt 3 proved patch 27's remaining assumption was also wrong:
  APC-on can enter `continue_decode` before four standard tensor strata have
  been captured. Append-only patch 28 admits that registered M15 program path
  from its first call and writes only the dedicated host replay envelope;
  standard tensor capture and generic request/incident evidence stay
  standard-only. The M15-only incident/replay ceiling remains 2 GiB. A frozen
  red must attest A=`standard+continue_decode` and B=`standard`; unknown paths
  and any non-M15 use remain fatal, while a B-side continue path is rejected
  by packaging.
- Attempt 4 (`canon-v1-apc-m15-on-d10-618eb775`) proves patch 28 reached the
  end of the real rollout: 2,560 requests completed, prefix-cache hit rate was
  92.5%, and solve ratio was 0.203. It then failed before A/B/C because the
  generic alignment gate did not admit this carrier's signed
  `sampler_is=None` recipe. Its two committed files pass `SHA256SUMS`; they
  prove the fatal admission boundary but are not a complete replay package.
- Attempt 5 paired run (`d11-a909fda1`, source `a909fda1`) produced hash-valid
  Git snapshots for both arms.  The snapshots show 0.0% cache hits off and
  approximately 89.4%--97.5% on, but contain no alignment, sampler,
  controlled-exit, classification, or GCS-terminal markers.  The accompanying
  receipt is an unverified summary until the GCS-side audit returns.

Claim ceiling: `FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN`.

The exact remote procedure is in [RUNBOOK.md](RUNBOOK.md). The execution agent
must run those commands rather than constructing a new carrier by hand.

## Prepared bounded carrier

The new renderer creates a matched pair from one committed source tree:

- `off`: APC-off shared-serving control;
- `on`: production-congruent cache-read treatment.

Both use the exact M15 main geometry: DP8xTP8, 32 prompts, 8 generations,
256 trajectories, concurrency 256, `vllm_max_num_seqs=32`, batched tokens 256,
15 turns, prompt 4096, response 8192, temperature 0.7, seed 42. Both stop after
one strict pre-alignment round with zero backward and zero optimizer commit.
Both deliberately use `--sampler_is=none`: A supplies rollout logprobs as the
old-policy source and no token-IS correction weights may exist.

The only intended cross-arm values are
`CANON_APC_M15_TARGET_DEBUG=off|on` and derived
`CANON_VLLM_ENABLE_PREFIX_CACHING=0|1`; a structural test rejects any other
document difference after arm-path normalization.

A must attest:

```text
prompt_logprobs=None
logprobs=1
skip_reading_prefix_cache=False
```

B must attest `reset_prefix_cache=True` and zero cached tokens. The classifier
rejects any B-C byte difference, optimizer marker, wrong source, wrong command,
missing capsule/journal/incident join, or M15 classifier failure hidden behind
the expected controlled exit code 42.

For every arm, postflight/GCS collection preserves:

- `m15_producer_unit.npz`: all 256 final token/logprob rows;
- `m15_replay_envelope.jsonl`: every serving call's exact host-side dispatch,
  request, DP slot, prefix hash/position and physical page table.

For a fresh red, postflight additionally creates
`p38_serving_capture/m15_first_red_replay/` with:

- `first_red_capsule.npz`: the earliest exact incident row's complete prompt,
  completion, masks, A/B/C, policy version, and sampling values;
- `first_red_contract.json`: request/call/position, DP slot, physical pages,
  page generations, and co-batch request IDs;
- `SHA256SUMS`.

It then creates `m15_full_replay_carrier/`. Its request-row join proves that
every scheduled token history comes from the saved 256-row producer and that
the first-red request/call/pages match the incident ledger. The full carrier
records scheduler dispatch order but has not yet forced that order through a
replay harness, so it remains input evidence rather than a mechanism verdict.

## CPU validation already run

From the independent worktree:

```bash
cd /home/yuxuan/code_rl_repro/worktrees/m15_eval_fix_0825
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_analyze_m15i_evidence.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_classify_m15_apc_target_run.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_first_red_replay.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_full_replay_carrier.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py -v
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_resolved_env.py -v
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py -v
python3 canon-zero-tim/tests/p3_prefix_cache/test_contract.py -v
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base ff913a84
bash -n canon-zero-tim/cluster/steps/00_env.sh canon-zero-tim/cluster/steps/90_run.sh
git diff --check
```

All task-specific tests and the flag audit pass. After the Attempt-4 repair,
the task carrier is 46/46, P57 is 146/146, V1 CPU is 67/67, and flags are
378/378. The pinned image
`sha256:418dc632...e53a` was then run on the final runtime/test tree. The first
attempt exposed an image-only test PATH defect (`python3` lives under
`/usr/local/bin`); after the test inherited the active interpreter directory,
the full rerun terminated with `V1_HP_EXACT_IMAGE_PASS ...
apc_m15_carrier=46 ... manifests=3`. This is exact-image admission, not a
one-host numerical replay or DP8xTP8 target result.

The worktree was initially created at reference `687b2bd6...`. The operator
branch later advanced through `ff913a84` to `9f79cc56`; the intervening raw-log,
P58 seed, and P64 shared-entrypoint changes were reviewed, then the release
commit was rebased without conflict before the final aggregate gate.

## Historical — Attempt-4 next-launch instructions (superseded)

This section records how the earlier paired carrier was admitted.  It is not
the current operation.  Follow the `START HERE` Attempt-12 GCS audit instead.

Do not relaunch source `eb58954f...`; its missing signed identity is
deterministically invalid. Patch 28 has passed the targeted and aggregate
exact-image gates on the current tree. First publish the observer repair, then
verify that the committed tree is identical to the admitted tree and render
only from that new full SHA. One paired-launch approval covers both target
arms. Use a unique label and a new output directory:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<full-committed-sha>
RUN_ID=<new-unique-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Do not edit the rendered YAML. Do not render from a dirty or abbreviated SHA.

If any runtime or test file changes before publication, rerun the dependency-
complete pinned-image gate against the immutable production image:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

The post-fix terminal must include `apc_m15_carrier=46`. The current
Attempt-4 repair tree produced that terminal with exit 0. This remains a CPU/image
admission gate, not a DP8xTP8 numerical result.

The installed-runner test is not a string-only predicate check: with zero
captured records and zero strata it executes `_p38_serving_begin`, requires the
M15 replay ledger call, and requires generic incident capture to remain absent.

Before applying the YAML, the rendered environment must contain all four
members of one workload identity:

```text
CANON_P57_WORKLOAD_CANDIDATE=m15
CANON_P57_DATA_SPLIT=main
--p57_workload_candidate=m15
--p57_data_split=main
--sampler_is=none
```

The checked-in renderer and real Step-00 resolver now enforce this and reject
wrong identity or file-path-entrypoint negatives before TPU work begins.
It must also contain `CANON_CONTINUE_DECODE=8`,
`CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard`, and
`CANON_P38_INCIDENT_MAX_BYTES=2147483648`. This combination is intentional:
the tensor records stay single-path while the replay envelope attests the
mixed production tail.

Do not relaunch Attempt-4 source `618eb775...`: it deterministically lacks the
new sampler admission. Attempt 4 has no matched fresh APC-off arm, so it cannot
substitute for either member of the new pair below.

## Historical — paired-launch contract used by Attempt 12

After one explicit paired-launch approval, issue both standalone commands
immediately. Do not append a pipe, `tee`, `&&`, or a monitor to either command:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off.yaml"
```
```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on.yaml"
```

Do not wait for off to finish before submitting on. The arms may execute and
fail concurrently; they still use distinct JobSets, logs, and JobSet-derived
GCS roots. A failure in one arm must not cancel or delete the other arm.

Interpretation remains control-first even though execution is concurrent.
First classify off and require `CONTROL_GREEN`, B-C zero, plus
`PREFLIGHT.json`, `COLLECTED.json`, and `COMPLETE.json`. If off is red or
inconclusive, preserve and report the on package, but do not use on to make an
APC-specific causal claim.

After a green control, the treatment has two admissible outcomes:

- `FRESH_TARGET_RED_FROZEN`: proceed to Phase C/D using the bundled first-red
  carrier; do not infer RoPE/page/cache mechanism yet;
- `TARGET_NOT_REPRODUCED`: this one bounded target observation was exact at
  representative depth/cache occupancy; it does not prove APC fixed.

Any `INCONCLUSIVE`, B-C red, missing join, missing GCS terminal marker, or
unexpected optimizer/backward evidence is a hard stop.

## Historical — generic replay return contract

Large GCS evidence remains durable and must not be added wholesale to Git. On
the machine that can read the bucket, run the checked-in GCS audit:

```bash
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_replay_gcs_audit.sh \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Return these small outputs plus their SHA values:

1. `PREFLIGHT.json`, `COLLECTED.json`, `COMPLETE.json`, and remote
   `SHA256SUMS`;
2. `serving-classification.json`;
3. the derived audit URI and `RETURN_RECEIPT.json`;
4. from the derived audit:
   - `m15-classification.json`;
   - on a red, `first-red-contract.json`, `replay-contract.json`, and
     `request-row-joins.jsonl`;
5. the raw log, or if it is too large, an immutable raw-log URI/SHA plus every
   line containing the alignment-pre marker, M15 APC marker prefix,
   `P3_APC_CONFIG`,
   `Prefix cache hit rate`, `CONTROLLED_EXIT`, and `FATAL`;
6. the exact source SHA, JobSet name, attempt number, GCS prefix, and Kubernetes
   terminal status.

The audit script verifies the remote root manifest and both nested replay
manifests before uploading its own `SHA256SUMS` last. Do not call a hash-valid
subset a complete upload; the three terminal markers and required
classifications remain separate completeness gates.

## Rollback

The new selector is default-off, so publication does not change production
behavior. Reverting this bounded carrier must remove the
renderer/profile/classifier/marker additions as one concern. Production
recipes remain APC-off throughout.


## Attempt 7 M15 Target Debug Runs (d13-663cb547)

Attempt 7 dual-arm execution (`d13-663cb547`, source commit `663cb5474490173cfaf33fce3a323d95e5fc2ee1`) was launched on dual 64-TPU allocations:
- **Control Arm (`canon-v1-apc-m15-off-d13-663cb547`)**: Successfully uploaded `PREFLIGHT.json`, terminated during startup.
- **Treatment Arm (`canon-v1-apc-m15-on-d13-663cb547`)**: Successfully uploaded `PREFLIGHT.json`, terminated during startup.
- Retained evidence: `evidence/v1_apc_m15_attempt7_d13_20260826/`.

## Attempt 8 M15 Target Debug Runs (d14-3820b168 Phase D Wide Layer Observer)

Attempt 8 dual-arm execution (`d14-3820b168`, source commit `3820b168e37080ea9c4e2e2832810a950a7c493f`) ran on dual 64-TPU allocations (DP8xTP8) with all 36-layer observers attached:
- **Control Arm (`canon-v1-apc-m15-off-d14-3820b168`)**:
  - Rollout: 256 trajectories completed, 0.0% prefix cache hit rate, solve rate 15.2%.
  - Pre-alignment: `verdict=PASS`, 0 differing bytes on A-B and B-C.
  - Collected >2,420 wide observer records across all 36 layers.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d14-3820b168`)**:
  - Rollout: 256 trajectories completed, **93.1%** prefix cache hit rate, solve rate 20.7%.
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=122951 bounds=[('S_decode_vs_S_prefill', 1191), ('S_prefill_vs_T_old', 0)]` (Reproduced 1,191 diff bytes between $S_{\text{decode}}$ and $S_{\text{prefill}}$).
  - Evidence: `evidence_sha256=740a34978c4519a0cd696aa6dc283ad111dcdb8f0bf8cbbe02a4c62722426854`.
  - Collected >2,112 wide observer records across all 36 layers.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- Retained evidence: `evidence/v1_apc_m15_attempt8_paired_d14_20260826/`.

## Attempt 9 M15 Target Debug Runs (d15-3f159250 Phase D Wide Layer Observer)

The historical receipt under
`evidence/v1_apc_m15_attempt9_paired_d15_20260826/` claimed a completed paired
run, APC-off `0/0`, APC-on A-B red by 1,329 bytes, and 2,313 tensor records.
That claim is **superseded as unsigned prose**:

- its full source SHA does not exist in the repository;
- both real GCS preflight markers instead identify the valid commit
  `3f159250c4781b3faafde238f768457a0478446b`;
- the expected-object GCS audit found no `COLLECTED`, `COMPLETE`, root manifest,
  classifier, or compact bundle in either arm.

Only bucket writability at startup and the runtime-marker source identity are
currently verified. The historical numerical values must not select a layer,
close a gate, or justify a repair. See
`evidence/v1_apc_m15_attempt9_gcs_salvage_20260827/`; the full object-name
inventory at the top of this handoff is the only admitted next operation.

## Attempt 12 M15 Target Debug Runs (d20-395c0e0d Phase D Wide Layer Observer)

Attempt 12 paired dual-arm execution (`d20-395c0e0d`, source commit `395c0e0de8626c96e85457b997efddd2dd2dec48`) ran on dual 64-TPU allocations (DP8xTP8) with all 36-layer observers attached:
- **Control Arm (`canon-v1-apc-m15-off-d20-395c0e0d`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **18.4%** (47/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=118186 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Seam records: 2,474 pairs across all 36 layers verified bitwise exact.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- **Treatment Arm (`canon-v1-apc-m15-on-d20-395c0e0d`)**:
  - Rollout: 256 trajectories completed, **92.5%** prefix cache hit rate, solve rate **22.7%** (58/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115908 bounds=[('S_decode_vs_S_prefill', 477), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 477 differing bytes across 227 elements).
  - Layer Fingerprint Comparison:
    - Layer 0 `layer_input`: 100% bitwise exact between uncached prefill writer (Gen 0) and cached readers (Gen 1..7).
    - Layer 0 `layer_output`: First red boundary identified (`first diff=(0, 'layer_output')`).
    - Cached readers (Gen 1 vs Gen 2 vs ... vs Gen 7): 100% bitwise identical to each other (`total differing = 0`).
  - Classification: `M15_LAYER_FIRST_RED_LOCALIZED`, `gate=COARSE_FIRST_RED_INTERVAL`, `selected_layer=0`.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- Retained evidence: `evidence/v1_apc_m15_attempt12_paired_d20_20260827/`.

### Follow-up action after the `START HERE` audit passes

Do not run the command below from the currently checked-in four-member summary
alone.  The read-only Attempt-12 GCS audit at the top of this Handoff must first
return `LAYER_SELECTED`, both arms evidence-bound, and no source conflict.
Only then, with separate user approval for the paired target launch, render:
Render and launch the Layer 0 full 15-checkpoint observer:
```bash
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "395c0e0de8626c96e85457b997efddd2dd2dec48" \
  --run-id "d21-full-l0" \
  --observer full \
  --seam-layer 0 \
  --output-dir "/tmp/v1-apc-m15-d21-full-l0"
```

## Attempt 13 M15 Target Debug Runs (d32-7d30f382 Phase D Layer-0 Full Observer)

The five checked-in files report that Attempt 13 (`d32-7d30f382`, source
commit `7d30f3827480e6f9d5ae972f55ca4d16f07de6df`) executed a paired dual-arm
DP8xTP8 Layer-0 full-observer run.  The following values are retained as the
submitted summary, not as a replayed official classification:

- **Control Arm (`canon-v1-apc-m15-off-d32-7d30f382`)**:
  - Rollout: 256 trajectories completed, **0.0%** prefix cache hit rate, solve rate **16.0%** (41/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=112544 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` ($A-B=0, B-C=0$).
  - Classification: `M15_OBSERVER_CONTROL_EXACT`, `gate=OBSERVER_REACHED_EXACT_ENDPOINT`.
  - Seam records: 2,474 pairs across Layer 0 verified bitwise exact.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.

- **Treatment Arm (`canon-v1-apc-m15-on-d32-7d30f382`)**:
  - Rollout: 256 trajectories completed, **92.7%** prefix cache hit rate, solve rate **19.9%** (51/256).
  - Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=115396 bounds=[('S_decode_vs_S_prefill', 239), ('S_prefill_vs_T_old', 0)]` ($B-C=0$ exact, captured 239 differing bytes).
  - Submitted fingerprint checkpoint summary:
    - `[0] layer_input`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[1] input_norm`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[2] q_proj`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[3] k_proj`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[4] v_proj`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[5] q_norm`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[6] k_norm`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[7] q_post_rope`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - `[8] k_post_rope`: 🟢 EXACT MATCH ($\Delta = 0.0$)
    - **`[9] rpa_output`**: red fingerprint; the reported
      `7.1857e8` is an integer-fingerprint delta, not an activation `max_abs`.
    - `[10..14] o_proj, residual, post_norm, mlp, layer_output`: 🔴 RED (downstream propagation).
  - Classification: `M15_INTERNAL_FIRST_RED_LOCALIZED`, `gate=INTERNAL_FIRST_RED_LOCALIZED`, `selected_layer=0`.
  - Terminal: Controlled exit code 42, zero backward, zero optimizer commits.
- Retained evidence: `evidence/v1_apc_m15_attempt13_paired_d32_20260828/`.

### Evaluator correction — Attempt 13 is not classifier-replayable yet

- The local subset's four hashed payloads are intact, but the complete
  three-round and terminal evidence chain is absent.
- Both submitted classifiers omit `diagnostic_round`; the on-arm classifier
  also omits the official anchor/signature/replay-ledger fields.
- The classifier in the claimed runtime source fails while resolving the
  reported `rpa_output` source anchor.  The local correction points to the
  observer patch, but has not yet been run against the private compact bundle;
  the submitted `source_interval` therefore remains unverified.
- Fingerprint equality is not full-tensor byte equality.  The available subset
  supports only an RPA/attention-call **interval hypothesis**, not a proven
  block-table or cached-KV-read defect.
- Claim ceiling:
  `ATTEMPT13_SUBSET_HASH_VALID / OFFICIAL_CLASSIFIER_NOT_REPLAYABLE /
  RPA_ATTENTION_CALL_INTERVAL_HYPOTHESIS`.
- Next action: follow the top `START HERE` section and recover d32's six
  per-round classifiers from GCS.  Do not launch again and do not enter Phase E
  until that return is independently hashed and the official classifier can be
  replayed.
