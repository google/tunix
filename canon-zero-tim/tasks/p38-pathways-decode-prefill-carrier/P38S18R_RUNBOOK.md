# P38s18r2 replacement terminal seam-and-tail runbook

This is the only current operator card for the P38 root-cause lane. P38s18r2
is the fresh successor to the failed P38s18r durability-seal attempt. It is a
64-TPU (`DP16xTP4`) FrozenLake diagnostic, not full training: three frozen-
weight rounds, backward zero, optimizer commits zero, stock arm only.

Do not launch from an uncommitted tree. Do not edit the rendered YAML or add
environment variables by hand. Do not rerun P38s18l/P38.2q. Do not reuse
run-id `p38s18r`, delete its GCS prefix, or cite its incomplete round as a
completed target verdict.

## 1. Publication gate

Source `ae63d44edc67cfcd5b19d34abc82feb681284c67` established the
observer-neutrality baseline on local v5p. Both observer-off and combined-
observer arms completed three rounds from that clean source:

```bash
set -euo pipefail
cd /home/yuxuan/code_rl_repro/sequence_packing/p48p49_integration
SOURCE_COMMIT=ae63d44edc67cfcd5b19d34abc82feb681284c67
test "$(git rev-parse "$SOURCE_COMMIT^{commit}")" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"

bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_incident_onehost.sh \
  off p38-2r-ae63-off
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_incident_onehost.sh \
  seam-tail p38-2r-ae63-on
```

Both commands must end with `PASS ... rounds=3 backward=0
optimizer_commits=0`. The `seam-tail` arm must also print
`TERMINAL_TAIL_PASS`. Because all local A-B boundaries are exact, the mismatch
writer correctly emits no mismatch capsule. Compare the three complete
alignment records with the registered hash-based byte-level classifier:

```bash
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/\
classify_p38_seam_neutrality.py \
  --off /mnt/disks/tunix-data/logp_probe_1host/\
p38_incident_p38-2r-ae63-off_off/pre_alignment.jsonl \
  --observed /mnt/disks/tunix-data/logp_probe_1host/\
p38_incident_p38-2r-ae63-on_seam-tail/pre_alignment.jsonl \
  --output /tmp/p38_2r_ae63_neutrality.json
test "$(python3 -c 'import json; print(json.load(open("/tmp/p38_2r_ae63_neutrality.json"))["status"])')" = PASS
```

The classifier compares the entire alignment contract except its wall-clock
timestamp, including hashes of the original arrays and all action-masked
endpoints. Any different token stream, geometry, denominator, metric, or
endpoint is a failed neutrality gate. Stop; do not launch merely because the
observer produced files. The admitted receipt is recorded in
`artifacts/p38_2r_onehost_neutrality_0816.md`.

The replacement round-scope source is not yet published. Before launch, the
operator must receive the user's explicitly approved full commit SHA. The
replacement changes only host evidence control: it derives frozen round scope
from `p38_diagnostic_round_index()`, strictly filters scoped records, and
admits only the schema-validated request journal as cumulative. Its focused
tests, fake-GCS two-round isolation/abrupt-exit test, pinned-image alignment
tests, and complete P33 CPU gate must be green. These receipts do not authorize
using an arbitrary checkout tip.

## 2. Render exactly one target JobSet

Use a fresh output directory and source SHA. The renderer is the only admitted
way to enable the combined observer:

```bash
set -euo pipefail
cd /home/yuxuan/code_rl_repro/sequence_packing/p48p49_integration
: "${SOURCE_COMMIT:?export the explicitly approved full P38s18r2 fix SHA}"
test "$(git rev-parse "$SOURCE_COMMIT^{commit}")" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
RUN_ID="${RUN_ID:-p38s18r2}"
test "$RUN_ID" != p38s18r
OUT="$(mktemp -d /tmp/p38s18r2.XXXXXX)"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only \
  --seam-mode layer \
  --terminal-tail

STOCK="$OUT/jobset-p38-serving-stock.yaml"
test -s "$STOCK"
test ! -e "$OUT/jobset-p38-serving-unified.yaml"
kubectl apply --dry-run=server -f "$STOCK"
```

Before apply, the manifest must show stock, concurrency 256, diagnostic rounds
3, seam mode `layer`, terminal tail `1`, and `maxRestarts: 0`. Prefix cache
remains disabled by the profile. There is no KV-unified arm and no
full-training admission. The rendered source label and checkout-fetch command
must contain the exact approved SHA.

## 3. Apply and observe

```bash
kubectl apply -f "$STOCK"
```

Do not delete or restart the JobSet after a numerical red; A-B red is the
expected diagnostic carrier. The process seals each round before the next
begins. Require, in order:

```text
[CANON_P38] PRECHECK_ROUND_COMPLETE ...       # exactly 3
[CANON_P38] ROUND_SEAL_REQUESTED ...          # exactly 3
[CANON_P38] ROUND_SEAL_ACKNOWLEDGED ...       # exactly 3
[P38.GCS] ROUND_COMPLETE ...                  # exactly 3, worker log
[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD rounds=3
[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0
[P38.GCS] LIVE_ACTION_ACKNOWLEDGED action=complete
```

Postflight must report one seam init, one tail init, positive A/B record counts,
exact B-C in every round, zero backward, and zero optimizer commits.

## 4. Required returned bytes

The GCS root is printed in the rendered env and has the form:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Return the complete root, not selected excerpts. It must contain:

```text
PREFLIGHT.json
COLLECTED.json
COMPLETE.json
SHA256SUMS
serving-classification.json
seam-classification.json
serving-capture.tar
mismatch-capsule.round-000000.npz
mismatch-capsule.round-000001.npz
mismatch-capsule.round-000002.npz
rounds/000000/ROUND_COMPLETE.json
rounds/000001/ROUND_COMPLETE.json
rounds/000002/ROUND_COMPLETE.json
```

Each round directory must also include its own mismatch capsule,
pre-alignment, request journal, incident ledger, all seam/tail JSON and NPZ
pairs, `ROUND_INVENTORY.json`, and `SHA256SUMS`. Verify every manifest after
download. A root SHA only proves the files listed there; it does not replace
the three round manifests.

## 5. Decision

- first hidden layer input/output red: localize that earliest layer;
- all hidden/final-norm fingerprints exact and raw target logit red: lm_head;
- raw target exact and raw normalizer red: raw vocabulary reduction;
- raw path exact and processed target/normalizer red: logits processing;
- all prior values exact and production endpoint red: production
  gather/subtraction/program tail;
- any missing key/file/round, endpoint/capsule mismatch, observer drift, or
  incomplete marker: `INCONCLUSIVE`, no cause claim and no repair selection.

## 6. Rollback

Do not pass `--terminal-tail` or any seam option. All new observers are
default-off and are not part of P45 FrozenLake full training.
