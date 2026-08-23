# P58.6/P58.7 operator runbook

This file does not authorize execution. Commit/push, direct-host runs, image
publication, Kubernetes apply, and 128-chip launch are separate approvals.

## P58.6 matched one-host XProf pair

Run both arms serially on the same direct-attached four-chip host from one
clean `local/*` source tree. Labels and artifact directories are immutable and
must be fresh.

```bash
export P58_ONEHOST_EXPECT_HOSTNAME='<exact-hostname>'
export P58_ONEHOST_EVIDENCE_ROOT='/mnt/disks/tunix-data/deepswe-onehost-xprof'
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_xprof_native.sh '<fresh-native-label>'
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_xprof_zero_hp.sh '<fresh-zero-label>'
```

Do not append a pipe to either run command. The drivers write `raw.log`
directly. After both arm classifiers say PASS, write the pair decision outside
both sealed arm roots:

```bash
python3 canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/classify_onehost_xprof_pair.py \
  --native '<native-root>/classification.json' \
  --zero-hp '<zero-root>/classification.json' \
  --output '<evidence-root>/pair-<fresh-pair-label>.json'
```

Require pair `PASS` before making a causal operation-attribution comparison.
`INCONCLUSIVE_INPUT_MISMATCH` preserves both packages but supports no speed
claim. Analyze captures under the `xprof-trace-analysis` discipline: first
validate the capture window, then use semantic Perfetto phases to locate the
update, then attribute XLA ops; never turn summed overlapping op time into
wall time.

## P58.7 Zero-HP full render

Only after publication/readback and explicit launch approval, use a fresh
output and run id:

```bash
python3 canon-zero-tim/cluster/render_p58_deepswe_tim.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output '<fresh-output-dir>' \
  --source-commit '<pushed-40-char-sha>' \
  --source-branch 'yuxzhang/canon-zero-tim' \
  --client-image '<immutable-registry-image>' \
  --run-id '<fresh-zero-hp-full-id>' \
  --stage full \
  --arm zero \
  --cpu-nodepool cpu-np \
  --worker-nodepool tpu-v5p-slice \
  --model-pvc '<production-model-pvc>' \
  --high-performance
```

Before apply, require renderer PASS, exact `4x4x8`, DP8 x TP8 role meshes,
`CANON_V1_HP_FULL=1`, the P58 HP profile, APC off, P59 on, and a server-side
dry run. Monitor updates 1–3 inside the same full job. Any real alignment FAIL
stops the run. A healthy job continues to 1,000 commits.

Postflight is automatic from `90_run.sh`; preserve its base P58 classification,
`p58_zero_hp_full_classification.json`, update report, trajectory journals,
checkpoints, XProf, Perfetto, fixed-head receipts, complete raw log, resolved
environment, rendered YAML, and their hashes.
