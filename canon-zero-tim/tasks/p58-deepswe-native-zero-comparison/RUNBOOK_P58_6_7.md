# P58.6/P58.7 operator runbook

This file does not authorize execution. Commit/push, direct-host runs, image
publication, Kubernetes apply, and 128-chip launch are separate approvals.

## Common TiTO preflight

Every DeepSWE arm is token-in/token-out. Native is not a legacy
re-tokenization control. Before accepting a run, require exactly one
`[DEEPSWE.TITO] ADMISSION_PASS` receipt with
`mode=token-in-token-out retokenize_sampled_tokens=0`. For an ordinary P58
training run, require at least one `[DEEPSWE.TITO] CONTINUATION` receipt with a
positive turn, positive prompt-token count, and SHA-256.

The continuation prompt must reuse sampled assistant token IDs exactly and
encode each new R2E observation once. It must reach rollout as token IDs with
`apply_chat_template=False`. Missing IDs, shape/dtype drift, width drift, or a
caller-supplied token override is a hard error. Decoded text remains legal for
action parsing only; it must not become the source of later model prompt IDs.

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

Require the exclusive-topology annotation exactly once on JobSet metadata and
never on the worker Pod template. The Kueue sentinel omits literal nodepool
affinity; a separately supplied real pool remains exact. Before apply,
require renderer PASS, server-side dry-run, exact `4x4x8`, DP8 x TP8 role meshes,
`CANON_V1_HP_FULL=1`, the P58 HP profile, APC off, P59 on, and a server-side
dry run. Monitor updates 1–3 inside the same full job. Any real alignment FAIL
stops the run. A healthy job continues to 1,000 commits.

Postflight is automatic from `90_run.sh`; preserve its base P58 classification,
`p58_zero_hp_full_classification.json`, update report, trajectory journals,
checkpoints, XProf, Perfetto, fixed-head receipts, complete raw log, resolved
environment, rendered YAML, and their hashes.
