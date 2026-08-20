# P38 TP8 fixed output-head runbook

This runbook is for Qwen3-4B and Qwen3-32B only. The implementation is CPU and
pinned-image gated but has not run on TPU. Do not call it certified until the
target gates below pass.

## Render one bounded Qwen3-4B update vehicle

Use the normal P46 arguments for your cluster; the only numerical delta from
the stock render is the final flag:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output /tmp/p38-q4-fixed.yaml \
  --workload q4-debug --topology 64 \
  --source-commit "$SOURCE_SHA" \
  --client-image "$PINNED_IMAGE" --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_POOL" --worker-nodepool "$TPU_POOL" \
  --model-pvc "$MODEL_PVC" --fixed-lm-head
```

P46 q4-debug is a three-update vehicle. For the smallest existing P44 vehicle,
render `render_p44_deepswe_parity.py --stage one-update --topology 64` with the
same `--fixed-lm-head` flag.

## Render Qwen3-32B

The P46 Qwen3-32B lane is full training, so first launch only if its normal
full-training contract is intended:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base canon-zero-tim/cluster/jobset-64chip.yaml \
  --output /tmp/p38-q32-fixed.yaml \
  --workload q32-train --topology 64 \
  --source-commit "$SOURCE_SHA" \
  --client-image "$PINNED_IMAGE" --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_POOL" --worker-nodepool "$TPU_POOL" \
  --model-pvc "$MODEL_PVC" --fixed-lm-head
```

Use `--topology 256` with the corresponding 256-chip base manifest only after
the 64-chip target passes. For an explicitly bounded Qwen3-32B construction
run, use `render_p34_jobset.py` with `--stage one-update --fixed-lm-head`.

## Pre-apply inspection

```bash
python3 - "$YAML" <<'PY'
import pathlib, sys, yaml
d = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
head = d["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
main = next(x for x in head["containers"] if x["name"] == "jax-tpu")
env = {x["name"]: x.get("value") for x in main["env"] if "value" in x}
assert d["metadata"]["labels"]["canon.zero-tim/fixed-lm-head"] == "1"
assert env["CANON_P38_FIXED_LM_HEAD"] == "1"
assert env.get("CANON_QWEN3_TP_SIZE") in (None, "8")
print(env["CANON_PROFILE_FILE"], env["CANON_P38_FIXED_LM_HEAD"])
PY
```

Do not hand-edit the rendered YAML.

## Required returned evidence

Return the complete worker log and the generated
`p38_fixed_lm_head_receipts.json`. A valid target must contain:

- Qwen3-4B: `K=2560 TP=8 local_N=18992 fixed_N=19200 endpoint=tied_embed`;
- Qwen3-32B: `K=5120 TP=8 local_N=18992 fixed_N=19200 endpoint=untied_lm_head`;
- request `semantic_M=16,32,64,128,256` (plus M8 when exercised), learner
  `semantic_M=4096`, and
  `CANON_P38_FIXED_LM_HEAD_VJP=1`;
- receipt classifier PASS, exact B-C, and the normal gradient/reducer/update
  gates.

An env value of 1 without these receipts is not execution evidence.

## Rollback/control

Render the identical command without `--fixed-lm-head`. The renderer emits 0;
never delete the env entry or hand-change the YAML.
