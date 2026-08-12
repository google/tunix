# P45.2b — Qwen3-8B TP8 engine overlay

- Status: completed

## Objective

Make the P45 DP8xTP8 carrier select an isolated Qwen3-8B TP8 canonical engine
overlay without changing the existing Qwen3-8B TP4 overlay or its evidence.

## Established failure

Target attempt `p45r3` from source `b26135f2` selected `model_dir=qwen8b` and
failed while importing `linear_p22xf.py`:

```text
RuntimeError: Qwen3-8B P22.XK model contract mismatch:
CANON_QWEN3_TP_SIZE='8'
```

The failure happened before any canonical PATHTRACE, rollout, optimizer
construction, evaluation, or training update. It is a model-overlay admission
failure, not an OOM or an optimizer-residency result.

## Deliverable

1. Add `src/engine_shims/models/qwen8b_tp8/` while leaving `qwen8b/` intact.
2. Register all seven TP8-local projection sites:
   `q=4096x512`, `k/v=4096x128`, `o=512x4096`,
   `gate/up=4096x1536`, and `down=1536x4096`.
3. Use BM/BN/BK `128/128/128`. Admit no matmul or SwiGLU feature padding.
4. Bind only the P45 profile to `CANON_MODEL_DIR_NAME=qwen8b_tp8`.
5. Add a target-image gate that installs the exact overlay, verifies its SHA
   manifest, covers every projection site, runs canonical forward/VJP checks,
   and proves a TP4 model environment is rejected.

## Exit gate

From the pinned P45 image:

```bash
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
```

The gate must print explicit PASS markers for overlay installation, seven-site
contract coverage, forward/VJP coverage, and the TP4 negative control. The
existing P45 CPU/render gate and adjacent P33 workload gate must also pass.

Verified on image ID
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:
29 installed files matched their manifests; the full `linear_p22xk` import
passed under TP8; all seven site shapes and the TP4 negative passed; Pallas
interpret forward/VJP were bitwise exact; P45 reported 83 workload/render tests
and 31 alignment tests passing; the complete adjacent P33 CPU gate passed.

## Non-goals and claim boundary

- Do not modify or re-sign the existing `qwen8b` TP4 overlay.
- Do not claim resident HBM capacity, rollout success, evaluation success, or
  training success from this local gate.
- Do not launch P45.3 until this phase is complete.

## Rollback

Stop selecting `qwen8b_tp8` from the P45 profile and remove only the new overlay
and its P45 tests. The TP4 carrier remains unchanged.
