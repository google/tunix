# P44.9 — r04 model-pinned SwiGLU feature padding

- Status: passed locally; unpublished; target retry not run

## Finding

- Confirmed: `p44r04` discovered all 256 devices, constructed the dynamic
  model mesh, loaded Qwen3-4B, connected W&B, and reached the MLP before
  `p22xj_padded_swiglu.py` rejected `(M, F)=(4096, 1216)`.
- Confirmed: Qwen3-4B intermediate size `9728` gives TP8-local `F=1216`,
  which is not divisible by the unchanged P22.XG `BF=256` feature tile.
- Confirmed: Qwen3-32B would encounter the same class of failure because
  `25600/8=3200` is not divisible by 256. The existing Qwen3-8B overlay uses
  `12288/4=3072`, which is already divisible by 256.
- Decision: Keep the BF256 Pallas kernel, bf16 precision, SwiGLU formula, and
  custom VJP unchanged. SwiGLU is elementwise across the feature dimension,
  so zero-padding an explicitly admitted trailing feature region and slicing
  it after the kernel does not mix with or alter semantic features.

## Execution

1. Add exact `SWIGLU_FEATURE_PADDING` mappings to the model overlays:
   Qwen3-4B `1216->1280`, Qwen3-32B `3200->3328`, and an explicit empty
   mapping for the already-aligned Qwen3-8B path.
2. Extend the existing row-padding wrapper to validate the model mapping,
   pad rows and features with zeros, call the unchanged base kernel, and
   return only the original `M x F` result.
3. Emit `F`, `Fp`, and independent row/feature padding bits in the runtime
   PATHTRACE. Require the 4B `F=1216 Fp=1280 feature_padded=1` evidence in
   the P44 classifier.
4. Add immutable-image Pallas interpret probes comparing both forward output
   and custom VJP exactly against the canonical SwiGLU. Reject an adjacent
   unregistered feature width.
5. Rerun P44 and adjacent P43/P39/P34 CPU and exact-image gates.

## Exit gate

- P44: 34 CPU cases and exact-image marker
  `SWIGLU_FEATURE_PADDING_INTERPRET_PASS model=qwen3-4b-tp8
  shape=129x1216 padded=256x1280 forward_exact=1 vjp_exact=1 negative=1`.
- P43: 22 CPU cases and the same probe at unpadded `3072->3072`.
- P34: 55 unit cases, two Pallas cases, and the same probe at
  `3200->3328`.
- P39 pilot, P34 static/trajectory/update, compile, shell syntax, and
  `git diff --check` remain green.

## Result

PASS locally on latest operator baseline
`e4ead609498771987c011a9cbc16fec7e4b17f69` in immutable local image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.

- Qwen3-4B: exact forward/VJP at `129x1216`, padded to `256x1280`, with an
  adjacent-width negative control.
- Qwen3-8B: exact forward/VJP at `129x3072`, row padded only to `256x3072`.
- Qwen3-32B: exact forward/VJP at `129x3200`, padded to `256x3328`, with an
  adjacent-width negative control.
- The worktree was safely fast-forwarded from
  `a9dc5f296a5cd1225efba7a66a7249113baefe00` to the baseline above. The two
  intervening commits modify only the independent P38 task ledger; no P44,
  engine, test, manifest, or runbook file overlaps. All CPU and exact-image
  gates were rerun after the fast-forward.
- No TPU target, rollout, backward, optimizer update, cloud mutation, commit,
  or push occurred for P44.9.

## Next

Rerun the gates at the latest non-overlapping baseline, then publish only
after explicit approval. The launch agent must start a fresh `rollout-only`
attempt and return the required feature-padding PATHTRACE; `p44r04` cannot be
reclassified.
