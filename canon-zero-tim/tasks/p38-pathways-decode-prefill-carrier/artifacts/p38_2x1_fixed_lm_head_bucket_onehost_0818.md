# P38.2x1 fixed-lm-head bucket gate — 2026-08-18

Scope: construction-only on the directly attached four-device v5p host. This
repairs the P38s23 warmup-contract omission; it is not Pathways repair evidence.

## Entering failure

P38s23/source `32caa773a057ccc2604ee6c1c5ce845f63346bbd` stopped inside
vLLM `CompilationManager.capture_model()` before rollout. The pinned request
bucket M32 reached `compute_logits`, while the first P38.2x contract admitted
only M16/M256. No A-B, B-C, backward, optimizer, or training claim exists for
that attempt. The separately committed hand report spells the full source SHA
incorrectly after its eight-character prefix; only the traceback/code match is
used here.

## Registered construction

- admitted outer request buckets: M8/16/32/64/128/256, and no other M;
- all buckets zero-pad to fixed M256 before Pallas;
- real Qwen3-8B BF16 `lm_head.weight`: `[4096,151936]`;
- TP4-local vocabulary N37984, padded to N38144;
- fixed tiles BM128/BN256/BK256; and
- four deterministic BF16 hidden-input seeds.

## Result

- 24/24 bucket-versus-M256 comparisons are bitwise exact;
- every comparison has `max_abs=0.0`;
- all six lowering receipts contain a custom call;
- non-bucket M1/M7/M24/M257 fail the CPU contract;
- fixed versus stock M16 differs at 249/211/268/219 elements, so the
  intervention remains non-empty; and
- the one-bit negative reports exactly 1 differing element.

Verdict: `FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS`.

## Local artifact hashes

```text
8c738f8b88410633733a422b7f537d1558096d50b805524bba2533482d7d4b88  p38_fixed_lm_head_p38x_bucketfix1_0818.raw.log
cb86eb8cd1be1b442114445c0a1209b73703b5e7ce0d077c97be37f1f302a39d  p38_fixed_lm_head_p38x_bucketfix1_0818.result.json
```

Local paths are under `/mnt/disks/tunix-data/logp_probe_1host/`. The source
worktree was dirty by construction; the raw log records the executable script
and shim hashes. Publication review must rerun this gate if those executable
files change.

## Claim ceiling

This admits only the full request-bucket construction and negative control.
It does not show that P38s23r1 starts successfully or repairs A-B on Pathways,
and it does not admit backward, optimizer, training, or a production default.
