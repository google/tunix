# V1.P4.19 — Admit zero-action rows in shared grouped reverse

- Status: active

## Finding

- Confirmed: M15 run `canon-p57-fl-zero-m15-mw21-6c701164` completed 61
  optimizer updates, then stopped before backward in `_p32_group_spec`. One
  DP8 group had completion lengths
  `[3686,4081,3377,5766,7967,5038,0,6402]`; rank 6 was prompt-only. The
  immutable incident is
  `../../../evidence/m15_step61_empty_completion_incident/`.
- Confirmed: the production caller admitted prompt-only rows only when the
  workload was P34 DeepSWE, even though the earlier shared D3b0 check already
  proves `completion_mask` is a subset of `completion_valid_mask` for every
  registered grouped workload.
- Confirmed: a prompt-only row therefore has no policy-action token. The
  grouped loss excludes it, and `_p32_reverse_group` zeros both logprob and
  entropy cotangents outside `completion_valid` before they enter the packed
  sequence.
- Confirmed: this failure happened during construction, before engine VJP,
  gradient reduction, clipping, or AdamW. It is not an alignment, backward,
  reward, or optimizer red.

## Shape ledger

| Quantity | M15 Step 61 |
|---|---:|
| Caller-global trajectories | 256 |
| Trainer mesh | DP8 x TP8 |
| Rank-major reverse groups | 32 |
| Rows in each group | 8 |
| Prompt/completion compiled widths | 4096 / 8192 |
| Local canonical M | 256 |
| Global engine M | 2048 |
| Largest real sequence in the failing group | 9193 |
| Chunks in the failing group | 36 |
| Prompt-only rows in the failing group | 1 |

No topology, compiled width, scheduler capacity, loss denominator, reduction,
or optimizer setting changes.

## Repair

1. Preserve `_p32_group_spec(..., allow_empty_completion=False)` for direct
   low-level callers.
2. Let the registered production grouped-reverse caller explicitly admit
   prompt-only rows for every workload after the shared action-mask-subset
   proof.
3. Preserve the historical P34 marker for DeepSWE and emit the generic P32
   marker for other grouped workloads. Both state
   `semantics=zero-loss-zero-gradient`.
4. Fail closed when every global trajectory is prompt-only. That case has no
   policy-action loss or gradient to commit and is not silently converted
   into an optimizer transaction.
5. Do not fabricate a completion token, drop/resample a trajectory, change a
   reward or advantage, change normalization, or bypass the fixed DP
   transaction.

## Regression gates

- Replay the exact M15 Step-61 DP8 vector and require `host_n_real`, the zero
  completion row, and `num_chunks=36` exactly.
- Prove a fully nonempty group returns identical group-spec fields with the
  admission disabled and enabled.
- Retain the P58 K11 exact DP8 vector and forced-16-device zero-output / zero-
  cotangent test.
- Require the production call to opt in explicitly and require the global
  all-prompt-only fail-closed guard.
- Run P57 and V1 host suites, then the complete P58 and V1 pinned-image gates.

## Exit gate

- Commands:

  ```bash
  bash canon-zero-tim/tests/p58_deepswe_native_zero/run_exact_image.sh \
    <approved-pinned-image-id>
  bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
    tunix_frozenlake_image:vllm-tpu0.25.0
  ```

- Pass: focused semantic tests, P57/V1 host suites, both complete pinned-image
  gates, syntax, and diff hygiene are green. A later DP8xTP8 target must emit
  the expected empty-row marker, retain every registered alignment/backward/
  optimizer gate, and cross at least one real optimizer commit.
- Fail: any nonzero cotangent from a prompt-only row, any changed nonempty
  group spec, any missing all-empty guard, or any existing gate regression
  stops publication. A target red remains at its first failing boundary.

## Performance observation only — no change in this phase

- Current incident evidence contains only the Step-61 tail: M15 was reported
  at about 10.1 minutes/update, Step-61 had 648,543 action tokens, and Rescore-B
  took 251.176 seconds. It does not contain the full current per-stage timing
  history, so current rollout versus reverse attribution is not yet signed.
- The early historical Zero evidence carried roughly 110k--129k action tokens
  per step. Step 61 therefore contains about 5--6x more policy-action work.
  The most likely explanation for an increasing raw wall time is that the
  improving multi-turn policy survives into more/later turns and produces
  longer trajectories. This is an inference; only a per-step
  `seconds/action-token` series can distinguish workload growth from a true
  throughput regression.
- Historical optimized M15 evidence before later P70/P71 work took 70--86
  minutes/update. Its 32 grouped reverse calls scheduled roughly 513--640
  M256 chunks/update; reverse alone took 2,762--3,515 seconds on steady steps.
  Chunk count and reverse time correlate at about 0.998. Rank-major grouping
  used only about 52--61% of scheduled token capacity, a 1.65--1.91x padded-
  work factor.
- The current profile uses `CANON_P71_SCAN=fwd`. Qwen3-8B TP8 still dispatches
  the 36 layer pullbacks separately for every chunk. The existing
  `CANON_P71_SCAN=bwd` block path is explicitly TP1-only and target-unverified;
  it is not enabled or widened here.
- A historical Native log shows Rescore-B at 27.114 seconds, but it is not a
  matched comparison: model state, generated token count, trajectory lengths,
  and run completion differ. The apparent roughly 10x wall gap is therefore a
  hypothesis, not a certified ratio.

The next performance phase must first replay the same frozen M15 trajectories
and weights on Native and Zero at DP8xTP8, normalize every stage by tokens, and
capture matched XProf/Perfetto. Only then should it choose between TP8 backward
block fusion, length-aware grouping, or serving-path work.

## Result

- Local implementation and focused pinned-image semantic tests are green.
- P57 host gate: 184/184 PASS.
- V1 Phase4 host gate: 93/93 PASS.
- Complete P58 pinned-image gate: PASS with terminal
  `P58_EXACT_IMAGE_CPU_PASS ... p32_empty_completion=4 regressions=1`.
- Complete V1 pinned-image gate: PASS with terminal
  `V1_HP_EXACT_IMAGE_PASS ... p32_empty_completion=2 ... manifests=3`.
- A repaired DP8xTP8 target has not run. The complete image outputs were
  observed directly and were not redirected into new durable evidence files.
  Runtime/test source is isolated in local CL
  `813bb7c5cb229df3bf9890d19959c988c8b9341e`; exact remote readback remains
  required. No image publication, manifest render, or TPU/Kubernetes mutation
  has occurred.
