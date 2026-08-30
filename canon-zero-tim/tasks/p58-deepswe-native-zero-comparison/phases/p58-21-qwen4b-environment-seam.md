# P58.21 — Qwen3-4B one-host environment-seam discriminator

Status: complete as a causal alignment control; backward admission incomplete.

## Bound RED

Direct-v5p artifact
`/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s20dev_20260829t0330z`
proved the P58.20 seven-target TP4 stack and exact B-C, then returned finite
A-B RED over 3,300 action tokens. The first mismatch is the first assistant
token after an environment result. The initial action is exact and token
shift +/-1 controls are much worse than shift 0. Backward and optimizer commit
were unreachable.

## Single-variable control

`CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC=standard-decode` is default-off and legal
only under P58.20 admission. It derives an empty `CANON_CONTINUE_DECODE`
instead of baseline `8`. It must preserve the exact source/model snapshot,
Pillow whitelist and task image, seed, G2 trajectory limits, explicit
temperature/top-k/top-p `0.7/0/1.0`, seven overlay targets, `qwen4b_tp4`
manifest, fixed tied TP4 head, prefix-cache off, strict A=B=C, and
backward-no-commit identity.

The selector and resolved continue-decode value must be present in the
durable manifest and classifier. Missing, unknown, foreign-workload, TP8,
Native, production-P58, or manually contradictory values fail closed.

## Decision

- Control exact: continue-decode cache state is a necessary cause on this
  carrier. A control backward PASS remains diagnostic; repair the value-8
  path and rerun P58.20 before TP8.
- Control RED at the same environment seam: continue-decode is not sufficient;
  add bounded per-page KV fingerprints at the injection boundary and compare
  cache write/content versus subsequent read.
- Any B-C RED, non-finite value, missing artifact, cleanup failure, or changed
  treatment is an invalid control and must not be interpreted.

## Target result

Direct-v5p run `p58s21std_20260829t0357z` preserved the P58.20 model, task,
seed, sampling, seven-target TP4 overlay, fixed head, prefix-cache, and strict
alignment contracts while deriving only `CANON_CONTINUE_DECODE=`.  Its two
real trajectories admitted 2,553 action tokens.  Both structured boundaries
were bitwise exact:

- `S_decode_vs_S_prefill`: 0 differing elements, 0 differing bytes,
  `max_abs=0.0`;
- `S_prefill_vs_T_old`: 0 differing elements, 0 differing bytes,
  `max_abs=0.0`.

The immutable artifact is
`/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s21std_20260829t0357z`.
Its trajectory, batch metrics, manifest, and pre-alignment checksums are in
`decode_prefill_probe.classification.json`.

The process then spent the remaining 7,200-second bounded runtime compiling
the first full 8,192-token backward and received the wrapper's SIGTERM.  No
gradient or state-identity record was produced, so the classifier correctly
returned `ZERO_TIM_BACKWARD_INCOMPLETE`.  This is not a backward PASS or a
backward numerical failure.  It does not weaken the exact alignment control:
standard decode removed the P58.20 environment-seam RED, making the
continue-decode program a necessary cause on this carrier.

## Claim ceiling

This phase can only classify one direct-host Qwen3-4B DP1xTP4 mechanism. It
does not certify the baseline high-performance arm, TP8, Pathways,
disaggregated roles, optimizer commit, or convergence.
