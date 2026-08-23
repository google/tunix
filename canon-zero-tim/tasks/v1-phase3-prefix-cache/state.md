# Phase3 prefix-cache state

- Branch: `local/v1-phase3-apc`, base `bc5d1141`.
- Scope: certify vLLM APC without weakening zero-TIM.  A is APC-on rollout,
  B is full engine prefill with `reset_prefix_cache=True`, and C is trainer
  old-policy forward.
- Current gate: G-E target-topology canary, which remains user-run and requires
  explicit launch approval.  No 64-chip launch has been made.
  P3.1's historical red did not reproduce on the current pinned stack, so no
  speculative numerical repair was made.  Bounded G-A evidence, deterministic
  repeat, and one-host production-shape G-B/G-C are green.
- Shape ledger for the one-host reproducer: caller-global rollout rows are the
  FrozenLake action tokens from batch 2 x generations 2; engine DP is 1;
  scheduler capacity is 4 sequences and 256 batched tokens; canonical kernel
  M is 256; semantic valid rows are the action mask; topology is DP1 x TP4.
- Treatment: only `CANON_VLLM_ENABLE_PREFIX_CACHING=1`.  HBM utilization and the
  stateless gate-only optimizer are held vehicle constants and perform zero
  optimizer commits.
- Last verified fact: dirty-page run `p3gd1_20260823T0506Z` deliberately
  changed one real layer-0 BF16 KV page.  The authoritative A-B gate detected
  30 differing bytes / 13 elements on the targeted first case, while the ten
  later clean cases stayed zero; status is `DIRTY_PAGE_GATE_CAUGHT` and B
  remained full reset.
- Deep-prefix control `p3bc4_20260823T0301Z` and APC-on treatment
  `p3br2_20260823T0306Z` use the production-decode v2 carrier.  All 11 prefixes
  from 1535 through 2049 are A-B exact; APC-on A consumed 1280...2048 cached
  tokens while B consumed zero.
- V2 APC-off `p3bc4_20260823T0301Z` is `BOUNDARY_CONTROL_GREEN`: 11/11
  production-decode A versus full-reset B comparisons are zero bytes, with
  exactly 16 A-returned tokens per case.  This is the matching control for the
  next APC-on v2 treatment.
- V2 APC-on `p3br2_20260823T0306Z` consumed 1280...2048 cached tokens while B
  consumed zero, yet all 11 A-B byte counts were zero.  Either the historical
  label conflated APC with a different at-scale serving KV-read carrier, or an
  old-version/at-scale topology precondition is missing here.  P3.1 first-red
  remains unverified; do not modify numerical code without a red.
- Determinism: repeat `p3br3_20260823T0327Z` produced a byte-identical complete
  boundary report to `p3br2`; report SHA-256 is
  `fa80dfd5f8be52017164169c7f0747a194fd90d0ef3b458ebdd1e1fb24fd7fec`.
- Claim ceiling: one-host G-A/G-B/G-C/G-D, including deterministic repeat and
  dirty-page gate sensitivity, are green for Qwen3-8B DP1xTP4.  This is not an
  upstream APC bug fix, a production/default-on decision, a stochastic
  long-run result, or G-E/64-chip certification.  APC remains default off and
  performance remains a bounded one-host proxy KEEP only.
- Performance result: `p3pc2` (off) versus `p3pa1` (on) passed all six
  per-round cross-arm hashes and both byte gates.  Steady rollout time fell
  `53.152s -> 49.999s` (`+5.932%`), with both steady rounds individually
  faster; `p3pp2` therefore records `KEEP_ONEHOST_PROXY`.  Full wall was
  `483s -> 482s` because initialization/C compilation dominates.
- Profile result: matched `p3xc1` (off) and `p3xa1` (on) are both
  `PROFILE_GREEN`, retain all three A-B/B-C byte gates, and emit device XProf
  plus semantic Perfetto.  Representative TPU module/operator counts are
  unchanged (`jit_run_model` 35/35, `jit_compute_and_gather` 34/34,
  `jit_run_compute_logits` 34/34); both semantic traces contain 92 packets,
  8 tracks, and 84 begin/end events with the same labelled operations.  This
  rules out a changed compiled operator graph as the primary one-host benefit;
  the host/scheduler/prefill explanation is an inference, not a localization.
- Next action: user runs G-E as an APC-only canary on the target topology; the
  agent evaluates its A-B/B-C hashes and writes the verdict.  Do not promote
  the flag or begin Phase4 before that gate.
- Rollback: leave `CANON_VLLM_ENABLE_PREFIX_CACHING` absent/empty/0 and stop
  invoking the phase3 runner.
