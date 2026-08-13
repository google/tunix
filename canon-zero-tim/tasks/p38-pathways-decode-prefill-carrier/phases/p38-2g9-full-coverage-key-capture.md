# P38.2g9: full-coverage alignment and typed-key capture

- Status: implemented locally; target P38s11 not run.

## Evidence correction

P38s10 is a valid bitwise PASS only for the subset it measured. It processed
four prompts x eight generations (`32` trajectories, `N_action=2731`) and
reported exact A-B and B-C. Historical stock P38s1/P38s2 processed the full
32 prompts x eight generations (`256` trajectories, about 46k actions) and
placed their sparse A-B mismatches predominantly in high global rows outside
the P38s10 subset. P38s10 therefore does not prove that the carrier vanished
and must not be called a repair.

P38s10 also emitted three
`CANON_P38_SERVING_CAPTURE_ERROR` records. The failure was mechanical: JAX
typed PRNG keys intentionally cannot be converted directly to NumPy. No
serving archive was admitted, so D1 remains incomplete.

## Deliverable

1. Keep the producer unit at four prompts x eight generations. Every unit is
   32 trajectories and remains divisible by DP16.
2. In P38 precheck-only mode, let the consumer wait for all 32 prompt groups
   (eight producer units) before one `_process_results` call. Alignment then
   covers all 256 trajectories and still stops before backward.
3. Reject a partial consumer tail rather than producing another subset PASS.
4. Serialize typed PRNG keys only in the capture copy through
   `jax.random.key_data`, recording the logical key dtype and implementation.
   The live sampling key, RNG splitting, model inputs, and numerical program
   are unchanged.
5. Make postflight require exactly one full-coverage marker and zero internal
   capture-error markers.

## Local gates

- learner contract: full coverage accepted; subset geometry and partial tail
  rejected; the normal non-P38 partial-tail behavior remains unchanged;
- renderer contract: 32 prompts, eight four-prompt units, eight generations,
  DP16, 256 covered trajectories;
- exact image: both overlays contain the typed-key serializer, preserve key
  bits, and match their manifests;
- postflight: a missing coverage marker or any capture error is rejected;
- shell/static checks and `git diff --check` pass.

## Target gate

One stock-only P38s11 must print the full-coverage marker, contain no capture
errors, cover 256 trajectories, return four serving records, and reach the
classifier/archive postflight. Its A-B result may be red or exact; either is
data. A full-coverage exact result would require repetition before any repair
claim because the rollout is stochastic. A red result proceeds to exact E0
replay and first-divergence localization.

## Claim ceiling and rollback

This phase fixes evidence coverage and capture serialization. It changes no
canonical numerical kernel and proves no numerical repair. Rollback is to
leave `CANON_P38_PRECHECK_ONLY`/capture variables unset or revert this phase;
ordinary training consumer behavior is unchanged.
