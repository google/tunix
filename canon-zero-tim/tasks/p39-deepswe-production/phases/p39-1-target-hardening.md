# P39.1 target hardening

Status: local gates pass; target not run.

## Deliverable

Make the first 4x8x8 DeepSWE `backward-no-commit` attempt self-identifying
and fail-closed before any numerical boundary is interpreted:

1. route P38 and P39 operators to separate handoffs;
2. serialize ambiguous source-SHA labels as explicit YAML strings;
3. compare all mapped trainer-anchor and live rollout-engine leaves bitwise on
   device before every A/B/C rescore;
4. persist exactly one weight-attestation record per update and require it in
   the P34 classifier;
5. preserve the weight evidence in stdout when a later gate exits nonzero.

The weight gate reuses `attest_actor_anchor_matches_engine`. It transfers only
the reduced verdict and metadata to the host; model leaves are not dumped.

## Local exit gates

- `P34_STATIC_PASS suites=7`
- `P34_TOXIC_SHA_ROUNDTRIP_PASS type=str value=022893e2`
- `P34_EXACT_IMAGE_CPU_PASS unit_cases=54 pallas_cases=1 contract_cases=5 scheduler_cases=1 overlay=qwen32b`
- `git diff --check`, Python compilation and shell syntax pass

## Target gate

Render only `backward-no-commit` from the final published
`yuxzhang/canon-zero-tim` SHA. Target promotion requires Attempt 0, one exact
weight record, one exact pre-alignment record, four exact post-backward
records, deterministic repeated gradients and zero optimizer commits.

## Claim boundary

Local completion does not prove Qwen3-32B numerical alignment, backward
correctness, memory fit or trainability on Pathways. The target remains
`TARGET NOT RUN`.

## Rollback

Do not apply P34, or revert the additive hardening commit. Existing P33 launch
profiles and the default-off P34 path remain unchanged.
