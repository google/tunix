# P38.2o one-host seam-observer gate — 2026-08-15 UTC

## Scope

Real Qwen3-8B DP1xTP4 FrozenLake diagnostic on local v5p. This is an
observer-neutrality and packaging gate, not a Pathways carrier verdict.

## Runs

- observer off: label `p38_2o_seam_r1`, state
  `/mnt/disks/tunix-data/logp_probe_1host/p38_incident_p38_2o_seam_r1_off`;
- layer observer on: state
  `/mnt/disks/tunix-data/logp_probe_1host/p38_incident_p38_2o_seam_r1_seam-layer`.

Both runs completed three diagnostic rounds with backward 0 and optimizer
commits 0. A-B and B-C were exact locally. The layer observer emitted 130 valid
records in diagnostic rounds 0 and 2; round 1 had no selected row in the
registered 1400..3072 source-position band.

## Endpoint-neutrality verdict

`classify_p38_seam_neutrality.py` returned
`observer_endpoint_bitwise_neutral` PASS. For every round, observer-off/on had
identical token IDs, action mask, S_decode, S_prefill, and T_old masked hashes.
Token hashes were:

- round 0: `eb479345701757495e609bdbe7f7dbb5617572485819e8f889295089bd78feeb`;
- round 1: `c6e77abe7192f54b5fa2669103dfcefc9b3aa66f276c5caae23e3f7f8cc8bb8a`;
- round 2: `de7198724e01f1c77dd949ca2ec343be473f9f2faa8d0487314c7e3cab2a625c`.

Endpoint hashes were:

- round 0: `c133007724cc7c41c04e2ecd23493972f5354f71c61f87bad3c7e072866e956c`;
- round 1: `f6ffb6c2aff30552dd7d2d8c830bf3708f978f959e7a65c4c4b5670ae2d364d5`;
- round 2: `b5a12c7e99a3b803b61504f31819bf6abe8c48c809f44a5fb536192392c520c3`.

The local data mount became read-only after the run, so the classifier PASS was
printed to stdout rather than written back under that mount. This is local
artifact-storage behavior, not a model or observer failure.

## Packaging gates

- Qwen3-1.7B and Qwen3-8B pinned overlays: all 31 manifest entries verified;
- runner tests: 29 PASS per overlay;
- seam capture/classifier/neutrality unit suites: PASS;
- controlled-exit/GCS postflight with seam classification: PASS.

## Claim ceiling

The observer is endpoint-neutral on one-host v5p and ready for a source-pinned
Pathways layer run. Local A-B exactness does not locate the production carrier.
No layer, RoPE site, or repair is selected by this artifact.
