# P38.2r same-source one-host neutrality receipt

Date: 2026-08-16 UTC

Executable source: `ae63d44edc67cfcd5b19d34abc82feb681284c67`

Worktree diff SHA-256 in both arms:
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The local v5p ran two independent Qwen3-8B `DP1xTP4` arms for three frozen
rounds each with prefix cache disabled:

- observer off: `PASS mode=off rounds=3 backward=0 optimizer_commits=0`;
- combined observer: `SEAM_LAYER_PASS records=130`,
  `TERMINAL_TAIL_PASS records=130`, and
  `PASS mode=seam-tail rounds=3 backward=0 optimizer_commits=0`.

The combined arm observed eligible rows in rounds 0 and 2. Round 1 did not
reach the configured deep band; its endpoint remains part of the three-round
neutrality comparison.

`classify_p38_seam_neutrality.py` returned `PASS` and
`observer_endpoint_bitwise_neutral`. For every round, the complete alignment
record excluding only its wall-clock timestamp was equal. This includes
`N_action`, action geometry, token/action-mask/full-array hashes, all three
action-masked endpoint hashes, both byte/element boundary contracts, and the
admission verdict.

Input report SHA-256 values:

- off: `3701d8c303c2665af7a6bde5d3f5845128d708df73b6299d71e0c8e1a26431bf`;
- seam-tail: `7a1eb5a2dbc290a1bba92a154d561079578884e3363bcd8fa7cffedf184498b6`.

No mismatch capsules exist because all local A-B action boundaries were
exact. That absence is expected and is not missing evidence. The native
alignment hashes are the byte-level endpoint contract for an exact run.

Verdict: the combined observer is endpoint-neutral on one-host v5p. Exactly
one stock P38s18r target run is authorized from the executable source above.
This receipt does not prove the 64-TPU carrier's location.
