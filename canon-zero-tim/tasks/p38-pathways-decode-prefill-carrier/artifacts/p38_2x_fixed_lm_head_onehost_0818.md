# P38.2x fixed lm-head one-host receipt — 2026-08-18

Scope: construction-only on the directly attached four-device v5p host. This
is not Pathways repair evidence.

## Construction

- real Qwen3-8B BF16 `lm_head.weight`: `[4096,151936]`;
- TP4 vocabulary sharding: local N37984;
- semantic rows: M16 decode and M256 prefill;
- fixed Pallas shape in both arms: M256/K4096/N38144;
- tiles: BM128/BN256/BK256; and
- four deterministic BF16 hidden-input seeds.

## Result

- fixed M16 versus the first 16 fixed M256 rows: 0 differing elements and
  `max_abs=0.0` for 4/4 seeds;
- stock M16 versus stock M256: also exact for 4/4 seeds, as in P38.2w;
- fixed versus stock selected rows: 249 / 211 / 268 / 219 differing elements,
  proving that the fixed candidate executed and changed the absolute program
  result rather than acting as an empty flag;
- one-bit negative: exactly 1 differing element; and
- verdict: `FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS`.

The decode and prefill outer StableHLO modules differ because their semantic
output M differs, but both receipts contain a custom call and both PATHTRACEs
name the same fixed Pallas M/K/N and tile sizes. The causal target remains one
P38s23 three-round arm.

## Local artifact hashes

```text
11895a58739e9a32b186fa66aa9a10ac99280fdd3bf7d0ffd89c50d0ecc0b53c  p38_fixed_lm_head_p38x_dev1_0818.raw.log
24d3ee81496771956b983617352e20d2f4c4ff266cb2ae4097d867c5e694a742  p38_fixed_lm_head_p38x_dev1_0818.result.json
```

Local paths are under `/mnt/disks/tunix-data/logp_probe_1host/`. The source
worktree was dirty by construction, so the result is tied to the checked-in
script hashes printed in the raw log and must be repeated from a published SHA
only if publication review changes executable files.

## Claim ceiling

This admits the fixed operator construction and its negative control only. It
does not admit A-B repair on 64 TPU, backward, optimizer, training, or a
production default.
