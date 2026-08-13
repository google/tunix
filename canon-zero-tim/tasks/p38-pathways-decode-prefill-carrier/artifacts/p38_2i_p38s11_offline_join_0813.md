# P38s11 offline exact-join audit

Date: 2026-08-13 UTC

Inputs:

- `debug_logs/p38s11_mismatch_capsule.npz`
- `debug_logs/p38s11_serving_capture.tar`
- `debug_logs/p38_p38s11_frozenlake_full_coverage.raw.log`

The capsule token streams were joined to each serving request only when the
captured `token_ids` were an exact array prefix and the little-endian int64
SHA-256 matched. The resulting request mappings are:

| source row | sequence | request | DP rank | computed | block IDs |
|---:|---:|---|---:|---:|---|
| 199 | 0 | `372-9b9cf482` | 14 | 1179 | `45,43,42,41,40` |
| 206 | 0 | `390-9ab9fff6` | 15 | 1230 | `62,61,60,59,33` |
| 199 | 1 | `372-9b9cf482` | 14 | 1216 | `45,43,42,41,40` |
| 206 | 1 | `390-9ab9fff6` | 15 | 1267 | `62,61,60,59,33` |
| 199 | 2 | `529-ac6158ef` | 2 | 1348 | `65,60,14,72,73,75` |
| 206 | 2 | `532-baab38d4` | 3 | 1463 | `34,35,51,1,2,18` |

The audit establishes that stable joins are possible without a new rollout
and that one serving snapshot may legitimately join multiple selected rows.
It does not capture either row at its mismatch time, does not attest KV page
contents, and does not prove page ownership causality. The remaining five
selected red rows were absent from the two-row P38s11 capsule, which is why
P38.2i expands the bounded capsule to eight rows and adds a per-request journal.
