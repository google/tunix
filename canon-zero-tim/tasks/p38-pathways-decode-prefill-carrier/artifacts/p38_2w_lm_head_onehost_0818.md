# P38.2w real-weight lm-head one-host receipt — 2026-08-18

## Scope

Construction-only operator screen on the local four-device v5p. The probe
loaded the real Qwen3-8B `lm_head.weight` from snapshot
`b968826d9c46dd6066d109eabc6255188de91218`, sharded vocabulary over TP4,
and compared identical first-16 BF16 hidden rows under local M=16 and M=256.
It tested default einsum and explicit `BF16_BF16_F32` for four deterministic
input seeds.

## Result

```text
verdict=BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE
default M16 vs M256: 0 differing elements, max_abs=0.0 (4/4 seeds)
algorithm M16 vs M256: 0 differing elements, max_abs=0.0 (4/4 seeds)
default vs algorithm at M16: 0 differing elements (4/4 seeds)
default vs algorithm at M256 selected rows: 0 differing elements (4/4 seeds)
negative control: 1 differing element
```

The StableHLO intervention is real rather than silently ignored. Default dots
contain no algorithm attribute; algorithm dots contain:

```text
lhs_precision_type=bf16
rhs_precision_type=bf16
accumulation_type=f32
allow_imprecise_accumulation=false
```

The default and algorithm StableHLO SHA values differ at both M=16 and M=256.
Numerical equality on these one-host inputs therefore means only that the
screen did not expose the production carrier. It is not evidence that the
Pathways lm-head programs are equal or that the preset repairs them.

## Evidence hashes

```text
raw.log   43f72e3cc168abee8e1d1838923c44ac7573099fc8e6ae651fc9ec977aa1a813
result    05ac723258938cac7b5e5d6280e68751e7819c7dea33cdabe32ff95200dcb1cd
runner    8ea6bd0a3dd0e2d8d01d7187c89a0e0af19c8ff144ca9b1894e11ea42d5db0f4
probe     06e3c65ae6206fcafd8600c3c61fa618c8acb482bcc609558b80719084dbf0c3
```

Local paths remain under
`/mnt/disks/tunix-data/logp_probe_1host/p38_lm_head_p38_2w_20260818b.*`.

## Adjacent gates

- verdict unit test: 1/1 PASS;
- pinned-image focused renderer plus verdict tests: 16/16 PASS;
- pinned-image complete P38 serving suite: 93/93 PASS;
- shell syntax and Python compilation: PASS.

## Claim ceiling

This is a real-weight, real-shape, real-sharding one-host operator screen. It
does not reproduce Pathways compilation, the production final-hidden values,
or the complete decode/prefill envelopes. The only admitted next target is the
single-variable P38s22 arm in `P38S22_RUNBOOK.md`.
