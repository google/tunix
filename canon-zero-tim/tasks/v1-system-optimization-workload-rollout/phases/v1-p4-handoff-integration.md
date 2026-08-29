# V1.P4 — integrate the system bundle into canonical workload handoffs

## Status

`DONE / OFFLINE GREEN / TARGET NOT RUN`

## Objective

Make the owning FrozenLake and DeepSWE handoffs select the renderers that
carry the reviewed system-optimization tuple. Prevent an operator from using
the historical P45 resident carrier, baseline P57 Zero profile, old DeepSWE
YAML, or a diagnostic carrier as a production full-training substitute.

## Routing decision

| Workload | Authoritative render-only entry | Preserved exclusion |
|---|---|---|
| FrozenLake P45 strict Zero/300 | `prepare_p67_frozenlake_two_full_wave.sh` | legacy P45 resident/checkpoint/warning-only carrier |
| FrozenLake M15/main strict Zero/300 | same two-full wrapper | baseline P57 Zero and APC diagnostics |
| DeepSWE Qwen3-4B Zero/full/HP/1,000 | `prepare_deepswe_zero_hp_full.sh` | Native, Native+IS, ordinary Zero, three-update, checked-VMA and seam diagnostics |

The DeepSWE route is prepared but not launch-ready. P58.19 seam localization
remains the active numerical queue; the handoff explicitly preserves that
block rather than promoting offline wiring to a training authorization.

## Files integrated

- `tasks/v1-phase4-three-full-recipes/HANDOFF.md` and `RUNBOOK.md`;
- `tasks/p57-frozenlake-tim-causal-study/HANDOFF.md` and `RUNBOOK.md`;
- `tasks/p45-frozenlake-dp8-tp8-resident/HANDOFF.md` and the legacy cluster
  runbook redirect;
- `tasks/p58-deepswe-native-zero-comparison/HANDOFF.md` and
  `cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## Gate and result

- handoff-routing regression: 4/4 host and 4/4 pinned-image CPU;
- FrozenLake adjacent renderer: 5/5;
- P58 adjacent renderer: 31/31;
- deterministic flag audit: 2/2;
- Python compile, Bash syntax, and diff hygiene: PASS.

No target was launched. Commit/push, image publication, server dry-run,
Kubernetes apply, and target monitoring remain separately approval-gated.
