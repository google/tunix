# V1.P1 — roll P74-era system optimization into FrozenLake and DeepSWE full

## Status

`DONE / OFFLINE GREEN / TARGET NOT RUN`

## Objective

Apply the already reviewed P74-era reverse-path system optimization to the two
registered FrozenLake strict full workloads and the registered DeepSWE
Qwen3-4B strict Zero-HP full workload. Reuse one exact selector tuple, preserve
the checked-VMA safety path, and prevent admission from leaking into Native,
diagnostic, non-HP, or neighboring profiles.

## Evidence and source coordinates

- P74 checked-VMA dispatch repair: `tunix/rl/canonical_qwen3_adapter.py`,
  selected by the existing checked-VMA path and requiring no new flag.
- Current Phase4 tuple source:
  `tasks/v1-phase4-three-full-recipes/scripts/render_three_full_recipes.py`,
  `_optimization_additions`.
- FrozenLake production renderer:
  `tasks/v1-phase4-three-full-recipes/scripts/render_p67_frozenlake_two_full_recipes.py`.
- DeepSWE production renderer and admission validator:
  `cluster/render_p58_deepswe_tim.py`.
- Workload profiles:
  `cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env` and
  `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env`.
- Authoritative persistence boundary: `cluster/steps/00_env.sh`.

## Mechanism

The P74 fix eliminates the checked-VMA seed-to-head host round trip whenever
the registered checked-VMA arm is active. The remaining bundle reduces host
receipt synchronization and consolidates each chunk's forward tape into
`CANON_P71_SCAN=fwd`; none of these selectors changes gradient arithmetic.
The DP collective reducer is excluded because its DP8/target oracle is still
missing.

## Exact rollout matrix

| Renderer arm | Receipt tuple | P71 fwd | checked-VMA/P67 | Collective reducer |
|---|---:|---:|---:|---:|
| FrozenLake P45 full HP | on | on | on | absent |
| FrozenLake M15/main full HP | on | on | on | absent |
| DeepSWE Zero full HP | on | on | on | absent |
| DeepSWE Native / Native+IS / ordinary Zero | absent | absent | historical | absent |
| DeepSWE checked-VMA or seam diagnostic | absent | absent | diagnostic tuple only | absent |

## Implementation

1. Add a small `cluster` helper returning an immutable copy of the exact
   performance selector tuple and validating allowed workload keys.
2. Replace the private Phase4 tuple copy with the helper.
3. Extend the FrozenLake two-full renderer and its receipt validator with the
   helper output.
4. Extend only P58 production Zero-HP full with the helper output; validate
   positive values and absence on every neighboring arm.
5. Add focused tests for render, authoritative env reload, profile isolation,
   forbidden reducer absence, and helper immutability/fail-closed behavior.

## Gates

- Focused host tests: all positive values exact, all negative arms absent.
- Real environment: rendered env survives `00_env.sh`, persisted `env.sh`, and
  reload without tuple loss or leakage.
- Adjacent suites: FrozenLake, P58, Phase4, flag registry.
- Exact-image: FrozenLake and P58 aggregate gates with read-only mount.
- Target: separately approved fresh DP8xTP8 runs; not part of source
  construction and not implied by offline passes.

## Result log

- Added `cluster/v1_full_system_optimization.py` as the exact source of truth
  for the registered tuple. Its API returns a fresh copy and rejects
  unregistered workload identities.
- FrozenLake P45/M15 and DeepSWE Zero/full/HP now receive the exact tuple.
  Render and runtime validators reject partial values, neighbor leakage, and
  any presence of `CANON_DP_COLLECTIVE_REDUCE`.
- DeepSWE Native, Native+IS, ordinary Zero, three-update, checked-VMA
  diagnostic, and seam-localization arms remain outside the production tuple.
- Focused host suites: 48/48 passed. Flag-registry audit: 2/2 passed. P70/P71
  pinned-image mechanism suite: 40 passed, 3 skipped because no TPU device was
  exposed to the CPU-only container.
- FrozenLake aggregate exact-image gate: exit 0 with
  `V1_HP_EXACT_IMAGE_PASS ... frozenlake_system_optimization=1`.
- DeepSWE aggregate exact-image gate: exit 0 with
  `P58_EXACT_IMAGE_CPU_PASS ... system_optimization=1`.
- The complete command ledger is in `../validation.log`; render-only operator
  commands and required target receipts are in `../RUNBOOK.md`.
- FrozenLake/DeepSWE DP8xTP8 target performance and convergence are
  `NOT RUN`. No TPU/Kubernetes launch, commit, push, or remote mutation was
  performed.
