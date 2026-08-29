# P1 — GSM8K Native/mismatch full control

## Goal

Prepare one immutable render-only DP16xTP4 GSM8K full-training control that
reuses the repository's P56 stock vanilla numerical path and is easy
to compare with the optimized Phase4 Zero full run in the same W&B project.

## Source evidence

- Executed stock oracle:
  `tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_common.sh`,
  Native arm. Its classifier requires `CANON_GSM8K_VANILLA=1`, no P32/P59/G6,
  no canonical adapter/program marker, and no alignment verdict.
- P33 supplies only the common `_gsm8k_command(200)` and registered full
  restart policy; its canonical baseline profile is not the Native runtime.
- Optimized carrier:
  `tasks/v1-phase4-three-full-recipes/scripts/render_gsm8k_full_dp16tp4_p74.py`.
- Shared scientific command: `render_p33_jobsets.py::_gsm8k_command(200)`.
- Shared deterministic data seed: `examples/math_gsm8k/qwen3_grpo_demo.py`
  declares `SEED = 42` and uses it for the shuffled training dataset.
- Shared W&B project/group: both the stock Native and V1-HP profiles resolve to
  `zero-tim-gsm8k-dp16-tp4` / `qwen3-1p7b-dp16-tp4`.

## Implementation decision

Load the existing P33 full spec only to inherit its command and key, then
replace the profile with the dedicated stock profile and use the untreated lm
head. After P33 performs its structural validation, remove every canonical
alignment/P32/P59/V1 selector and evidence path, zero the three P33 admission
gates, add the signed P56 vanilla selector, and remove the proxy compiler
precision pin.

The entrypoint recognizes only this exact stock identity, skips the canonical
install/overlay path, verifies all six stock engine hashes, imports the normal
driver, and emits a stock-path receipt. There is no alignment observer.

## Offline gates

1. Render one fresh manifest and an immutable hash index.
2. Prove Native and Zero have byte-identical `CANON_RUN_CMD`, DP/TP geometry,
   full stage, commit mode, optimizer placement, project, group, and source
   SHA. The lm-head difference is intentional: stock Native versus fixed Zero.
3. Prove every Zero numerical selector is absent from raw Native, source the
   real profile through `00_env.sh`, and inject P32/P71 mixed-arm negatives.
4. Prove the wrapper rejects a dirty/wrong source and reused output and never
   executes `kubectl`.
5. In the pinned image, verify the six stock engine hashes and import the real
   normal-training driver without a canonical overlay.
6. Run adjacent Phase4 tests, flag audit, syntax, and diff hygiene.

## Target acceptance (not executed in this phase)

- Renderer receipt and source SHA match the reviewed manifest.
- W&B project/group match the Zero arm; the Native-specific run name remains
  distinct.
- Environment receipts show stock profile, P56 vanilla, P32 absent,
  canonical engine/alignment/P59/V1 off, DP16xTP4, and 200 steps.
- Stock preflight reports six unchanged engine files and
  `canonical_overlay=absent alignment=off`.
- Training emits the two P56 vanilla receipts and no canonical adapter,
  Zero backward program, or alignment verdict.
- Performance is reported from `p32_vag_reverse` and whole-step wall time, not
  the gradient-accumulator sub-timer.

## Result log

`OFFLINE COMPLETE / TARGET NOT RUN`.

The first warning-only canonical draft is rejected and its earlier validation
counts are non-authoritative. The corrected host task suite passed with nine
tests and one pinned-image-only skip; the pinned image passed all ten Native
contracts plus one optimized-Zero neighbor. FrozenLake and DeepSWE pinned
image gates also passed, and the aggregate render returned
`FOUR_CARRIER_RENDER_PASS manifests=4 optimized_zero=3 stock_native=1`.

No clean post-change SHA, Kubernetes server dry-run/apply, target training,
target XProf/performance, convergence, commit, push, or image publication was
executed.
