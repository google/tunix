Prepare a stock GSM8K Native full control

Reuse the registered P33 `gsm8k-full` command and restart policy as a
separately named Native/mismatch DP16xTP4 control for the optimized GSM8K
V1-HP full recipe. Keep the model, data seed, 200-step horizon, optimizer
placement, training command, and W&B project/group identical between the
arms, while retaining the stock lm head and ordinary Tunix trainer in Native.

Add a fail-closed renderer, dedicated stock profile, stock-engine preflight,
and clean-SHA render-only wrapper. Native selects `CANON_GSM8K_VANILLA=1`,
keeps `CANON_P32_WORKLOAD` absent, skips the canonical engine overlay and
alignment observer, and rejects checked-VMA, first-update, P70 receipt, P71
scan, P67, P63, and collective-reducer selectors. Add immutable manifest
receipts, live mixed-arm negatives, read-only exact-image gates, and operator
handoffs for the four full-training carriers.

Verified by running:

- `python3 canon-zero-tim/tests/v1_gsm8k_native_full/test_renderer.py` (9 pass,
  1 pinned-image-only skip on host; 10/10 in the pinned image)
- `python3 canon-zero-tim/tests/v1_system_optimization/test_workload_rollout.py` (4/4)
- `python3 canon-zero-tim/tests/v1_phase4/test_p67_frozenlake_two_full_renderer.py` (5/5)
- `python3 canon-zero-tim/tests/p58_deepswe_native_zero/test_renderer.py` (31/31)
- `bash canon-zero-tim/tests/v1_gsm8k_xprof_pair/run_cpu.sh` (25/25)
- `python3 canon-zero-tim/tests/manage_canon_flags/test_audit_flag_registry.py` (2/2)
- the pinned-image Native, FrozenLake, and DeepSWE exact-image gates
- a four-manifest aggregate render (3 optimized Zero, 1 stock Native)
- Python/Bash syntax and `git diff --check`

Not verified because no target run was launched: DP8xTP8 FrozenLake/DeepSWE
or DP16xTP4 GSM8K full training, target performance, convergence, target
XProf, Kubernetes server dry-run/apply, and live W&B comparison.

本方案的缺点

This adds a second operator-facing GSM8K carrier and receipt path that must
remain in sync with the registered P33 scientific command. The renderer
fail-closes on drift, but any intentional future command change requires a
reviewed update here. Stock Native and strict Zero are matched configurations,
not guaranteed bitwise paired rollouts; Native deliberately does not receive
the Zero arm's safety/performance selectors, canonical lm head, or alignment
observer.
