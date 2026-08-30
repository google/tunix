# GSM8K Native/mismatch full-control log

## 2026-08-29 — task bound and source audit complete

> Superseded: this first audit incorrectly treated the warning-only P33
> canonical program as Native. It is retained as a decision record, not an
> accepted conclusion.

- Bound the task to source `d4128940464054866d466a6cce5adf326f513caf`
  in the named P57 worktree and read the repository rules plus canonical
  branch, flag, and phase-workflow skills.
- Located the existing original full carrier in
  `cluster/render_p33_jobsets.py::_SPECS[key=gsm8k-full]`. It is DP16xTP4,
  200 steps, resident optimizer, fixed lm head, warning-only GSM8K alignment,
  `CANON_V1_HP_FULL=0`, and rank-parallel backward off.
- Located the optimized comparison carrier in the Phase4 single GSM8K P74
  renderer. Both arms call the exact same `_gsm8k_command(200)` and the driver
  fixes `SEED=42`.
- Verified both profiles route to W&B project
  `zero-tim-gsm8k-dp16-tp4` and group `qwen3-1p7b-dp16-tp4`. The control will
  preserve those values and distinguish the arm through its JobSet/run name
  and labels.
- Decision: wrap the existing P33 spec instead of inventing a new Native
  program. The renderer may replace only `job_prefix`; it must leave the key
  `gsm8k-full` intact so the original full restart policy is preserved.
- No launch, commit, push, image publication, or remote mutation was
  performed.

## 2026-08-29 — implementation and offline handoff complete

> Rejected: these results exercised the wrong warning-only canonical arm and
> cannot validate the stock Native control requested by the user.

- Added a dedicated renderer that locates the registered P33 `gsm8k-full`
  spec, fail-closes on any scientific drift, and changes only the JobSet/run
  prefix. The original `gsm8k-full` key remains, preserving its registered
  restart policy.
- Added comparison labels, an immutable manifest index, and a clean-SHA,
  fresh-output render-only wrapper. The wrapper prints one unpiped apply
  command but never executes it.
- Added seven task tests. They prove the Native/Zero training commands and
  core full-run geometry are identical, source both real profiles to prove the
  same W&B project/group, pin the driver seed, reject every Zero-only selector,
  and exercise a live injected-selector negative control.
- Host results: task 7/7, system-optimization handoff 4/4, adjacent Phase4
  9/9, and deterministic flag audit 2/2. Python/Bash syntax and
  `git diff --check` passed.
- Pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  passed the read-only task gate plus the optimized Zero neighbor test and
  emitted `V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS`.
- Added the self-contained task `HANDOFF.md` and routed the Phase4 handoff and
  runbook to both matched render-only entry points.
- No clean post-change source SHA exists yet, so the success path of the
  clean-worktree wrapper was not invoked. No TPU/Kubernetes launch, target
  XProf, target performance/convergence result, commit, push, or image
  publication occurred.

## 2026-08-29 — semantic correction to true stock Native

- Compared the rejected draft against the executed P56/P60 one-host Native
  carrier. The decisive contract is stock vanilla with no P32 workload,
  P59/G6, canonical adapter/program, or alignment verdict.
- Added a dedicated Native profile that removes every presence-sensitive
  canonical engine selector, zeros shared boolean paths, uses the untreated lm
  head, and retains only ordinary training/orchestration settings.
- Changed the renderer to strip all raw Zero selectors and report paths, zero
  P32/P33 admission gates, add `CANON_GSM8K_VANILLA=1`, and remove the proxy
  excess-precision pin.
- Added an exact entrypoint branch that skips canonical installation, verifies
  six stock engine hashes, imports the normal GSM8K driver, and emits
  `GSM8K_NATIVE_STOCK_PATH ... canonical_overlay=skipped alignment=off`.
- Extended W&B routing so the vanilla arm uses the same project/group and its
  distinct JobSet run name without activating P32.
- Real `00_env.sh` reload passes the corrected stock contract. Injecting
  `CANON_P32_WORKLOAD=gsm8k` is rejected as a mixed-arm caller contradiction.
- No TPU/Kubernetes launch, commit, push, or image publication was performed.

## 2026-08-29 — corrected stock carrier and four-workload offline gates pass

- The corrected host task suite passed nine tests with one pinned-image-only
  skip. The pinned image then ran that skipped stock preflight and passed all
  ten Native contracts plus one optimized-Zero neighbor.
- The existing GSM8K one-host XProf contract passed 25/25, proving the P56
  Native oracle still accepts the shared driver W&B routing change.
- The complete FrozenLake pinned-image gate passed with
  `frozenlake_system_optimization=1`; the complete DeepSWE pinned-image gate
  passed with `zero_hp_full=1 system_optimization=1 regressions=1`.
- Focused shared/FrozenLake/DeepSWE/flag tests passed 4/4, 5/5, 31/31, and
  2/2 respectively. Python and Bash syntax checks passed.
- A final fail-fast aggregate render produced P45 Zero/full, M15/main
  Zero/full, DeepSWE Zero/full/HP, and GSM8K stock Native/full and emitted
  `FOUR_CARRIER_RENDER_PASS manifests=4 optimized_zero=3 stock_native=1`.
- Two preceding aggregate checks were rejected as harness errors: one used a
  path instead of the registered short campaign tag, and one searched an
  adjacent YAML name/value pair as if it were one line. Neither is counted as
  a product failure or PASS; the final fail-fast check is authoritative.
- No clean post-change source exists, so the production wrappers' positive
  clean-worktree path remains unrun. No Kubernetes server dry-run/apply,
  TPU target training, target XProf/performance, convergence, W&B comparison,
  commit, push, or image publication occurred.

## 2026-08-30 — Attempt 03 Splash input-layout repair implemented

- Fast-forwarded the clean task worktree to published source
  `2af1197f4d0bb604d7c423f703251fc5187b4594` and evaluated the immutable
  Attempt 03 report and raw error. The first failing boundary is the learner's
  Qwen3 Splash `shard_map` admission: a replicated `int8[4,8,8]` kernel-mask
  leaf does not match the declared TP input `P('model', None, None)`. No model
  math or optimizer update ran in that attempt.
- Added an Explicit-mesh-only helper that maps the real Splash kernel pytree
  to the `manual_sharding_spec` it already declares. Auto meshes return the
  original kernel object, so their historical program is untouched. No loss,
  attention math, precision, gradient, optimizer, profile, renderer, YAML, or
  Zero selector changed.
- Added a forced eight-device CPU gate using a real Splash kernel leaf. The
  negative reproduces the production `in_specs ... does not match` error; the
  repaired leaf passes the same `shard_map`, every kernel leaf has the
  normalized intended spec, and all values are byte-identical. The Auto-mesh
  negative proves object identity.
- The first pinned-image run stopped on a test-only assertion that compared
  equivalent short and rank-normalized `PartitionSpec` spellings. The product
  helper had already produced the intended normalized placement. The assertion
  was corrected to compare normalized specs, then the complete gate passed:
  Native 10/10, Qwen sharding 9/9, Zero neighbor 1/1.
- This establishes `PINNED-IMAGE PASS`, not target success. This entry ships
  with the repair CL; no post-fix TPU run, optimizer commit, image publication,
  or Kubernetes mutation occurred during validation.

## 2026-08-30 — Attempts 04/05 and Auto/Manual output-sharding repair

- Pulled exact operator tip
  `89ef0ad567d5abe33074a53c6655a6b8bc80cf6e` and verified both immutable
  incident packages. Attempt 04 crossed Splash and failed at an Explicit-axis
  doubly-sharded output projection. Attempt 05 used Native Auto axes, completed
  rollout at 5,668.9 tokens/s, then failed before trainer math because the
  embedder passed a spec naming Auto axes to `.get(out_sharding=...)`.
- `_activation_out_sharding()` now returns a named placement only when every
  axis referenced by the spec is Explicit. Auto/Manual named axes return
  `None`; unknown names remain fail-closed. Explicit projection/gather behavior
  from Attempt 04 remains intact.
- Added Auto and Manual controls plus a mixed-axis referenced-name control.
  The first exact-image run exposed a pre-existing harness omission:
  `absltest.main()` occurred before the Attempt-04 class, so only 11 of 13
  tests ran even though the terminal count was static. The entrypoint moved to
  EOF and the authoritative rerun executed 13/13.
- Final gates: Native image contracts 12/12, Qwen sharding 13/13, Zero neighbor
  1/1, host P34 static ten suites, and flag audit 409/409 with
  `changed_names=0`. Terminal:
  `V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS native_contract=12
  qwen_sharding=13 auto_out_sharding=2 zero_neighbor=1`.
- No model math, partition spec, mesh shape, loss, precision, gradient,
  optimizer, flag, profile, arm identity, or update horizon changed. No
  repaired target, optimizer commit, commit/push, image publication,
  Kubernetes mutation, or TPU launch occurred.

## 2026-08-30 — P5 post-rebase admission

- Fast-forwarded the operator worktree to
  `98d102eb27fe05fcee327688d0aa6d236b32be4a` and reran the complete pinned
  image gate. Native contracts pass 12/12, the Qwen suite executes and passes
  all 13/13 tests, and the Zero neighbor passes 1/1.
- Exact terminal:
  `V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS native_contract=12
  qwen_sharding=13 auto_out_sharding=2 zero_neighbor=1`.
- This is transcript-only construction evidence. No repaired DP16xTP4 target,
  optimizer commit, commit/push, image publication, Kubernetes mutation, or
  TPU launch occurred.
