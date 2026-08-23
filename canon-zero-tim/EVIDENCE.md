# Evidence

Every claim below names the artifact that carries it. `./verify_evidence.sh` checks that each
one still exists and still hashes to what was recorded — run it before citing anything.

Release artifacts live on the probe host under `/mnt/disks/tunix-data/logp_probe_1host`
(override with `CANON_ARTIFACT_ROOT`). Package-local cluster evidence is under `debug_logs/`.
Machine-readable lists: `evidence/artifacts.sha256` and
`evidence/package_artifacts.sha256`.

## What is signed

| Claim | Number | Artifact |
|---|---|---|
| Forward `A = B = C` bitwise at release depth | logits `0/303872`, hidden `0/10240` | `p19x6_t2decomp_n500_nl64.raw.log` |
| Forward-only and `value_and_grad` primal are the same program | THIRDPROG `0/303872`, 64 layers | `p19x6_t2decomp_n500_nl64.raw.log` |
| Step-0 exactness (four boundaries, ratio, clip/TIS) | 0 bytes; `w = r = w·r = 1`; clip/TIS `0` | `p19x6_xbucket_canon_256_1024.raw.log` |
| Gradient correctness of the non-standard pullback, in-domain | 20/20 records; worst route rel-L2 `3.59e-5`; `g[1]` fault rejected | `std_p22_t2a_candidate_r1_0804.raw.log`, `p22_t2a_candidate_r1_0804.json` |
| 200 optimizer steps of real GSM8K at zero TIM | 3200/3200 reports, `N_action = 2,745,204`, all boundaries 0 bytes, 200 finite nonzero gradients, 200 commit/sync | `p26_gsm8k_g3_0804_g3_r1_evidenceonly.{raw.log,alignment.jsonl,classification.json}` |
| The exact command that produced it | — | `p26_runner_freezes/run_p26_gsm8k_train_g3_0804_r1_evidenceonly.sh` |
| 64-chip single-slice Pathways bounded admission | Attempt 0; P1a PASS; P1 18/18 advisory dirty; P1b 4/4 bitwise with live gradients; toy T2 7/7 | `debug_logs/head_jax_tpu.{log,classification.json}` |

The training branch tip (`3a00d951`) is byte-identical to the sources that run recorded: all
25 files in its manifest match. See `docs/phase0.md` finding F4.

## Where the chain is weak

Stated plainly, because a reader who discovers this later will discount everything else.

**The 200-step run recorded SHA-256 for only 1 of the 28 engine-side runtime files.**
`tpu_runner_p21_l30.py` (`b8b1e118…`) is covered and verified. The other 27 — the shim chain,
the Pallas ops, the contracts, the attention wrapper — were not hashed by that run.

The substitute evidence is timestamps: all 28 files have mtime ≤ `2026-08-03 22:51`, while the
run's frozen runner has mtime `2026-08-04 17:48` and its log completed `2026-08-04 20:29`.
Every file predates the run by at least 19 hours and none was touched after.

**That is mtime-grade evidence, not cryptographic.** It is good enough to believe nothing
changed and not good enough to prove it. The fix is one line in the runner: extend its
`sha256sum` block to cover all six engine mount points and their dependency closure. Until
that is done and a fresh run is recorded, treat the engine side of the 200-step claim as
strongly-supported rather than proven.

## Pathways regime boundary (2026-08-10)

P36 established that every Pathways result recorded before `envon1` ran with
`--xla_allow_excess_precision=false` **undelivered** (client-env only; the server-side
compiler never saw it — KNOWN_FOOTGUNS #13). Those results remain valid **flag-off
baselines** and are not retroactively edited. From `envon1` onward the canonical Pathways
regime delivers the flag through the `pathways-proxy` container environment; renderers
enforce it and `tests/t0_cpu/test_cluster_contracts.py` locks both static manifests.
Flag-off and flag-on numbers must never be compared as if from one regime.

## What is *not* claimed

- **Convergence.** The 200-step campaign proves a stable training loop and exact numerics.
  Reward and solve rate moved; that is a trend over a noisy `n=32`, not convergence.
- **A model artifact.** That branch ran with checkpointing disabled by explicit choice. There
  is no resumable checkpoint, and the in-memory state left with the container.
- **Cross-bucket bitwise as a solved mechanism.** The gate is green *inside the pinned-bucket
  configuration*. The mechanism behind cross-bucket drift is still unknown; pinning avoids it
  rather than fixing it, and pinning is not a production posture (decode pays for a 256-token
  bucket regardless of how few tokens it has).
- **A full-model or training claim on another topology.** The signed release numbers above still
  come from one directly-attached 4-chip v5p host. A 64-chip single-slice Pathways Attempt 0
  passes bounded P1b/T2 target gates, but full-model initialization, segmented backward,
  optimizer commit, multi-slice behavior and training remain unverified — see
  `CLUSTER_ADMISSION.md`.
- **The production `nnx` training forward.** The signed `C` is the engine-module differentiable
  forward. The production tunix `nnx` path still differs by `0.0267`.

## Run index (2026-08 campaigns; canonical claim→run→path map)

Added 2026-08-16. New runs land under a per-run directory with their own `SHA256SUMS`
(packaging contract: `tasks/canon_system_redesign/phase1_run_contract.md` in the outer repo);
this table is the navigation layer. Spot-verified at basis `a94d6c0c`:
`p38s1` raw log `99166ff1…`, `p38s11` capsule `037f5b24…`, and `sha256sum -c` OK for
p38s12f / p38s15 / p38s16.

| Claim | Run | Path |
|---|---|---|
| At-scale stock carrier, first admitted full coverage | p38s11 | `debug_logs/p38s11_*` (log, capsule, jsonl, tar) |
| Exact-call joins + journal first production pass | p38s12(a-semantics) | `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s12b/` |
| Concurrency-32 exclusion (still red 11/46,390) | p38s12f | `…/evidence/p38s12f/` |
| Fingerprint revision (floor 1498, turn-0 cases) | p38s13a | `…/evidence/p38s13a/` (analysis-grade) |
| Terminal forensics ×3 rounds, 64 joined mismatches | p38s15 | `…/evidence/p38s15/` |
| Single-active fixed-M discriminator; golden incident call 4223 | p38s16 | `…/evidence/p38s16/` (classification absent → analysis-grade) |
| Live-KV fingerprint EQUAL on red row (corrected; initial "differs" withdrawn) | p38s17 | `…/evidence/p38s17/` + task `log.md` 2026-08-15 correction |
| Layer seam reduction sealed; ambiguous join on red row (analysis-grade partial) | p38s18l | `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s18l/` + GCS derived |
| Round-0 seam+tail corpus sealed but direct classifier rejects duplicate seam keys; B-C exact, A-B 45 bytes / 32 elements (analysis-grade) | p38s18r2 | `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s18r2/` + `artifacts/p38s18r2_round0_seam_tail_report.md`; immutable GCS reduction pending P38.2s publication |
| Aborted-launch stub | p38s14 | `…/evidence/p38s14/` (INCONCLUSIVE, head log only) |
| One-host perf ledger (8-run bitwise hash chain, decode-tax ablation) | p48/p49 T0–T5, F0–F12 | outer repo `tasks/p48*`, `tasks/p49*`; worktree `local/p48-profiling` |
| Cross-arm A identity for prompt_logprobs=None | P47a | landed CL 10242fa1 lineage; evidence dirs `p41_optimizer_p48c_p47a_*` on probe host |
| One-host GSM8K dispatch fabric (Execute 228k/step, add 105k, layer jits only 6.7s) | p51r6 | probe host `p51_gsm8k_xprof_p51r6_20260815/` (host-only capture; superseded for device planes) |
| Same capture with device planes (8 planes, ~23M device events; python_tracer=0) | p51rx | probe host `p51_gsm8k_xprof_p51rx_devplane_20260818/` |
| P52 reverse-scaffold consolidation on GSM8K one host | p52ab_{off,on} | probe host `p51_gsm8k_xprof_p52ab_{off,on}_20260815/` — warm 94.34±6.9 → 81.80±5.6s, issue 22.6→12.4s, 102/102 ALIGN zero |
| P52 byte gates | p52rv + pair | `…p52rv_20260815/` (verify, 0 mismatch) + `p41_optimizer_p52_neutral_20260815/` (103/103 fingerprints identical to the certified pair) |
| Official tunix.perf stack on one host: semantic v2 timeline + Profiler-windowed device capture, zero-diff intact | p54final | probe host `p51_gsm8k_xprof_p54final_20260819/` — 3/3 steps, 51/51 alignment zero, window steps 2→3, xplane `0336c08cc768e85f…` (1852MB, device planes), trace.json.gz `4f960988fd8f2af5…`, perfetto_trace_v2 `209a8040ee9d4a6c…` (503 packets / 480 track events) |
| Whole-step device capture holds only the first ~25s of engine decode: the TPU device trace buffer (~2.8M op events/core) fills and silently drops the rest; wall-clock window trims are a VETOED lever | p54final,p55a,p55a2 | per-core XLA Modules line: 13 modules all decode-family, span 24.9s, zero trainer modules; windows ~95s/~94s/~66s → xplane 1941/1995/1887MB with trace.json.gz constant ~41MB; rollout_update delay mode landed-and-reverted (cf96994f), do not retry time-shift trims |
| phase=update capture retains the whole backward: window anchored at the G6 update entry | p55c | `p51_gsm8k_xprof_p55c_20260819/` — attestation steps armed/started/stopped = 2/2/3, 3/3 steps, 51/51 zero, xplane 1534MB `fd95ac5340170334…`, trace.json.gz `ce7d22a6efb2ec4a…`, v2 `29c28750c7b7748f…`; device module census: block_pullback×1758, pullback_local_{head,norm,embed}, adjoint×17, _precomputed_gradient_step×8, decode family ABSENT on ALL 8 TensorCore planes (hardened census: per-plane backward-present + decode-absent, rc gates); census negative control on p54final → RED rc=1 |
| Final-tree step-mode regression: official Profiler window intact beside the update anchor | p55b2 | `p51_gsm8k_xprof_p55b2_20260819/` — absl window steps 2→3, 3/3 steps, 51/51 zero, xplane 2083MB `58f634a13abdc40a…` (buffer-cap fingerprint again), v2 trace `26989ca7fa6acb52…` |
| The G6 training phase is on the semantic timeline as one flat official peft_train span, placed exactly like weight_sync (cluster lane + one device lane, baseline 19 tracks, no custom names) | p55d | `p51_gsm8k_xprof_p55d_20260820/` — 3/3 steps, 51/51 zero, update-window attestation 2/2/3, backward census 8/8 planes, xplane 1542MB `6d0a049c5a0f2bcb…`, trace.json.gz `bcbc63931155f904…`, v2 `3b8421c5759ef698…`; census controls: p55d GREEN rc=0 / p55c (reverted nested-span smear, 20 tracks + custom leftovers) RED rc=1 / p54final (no span) RED rc=1 |
| Direct-TPU checkpoint mechanics: sharded model, device-resident Adam and metadata restore exactly; interval 10 and `LatestN(1)` enforced | P45.3c one-host | `tasks/p45-frozenlake-dp8-tp8-resident/evidence/p45_onehost_checkpoint_v5p.txt` (`sha256:1a84774925f311674c8eb9693889e5e6a9898d5473c529452561add168c6e1e0`; mechanism-only, Pathways/GCS target not run) |
| Same-input canonical log-softmax across two outer TPU programs is exact at production tail shape; injected bit flip is observed | P38.2t one-host | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2t_onehost_tail_construction_0817.md` (`0/38,895,616` elements; negative `1`; construction-only, not 64-chip Pathways evidence) |
| Two sealed P38 terminal-discriminator rounds; 54/54 selected red points have exact captured final-hidden rows and first measured red interval at lm_head logits; run incomplete at 2/3 | p38s21 | `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s21/` + `artifacts/p38s21_analysis_0818.md`; `ANALYSIS_GRADE_PARTIAL_2_OF_3`, no controlled exit/root COMPLETE |
| Real Qwen3-8B weight lm-head screen at TP4 M16/M256: default and explicit BF16/FP32-accumulation preset both exact for 4/4 seeds; lowering intervention present; negative=1 | P38.2w one-host | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2w_lm_head_onehost_0818.md`; `BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE`, construction-only |
| Real Qwen3-8B TP4 fixed Pallas lm-head construction: M16/M256 both use M256/K4096/N38144, shared rows exact 4/4, negative=1, and fixed differs from stock | P38.2x one-host | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2x_fixed_lm_head_onehost_0818.md`; construction-only, P38s23 target not run |
| P38s23 warmup omission repaired: exact request buckets M8/16/32/64/128/256 all use M256/K4096/N38144; 24/24 real-weight bucket comparisons exact, max_abs=0, negative=1 | P38.2x1 one-host | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2x1_fixed_lm_head_bucket_onehost_0818.md`; request-bucket construction-only |
| P38s23r1 passed all six request warmups and all 256 rollout trajectories, then learner rescore M4096 failed the narrow shape contract before A-B/B-C precheck | P38.2x1 target | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38s23r1_prefill_m4096_shape_error_report.md`; `INCONCLUSIVE_LEARNER_SHAPE_CONTRACT`, no numerical claim |
| GSM8K full bootstrap selected `data,model` PartitionSpecs for an actual `dp,tp` mesh, then later retries collided with Attempt-0 evidence | p38y6 | `tasks/p38-pathways-decode-prefill-carrier/debug_logs/p38_p38y6_gsm8k_full_sharding_axis_error.raw.log` + `tasks/p38-pathways-decode-prefill-carrier/P38_GSM8K_FULL_SHARDING_AXIS_ERROR_REPORT.md`; `INCONCLUSIVE_BOOTSTRAP_SHARDING_AXIS`, no model-load or numerical claim |
| Three independently sealed 64-TPU lm-head algorithm rounds: 3/3 tar archives and 30/30 logical members verify; A-B red in all rounds (66 elements / 111 bytes across 143,464 actions), B-C exact; generic preset rejected | p38s22 | `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s22/round-salvage-v1/` + `artifacts/p38s22_analysis_0818.md`; P38.2w2 forward-discriminator PASS only. Root manifest/COLLECTED/COMPLETE, returned terminal classification, backward, and optimizer remain unadmitted. |
| P58 native full attempt completed 128 real trajectories and exact 398-leaf live-weight attestation, then failed before trainer forward on the processed-S_prefill arm contract | p58f04 | `tasks/p58-deepswe-native-zero-comparison/evidence/p58f04/run.log` (`a7b0cda5e7d359c7e320b29f8af197db0dd6c46dc34850aa55ffb350fb766fdd`); `INCONCLUSIVE`, no forward/backward/update/checkpoint |
| P58 native full reached the first real value-and-grad/backward after 128 RepoEnv trajectories and finite A-B/B-C warning admission, then the over-strict observer-only T_old/T_current gate stopped before a durable update | p58f07 | `tasks/p58-deepswe-native-zero-comparison/evidence/p58f07/run.log` (`147332c0d9ffc6a4e5016963b18f427efeee683adb2a31defcd671941a1c58ef`); `INCONCLUSIVE`, no optimizer receipt/checkpoint |
| P58 native full head bootstrapped stock engine/R2E/data, but worker reached a foreign host-network Pathways RM and failed CL strict compatibility before rollout | p58f08 | `tasks/p58-deepswe-native-zero-comparison/evidence/p58f08/run.log` (`87d4386f1818ab40c87817819549df56d6e7de3995e333665b0021ff111a2f0e`); `INCONCLUSIVE`, no trajectory/update/checkpoint |
| P58 native full attached to the correct 128-device Pathways system and completed all 128 Step-0 rollout slots in 1,699.1s, then learner merge dereferenced a reset-timeout row with `original_input=None` before journal/alignment/training | p58f09 | `tasks/p58-deepswe-native-zero-comparison/evidence/p58f09/run.log` (`8977eefcb2ef34bc17c4dbb6e129b1d02cacba6b63041ab42d43a3aa8b5f4d0b`); `INCONCLUSIVE`, no durable journal/update/checkpoint |
| P58 native full proved the concurrency-128 one-wave geometry by completing all 128 trajectories and 8/8 groups in 1,209.2s, then one reset-timeout fallback lacked `prompts` and stopped learner processing | p58f11 | legacy incoming `debug_logs/p58_p58f11_deepswe_missing_prompt_key.raw.log` (`9bd1ca7526f38df32bde01cb4f811c464b76cc253d029b2f448cdc80164fee74`) plus classification (`1ba5c0f34171d219a4a6716c7368f86fc8968015649d577384e15ad6b4328454`); `INCONCLUSIVE`, no durable journal/update/checkpoint |
| Four P57 450-update P45/M15 native-no-IS/token-IS attempts completed real Step-0 rollout, then shared trajectory packaging replaced the FrozenLake prompt-bearing trajectory task with a policy-metadata-only environment task | n45c/n15c/i45c/i15c | legacy incoming `debug_logs/p57_frozenlake_missing_prompts_key.raw.log` (`d37f26a109620131325d1f0e8343a20d76e2dddb01784bf6066fef92898a4799`) plus classification (`989ee6dc9f289213306b83004fa17db4a3f26a08d4812cd6d81969e1d525ce6b`); all four `INCONCLUSIVE`, no trainer alignment/backward/update/checkpoint |
| Phase3 APC production-decode boundary carrier is exact through prefixes 1535…2049 with real hits; independent repeat is byte-identical | p3bc4 / p3br2 / p3br3 | probe host `p3_apc_boundary_p3b{c4,r2,r3}_*`; p3br2/p3br3 complete report SHA `fa80dfd5f8be5201…`, one-host G-A + deterministic G-D only |
| Phase3 APC production-shape A-B/B-C is exact for three rounds with hits up to 86.3%; matched greedy steady rollout is 53.152s→49.999s (+5.932%) | p3gbc2 / p3pc2 / p3pa1 / p3pp2 | probe host `p3_apc_p3gbc2_*`, `p3_apc_p3{pc2,pa1}_*`, `p3_apc_perf_pair_p3pp2_20260823T0414Z/`; comparison SHA `5248d0608517ca54…`, `KEEP_ONEHOST_PROXY` |
| Phase3 matched diagnostic XProf/Perfetto retains all byte gates and the same principal module/op counts; semantic operation labels are structurally present in both arms | p3xc1 / p3xa1 / p3xp2 | probe host `p3_apc_p3{xc1,xa1}_*`, `p3_apc_xprof_pair_p3xp2_20260823T0453Z/`; summary SHA `307c2f4ae46d1f40…`, shape attribution only |
| Phase3 dirty one real layer-0 KV page; authoritative A-B gate caught 30 bytes / 13 elements and ten clean cases stayed zero | p3gd1 | probe host `p3_apc_boundary_p3gd1_20260823T0506Z_dirty/`; `DIRTY_PAGE_GATE_CAUGHT`, report SHA `ff08495ebb66a9ea…`, intended negative not a zero-TIM failure |

Legacy flat pile `debug_logs/` is frozen read-only; anything new must use a run directory.
