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
| Direct-TPU checkpoint mechanics: sharded model, device-resident Adam and metadata restore exactly; interval 10 and `LatestN(1)` enforced | P45.3c one-host | `tasks/p45-frozenlake-dp8-tp8-resident/evidence/p45_onehost_checkpoint_v5p.txt` (`sha256:1a84774925f311674c8eb9693889e5e6a9898d5473c529452561add168c6e1e0`; mechanism-only, Pathways/GCS target not run) |
| Same-input canonical log-softmax across two outer TPU programs is exact at production tail shape; injected bit flip is observed | P38.2t one-host | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2t_onehost_tail_construction_0817.md` (`0/38,895,616` elements; negative `1`; construction-only, not 64-chip Pathways evidence) |
| Two sealed P38 terminal-discriminator rounds; 54/54 selected red points have exact captured final-hidden rows and first measured red interval at lm_head logits; run incomplete at 2/3 | p38s21 | `tasks/p38-pathways-decode-prefill-carrier/evidence/p38s21/` + `artifacts/p38s21_analysis_0818.md`; `ANALYSIS_GRADE_PARTIAL_2_OF_3`, no controlled exit/root COMPLETE |
| Real Qwen3-8B weight lm-head screen at TP4 M16/M256: default and explicit BF16/FP32-accumulation preset both exact for 4/4 seeds; lowering intervention present; negative=1 | P38.2w one-host | `tasks/p38-pathways-decode-prefill-carrier/artifacts/p38_2w_lm_head_onehost_0818.md`; `BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE`, construction-only |

Legacy flat pile `debug_logs/` is frozen read-only; anything new must use a run directory.
