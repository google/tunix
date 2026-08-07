# Evidence

Every claim below names the artifact that carries it. `./verify_evidence.sh` checks that each
one still exists and still hashes to what was recorded — run it before citing anything.

Artifacts live on the probe host under `/mnt/disks/tunix-data/logp_probe_1host` (override with
`CANON_ARTIFACT_ROOT`). Machine-readable list: `evidence/artifacts.sha256`.

## What is signed

| Claim | Number | Artifact |
|---|---|---|
| Forward `A = B = C` bitwise at release depth | logits `0/303872`, hidden `0/10240` | `p19x6_t2decomp_n500_nl64.raw.log` |
| Forward-only and `value_and_grad` primal are the same program | THIRDPROG `0/303872`, 64 layers | `p19x6_t2decomp_n500_nl64.raw.log` |
| Step-0 exactness (four boundaries, ratio, clip/TIS) | 0 bytes; `w = r = w·r = 1`; clip/TIS `0` | `p19x6_xbucket_canon_256_1024.raw.log` |
| Gradient correctness of the non-standard pullback, in-domain | 20/20 records; worst route rel-L2 `3.59e-5`; `g[1]` fault rejected | `std_p22_t2a_candidate_r1_0804.raw.log`, `p22_t2a_candidate_r1_0804.json` |
| 200 optimizer steps of real GSM8K at zero TIM | 3200/3200 reports, `N_action = 2,745,204`, all boundaries 0 bytes, 200 finite nonzero gradients, 200 commit/sync | `p26_gsm8k_g3_0804_g3_r1_evidenceonly.{raw.log,alignment.jsonl,classification.json}` |
| The exact command that produced it | — | `p26_runner_freezes/run_p26_gsm8k_train_g3_0804_r1_evidenceonly.sh` |

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
  come from one directly-attached 4-chip v5p host. A separate 64-chip single-slice Pathways
  discovery process passed bounded P1b/T2 probes, but its pinned-repeat gate, full model,
  multi-slice behavior and training remain unverified — see `CLUSTER_ADMISSION.md`.
- **The production `nnx` training forward.** The signed `C` is the engine-module differentiable
  forward. The production tunix `nnx` path still differs by `0.0267`.
