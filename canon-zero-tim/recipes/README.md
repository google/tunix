# T2 / T3 recipes

These need the pinned image, a model checkpoint and a 4-chip v5p host, so they are not
runnable from this package alone. What is recorded here is enough to (a) reproduce them where
those things exist and (b) check a run elsewhere against the signed one.

The comparison that matters is **byte identity to the recorded numbers**, not "it ran".

Artifacts referenced below live under `/mnt/disks/tunix-data/logp_probe_1host`; their SHA-256s
are in `../evidence/artifacts.sha256` and `../verify_evidence.sh` checks them.

---

## Common configuration

Sourced from `claude_work/canon_env.sh` on the probe host; the same set is expressed in
`../cluster/profiles/_canonical_engine.env`.

```
CANON_FIXED_AR=1  CANON_FIXED_AR_EMBED=1  CANON_RPA_VJP2=1  CANON_VJP2_MAX_SEQS=<n_seqs>
CANON_RPA_D/P/M=128,512,128,512   MIN_TOKEN_BUCKET=256   NEW_MODEL_DESIGN=1
CANON_LOGPROB_M=256   CANON_PROMPT_PROCESSED_LOGPROBS=1
VLLM_ENABLE_V1_MULTIPROCESSING=0
XLA_FLAGS="--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false"
```

Two preflight rules, both learned the hard way:

- **Refuse to launch** on an incomplete set. A dropped `XLA_FLAGS` entry once removed
  `--xla_allow_excess_precision=false` silently; THIRDPROG went red and the resulting numbers
  were read as a finding before anyone noticed.
- **Read the postflight before any number.** If `THIRDPROG` is red, every downstream value in
  that run is void — not degraded, void.

---

## T2.1 — forward equality and third-program identity

```
K2.abc         hidden   0/10240      logits   0/303872
THIRDPROG                             primal   0/303872     (64 layers)
GRAD-HEALTH    determinism_exact=True
```

Artifact: `p19x6_t2decomp_n500_nl64.raw.log`

`THIRDPROG` is a hard gate. It compares the standalone forward's output with the primal
returned by `value_and_grad`. Those are two different executables and JAX never promises they
agree; without the fixed-order reductions they differ by ~47% of bytes past depth 15.

## T2.2 — step-0 exactness

```
four boundaries   0 differing bytes, max abs 0
ratio             w = r = w*r = 1   exactly
clip_hits = 0     tis_hits = 0
```

Artifact: `p19x6_xbucket_canon_256_1024.raw.log`

## T2.3 — gradient correctness of the non-standard pullback

Compares VJP2's pullback against an independent forward-mode JVP and a reference VJP of the
pure-JAX contract, over the real Mosaic primal, at production cache extent — the adjoint
identity `<g, Jv> == <J^T g, v>` computed along two independent routes.

```
20/20 records      worst primal rel-L2 6.56e-3     worst route rel-L2 3.59e-5
sign mismatch 0    g[1] fault arm correctly rejected
```

Artifacts: `std_p22_t2a_candidate_r1_0804.raw.log`, `p22_t2a_candidate_r1_0804.json`

The `g[1]` fault arm is the important line: a deliberately mis-routed cache cotangent must make
the gate go red. A gate nobody has tried to break is not a gate.

---

## T3 — GSM8K training ladder

Promotion order, each stage gated on the previous:

```
G0   exact-image CPU admission
G1a  1 step,  batch1 x gen2       N_action=248,  gradient 0.2502315640449524
G1b  2 steps, batch1 x gen8       8 reports,  N_action=10580
G1c  2 steps, batch4 x gen8       32 reports, N_action=52077
G2   10 steps                     160/160 reports, N_action=240796, warm step ~35.21s
G3   200 steps                    3200/3200 reports, N_action=2745204
```

Recorded command (no secrets; `WANDB_API_KEY` inherited from the environment):

```bash
test -n "${WANDB_API_KEY:-}"
P26_TIMEOUT_SECONDS=14400 \
  bash tasks/logprob_diff_operator_alignment/scripts/run_p26_gsm8k_train.sh g3 <fresh-label>
```

Per-microbatch contract — every one of the 3200 reports satisfies all of it:

```
S_decode == S_prefill == T_old == T_current     bitwise
w = r = w*r = 1                                 exactly
clip_hits = tis_hits = 0
gradient finite and nonzero
one commit and one weight sync per global step
```

Artifacts: `p26_gsm8k_g3_0804_g3_r1_evidenceonly.{raw.log,alignment.jsonl,classification.json}`
Frozen runner: `p26_runner_freezes/run_p26_gsm8k_train_g3_0804_r1_evidenceonly.sh`

**Read `../EVIDENCE.md` before citing G3.** It ran with checkpointing disabled by explicit
choice, so there is no model artifact; reward movement over `n=32` is a trend, not convergence;
and the engine-side file set has mtime-grade rather than cryptographic coverage.

### G1a — the cheapest full round-trip

One global step, ~32 minutes, and it exercises the entire path: engine patches, shim chain,
training-side hooks, all four boundaries, and the release classifier. It is the recommended
end-to-end check after any change to this package.

```
verdict           P26_GSM8K_G1A_PASS
N_action          248
boundaries        three, 0 differing bytes each
gradient          0.2502315640449524      (exactly; not "close")
policy sync       version advanced to 1
```
