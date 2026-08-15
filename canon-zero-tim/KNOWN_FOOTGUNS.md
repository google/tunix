# Known footguns

Every entry here has actually happened. They share one shape: **the run goes green while the
thing you meant to test never ran.**

## In this package

### 1. `run_p19x6.sh` hardcodes the *old* differentiable attention

`run_p19x6.sh:23` passes `-e CANON_RPA_VJP=1` — the prefill-only contract whose backward gives
**identically zero gradient** to q/k/v on the cached path. `CANON_RPA_VJP2=1` must be passed
explicitly; it is the switch, while `CANON_VJP2_MAX_SEQS` is only its parameter. Passing the
parameter without the switch silently measures the unfixed path.

That runner is left untouched on purpose: it carries signed evidence, and editing it would
make old artifacts unattributable. This package ships its own entry points
(`cluster/entrypoint.sh`, `tests/*/run.sh`) which never set the old switch.

Note both release configurations set `CANON_RPA_VJP=1` *alongside* VJP2. VJP2 wins in the
engine's branch order, so it is currently harmless — but if VJP2 were ever unset, the broken
path would activate silently. `cluster/steps/00_env.sh` prints a notice when it sees this.

### 2. The config guard defaults to *off*

The same line carries `-e P18_SKIP_ENV_CHECK=${SKIPENV:-1}`. The default `1` downgrades the
configuration assertion to a warning, and a warning inside a running log only works if someone
reads it. This package's `00_env.sh` refuses to launch instead.

### 3. Chain members are resolved by path — a missing one does not raise

The shim chain loads each layer by absolute path and each helper by module name. A missing or
stale member means the engine quietly uses its stock module. Every switch still reads "on".
Hence: `install.sh` fails on a `MANIFEST.sha256` mismatch, `50_verify_overlay.sh` checks byte
identity *and* imports the chain, and `90_run.sh` refuses a run with no `[PATHTRACE]`.

### 4. Diagnostic code is baked into two of the patches

`04-qwen3` and `05-qwen2` add *only* diagnostic instrumentation — `CANON_CUT` bisection cut
points, `P16_NUM_LAYERS` depth reduction, `CANON_TAIL`, `CANON_BARRIER_ALL`. No canonical fix.

Two consequences. Reading the patch numbering as "six fixes" is wrong — there are four.  And
the instrumentation cannot be stripped: the shim chain bottoms out in these files, so removing
it breaks byte-identity with the signed sources. What *is* excluded is the switches: none of
them appear in `cluster/profiles/`, so they stay off unless someone sets them deliberately.

`CANON_CUT` in particular replaces the attention output with a constant or with an earlier
tensor. A run with it set is a bisection probe, not a measurement of the model.

## General

### 5. Absence of output is never a pass

A gate that printed nothing did not run. Treat a missing line as red and find out why.

### 6. `grep` silently drops every match on a log with control characters

Progress bars make `grep` treat a log as binary and report nothing — indistinguishable from
"the intervention never fired". Always `grep -a`. Verify a negative claim ("this line never
appears") with a second tool: `sha256sum`, `wc`, `file`.

### 7. A hard gate going red voids the whole run

If `THIRDPROG` is red, the forward-only and forward+backward programs are not the same family.
Nothing else measured in that run means anything. Read the postflight *first*, then the
numbers — not the other way round.

### 8. `os.environ.get(k, default)` never sees its default under docker/k8s

`-e K=` always sets the key. An unset source variable becomes an empty string, not a missing
key, and `int("")` explodes far from the cause. Diagnostic switches are therefore exported
explicitly empty, and parsing treats empty as absent.

### 9. `differing_bytes` saturates

Once a perturbation touches most elements the count parks near ~46% regardless of magnitude.
It answers "identical or not" and nothing else. To *rank* two nonzero differences use relative
L2 or `1 - cos`, and first confirm on a known-nontrivial perturbation that your metric is not
already saturated.

### 10. `MIN_TOKEN_BUCKET` is a **global** token count

Under data parallelism the runner divides it by `dp_size`. Copying `256` from the `dp=1` recipe
to a `dp=64` deployment gives each replica a bucket of 4 — the pinning the entire result rests
on would be gone while every switch still reads "on". Derive it with
`tests/t1_tpu/probe_bucket_contract.py`.

The TPU inference scheduler limits have the opposite unit: `max_num_seqs` and
`max_num_batched_tokens` are interpreted **per DP rank**, then multiplied by `dp_size` while
building the global precompile shapes. P34 originally supplied `64/8192` to DP16 and therefore
requested 1024 global sequences plus token paddings
`[4096, 8192, 16384, 32768, 65536, 131072]`. The signed P34 values are `4/256`: they yield 64
global sequences and exactly `[4096]`. Verify this with
`tests/p34_deepswe/probe_scheduler_contract.py`; do not infer the unit from the CLI name.

### 11. Device order is not what you passed in

Topology-aware mesh construction permutes it — on the 4-chip probe host `[0,1,2,3]` comes back
as `[0,2,1,3]`, and two different mesh *shapes* produce different permutations. So "both sides
use the same expression" does **not** imply the same order. Read the order after the mesh is
built, and assert it on both sides with `CANON_EXPECT_MODEL_MESH_IDS`.

### 12. Never let a secret near the repo

The tunix checkout's `git remote` URL has been seen carrying a plaintext `ghp_` token with push
rights. Existing secret scans cover W&B run trees, not `.git/config`. Never package `.git/`,
never paste `git remote -v` output into a report, and rotate anything that has been exposed.

### 13. Pathways initialization cannot be best-effort in proxy mode

Importing JAX before `pathwaysutils.initialize()` or swallowing its exception can leave the
probe on a different backend than the operator requested.  Every JAX-based T1 probe imports the
shared `tests/t1_tpu/pathways_bootstrap.py` first.  When `JAX_PLATFORMS` contains `proxy`, or a
Pathways endpoint is configured, import/initialization failure is fatal and a successful run must
contain exactly one `[T1.PATHWAYS]` marker per JAX probe.  Proxy runs require
`required=1 initialized=1`; a directly attached TPU may report `required=0 initialized=0`.
The worker readiness loop uses the same bootstrap and rejects a timeout or the wrong visible
device count; an elapsed sleep is not readiness evidence.

### 14. A device prefix is not a production TP mesh on Pathways

`devices[:4]` was a convenient TP4 probe on the original directly attached host. On a 64-device,
16-host Pathways slice it requested a physical `4,1,1` subset that crossed `2,2,1` host bounds,
then failed at compile time. Even if a client flag forced it through, it would still not attest
the production DP16×TP4 placement. Construct the topology-aware full-slice `(16,4)` mesh, prove
all 64 device ids occur exactly once, and print the actual TP groups. Never use a reshape fallback
for topology admission.

### 15. A JAX runtime failure taints the rest of a shared Pathways session

The first 64-device runner caught the P1 exception and continued into P2-P4 and H1-H4. Those
rows were useful for triage but are not release evidence: the client session had already failed,
several legacy probes failed again internally, and some scripts swallowed their own exceptions.
The unified runner now stops at the first nonzero exit or exception and prints
`SKIP_TAINTED after=<probe> skipped=<list>`. Fix the first failure and rerun in a fresh JobSet
attempt; never promote downstream rows from the contaminated session.

### 16. Backend flags do not belong in application `sys.argv`

The first Pathways bootstrap appended two speculative subslice flags to `sys.argv`. They did not
change the live client guard, and `probe_dp_update.py` later rejected them as unknown argparse
arguments before any DP measurement ran. The bootstrap no longer mutates application arguments.
More importantly, the probes no longer need the flag: they construct host-valid full-slice
meshes. A runtime safety check should be satisfied structurally, not disabled speculatively.

### 17. A truthy workload flag is not a workload-specific exception

`CANON_P32_WORKLOAD` names a semantic contract; it is not a generic admission boolean. A broad
`if CANON_P32_WORKLOAD` exception intended for GSM8K also admitted FrozenLake with
`sampler_is=None`, bypassing the token-correction requirement until the runtime alignment gate
failed. Compare the exact workload name and test both the intended exception and a neighboring
workload negative control. A rollout-quality metric or completed model trace before that failure
does not promote the run.

### 13. Client-container `XLA_FLAGS` never reaches the Pathways TPU compiler

Under Pathways, HLO is compiled on the server side. An `XLA_FLAGS` value exported in the
JAX client container configures nothing but the client's own CPU backend — while every
launch-time check that greps the client environment happily reports the flag as present.
That is how `--xla_allow_excess_precision=false`, load-bearing for third-program closure,
ran effectively unset through every Pathways campaign before P36 (replicated-arm drift
`~90k/262144`; with delivery it is `0/262144`).

Delivery contract, verified by P36 flagon1/envon1:

- the pinned proxy **rejects** the flag as a command-line argument
  (`ERROR: Unknown command line flag`) — args are the wrong channel;
- the working channel is the **proxy container environment**:
  `XLA_FLAGS=--xla_allow_excess_precision=false` on `pathways-proxy`;
- every renderer enforces exactly one such env entry and refuses a raw argv flag
  (`ensure_proxy_xla_env`); the static manifests are locked by
  `tests/t0_cpu/test_cluster_contracts.py`.

Corollary: a "flags OK" assertion is only as good as the process it inspects. Assert the
rendered manifest, not the launcher's own environment.

### 18. A renderer test does not prove the rendered workload reaches the learner

P34 originally passed its static renderer and exact-image overlay tests while
the real learner would reject the first batch: the recipe deliberately disabled
sampler IS and TIS, but the generic alignment guard admitted
`sampler_is=None` only for GSM8K. None of the old tests fed the rendered
environment through `00_env.sh` and then exercised the workload-specific
learner policy.

Every production recipe now needs both layers: run the renderer output through
the real environment preflight, and test the exact workload-specific policy
with a neighboring negative control. A green manifest or overlay import alone
is only `STATIC PASS`; it is not launch readiness.

### 19. `envelope_probe` identifies A-records by a side effect the sampler no longer produces

Since the rollout sampler requests `prompt_logprobs=None` (P47a, the c4ec573d semantics),
A requests no longer allocate prompt-logprob structures. `tunix/rl/envelope_probe.py:362-366`
still enumerates A request ids from the `num_prompt_logprobs` capture metadata — a byproduct
of the old `=0` quirk — so its `native_A_observed` attestation goes **false-red** on any
post-P47a capture. The B rescore path still requests `=0` and is untouched. Adapt the probe's
A-identification (see `tasks/p35-envelope-discriminator/PRECONDITIONS.md`) before the next
P35 envelope run; a red from this cause is a stale heuristic, not a boundary finding.

### 20. `CANON_P28_SEGMENTED_VJP` does not gate the segmented training update

Corrected 2026-08-15; the first version of this entry claimed the L3 pair
never executes the segmented value-and-grad because it sets
`CANON_P28_SEGMENTED_VJP=0`. That reason was wrong: the flag gates only
the standalone `run_p28_block_vjp_gate` diagnostic
(canonical_qwen3_adapter.py:5154); the training update runs the segmented
path whenever `CANON_P28_SEGMENTED_TRAIN=1` (peft_trainer.py:373), and
the pair's own logs carry `stage=vag_reverse` lines. The durable footgun
is the env name itself: reading `SEGMENTED_VJP=0` as "segmented VJP off"
mis-classifies which code a vehicle exercises — verify against the run
log (`grep -a "stage=vag_"`), not the env list. A pair with the new perf
envs UNSET still does not execute those branches (defaults off), so
default-off neutrality certification via the pair stands; in-run
dual-compute verify modes remain the byte gates of choice because they
also dodge step-0 sampling nondeterminism.
