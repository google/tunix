# canon-zero-tim

> **New here? Read `START_HERE.md`.** This file explains the mechanism; that one tells you
> what to do.

Make the rollout engine, its re-scorer, and the training forward produce **bit-identical**
logprobs — then keep them that way while the model trains.

```
A = engine decode logprobs      the behaviour policy the tokens were actually sampled from
B = engine prefill re-score     the same engine, same tokens, scored in one pass
C = differentiable training forward
goal: A = B = C, bitwise, over the full distribution
```

---

## Why bitwise, and not "close enough"

Two amplifiers make any nonzero seed useless to shrink.

**Depth.** A ULP-scale difference entering the layer stack is amplified fast and saturates
within 3–4 layers: measured 1 layer `3.4e-07` → 2 layers `1.5e-03` → end-to-end logprob
difference `0.015–0.024`. Shrinking the seed from `1e-2` to `1e-8` lands in the same place.
Only exactly zero is different in kind.

**Time.** A trajectory's log-probability is a sum, so per-token differences accumulate over
its length, and Gibbs' inequality fixes the sign — self-scored tokens drift systematically
high, they do not cancel. Long-horizon agentic RL feels this first.

## Why it happens

Not because anyone's operator is wrong. Measured with both engines' real modules in one
process, fed identical inputs: RMSNorm `0`, matmul `0`, RoPE `0`, attention kernel `0` —
and the end-to-end difference was still there.

The cause is **program identity**. An XLA executable's numerics are fixed by more than the
operators:

```
executable = operators + SHAPE + JIT BOUNDARY + REDUCTION ORDER
```

Change the token bucket, wrap a call in a nested `jit`, or ask for a gradient, and you get a
different program — and bf16 addition is not associative, so a different program is a
different number. The package pins every one of those degrees of freedom.

A useful corollary: two *different implementations* of the same maths differ by ~1% in bf16
no matter how carefully written (measured: swapping only the attention implementation moved
logprobs by `0.0115`, the same order as the whole A/B/C gap). You cannot align two
implementations below that floor. You can only run **the same code**. That is why `C` calls
the engine's own attention kernel rather than a training-side equivalent.

---

## What is changed

Six patches against the engine, one file each, **1252 lines total**. Every switch defaults
off; with the environment unset the patched files behave byte-identically to stock.

| # | Patch | Root cause it addresses | Switch |
|---|---|---|---|
| 01 | `attention-interface` | RPA block sizes are chosen by a shape- *and* concurrency-dependent lookup, so decode and prefill silently accumulate at different granularity; and the kernel is not differentiable | `CANON_RPA_D/P/M`, `CANON_RPA_VJP2` |
| 02 | `embed` | The vocab-sharded embedding gather is the last 4-way reduction in the forward, and 4-way reductions lower differently in a forward-plus-backward program | `CANON_FIXED_AR_EMBED` |
| 03 | `linear` | The bf16 all-reduce closing a contract-sharded projection sums in **ring-position order**, so the same token at a different padded row gets different bits | `CANON_FIXED_AR` |
| 04 | `qwen3` | **none — diagnostic instrumentation only** (bisection cut points, depth reduction) | `CANON_CUT`, `P16_NUM_LAYERS` |
| 05 | `qwen2` | **none — diagnostic instrumentation only** (tail cut, optimization barriers) | `CANON_TAIL`, `CANON_BARRIER_ALL` |
| 06 | `tpu-runner` | Decode and prefill land in different token buckets; the vocab reduction splits by `M` **and** by nested-jit caller; prompt scoring does not inherit decode's processed-logprob semantics | `MIN_TOKEN_BUCKET`, `CANON_LOGPROB_M`, `CANON_PROMPT_PROCESSED_LOGPROBS` |

Patches 04 and 05 carry no fix. They are here because the shim chain bottoms out in those two
files, so removing their instrumentation would break byte-identity with the sources that carry
the signed evidence. The diagnostic *switches* are excluded from the canonical profile
(`cluster/profiles/`); the diagnostic *code* is not separable from the chain.

Plus a **shim chain** (`src/engine_shims/`) carrying the promoted Pallas canonical ops and the
differentiable attention wrapper, and one XLA flag,
`--xla_allow_excess_precision=false`, which removes a layer of forward-vs-forward+backward
lowering difference.

### The forward/backward split

The engine's attention kernel is fast, bit-exact, and **not differentiable** — JAX has no
transpose rule for it. Training needs a gradient, so the package supplies one:

```
forward   = the real kernel, verbatim          <- must be bitwise; this is what A, B and C compare
backward  = the Jacobian of a line-by-line
            pure-JAX transcription             <- must be correct; SGD does not care about 1 ULP
```

They meet only at the inputs: `custom_vjp` saves the arguments, and the transcription's own
output value is discarded. The transcription is faithful to `1.49e-08` in fp32 (a from-scratch
reimplementation is `7.06e-04`), and an fp64 oracle shows the chunked-with-cache and
full-prefill forms are the same mathematical function — so its VJP is the kernel's VJP, not a
surrogate.

---

## Running the tests

### T0 — pure CPU, seconds, no TPU

```bash
tests/t0_cpu/run.sh
tests/t0_cpu/negative_control.sh     # proves the gate rejects bad runs
```

Proves the differentiable contract is mathematically sound: value residual exactly `0`,
gradient agreement `~5e-16`, finite-difference cross-check `~1.1e-08`. Does **not** touch the
kernel — that is T1's job.

### T1 — needs ≥2 TPU chips, no model, no image build

```bash
tests/t1_tpu/run.sh                       # host: re-enters the pinned image
CANON_IN_CONTAINER=1 tests/t1_tpu/run.sh  # already inside a TPU container
```

Four topology admission probes plus the four historical minimal reproducers. **Run this first
on any new cluster** — it decides in seconds whether the switch set transfers at all. See
`CLUSTER_ADMISSION.md`.

### T2 / T3 — need the pinned image, a checkpoint, and a 4-chip host

Not runnable from this package alone. `recipes/` records the exact commands, the expected
output lines, and the artifact SHA-256s so a run elsewhere can be checked against the signed
ones.

---

## Installing the chain

```bash
./install.sh /somewhere --from-image tunix_frozenlake_image:vllm-tpu0.25.0 --model qwen1p7b
./install.sh /somewhere --from-path  /path/to/site-packages/tpu_inference --model qwen1p7b
```

`--from-path` needs no docker, no image on disk, and no network — that is the mode a pod uses.
Both verify every produced file against `MANIFEST.sha256` and **fail** on a mismatch, because
the chain is loaded by path: a stale member does not raise, it silently reverts you to stock.

For Kubernetes see `cluster/README.md`.

---

## The one thing to remember

> A run with the switches on and no `[PATHTRACE]` lines did not do what you think it did.

The chain is resolved by module name and by absolute sibling path. Neither raises when a
member is missing — the engine falls back to its stock module, every switch still reads "on",
and the run goes green having computed nothing canonical. The `[PATHTRACE]` tally is the only
evidence that the intervention executed. The exit code is not evidence.

---

## Layout

```
START_HERE.md           zero-context entry point: status, what your machine can run
RUNBOOK.md              step-by-step procedures for each runnable task
README.md               this file -- what the mechanism is and why
ANCHORS.md              the two versions everything is pinned to, and how to reach them
EVIDENCE.md             claim -> artifact -> SHA-256, including where the chain is weak
KNOWN_FOOTGUNS.md       traps that produce green runs
CLUSTER_ADMISSION.md    what to measure before trusting a new topology
install.sh              assemble the chain (two source modes)
MANIFEST.sha256         expected SHA of every installed file
STOCK_MANIFEST.sha256   expected SHA of the six upstream files the patches are cut against
patches/tpu_inference/  six real diffs, one per file
src/engine_shims/       the shim chain + promoted Pallas ops + model-specific modules
tests/t0_cpu/           CPU gates
tests/t1_tpu/           topology admission probes
tests/t2_dp/            DP reduction/update admission + negative control
cluster/                Pathways/GKE entry point, steps, profiles, manifest
recipes/                T2/T3 reproduction recipes and expected outputs
docs/                   the phase-by-phase record of how this package was built
```
