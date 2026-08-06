# Anchors

Everything in this package is pinned to exactly two upstream versions. Both are reachable;
neither is a floating tag. If you change one, every bitwise claim recorded against it becomes
a claim about a different system.

## 1. Engine — `tpu_inference`

```
image      tunix_frozenlake_image:vllm-tpu0.25.0
image id   418dc632edd8
package    /usr/local/lib/python3.12/site-packages/tpu_inference   (this image's layout)
```

The six patches in `patches/tpu_inference/` are diffs against the files in **this** build.
Their expected SHA-256s are in `STOCK_MANIFEST.sha256`; `cluster/steps/20_probe_image.sh`
checks them per file before anything else runs.

Other layouts seen in the wild (pass via `CANON_TPU_INFERENCE_PATH`):

```
/app/vllm_tpu_inference/tpu_inference/tpu_inference     GKE base image
```

**Known drift risk.** The GKE `tunix_base_image:latest` is a different build *and* a floating
tag. Two consequences, both real:
- the patches may not apply, or may apply with fuzz onto a file nobody has gated;
- the same manifest can run on a different engine tomorrow, which is incompatible with a
  bitwise contract.

Pin by digest before drawing conclusions.

## 2. Training — `tunix`

```
repo        https://github.com/google/tunix.git
base        9fa7e251                    reachable from origin/yuxzhang/fix_accum_fp32
tip         3a00d951                    = this branch's starting point
delta       64 files / 11644 lines       (reproduce with the command below)
```

`3a00d951` is byte-identical to the sources that produced the signed GSM8K 200-step evidence:
all 25 files recorded by that run's manifest match (`docs/phase0.md`, finding F4).

### Why the base is not `main`

`main` is not a valid anchor for this work, and not merely because the diff would be noisy:

- 10 of the 18 touched `tunix/` files already differ between `main` and `9fa7e251`;
- the largest, `sft/peft_trainer.py` (346+/40−), differs because of the fp32
  gradient-accumulation work — **which the training recipe depends on** (one fp32 accumulated
  commit per group). The change set is not merely inconvenient to rebase; it is not correct on
  `main`.

Rebasing onto `main` is a separate project. It would also break byte-identity with the signed
evidence, so it needs the full gate ladder re-run afterwards.

### Assembly

The branch already contains the training-side changes — check it out and it works. There is
no training-side patch file to apply, and deliberately so: a diff of this branch's own history,
committed into this branch, adds nothing that the command below does not give you, and it goes
stale the moment the branch moves.

```bash
git fetch origin yuxzhang/fix_accum_fp32     # makes the base commit reachable
git diff 9fa7e251 3a00d951 -- tunix/         # exactly what zero-TIM changed
```
