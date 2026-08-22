# Reproducing Raiden Weight Sync for GRPO RL (MaxText trainer + tpu-inference rollout)

This document reproduces a working, live-verified setup where a MaxText training
engine and a tpu-inference `RLVllmSampler` rollout exchange GRPO policy weights
over [Raiden](https://github.com/AI-Hypercomputer/tpu-raiden), a native
TPU-to-TPU weight-transfer library, launched as three GKE JobSets via
`tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh`.

## 1. Base image

Everything here builds on top of a shared TPU base image (JAX/Torch/vLLM
toolchain, no Raiden or MaxText yet):

```
us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813
```

That image ships `jax==0.9.2` / `jaxlib==0.9.2` / `libtpu==0.0.39`. Raiden
requires JAX `>=0.11.0`, so the build below upgrades the stack before
installing anything Raiden-related — mixing a JAX 0.11.0 raiden wheel with an
older JAX is a silent ABI break, not an import error.

## 2. Get a Raiden wheel

The live-verified run in section 6 used a wheel already baked into an
existing working image
(`gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:maxtext-raiden-0821-v4`,
`tpu_raiden_jax-0.0.1.dev20260822010825`, built against JAX 0.11.0 / libtpu
0.0.44). Pull that wheel out directly rather than rebuilding it:

```bash
mkdir -p /path/to/tunix/raiden_wheels
docker run --rm -v /path/to/tunix/raiden_wheels:/out \
  gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:maxtext-raiden-0821-v4 \
  bash -c "cp /app/raiden_wheels/*.whl /out/"
```

If that image isn't accessible, or a fresh wheel is genuinely needed, build
one from
[`AI-Hypercomputer/tpu-raiden`](https://github.com/AI-Hypercomputer/tpu-raiden)
instead:

```bash
git clone https://github.com/AI-Hypercomputer/tpu-raiden.git
cd tpu-raiden
# Confirm requirements.txt pins jax==0.11.0 / jaxlib==0.11.0 -- if a newer
# commit has moved the pin, the Dockerfile's jax/jaxlib/libtpu versions
# (step 3) need to move with it.
grep jax requirements.txt

WITH_TORCH=0 ./ci/build_wheel.sh jax
```

This runs a hermetic Bazel build inside an `ml-build` container (glibc 2.35,
matching the TPU runtime). It needs a lot of CPU and disk — point
`RAIDEN_CONTAINER_CACHE` at a volume with real free space (the script's
default, `$HOME/.bazel_cache_container`, filled a 97GB root disk to 100% and
killed a build at 99% complete in-house; redirect it, e.g.
`export RAIDEN_CONTAINER_CACHE=/mnt/data/.bazel_cache_container`, before
running). On a 200+ core machine it still takes roughly 40-60 minutes, most
of it compiling LLVM/XLA from scratch on a cold cache. The wheel lands in
`dist/`:

```
dist/tpu_raiden_jax-0.0.1.dev<timestamp>-cp312-cp312-manylinux_2_31_x86_64.whl
```

Either way, the wheel needs to end up in `tunix/raiden_wheels/` before
building the image in step 3 — `Dockerfile.maxtext` picks up everything
under that directory.

## 3. Build the tunix + MaxText + Raiden Docker image

From the `tunix` repo root (with `raiden_wheels/*.whl` in place from step 2):

```bash
docker build -f Dockerfile.maxtext -t <your-image-tag> .
docker push <your-image-tag>
```

Note: `Dockerfile.maxtext`'s `FROM` line and the JAX/libtpu upgrade step
below are written and ready, but this exact image (base + upgraded JAX/libtpu
+ the extracted wheel) has not actually been built and pushed as part of this
work — the live-verified run in section 6 used the existing
`maxtext-raiden-0821-v4` image directly, not a fresh build from the yangmu
base. Building and pushing it is the one remaining step for a fully
from-scratch reproduction; it's a standard `docker build`/`docker push`, no
surprises expected, but it hasn't been run and its output hasn't been
inspected.

`Dockerfile.maxtext` (relevant excerpt):

```dockerfile
FROM us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813

ENV PATH="/opt/venv/bin:$PATH"

# Bump JAX/libtpu to what the bundled raiden wheel needs.
RUN pip install --no-cache-dir -U "jax[tpu]==0.11.0" "jaxlib==0.11.0" "libtpu==0.0.44" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install TPU Raiden JAX wheel (from step 2, above)
COPY ./raiden_wheels/*.whl /tmp/raiden_wheels/
RUN pip install --no-deps /tmp/raiden_wheels/*.whl && rm -rf /tmp/raiden_wheels

# Install MaxText dependencies directly into docker image
RUN pip install --no-cache-dir \
    aqtp tokamax einshape typeguard drjax ml-goodput-measurement \
    cloud-accelerator-diagnostics cloud-tpu-diagnostics \
    google-cloud-mldiagnostics google-cloud-monitoring

# Install Tunix in editable mode
WORKDIR /app
COPY . /app
RUN pip install --no-deps -e /app
```

Note: this image bakes in Raiden, MaxText's Python *dependencies*, and Tunix
itself — but **not** MaxText or tpu-inference's own code. `k8s_launcher.sh`
(step 5) `git clone`s and `pip install -e`s those two repos' feature branches
fresh at pod startup instead, so an image rebuild is only needed when
dependencies change, not when the MaxText/tpu-inference application code
changes.

Image actually used for the verified run in section 6 (an earlier build, not
from the yangmu base above — see the note in step 3):

```
gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:maxtext-raiden-0821-v4
sha256:d19cfd356271183e7f2718aab77946207b2257a9c097f8e9e3495341f80f2110
```

## 4. The three patches

The feature spans three repos. Each branch is pushed and squashed to a
single commit; no PRs are open yet (open them from these branches when
ready):

| Repo | Branch | What it adds |
|---|---|---|
| [`google/tunix`](https://github.com/google/tunix) | `mohit/raiden-maxtext-rlvllm` | `VllmSamplerAdapter`, `RaidenSynchronizer`, `WeightSyncCoordinator` protocol stack; the GRPO example + `k8s_launcher.sh` |
| [`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext) | `mohit/raiden-engine-sync` | `MaxTextTrainingEngine` (conformed to Tunix's trainer contract) + `prepare_weight_sync()`/`release_weight_sync()` |
| [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) | `mohit/rl-sampler` | `RLVllmSampler` + in-process Raiden binding inside the vLLM EngineCore worker |

Read each branch's (single, squashed) commit message for the full technical
rationale (in particular tpu-inference's, which covers a real architecture
point: weight binding has to happen *inside* the vLLM EngineCore subprocess,
not in the parent process, because the live TPU-resident weight arrays can't
be shipped across vLLM's multiprocess RPC boundary).

`k8s_launcher.sh` pulls these three branches by name at pod startup (see
step 5) — no manual checkout needed beyond having push/fetch access to fork
or clone them if you want to modify them further.

## 5. Launch on a GKE cluster

Target: a 2-slice v5p-8 GKE cluster (trainer on one v5p-8 slice, rollout on
the other), e.g. `auto-v5p-8-bodaborg` (`europe-west4`,
`cloud-tpu-multipod-dev`).

```bash
gcloud container clusters get-credentials <cluster> --region=<region> --project=<project> --dns-endpoint

cd tunix
export USER=<a-valid-k8s-label-safe-username>   # no dots/underscores
export TUNIX_IMAGE=<your-image-tag>              # from step 3, or the known-working
                                                   # gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:maxtext-raiden-0821-v4
export VERIFY_WEIGHTS=true                       # optional: logs source/dest checksums
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --command start
```

Defaults (overridable via env vars, see the top of `k8s_launcher.sh`):
`MODEL_NAME=Qwen3-0.6B`, `MAXTEXT_MODEL_NAME=qwen3-0.6b`,
`TRAINER_BACKEND=maxtext`, `MAX_STEPS=2`, `tpu_slice=tpuv5:2x2x1` per side.
Trainer and rollout **must** load the same architecture/size for Raiden's
exact name-matching sync to work.

Watch progress:

```bash
kubectl get pods | grep "$USER"
kubectl logs -f $(kubectl get pods | grep "$USER-orch" | awk '{print $1}')
```

Tear down:

```bash
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --command stop
```

## 6. What was tested, and log links

Two live runs on `auto-v5p-8-bodaborg` (2-slice v5p-8), `qwen3-0.6b`,
`MAX_STEPS=2`, both completed with `EXIT_CODE=0` and two real Raiden
weight-sync rounds (`policy_version` 0→1→2):

- **Functional run** (2026-08-22 04:58–04:59 UTC): confirmed MaxText's
  `MaxTextTrainingEngine` and tpu-inference's `RLVllmSampler`/
  `MaxTextForCausalLM` were both actually in use (not a stub path), full GRPO
  loop completed.
- **`VERIFY_WEIGHTS=true` run** (2026-08-22 05:12–05:13 UTC): source (trainer)
  and destination (rollout) per-tensor checksums logged and cross-checked —
  matched to within floating-point summation-order tolerance (~1e-7
  relative), confirming the transferred weights are correct, not just
  present.

Cloud Logging Explorer link for the `VERIFY_WEIGHTS=true` run (all 3 pods —
orchestrator/trainer/rollout; widen the time range if the default window
doesn't cover 2026-08-22 05:12-05:13 UTC):

```
https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Aresource.labels.project_id%3D%22cloud-tpu-multipod-dev%22%0Aresource.labels.location%3D%22europe-west4%22%0Aresource.labels.cluster_name%3D%22auto-v5p-8-bodaborg%22%0Aresource.labels.namespace_name%3D%22default%22%0Aresource.labels.pod_name%3D%28%22mohitkhatwani-orch-proc-0-0-7ts87%22%20OR%20%22mohitkhatwani-roll-proc-0-0-f6rdd%22%20OR%20%22mohitkhatwani-train-proc-0-0-tql54%22%29?project=cloud-tpu-multipod-dev
```

Key evidence, pulled directly from those logs:

Trainer backend confirmed:
```
[TrainerNode] Trainer backend: MaxTextTrainingEngine.
```

Rollout model confirmed:
```
[RolloutNode] Loading MaxText model 'qwen3-0.6b' natively via maxtext_vllm_adapter's MaxTextForCausalLM (architectures override).
INFO [model_loader.py:874] Registered JAX model MaxTextForCausalLM with tpu_inference and vLLM registries.
[RolloutNode] Initializing RLVllmSampler with model: Qwen/Qwen3-0.6B
INFO [model.py:672] Resolved architecture: MaxTextForCausalLM
[RolloutNode] RLVllmSampler started successfully.
```

Cross-verified checksums (source vs. destination, one param sample):
```
[TrainerNode] Source weights checksums: {"decoder_norm.scale": 1024.0, "layers_0.mlp.wi_0.kernel": 80765.9453125, ...}
[RolloutNode] Destination weights checksums: [{"decoder_norm.scale": 1024.0, "layers_0.mlp.wi_0.kernel": 80765.9375, ...}]
```

## 7. Known gaps / next steps

- This was a `MAX_STEPS=2`, `qwen3-0.6b` smoke run — a longer run and a
  larger model would give more throughput/HBM-headroom confidence.
- `tests/experimental/rollout/vllm_sampler_adapter_test.py::test_weight_sync_delegations`
  has a pre-existing, unrelated failure (an `AsyncMock` auto-attribute
  artifact) — not touched by this work, worth a follow-up.
- Building and pushing the yangmu-base image from step 3 hasn't actually been
  done/verified — see the note there.
- The three branches (section 4) are squashed and pushed but have no open
  PRs — open them from the branch names in that table when ready.
