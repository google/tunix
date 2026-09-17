# Raiden weight-sync smoke test

End-to-end 2-step GRPO smoke test for the Raiden weight-sync path on
Qwen3.5-35B-A3B:

```
Pathways MaxText trainer   2x2x2 (8 chips, FSDP=8)  -- Raiden FFI
        |  weight sync (633 variables)
        v
mcJAX vLLM rollout         2x2x1 (4 chips, dp=2 tp=2) -- Raiden TCP
```

Checkpoints every step, `EXIT_CODE=0` expected, ~35 min wall clock.

## Quickstart

```bash
bash raiden_smoke_test/build.sh        # build + push image       (~8 min warm)
bash raiden_smoke_test/run.sh start    # launch                   (~25 min)

bash raiden_smoke_test/run.sh status   # progress at a glance
bash raiden_smoke_test/run.sh watch    # follow orchestrator, de-noised
bash raiden_smoke_test/run.sh stop     # tear down (guarded)
```

| File | Role |
| :--- | :--- |
| [`Dockerfile`](Dockerfile) | The image. Built from `projects/` so it can `COPY` all three repos. |
| [`build.sh`](build.sh) | Fetch the Raiden wheel, build, push. |
| [`run.sh`](run.sh) | `dryrun` / `start` / `status` / `watch` / `stop`. |
| [`config.sh`](config.sh) | Every pin and knob. Sourced by both scripts — **edit here only**. |

`config.sh` exists so the pins are not duplicated between build and run; it is
not meant to be executed. Every value is overridable from the environment:

```bash
IMAGE_TAG=my-experiment bash raiden_smoke_test/build.sh
MAX_STEPS=5 bash raiden_smoke_test/run.sh start
```

Use `run.sh dryrun` to render and assert the manifests without touching TPUs.

## Prerequisites

`maxtext/` and `tpu-inference/` must be **siblings of `tunix/`**, because the
Dockerfile `COPY`s all three and docker cannot copy from outside the build
context:

```
projects/                     <- the docker build context
├── tunix/
│   └── raiden_smoke_test/    <- you are here (Dockerfile lives here)
├── maxtext/
└── tpu-inference/
```

Override with `MAXTEXT_DIR`, `TPU_INFERENCE_DIR`, `DOCKERFILE` if your layout
differs. Also needs `docker`, `gcloud`, `gsutil`, and `kubectl` (the scripts add
`~/.local/bin` to `PATH`).

## Validated configuration

| Component | Pin |
| :--- | :--- |
| tunix | `032db7ea` |
| maxtext | `7c016bfe2` |
| tpu-inference | `9dc23977` |
| vLLM | `51da0ca66` (same as upstream `requirements/requirements.txt`) |
| Raiden wheel | `tpu_sync_jax-0.0.1.dev20260914193202` |
| Pathways server | `unsanitized_server:raiden_20260914_fix` |
| Pathways proxy | `unsanitized_proxy_server:raiden_20260914` |

Reference run `head-v19-0102` (2026-09-17) passed 11/11 gates: 633/633 variables
paired across 3 weight syncs, weight checksums mutating each policy version,
checkpoints at steps 1 and 2 (140/147 GiB), `EXIT_CODE=0`.

> [!NOTE]
> The Pathways server tag carries a `_fix` suffix and the proxy tag does not.
> That asymmetry is intentional. Both images also return **403 on `docker pull`
> from a cloudtop** — GKE nodes pull with a different service account, so a 403
> locally is expected, not a misconfiguration.

## Known upstream issues this config works around

### 1. `per_token_logps` is missing on the MaxText trainer backend — **blocker**

Upstream `9b24d9df` added a token-level importance-sampling correction and
shipped it **default-on** (`USE_ROLLOUT_LOGPS=true`), but only implemented
`Trainer.per_token_logps` for the tunix-native `peft_trainer_v2` backend.
`MaxTextTrainingEngine` has no such method, so **every** `TRAINER_BACKEND=maxtext`
GRPO run at HEAD dies at step 0:

```
AttributeError: 'MaxTextTrainingEngine' object has no attribute 'per_token_logps'
```

`config.sh` sets `USE_ROLLOUT_LOGPS=false` to route around it. **This disables
the off-policy TIS correction**, which matters here because the sampler (vLLM)
and trainer (MaxText) are different backends — precisely the regime TIS exists
to correct. Remove that line once upstream implements the method, and re-run to
validate the agreement path. (`perplexity: 1.0000` in the metrics is an artefact
of the workaround, not a model property.)

### 2. `MINI_BATCH_SIZE` changed units

`batch_size` and `mini_batch_size` are now both counted in **prompt groups**, and
`rl_program.py:202` enforces `batch_size % mini_batch_size == 0`. Older configs
using trajectory counts fail ~2 min in, *after* TPUs are allocated. The trainer's
own validation does not catch it — only the orchestrator's check fires.

### 3. Upstream's Raiden fallback installs a package that no longer exists

The repo-root `Dockerfile` falls back to
`pip install tpu-raiden-jax --extra-index-url .../tpu-raiden/simple/` when no
local wheel is present. The distribution was **renamed to `tpu_sync_jax`** (import
`tpu_sync`), so that fallback is stale. Not triggered here — we always supply a
local wheel — but it will bite anyone else.

### 4. `_STOP_TIMEOUT_S = 60 s` vs a ~236 s final save

The orchestrator stops waiting after 60 s and logs
`Failed to stop remote worker: TimeoutError()`, while the final checkpoint needs
~236 s. The trainer keeps finalizing, so the checkpoint normally survives — but
it is one scheduling accident from silent data loss. `run.sh stop` therefore
**refuses to tear down** until `checkpoints/<MAX_STEPS>/commit_success.txt`
exists (override with `FORCE_STOP=1`).

## Why not just use the repo-root `build_docker.sh`?

We reuse what we can — our [`Dockerfile`](Dockerfile) already calls upstream's
`scripts/install_tunix_vllm_requirement.sh`, overriding `REQ_FILE` /
`SPECIAL_REQ_FILE` to inject pinned commits. But three blockers live *inside* the
upstream Dockerfile and cannot be wrapped around:

| # | Need | Upstream | Here |
| :-- | :--- | :--- | :--- |
| 1 | JAX/libtpu pinned to the Raiden wheel's ABI | none | `jax[tpu]==0.11.0`, `jaxlib==0.11.0`, `libtpu==0.0.44` |
| 2 | maxtext from a **local** clone | `git+…/maxtext.git` @HEAD | `COPY maxtext /app/maxtext` |
| 3 | tpu-inference pinned to a commit | `special_requirements.txt`, carrying a `TODO: Re-align tpu-inference pin` | `ARG TPU_INFERENCE_COMMIT` + source overlay |
| 4 | Fail the *build* on a vLLM/tpu-inference mismatch | none | `modelopt` import assertion |
| 5 | Push to a registry | builds `tunix_base_image` locally | tags + pushes |

If (1)–(4) were added upstream as build args, `build.sh` would collapse to a thin
wrapper over `build_docker.sh` and the custom Dockerfile could be deleted. That
is the intended direction.

## Gotchas

- **`kubectl logs job/<trainer>` gives you the Pathways proxy**, not the trainer.
  All `[TrainerNode]` output — checksums, `devices_per_host`, checkpoint activity
  — needs `-c main`. `run.sh` handles this; remember it for ad-hoc debugging.
- **Cloud Logging outlives the pods** and is the better record: it includes
  crash-loop replicas that streaming misses and has no startup gaps. Default GKE
  retention is 30 days. The cluster is **shared**, so you must filter by your own
  pod name or you will read someone else's run:

  ```bash
  gcloud logging read --project cloud-tpu-shared-capacity --freshness=6h \
    'resource.type="k8s_container"
     resource.labels.cluster_name="bodaborg-v5p-nap"
     resource.labels.namespace_name="trellis"
     resource.labels.pod_name:"'"$USER"'"
     textPayload:"__grand_total__"'
  ```
