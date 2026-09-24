# Tunix Container & Dependency Installation Architecture

This directory contains the shared dependency installation and environment
verification scripts used by both `Dockerfile` and GitHub Actions CI
(`.github/workflows/build_and_test_tunix.yml`, `.github/workflows/cpu-tests.yml`,
`.github/workflows/tpu-tests.yml`, and
`.github/workflows/tpu-nightly-regression.yml`).

---

## 1. Why `Dockerfile` Builds `FROM python:3.12-slim` Instead of `vllm/vllm-tpu`

### How Upstream `vllm/vllm-tpu` Is Built

Upstream `vllm/vllm-tpu` (`vllm-project/tpu-inference/docker/Dockerfile`) starts
`FROM python:3.12-slim-bookworm` and installs all packages directly into the
global system Python via `pip` (without `uv` or `/opt/venv`):

1. Clones `vllm-project/vllm` at `VLLM_COMMIT_HASH`, removes `torch` pins from
   `requirements/tpu.txt`, and runs `pip install -r requirements/tpu.txt`,
   `pip install -e . --no-build-isolation`,
   `pip install lm-eval[api,math]==0.4.12`, and `pip install depyf`.
2. Copies `tpu-inference` into `/workspace/tpu_inference` and runs
   `pip install -r requirements.txt`,
   `pip install -r requirements_benchmarking.txt`, and `pip install -e .`
   (without `--no-deps`).
3. Sets `ENTRYPOINT ["/entrypoint.sh"]`.

### Why Our Split-Layer `FROM python:3.12-slim` Architecture Is Superior

1. **Isolated `/opt/venv` & `/app` Layout**: Installs all runtime packages into
   `/opt/venv` (`PATH="/opt/venv/bin:$PATH"`) with `WORKDIR /app` and
   `CMD ["bash"]`, without inheriting `/entrypoint.sh` or locking into a single
   static `vllm` + `tpu-inference` commit pair.
2. **Standardized `pyproject.toml` Packaging (PEP 621, PEP 735 & `[tool.uv]`)**:
   All Python dependencies, git-pinned integration groups, and resolver
   overrides live declaratively in `pyproject.toml` while `Dockerfile` orders
   layers strictly by rate of change and build cost:
   - **Layer 1 (Base vLLM + TPU-Inference Compilation)**: Copies *only*
     `scripts/install_tunix_vllm_requirement.sh` before `pyproject.toml` and
     compiles `vllm` (`VLLM_COMMIT`) + `tpu-inference`
     (`TPU_INFERENCE_COMMIT`) inside `/opt/venv`. Routine edits to
     `pyproject.toml` or Tunix source **never** invalidate this ~15-minute C++
     compilation layer.
   - **Layer 2 (Declarative `pyproject.toml` Runtime)**: Copies `pyproject.toml`
     and runs `uv pip install -e "/app[tpu,cli,test,experimental]"` from cached
     wheels in ~15 seconds.
   - **Layer 3 (Optional Kubernetes CLI Tools)**: Runs `install_k8s_tools.sh`
     when `INSTALL_K8S_TOOLS=true`.
   - **Layer 4a (MaxText Third-Party PyPI Wheels)**: Runs
     `uv pip install -e "/app[maxtext]"` and stays cached when `MAXTEXT_REF` or
     Tunix source changes.
   - **Layer 4b (MaxText Source Only, ~15s)**: Installs
     `--group maxtext-git` from `pyproject.toml` by default (or overrides via
     `--build-arg MAXTEXT_REF=<ref>`), rebuilding MaxText in **~15 seconds**
     without `--no-cache`.
   - **Layer 5 (Tunix Source & Distributed Discovery Proto)**: Compiles
     `tunix/experimental/distributed/runtime/discovery/discovery_service.proto`
     explicitly, runs `uv pip install --no-deps -e /app`, and verifies
     `maxtext.checkpoint_conversion.to_maxtext` and `maxtext_vllm_adapter` at
     build time via `scripts/verify_environment.py`.
3. **Automatic Transitive Override Enforcement (`[tool.uv]`)**:
   `[tool.uv] override-dependencies` in `pyproject.toml` automatically enforces
   `flax>=0.12.5`, `numpy==2.3.5` (required by `tpu-inference` / Numba), and
   `protobuf>=7.35.1` (required by Tunix gRPC runtime discovery) across every
   `uv pip install` invocation, while `ENV UV_TORCH_BACKEND=cpu` ensures `uv`
   always selects CPU PyTorch wheels inside the TPU container.

---

## 2. Single Sources of Truth (Where to Update Dependencies & Pins)

| Category | Single Source of Truth | How It Is Installed |
| :--- | :--- | :--- |
| **Core & Optional PyPI Dependencies** | `pyproject.toml` (`[project.dependencies]` & `[project.optional-dependencies]`) | `uv pip install -e ".[tpu,cli,test,experimental]"` / `uv pip install -e ".[maxtext]"` |
| **vLLM & TPU-Inference Commits** | [`install_tunix_vllm_requirement.sh`](install_tunix_vllm_requirement.sh) (`VLLM_COMMIT` & `TPU_INFERENCE_COMMIT`) | `bash scripts/install_tunix_vllm_requirement.sh` |
| **MaxText & DeepSWE Git Commits (`maxtext-git`, `deepswe-git`)** | `pyproject.toml` (`[dependency-groups]`) | `uv pip install --no-deps --group maxtext-git` / `bash examples/deepswe/install_deepswe.sh` |
| **Cross-Stack Version Pins (`flax`, `numpy`, `protobuf`)** | `pyproject.toml` (`[tool.uv] override-dependencies`) | Applied automatically by `uv` |
| **Raiden (`tpu_sync_jax` GCS Wheel)** | [`install_raiden.sh`](install_raiden.sh) (`RAIDEN_WHEEL_URL` & `RAIDEN_WHEEL_SHA256`) | `bash scripts/install_raiden.sh` |
| **Kubernetes CLI Tools** | [`install_k8s_tools.sh`](install_k8s_tools.sh) (`INSTALL_K8S_TOOLS=true`) | `bash scripts/install_k8s_tools.sh` |

When adding a new `install_<component>.sh` script to `Dockerfile`, also add its
path to the `ENV_HASH` `sha256sum` list in
`.github/workflows/build_and_test_tunix.yml` so changes to the script trigger a
fresh container build in CI.

---

## 3. Script Reference

### [`install_tunix_vllm_requirement.sh`](install_tunix_vllm_requirement.sh)

Installs the pinned `vllm` (`VLLM_COMMIT`), `tpu-inference`
(`TPU_INFERENCE_COMMIT`), and `jax-smi` commits with CPU PyTorch wheels
(`--torch-backend=cpu`), and—when `pyproject.toml` is present in the repository
root—installs `.[tpu,cli,test,experimental]`. Used both in Layer 1 of
`Dockerfile` (before `COPY pyproject.toml`) and by developers setting up a local
TPU VM.

### [`install_raiden.sh`](install_raiden.sh)

Installs the native TPU weight-synchronization wheel (`tpu_sync_jax`) from
`./raiden_wheels/*.whl` (`RAIDEN_WHEEL_DIR`) or `RAIDEN_WHEEL_URL` and compiles
`tunix/experimental/distributed/runtime/discovery/discovery_service.proto`.

### [`install_k8s_tools.sh`](install_k8s_tools.sh)

Installs `google-cloud-cli`, `google-cloud-cli-gke-gcloud-auth-plugin`,
`kubectl`, `k9s`, and debugging utilities when `INSTALL_K8S_TOOLS=true`.

### [`verify_environment.py`](verify_environment.py)

Validates Python packages and `tunix.__version__` during the Docker build
(`Dockerfile`).

---

## 4. Building, Resolving & Publishing Container Images

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg INSTALL_MAXTEXT=true \
  -t tunix-mlperf:latest .
```

In `.github/workflows/build_and_test_tunix.yml` (`build_tunix_docker`), every CI
run resolves or builds an immutable `tunix/mlperf:<github.sha>` container image
before launching `tunix_cpu_unit_tests` (`run_vllm`) and `tunix_tpu_unit_tests`
(`run_dev`):

1. **Content-Addressed Environment Hash (`:env-<hash>` Fast Path, ~5s)**:
   Computes a SHA-256 hash (`ENV_HASH`) over all container-defining files
   (`Dockerfile`, `pyproject.toml`, `scripts/install_*.sh`,
   `scripts/verify_environment.py`, `examples/deepswe/install_deepswe.sh`, and
   `discovery_service.proto`). On pull requests where `:env-${ENV_HASH}` already
   exists in Artifact Registry, `build_tunix_docker` aliases `:env-${ENV_HASH}`
   to `:${GITHUB_SHA}` and `:${SHORT_SHA}` in ~5s without rebuilding; downstream
   jobs overlay the PR's Tunix code in ~1s at runtime.
2. **Build & Multi-Region Publish Path**: When any container-defining file
   changes (or on `main` / `schedule` / `workflow_dispatch`),
   `build_tunix_docker` builds and verifies `Dockerfile`
   (`INSTALL_MAXTEXT=true`) via BuildKit with Artifact Registry layer caching
   (`type=registry,ref=.../tunix/mlperf:buildcache`) and publishes
   `:${GITHUB_SHA}`, `:${SHORT_SHA}`, and `:env-${ENV_HASH}` to:
   - `us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/mlperf`
   - `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/mlperf`
3. **Immutable Image Passing & Concurrency Safety**: Both `cpu-tests.yml`
   (`run_vllm`) and `tpu-tests.yml` (`run_dev`) receive
   `needs.build_tunix_docker.outputs.image_uri` (`:${GITHUB_SHA}`), guaranteeing
   every test job in a workflow run executes against the exact same verified
   container image. The `:latest` tag is only updated on `main` branch runs,
   preventing concurrent PRs from overwriting `:latest`.

---

## 5. CI Caching & Test Throughput Optimizations

### Container-Compatible Model, Dataset & Checkpoint Caching (Zero Image Bloat)

To keep `tunix/mlperf` generic for Trellis and MLPerf workloads (and keep
container pull + initialization on TPU runners at ~13s), **zero model weights,
datasets, or checkpoints are baked into the Docker image**. Instead,
`.github/workflows/tpu-tests.yml` mounts and restores cached artifacts into the
container's `/root/.cache` hierarchy at runtime:

- **Pinned Container Cache Roots**:
  - `HF_HOME=/root/.cache/huggingface`: Hugging Face Hub weights & tokenizers.
  - `KAGGLEHUB_CACHE=/root/.cache/kagglehub`: Kaggle model weights (e.g.
    `gemma2-2b-it` used in SFT smoke tests).
  - `TFDS_DATA_DIR=/root/.cache/tensorflow_datasets`: Pre-built TFDS datasets
    (e.g. `gsm8k`).
  - `JAX_COMPILATION_CACHE_DIR=/root/.cache/jax_compilation_cache`: Persistent
    XLA TPU/CPU compiled executables across runs.
  - `ARTIFACT_ROOT=/root/.cache/tunix_artifacts/qwen3_dist_gsm8k`: Caches both
    the downloaded `Qwen/Qwen3-0.6B` safetensors (`models/`) and the
    pre-converted MaxText Orbax checkpoint (`maxtext_models/`), allowing
    `ci_smoke_gsm8k_qwen3_0p6b_maxtext.sh` to skip `convert_hf_to_maxtext` on
    warm runs while writing ephemeral training checkpoints to `/tmp`.
- **Selective Cache Exclusions (`xet/` and `>7B` Models)**:
  - `HF_XET_HIGH_PERFORMANCE=1` accelerates cold downloads via Xet chunk
    streaming, but writes duplicate staging chunks to
    `/root/.cache/huggingface/xet` alongside the final `hub/` files. Excluding
    `!/root/.cache/huggingface/xet` prevents archiving duplicate chunks.
  - `Qwen/Qwen2.5-7B` (~15 GB raw / ~6.5 GB compressed) streams from the Xet CDN
    to a GCE `v6e-8` VM in `us-central1` in ~7s, whereas restoring a 6.5 GB
    archive from GitHub Actions cache (Azure Blob Storage) takes ~52s and
    consumes most of GitHub's 10 GB repository cache quota. Excluding
    `!/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B*` keeps the
    `run_dev` cache compact (~1.5 GB) and fast to restore (~10s).

### Fast Dependency Resolution (`uv`) & Parallel Test Execution (`pytest-xdist`)

- **`uv pip install --system` vs. `pip install`**: In
  `.github/workflows/cpu-tests.yml`, replacing `python -m pip install` with
  `uv pip install --system` reduces wheel resolution and installation on
  ephemeral Ubuntu runners from ~70–127s to ~10–14s. When combining PyPI with
  `--extra-index-url https://download.pytorch.org/whl/cpu` (in `run_dev`),
  `--index-strategy unsafe-best-match` is passed so `uv` can select newer PyPI
  versions (such as `requests>=2.32.2` required by `datasets>=3.0.0`) rather
  than pinning to older versions hosted on the PyTorch CPU index
  (`requests==2.28.1`).
- **`pytest -n auto --dist=loadfile`**: Parallelizes multi-file CPU unit test
  suites across all available runner vCPUs while keeping all tests within a
  given module on the same worker (`--dist=loadfile`), avoiding redundant
  per-worker initialization of module-scoped JAX models and meshes.
