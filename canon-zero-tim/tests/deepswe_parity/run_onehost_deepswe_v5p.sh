#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

STAGE="${1:?usage: run_onehost_deepswe_v5p.sh <sandbox-only|rollout-only|backward-no-commit>}"
case "$STAGE" in
  sandbox-only)
    ROLLOUT_ONLY=1
    NO_COMMIT=0
    ;;
  rollout-only)
    ROLLOUT_ONLY=1
    NO_COMMIT=0
    ;;
  backward-no-commit)
    ROLLOUT_ONLY=0
    NO_COMMIT=1
    ;;
  *)
    echo "one-host runner admits rollout-only or backward-no-commit" >&2
    exit 2
    ;;
esac

SANDBOX_RUNTIME="${DEEPSWE_ONEHOST_SANDBOX_RUNTIME:-direct}"
AGENT_SANDBOX_COMMIT="7935857fee859bb18752ee04d8948b975e47ff20"
AGENT_SANDBOX_ROOT="${DEEPSWE_AGENT_SANDBOX_ROOT:-}"
SANDBOX_CAPACITY="${DEEPSWE_ONEHOST_SANDBOX_CAPACITY:-}"
SANDBOX_NODEPOOL="${DEEPSWE_ONEHOST_SANDBOX_NODEPOOL:-}"
case "$SANDBOX_RUNTIME" in
  direct)
    if [[ -n "$AGENT_SANDBOX_ROOT" || -n "$SANDBOX_CAPACITY" || \
          -n "$SANDBOX_NODEPOOL" ]]; then
      echo "Fleet-only one-host values are invalid with sandbox runtime direct" >&2
      exit 2
    fi
    ;;
  fleet)
    if [[ ! "$SANDBOX_CAPACITY" =~ ^[1-9][0-9]*$ ]] || \
       (( SANDBOX_CAPACITY < 8 )); then
      echo "one-host Fleet capacity must be an explicit integer >=8" >&2
      exit 2
    fi
    if [[ -z "$SANDBOX_NODEPOOL" ]]; then
      echo "one-host Fleet requires DEEPSWE_ONEHOST_SANDBOX_NODEPOOL" >&2
      exit 2
    fi
    if [[ -z "$AGENT_SANDBOX_ROOT" || ! -d "$AGENT_SANDBOX_ROOT/.git" ]]; then
      echo "one-host Fleet requires an exact DEEPSWE_AGENT_SANDBOX_ROOT checkout" >&2
      exit 2
    fi
    ACTUAL_AGENT_SANDBOX_COMMIT="$(git -C "$AGENT_SANDBOX_ROOT" rev-parse HEAD)"
    if [[ "$ACTUAL_AGENT_SANDBOX_COMMIT" != "$AGENT_SANDBOX_COMMIT" ]]; then
      echo "Agent Sandbox source changed: expected=$AGENT_SANDBOX_COMMIT actual=$ACTUAL_AGENT_SANDBOX_COMMIT" >&2
      exit 2
    fi
    for path in \
      "$AGENT_SANDBOX_ROOT/examples/agent-sandbox-rl/agent_sandbox_rl" \
      "$AGENT_SANDBOX_ROOT/clients/python/agentic-sandbox-client/k8s_agent_sandbox"; do
      if [[ ! -d "$path" ]]; then
        echo "Agent Sandbox exact checkout is incomplete: $path" >&2
        exit 2
      fi
    done
    ;;
  *)
    echo "DEEPSWE_ONEHOST_SANDBOX_RUNTIME must be exactly direct or fleet" >&2
    exit 2
    ;;
esac
if [[ "$STAGE" == "sandbox-only" && "$SANDBOX_RUNTIME" != "fleet" ]]; then
  echo "sandbox-only is an Agent Sandbox Fleet lifecycle probe" >&2
  exit 2
fi

PYTHON="${DEEPSWE_TRAIN_PYTHON:-/mnt/disks/tunix-data/venvs/train/bin/python}"
MODEL_PATH="${DEEPSWE_QWEN4B_MODEL_PATH:-/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
DATASET_CACHE="${DEEPSWE_DATASET_CACHE:-/mnt/disks/tunix-data/dataset_cache}"
R2EGYM_ROOT="${DEEPSWE_R2EGYM_ROOT:-/home/yuxuan/tunix/submodules/R2E-Gym}"
GOLD_WHITELIST="${DEEPSWE_ONEHOST_WHITELIST:-/home/yuxuan/code_rl_repro/google_dev/tunix/experimental/smoke_test_whitelist.jsonl}"
TASK_IMAGE="${DEEPSWE_ONEHOST_TASK_IMAGE:-namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d}"
RUN_ID="${CANON_RUN_ID:-onehost-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
ARTIFACT_DIR="${DEEPSWE_ONEHOST_ARTIFACT_DIR:-/mnt/disks/tunix-data/deepswe-onehost-evidence/${RUN_ID}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing DeepSWE training interpreter: $PYTHON" >&2
  exit 2
fi
for path in "$MODEL_PATH" "$DATASET_CACHE" "$R2EGYM_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "missing one-host prerequisite directory: $path" >&2
    exit 2
  fi
done
if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "Qwen3-4B snapshot is incomplete: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$GOLD_WHITELIST" ]]; then
  echo "missing reviewed DeepSWE whitelist: $GOLD_WHITELIST" >&2
  exit 2
fi
if [[ "$ARTIFACT_DIR" != /* ]] || [[ -e "$ARTIFACT_DIR" ]]; then
  echo "artifact directory must be a new absolute path: $ARTIFACT_DIR" >&2
  exit 2
fi

SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_BRANCH="${CANON_SOURCE_BRANCH:-$(git branch --show-current)}"
SOURCE_BRANCH="${SOURCE_BRANCH:-detached}"
SOURCE_DIRTY="$(git status --porcelain --untracked-files=no)"
if [[ -n "$SOURCE_DIRTY" ]] && \
   [[ "${DEEPSWE_ONEHOST_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "one-host evidence requires a clean tracked worktree" >&2
  echo "set DEEPSWE_ONEHOST_ALLOW_DIRTY=1 only for development evidence" >&2
  exit 2
fi
R2EGYM_SHA="$(git -C "$R2EGYM_ROOT" rev-parse HEAD)"
if [[ "$R2EGYM_SHA" != "0d94c4eb9431cd195c55a7ea3abd54006c9a1735" ]]; then
  echo "R2E-Gym source changed: $R2EGYM_SHA" >&2
  exit 2
fi

export CANON_DEEPSWE_ONEHOST_SMOKE=1
export CANON_DEEPSWE_ONEHOST_STAGE="$STAGE"
export CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY="$ROLLOUT_ONLY"
export CANON_DEEPSWE_ONEHOST_NO_COMMIT="$NO_COMMIT"
export CANON_DEEPSWE_ONEHOST_DEBUG_DIR="$ARTIFACT_DIR"
export CANON_DEEPSWE_ONEHOST_REPORT="$ARTIFACT_DIR/backward_no_commit.json"
export CANON_DEEPSWE_ONEHOST_TASK_IMAGE="$TASK_IMAGE"
export DEEPSWE_ONEHOST_REAL_B2=1
export CANON_EXPECT_COMMIT="$SOURCE_SHA"
export CANON_SOURCE_BRANCH="$SOURCE_BRANCH"
export CANON_RUN_ID="$RUN_ID"
export CANON_P34_DEEPSWE=0
export CANON_P39_64CHIP_PILOT=0
export CANON_P43_DEEPSWE_DEBUG=0
export CANON_P44_DEEPSWE_PARITY=0
export DATASET_CACHE
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export CANON_DEEPSWE_SANDBOX_RUNTIME="$SANDBOX_RUNTIME"
if [[ "$SANDBOX_RUNTIME" == "fleet" ]]; then
  export CANON_AGENT_SANDBOX_COMMIT="$AGENT_SANDBOX_COMMIT"
  export R2E_SANDBOX_CAPACITY="$SANDBOX_CAPACITY"
  export NODE_SELECTOR_KEY="cloud.google.com/gke-nodepool"
  export NODE_SELECTOR_VAL="$SANDBOX_NODEPOOL"
  export R2E_K8S_NAMESPACE="${DEEPSWE_ONEHOST_SANDBOX_NAMESPACE:-default}"
  export R2E_ACTIVE_DEADLINE_SECONDS=5100
  export R2E_POD_START_TIMEOUT_SECONDS=1200
  export R2E_POD_DELETE_TIMEOUT_SECONDS=300
  export R2E_K8S_CPU=2
  export R2E_K8S_MEM=4Gi
  export R2E_K8S_CPU_LIMIT=4
  export R2E_K8S_MEM_LIMIT=8Gi
  export IMAGE_PULL_SECRET="${DEEPSWE_ONEHOST_IMAGE_PULL_SECRET:-dockerhub-pro}"
fi
# TPU must remain the default backend, while vLLM's TPU weight loader needs one
# visible CPU device for its host-side staging mesh.
export JAX_PLATFORMS=tpu,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SKIP_JAX_PRECOMPILE=true
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [[ "$SANDBOX_RUNTIME" == "fleet" ]]; then
  export PYTHONPATH="$ROOT:$R2EGYM_ROOT:$AGENT_SANDBOX_ROOT/examples/agent-sandbox-rl:$AGENT_SANDBOX_ROOT/clients/python/agentic-sandbox-client${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$ROOT:$R2EGYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
unset DOCKER_HOST JAX_BACKEND_TARGET PATHWAYS_HEAD
unset CANON_ALIGNMENT_GATE CANON_ALIGNMENT_GATE_ONLY
unset CANON_ALIGNMENT_UPDATE_CANARY CANON_ALIGNMENT_TRAIN
unset CANON_P28_SEGMENTED_TRAIN CANON_P28_G6_UPDATE

echo "[DEEPSWE.ONEHOST.INVENTORY] source_sha=$SOURCE_SHA source_branch=$SOURCE_BRANCH tracked_dirty=$([[ -n "$SOURCE_DIRTY" ]] && echo 1 || echo 0) r2egym_sha=$R2EGYM_SHA stage=$STAGE sandbox_runtime=$SANDBOX_RUNTIME artifact_dir=$ARTIFACT_DIR"
"$PYTHON" -c '
import jax
devices = jax.devices()
assert len(devices) == 4, devices
assert all(device.platform == "tpu" for device in devices), devices
print(
    "[DEEPSWE.ONEHOST.DEVICES] PASS count=4 kinds="
    + ",".join(str(device.device_kind) for device in devices),
    flush=True,
)
'
if [[ "$SANDBOX_RUNTIME" == "direct" ]]; then
  "$PYTHON" -c '
import docker
import r2egym
client = docker.from_env()
assert client.ping() is True
print("[DEEPSWE.ONEHOST.R2E] PASS runtime=direct docker=1 import=1", flush=True)
'
else
  "$PYTHON" -c '
import inspect
import os
from pathlib import Path
import agent_sandbox_rl
from agent_sandbox_rl import FleetConfig, SandboxFleet
from agent_sandbox_rl.adapters.r2egym import make_fleet_repo_env
from examples.deepswe.sandbox_fleet import verify_fleet_permissions
from kubernetes import client, config

expected_root = Path(os.environ["DEEPSWE_AGENT_SANDBOX_ROOT"]).resolve()
loaded = Path(inspect.getfile(agent_sandbox_rl)).resolve()
if expected_root not in loaded.parents:
  raise RuntimeError(f"Agent Sandbox import escaped exact checkout: {loaded}")
FleetConfig(max_concurrent=4, max_warmpool_size=4)
assert callable(make_fleet_repo_env)
assert hasattr(SandboxFleet, "warm_images")
if os.environ.get("KUBERNETES_SERVICE_HOST"):
  config.load_incluster_config()
else:
  config.load_kube_config()
groups = client.ApisApi().get_api_versions(_request_timeout=60).groups
names = {group.name for group in groups}
required = {"agents.x-k8s.io", "extensions.agents.x-k8s.io"}
missing = required - names
if missing:
  raise RuntimeError(f"Agent Sandbox API groups unavailable: {sorted(missing)}")
verify_fleet_permissions(
    namespace=os.environ["R2E_K8S_NAMESPACE"],
    image_pull_secret=os.environ.get("IMAGE_PULL_SECRET") or None,
)
print(
    "[DEEPSWE.ONEHOST.R2E] PASS runtime=fleet exact_source=1 "
    "kubeconfig=1 api_groups=1 rbac=1",
    flush=True,
)
'
fi

if [[ "$STAGE" == "sandbox-only" ]]; then
  "$PYTHON" -c '
import concurrent.futures
import os
import time

from examples.deepswe.sandbox_fleet import DeepSWESandboxFleet

image = os.environ["CANON_DEEPSWE_ONEHOST_TASK_IMAGE"]
rows = [
    {"instance_id": "onehost-fleet-0", "docker_image": image, "prompts": "probe"},
    {"instance_id": "onehost-fleet-1", "docker_image": image, "prompts": "probe"},
]
manager = DeepSWESandboxFleet(
    rows, batch_size=2, num_generations=2, max_concurrency=4
)
batch_started = time.perf_counter()
try:
  manager.prepare_batch({"docker_image": [image, image]})
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(
            manager.acquire,
            rows[index % 2],
            batch_started_monotonic=batch_started,
        )
        for index in range(4)
    ]
    handles = [future.result()[0] for future in futures]
  for handle in handles:
    output = handle.exec(["/bin/sh", "-c", "printf canon-agent-sandbox"])
    if output != "canon-agent-sandbox":
      raise RuntimeError(f"unexpected sandbox exec output: {output!r}")
    manager.release(handle)
finally:
  manager.close()
print(
    "DEEPSWE_ONEHOST_SANDBOX_FLEET_PASS prompts=2 generations=2 "
    "claims=4 leaked_resources=0",
    flush=True,
)
'
  exit 0
fi

"$PYTHON" examples/deepswe/train_deepswe_nb.py \
  --model_version Qwen3-4B-Instruct-2507 \
  --model_absolute_path "$MODEL_PATH" \
  --gold_whitelist "$GOLD_WHITELIST" \
  --batch_size 2 \
  --mini_batch_size 2 \
  --train_micro_batch_size 1 \
  --compute_logps_micro_batch_size 1 \
  --rollout_micro_batch_size 1 \
  --num_generations 2 \
  --num_iterations 1 \
  --max_prompt_length 3584 \
  --max_response_length 512 \
  --max_turns 2 \
  --max_steps 1 \
  --num_epochs 1 \
  --eval_every_n_steps 10 \
  --max_concurrency 4 \
  --temperature 0.7 \
  --rollout_engine vllm \
  --vllm_utilization 0.3 \
  --rollout_vllm_max_num_seqs 4 \
  --max_num_batched_tokens 512 \
  --rollout_mesh_dp 1 \
  --rollout_mesh_tp 4 \
  --train_mesh_dp 1 \
  --train_mesh_tp 4 \
  --ckpt_dir none \
  --dtype bfloat16 \
  --param_dtype bfloat16 \
  --use_rollout_logps \
  --logging_level INFO

if [[ ! -s "$ARTIFACT_DIR/run_manifest.json" ]] || \
   [[ ! -s "$ARTIFACT_DIR/batch-000000.trajectories.jsonl.gz" ]] || \
   [[ ! -s "$ARTIFACT_DIR/batch_metrics.jsonl" ]]; then
  echo "one-host trajectory evidence is incomplete: $ARTIFACT_DIR" >&2
  exit 1
fi

"$PYTHON" -c '
import gzip
import json
import os
from pathlib import Path

root = Path(os.environ["CANON_DEEPSWE_ONEHOST_DEBUG_DIR"])
with gzip.open(root / "batch-000000.trajectories.jsonl.gz", "rt") as source:
  rows = [json.loads(line) for line in source if line.strip()]
if len(rows) != 4:
  raise RuntimeError(f"one-host B2xG2 requires four trajectory rows, got {len(rows)}")
with open(root / "batch_metrics.jsonl", encoding="utf-8") as source:
  metrics = [json.loads(line) for line in source if line.strip()]
if not metrics:
  raise RuntimeError("one-host batch metrics are empty")
last = metrics[-1]
if last.get("trajectories") != 4 or last.get("complete_trajectories") != 4:
  raise RuntimeError(f"one-host B2xG2 batch is incomplete: {last}")
print("[DEEPSWE.ONEHOST.ROWS] PASS trajectories=4 complete=4", flush=True)
'

if [[ "$STAGE" == "rollout-only" ]]; then
  echo "DEEPSWE_ONEHOST_ROLLOUT_PASS model=qwen3-4b-instruct-2507 devices=4 trajectories=4"
  exit 0
fi

VERDICT="$("$PYTHON" -c '
import json
import os
path = os.environ["CANON_DEEPSWE_ONEHOST_REPORT"]
with open(path, encoding="utf-8") as source:
  report = json.load(source)
assert report["commits"] == 0, report
assert report["gradient_finite"] is True, report
assert not report["model_changed_paths"], report
assert not report["optimizer_changed_paths"], report
assert not report["accumulator_changed_paths"], report
assert not report["reference_changed_paths"], report
print(report["verdict"])
')"
if [[ "$VERDICT" == "PASS" ]]; then
  echo "DEEPSWE_ONEHOST_BACKWARD_NO_COMMIT_PASS model=qwen3-4b-instruct-2507 devices=4"
  exit 0
fi
if [[ "$VERDICT" == "INCONCLUSIVE_NO_SIGNAL" ]]; then
  echo "DEEPSWE_ONEHOST_BACKWARD_INCONCLUSIVE_NO_SIGNAL model=qwen3-4b-instruct-2507 devices=4" >&2
  exit 3
fi
echo "one-host backward report failed with verdict=$VERDICT" >&2
exit 1
