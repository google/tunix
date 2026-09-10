# GKE deployment: multi-pod GSM8K GRPO + py-inference-scheduler

Runs the distributed GRPO demo across separate pods on GKE, one TPU host per
worker, with rollout routing decided by the py-inference-scheduler service.

Manifests live in `multi-pod/` and were rendered from the templates in
`tunix/experimental/distributed/deployment/yamls/`:

| File | What it is | Where it runs |
| --- | --- | --- |
| `00-scheduler.yaml` | `pyis-scheduler` Deployment + Service (`:8100`) | CPU default pool |
| `01-orch.yaml` | `orch` JobSet: orchestrator + discovery server (`:20000`) | n2-standard-8 |
| `02-train.yaml` | `train` JobSet: trainer, LoRA, model prefetch (`:20002`) | 1-chip v6e |
| `10-roll.yaml` | `roll` JobSet, `replicas: 6`: vLLM rollout workers (`:20001`) | 6 × 1-chip v6e |

Each rollout replica derives `--worker_id=roll-${JOB_INDEX}` from the JobSet
job-index label, so the workers register with discovery as `roll-0..roll-5`.

## 1. Cluster + node pools

```bash
PROJECT=<your-project>
CLUSTER=tunix-rl-demo
ZONE=asia-northeast1-b          # any zone with v6e capacity
REPO=us-central1-docker.pkg.dev/${PROJECT}/tunix-rl

gcloud container clusters create "$CLUSTER" \
  --project="$PROJECT" --zone="$ZONE" --release-channel=rapid \
  --machine-type=n2-standard-8 --num-nodes=1 --disk-size=100 \
  --workload-pool="${PROJECT}.svc.id.goog"

# This pool is configured for spot capacity, its cheaper and easier to find. If you have access to more reliable capacity, use that.
gcloud container node-pools create tpu-v6e-1t-spot \
  --project="$PROJECT" --cluster="$CLUSTER" --zone="$ZONE" \
  --node-locations="$ZONE" --machine-type=ct6e-standard-1t \
  --spot --num-nodes=0 --disk-size=200 \
  --enable-autoscaling --min-nodes=0 --max-nodes=8
```

Preflight: `gcloud compute machine-types list --zones="$ZONE"
--filter="name~ct6e"` must return results, and the project needs >= 7 chips
of TPU v6e (spot/preemptible) quota in the region.

Install the JobSet controller (the manifests use `jobset.x-k8s.io/v1alpha2`):

```bash
VERSION=$(curl -s https://api.github.com/repos/kubernetes-sigs/jobset/releases/latest | jq -r .tag_name)
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${VERSION}/manifests.yaml"
```

## 2. Build and push images

```bash
gcloud artifacts repositories create tunix-rl \
  --repository-format=docker --location=us-central1 --project="$PROJECT" || true

# TPU image (tunix repo root; includes vLLM/tpu-inference via the install script)
docker build -t "$REPO/tunix-tpu:latest" .
docker push "$REPO/tunix-tpu:latest"

# Scheduler image (py-rl-scheduler repo root)
docker build -f integration/tunix/Dockerfile -t "$REPO/py-rl-scheduler:latest" .
docker push "$REPO/py-rl-scheduler:latest"
```

## 3. W&B credentials (optional)

The orchestrator reads `WANDB_API_KEY` from the `wandb-credentials` secret;
without it the run still works, just unlogged:

```bash
kubectl create secret generic wandb-credentials --from-literal=api-key=<your-key>
```

## 4. Launch

The manifests hardcode the image project; point them at yours, then apply the
whole directory:

```bash
sed "s/PROJECT_ID/${PROJECT}/g" multi-pod/*.yaml | kubectl apply -f -

kubectl get pods -w
kubectl logs -f job/orch-proc-0             # orchestrator / training progress
kubectl logs -f deploy/pyis-scheduler       # per-request /schedule decisions
kubectl logs -f job/roll-proc-0             # a rollout worker (0..5)
```

Success looks like: the orch pod completes with `EXIT_CODE=0`, W&B shows the
run under `trellis-gsm8k`, and the scheduler log shows one scheduling decision
per rollout request, spread across `roll-0..roll-5`.

## 5. Teardown / cost control

```bash
kubectl delete -f multi-pod/
# TPU pool autoscales to 0 when idle; force it down immediately with:
gcloud container clusters resize "$CLUSTER" --node-pool=tpu-v6e-1t-spot \
  --num-nodes=0 --zone="$ZONE" --quiet
```

## Notes / knobs

- The manifests are a rendered snapshot (kept apply-able as a record of the
  proven run). To change topology/model/flags, either edit them directly or
  re-render from `tunix/experimental/distributed/deployment/yamls/`.
- Pods use `hostNetwork` on dedicated TPU hosts; discovery is at `orch:20000`,
  rollout workers serve gRPC on `:20001`, trainer on `:20002`. The orchestrator
  reaches the scheduler via the ClusterIP service (`http://pyis-scheduler:8100`).
- The trainer prefetches Qwen3-1.7B from HuggingFace into a `/tmp` hostPath
  before starting, so restarts on the same node skip the download.
- `--weight_sync_mode=none` and LoRA (rank 16) — this demo exercises the
  scheduler-routed rollout path, not weight sync.
- The TPU pool is spot: worker pods restart on preemption (huge
  `backoffLimit`), but the orchestrator Job has `backoffLimit: 0` — if the
  orch pod dies, delete and re-apply the JobSets.

## Single-host variant (legacy: `job.yaml`)

The original demo packed everything onto one `ct5lp-hightpu-8t` host
(2 trainer chips + 6 × 1-chip rollout workers, scheduler as a native sidecar):

```bash
sed "s/PROJECT_ID/${PROJECT}/g" job.yaml | kubectl apply -f -
kubectl logs -f job/tunix-gsm8k-grpo-6w -c tunix        # launcher + orchestrator
kubectl logs -f job/tunix-gsm8k-grpo-6w -c scheduler    # /schedule decisions
kubectl delete job tunix-gsm8k-grpo-6w
```

Needs a `tpu-v5e-8t` pool (`ct5lp-hightpu-8t`) and >= 8 chips of TPU v5 Lite
PodSlice quota; chip layout and sizing live as env vars in `job.yaml`
(`ROLLOUT_PORTS` / `ROLLOUT_TPU_CHIPS_LIST`). Native sidecars require
GKE >= 1.29.
