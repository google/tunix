# GKE deployment: 6-worker GSM8K GRPO + py-inference-scheduler sidecar

Runs the distributed GRPO demo on one TPU v5e 8-chip node (`ct5lp-hightpu-8t`,
us-central1-a): 2 chips for the trainer, 6 × 1-chip vLLM rollout workers, with
rollout routing decided by the py-inference-scheduler sidecar.

## 1. Cluster + node pools

```bash
PROJECT=<your-project>
CLUSTER=tunix-rl-demo
ZONE=us-central1-a
REPO=us-central1-docker.pkg.dev/${PROJECT}/tunix-rl

gcloud container clusters create "$CLUSTER" \
  --project="$PROJECT" --zone="$ZONE" --release-channel=rapid \
  --machine-type=n2-standard-8 --num-nodes=1 --disk-size=100 \
  --workload-pool="${PROJECT}.svc.id.goog" --addons=GcsFuseCsiDriver

# Spot capacity (~60-70% cheaper, preemptible; drop --spot for on-demand).
# Starts at 0 nodes: the TPU node only exists (and bills) while a Job runs.
gcloud container node-pools create tpu-v5e-8t \
  --project="$PROJECT" --cluster="$CLUSTER" --zone="$ZONE" \
  --node-locations="$ZONE" --machine-type=ct5lp-hightpu-8t \
  --spot --num-nodes=0 --disk-size=200 \
  --enable-autoscaling --min-nodes=0 --max-nodes=1
```

Preflight: `gcloud compute machine-types list --zones=us-central1-a
--filter="name~ct5lp"` must return results, and the project needs >= 8 chips of
"TPU v5 Lite PodSlice" quota in us-central1. Native sidecars require GKE >=
1.29 (the `rapid` channel is well past this).

## 2. Build and push images

```bash
gcloud artifacts repositories create tunix-rl \
  --repository-format=docker --location=us-central1 --project="$PROJECT" || true

# TPU image (tunix repo root; includes vLLM/tpu-inference via the install script)
docker build -t "$REPO/tunix-tpu:latest" .
docker push "$REPO/tunix-tpu:latest"

# Scheduler sidecar image (py-rl-scheduler repo root)
docker build -f integration/tunix/Dockerfile -t "$REPO/py-rl-scheduler:latest" .
docker push "$REPO/py-rl-scheduler:latest"
```

## 3. Launch

```bash
sed "s/PROJECT_ID/${PROJECT}/g" job.yaml | kubectl apply -f -

kubectl get pods -l app=tunix-gsm8k-grpo -w
kubectl logs -f job/tunix-gsm8k-grpo-6w -c tunix        # launcher + orchestrator
kubectl logs -f job/tunix-gsm8k-grpo-6w -c scheduler    # per-request /schedule decisions
```

Success looks like: the tunix container prints "Distributed GSM8K GRPO chain
demo (vLLM) finished successfully." and the scheduler log shows one scheduling
decision per rollout request, spread across `rollout-0..rollout-5`.

## 4. Teardown / cost control

```bash
kubectl delete job tunix-gsm8k-grpo-6w
# TPU pool autoscales to 0 when idle; force it down immediately with:
gcloud container clusters resize "$CLUSTER" --node-pool=tpu-v5e-8t \
  --num-nodes=0 --zone="$ZONE" --quiet
```

## Notes / knobs

- Chip layout and workload sizing live as env vars in `job.yaml`
  (`ROLLOUT_PORTS` / `ROLLOUT_TPU_CHIPS_LIST` are parallel lists; ports are
  comma-separated, chip groups semicolon-separated).
- `USE_LORA=1` is set because full fine-tuning 1.7B with fp32 Adam state is
  tight on the trainer's 2 × 16GB chips.
- The model is downloaded from HuggingFace into an emptyDir on first start;
  for faster restarts, swap the `artifacts` volume for a GCS bucket via the
  gcsfuse CSI driver (addon is already enabled on the cluster).
- The reference inference node (KL, `--beta != 0`) is not enabled: it would
  need chips beyond the 8 on this host.
- The TPU pool is Spot: a preemption kills the run mid-flight (the Job has
  `backoffLimit: 0`, so it won't retry automatically — re-apply it). Fine for
  this demo; use on-demand for anything long-running.
- This manifest was drafted alongside the integration and has not yet been
  run on a real cluster; expect first-run friction (image sizes, vLLM warmup
  timeouts — bump `WAIT_TIMEOUT_SECS` if node startup is slow).
