# FrozenLake Score Centering A/B on GKE (v5p-32, Pathways)

Runs `examples/frozenlake/train_frozenlake_qwen3.py` as a Pathways client on a
v5p-32 (2x2x4) slice, one JobSet per arm:

| Arm      | JobSet          | Flag                             |
|----------|-----------------|----------------------------------|
| SC on    | `fl-sc-on-s42`  | `--score_centering`              |
| SC off   | `fl-sc-off-s42` | `--score_centering_diagnostics`  |

`--score_centering_diagnostics` requests the sampler's top-k logprobs and logs
the `score_centering/*` metrics without changing the loss, so both arms report
the same metrics. The recipe splits the 16 chips with `--rollout_devices 8`:
the first 8 devices run vLLM (DP=2 x TP=4), the last 8 hold the actor and the
reference model (TP=8).

## Files

- `frozenlake_sc_ab_v5p32.yaml`: JobSet template (`{{PLACEHOLDER}}`s are filled
  in by the launcher).
- `launch_sc_ab.sh`: build / up / down / status.
- `cloudbuild.yaml`, `Dockerfile.pathways`: image = repo `Dockerfile` +
  `pathwaysutils` + `gymnasium`, with jax/flax pinned back to the versions
  tpu-inference expects. Before the image is pushed, `image_check.py` imports
  everything the recipe needs and the Score Centering unit tests run.

## Usage

```bash
# 1. One-time: W&B key as a Secret (never put the key in a file or YAML).
kubectl create secret generic linchai-wandb-api-key --from-literal=api-key=<KEY>

# 2. Image (async Cloud Build from a snapshot of the working tree).
examples/frozenlake/gke/launch_sc_ab.sh build
gcloud builds list --project=cloud-tpu-multipod-dev --limit=3

# 3. Smoke test (3 steps, eval + checkpoint at step 2), then clean up.
IMAGE=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/linchai-repo/tunix_base_image:<tag>
examples/frozenlake/gke/launch_sc_ab.sh up --image $IMAGE --smoke
examples/frozenlake/gke/launch_sc_ab.sh down --smoke

# 4. Full runs (200 steps each).
examples/frozenlake/gke/launch_sc_ab.sh up --image $IMAGE
examples/frozenlake/gke/launch_sc_ab.sh status
kubectl logs -f <jobset>-pathways-head-0-0-<suffix> -c jax-tpu
```

`up --dry-run` renders the YAMLs and runs a server-side dry run. Settings such
as the bucket, reservation, W&B project/group or seed can be overridden via
environment variables (see the top of `launch_sc_ab.sh`).

## Notes

- Data: `gs://linchai-bucket-dev/data/frozenlake/grid5to12/` is a copy of
  `gs://tunix/data/Frozenlake/grid5to12/` (5x5 to 12x12 grids), because the
  cluster's node service account cannot read `gs://tunix`.
- Checkpoints: every 20 steps to `CKPT_DIR`, ~16 GiB each (fp32 params + Adam
  state); all are kept (`--max_to_keep 100`). The node service account only has
  object access on the bucket, so orbax's bucket-metadata lookup
  (`storage.buckets.get`, used to list or delete steps) returns 403; the recipe
  treats that as a flat-namespace bucket.
- Restarts: `failurePolicy.maxRestarts=1`. A restarted JobSet resumes from the
  latest checkpoint in `CKPT_DIR` and keeps logging to the same W&B run
  (`WANDB_RUN_ID` is fixed per launch, `WANDB_RESUME=allow`). Interval
  checkpoints store the global step, so a resumed run continues where it
  stopped. The final checkpoint written at the end of training does not
  (`PeftTrainer._save_last_checkpoint` passes no custom metadata), so
  relaunching a finished run with the same `--run-tag` starts another
  `--num_batches` steps from the final weights; use a new `--run-tag` instead.
- `up --no-ckpt` launches without checkpoints (empty `CKPT_DIR`). A restarted
  JobSet then retrains from step 0 and logs to a new W&B run
  `<run id>-a<restart attempt>`, because W&B drops steps below a run's last
  logged step.
- `EXIT_AFTER_TRAIN=1` makes the controller exit right after training, so the
  JobSet completes and releases the TPUs even if rollout threads linger.
- Hotfix without rebuilding: `launch_sc_ab.sh code-tarball` uploads `tunix/`
  and `examples/` from the working tree; pass the printed path to
  `up --code-tarball`.
