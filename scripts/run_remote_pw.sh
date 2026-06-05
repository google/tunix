#!/bin/bash
set -euo pipefail
# Usage:
#   CLUSTER_NAME=<cluster> JOB_NAME=<job> TPU_TYPE=<tpu> TOPOLOGY=<topo> ./run_remote_pw.sh
# Examples:
#   ./run_remote_pw.sh                    # uses defaults
#   CLUSTER_NAME=lance-v5p-16 JOB_NAME=lancewang-v5p-pw-3 \
#     JOB_NAME=lancewang-v5p-pw-3 TPU_TYPE=v5p TOPOLOGY=2x2x4 PROXY_MEMORY_LIMIT=320G PROXY_CPU_LIMIT=50 \
#     ./run_remote_pw.sh

CLUSTER_NAME=${CLUSTER_NAME:-tunix-v5p-16}
REMOTE_PW_PORT=${REMOTE_PW_PORT:-8890}
JOB_NAME=${JOB_NAME:-tunix-${USER}-${TPU_TYPE}-${TOPOLOGY}-pw-0}

# cpu-np or cpu-np-large-mem-disk
CPU_POOL_NAME=${CPU_POOL_NAME:-cpu-np}


# https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
TPU_ARG=${TPU_TYPE:-v5p}

TOPOLOGY=${TOPOLOGY:-2x2x2}
TPU_SLICE="tpu-${TPU_ARG}-slice"

# Number of TPU chips per worker pod (`google.com/tpu: 4` below).
CHIPS_PER_WORKER=${CHIPS_PER_WORKER:-4}

# Parse TOPOLOGY (e.g. 2x2x2) into components TOPO_X, TOPO_Y, TOPO_Z
# Provide safe defaults of 1 when parts are missing.
IFS='x' read -r TOPO_X TOPO_Y TOPO_Z <<< "${TOPOLOGY}"
TOPO_X=${TOPO_X:-1}
TOPO_Y=${TOPO_Y:-1}
TOPO_Z=${TOPO_Z:-1}

# Compute total number of TPU chips.
TOTAL_TPU_CHIPS=$(( TOPO_X * TOPO_Y * TOPO_Z ))

# Compute worker pods from topology and chips per worker request.
# Example: 2x2x2 => 8 chips, CHIPS_PER_WORKER=4 => 2 worker pods.
if (( TOTAL_TPU_CHIPS % CHIPS_PER_WORKER != 0 )); then
  echo "ERROR: TOPOLOGY=${TOPOLOGY} yields ${TOTAL_TPU_CHIPS} chips, which is not divisible by CHIPS_PER_WORKER=${CHIPS_PER_WORKER}." >&2
  exit 1
fi
WORKER_PODS=${WORKER_PODS:-$((TOTAL_TPU_CHIPS / CHIPS_PER_WORKER))}

ZONE=${ZONE:-europe-west4-b}
REGION=${REGION:-europe-west4}
PROJECT=${PROJECT:-cloud-tpu-multipod-dev}

GITHUB_PATH=${GITHUB_PATH:-/github} # Specify your repos are in github folder
TEMP_BUCKET=${TEMP_BUCKET:-lancewang-dev-supercomputer-testing/tunix/pw}

JAX_TPU_IMAGE=${JAX_TPU_IMAGE:-gcr.io/cloud-tpu-multipod-dev/lance_deepswe:latest}
PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest}
PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest}
PROXY_CPU_LIMIT=${PROXY_CPU_LIMIT:-50}
PROXY_MEMORY_LIMIT=${PROXY_MEMORY_LIMIT:-320G}
JAX_CPU_LIMIT=${JAX_CPU_LIMIT:-50}
JAX_MEMORY_LIMIT=${JAX_MEMORY_LIMIT:-300G}
# TPU worker ephemeral storage knobs (to reduce disk-pressure evictions).
# Note: some Kueue ClusterQueues do not expose ephemeral-storage flavors.
# In that case, setting container ephemeral-storage resources causes admission failure:
#   "resource ephemeral-storage unavailable in ClusterQueue"
# Keep this empty by default (disabled) and rely on WORKER_TMP_SIZE_LIMIT.
# Choose a conservative /tmp cap to leave headroom for image layers, logs and system daemons.
WORKER_TMP_SIZE_LIMIT=${WORKER_TMP_SIZE_LIMIT:-25Gi}

if [ ! -d "${GITHUB_PATH}/experimental/.git" ]; then
  git clone sso://user/abhinavsing/experimental "${GITHUB_PATH}/experimental"
fi


cat > jobset.yaml <<EOF
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: ${JOB_NAME}
  labels:
    kueue.x-k8s.io/queue-name: multislice-queue
    xpk.google.com/workload: ${JOB_NAME}
spec:
  coordinator:
    replicatedJob: pathways-head
  failurePolicy:
    restartStrategy: Recreate
  network:
    enableDNSHostnames: true
    publishNotReadyAddresses: true
  startupPolicy:
    startupPolicyOrder: InOrder
  successPolicy:
    operator: All
    targetReplicatedJobs:
      - pathways-head
  replicatedJobs:
    - name: pathways-head
      replicas: 1
      template:
        metadata:
          annotations:
            alpha.jobset.sigs.k8s.io/exclusive-topology: kubernetes.io/hostname
        spec:
          backoffLimit: 0
          completionMode: Indexed
          completions: 1
          parallelism: 1
          template:
            metadata:
              annotations:
                alpha.jobset.sigs.k8s.io/exclusive-topology: kubernetes.io/hostname
            spec:
              hostNetwork: true
              dnsPolicy: ClusterFirstWithHostNet
              restartPolicy: Never
              nodeSelector:
                cloud.google.com/gke-nodepool: ${CPU_POOL_NAME}
              volumes:
                - name: shared-tmp
                  hostPath:
                    path: /tmp
                    type: DirectoryOrCreate
              initContainers:
                - name: pathways-rm
                  image: ${PATHWAYS_SERVER_IMAGE}
                  imagePullPolicy: Always
                  args:
                    - --server_port=29001
                    - --gcs_scratch_location=gs://cloud-pathways-staging/tmp
                    - --node_type=resource_manager
                    - --instance_count=1
                    - --instance_type=tpuv5:${TOPOLOGY}
                  env:
                    - name: ENABLE_PATHWAYS_PERSISTENCE
                      value: "1"
                    - name: REPLICATED_JOB_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
                    - name: JOBSET_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
                    - name: HOST_ADDRESS
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                    - name: TPU_SKIP_MDS_QUERY
                      value: "true"
                  ports:
                    - containerPort: 29001
                    - containerPort: 29002
                  resources:
                    limits:
                      cpu: "8"
                      memory: 16G
                  restartPolicy: Always
                - name: pathways-proxy
                  image: ${PATHWAYS_PROXY_IMAGE}
                  imagePullPolicy: Always
                  args:
                    - --server_port=29000
                    - --resource_manager_address=\$(PATHWAYS_HEAD):29001
                    - --gcs_scratch_location=gs://cloud-pathways-staging/tmp
                  env:
                    - name: ENABLE_PATHWAYS_PERSISTENCE
                      value: "1"
                    - name: PATHWAYS_HEAD
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                  ports:
                    - containerPort: 29000
                  resources:
                    limits:
                      cpu: "${PROXY_CPU_LIMIT}"
                      memory: ${PROXY_MEMORY_LIMIT}
                  restartPolicy: Always
              containers:
                - name: jax-tpu
                  image: "${JAX_TPU_IMAGE}"
                  imagePullPolicy: Always
                  command:
                    - bash
                    - -c
                    - sleep infinity
                  env:
                    - name: ENABLE_PATHWAYS_PERSISTENCE
                      value: "1"
                    - name: PATHWAYS_HEAD
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                    - name: JAX_PLATFORMS
                      value: proxy
                    - name: XCLOUD_ENVIRONMENT
                      value: GCP
                    - name: JAX_BACKEND_TARGET
                      value: grpc://\$(PATHWAYS_HEAD):29000
                  resources:
                    limits:
                      cpu: "${JAX_CPU_LIMIT}"
                      memory: ${JAX_MEMORY_LIMIT}
                  securityContext:
                    privileged: true
                  volumeMounts:
                    - mountPath: /tmp
                      name: shared-tmp
    - name: worker
      replicas: 1
      template:
        spec:
          backoffLimit: 16
          completionMode: Indexed
          completions: ${WORKER_PODS}
          parallelism: ${WORKER_PODS}
          template:
            spec:
              hostNetwork: true
              dnsPolicy: ClusterFirstWithHostNet
              priorityClassName: very-high
              restartPolicy: OnFailure
              terminationGracePeriodSeconds: 30
              nodeSelector:
                cloud.google.com/gke-tpu-accelerator: ${TPU_SLICE}
                cloud.google.com/gke-tpu-topology: ${TOPOLOGY}
              volumes:
                - name: shared-tmp
                  emptyDir:
                    sizeLimit: ${WORKER_TMP_SIZE_LIMIT}
              containers:
                - name: pathways-worker
                  image: ${PATHWAYS_SERVER_IMAGE}
                  imagePullPolicy: Always
                  args:
                    - --server_port=29005
                    - --resource_manager_address=\$(PATHWAYS_HEAD):29001
                    - --gcs_scratch_location=gs://cloud-pathways-staging/tmp
                  env:
                    - name: ENABLE_PATHWAYS_PERSISTENCE
                      value: "1"
                    - name: TPU_MIN_LOG_LEVEL
                      value: "0"
                    - name: TF_CPP_MIN_LOG_LEVEL
                      value: "0"
                    - name: XCLOUD_ENVIRONMENT
                      value: GCP
                    - name: MEGASCALE_GRPC_ENABLE_XOR_TRACER
                      value: "false"
                    - name: MEGASCALE_NUM_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: JOBSET_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
                    - name: REPLICATED_JOB_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
                    - name: MEGASCALE_SLICE_ID
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/job-index']
                    - name: PATHWAYS_HEAD
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                    - name: MEGASCALE_COORDINATOR_ADDRESS
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
                  ports:
                    - containerPort: 29005
                    - containerPort: 29006
                    - containerPort: 8471
                    - containerPort: 8080
                  resources:
                    limits:
                      google.com/tpu: ${CHIPS_PER_WORKER}
                  volumeMounts:
                    - mountPath: /tmp
                      name: shared-tmp
EOF

# export JOB_NAME=lancewang-v5p-pw-4; export GITHUB_PATH=/Users/lancewang/github; export TEMP_BUCKET=lancewang-dev-supercomputer-testing/tunix/pw

# 1) Kill anything listening on local 8888
# lsof -tiTCP:8888 -sTCP:LISTEN | xargs -r kill

# 2) Kill launcher/tunnel helpers if still alive
# pkill -f 'run_remote_pw.sh|remote-ide.py|kubectl port-forward.*8888'

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials $CLUSTER_NAME --zone $REGION

kubectl get pods; kubectl delete pathwaysjob "$JOB_NAME" --ignore-not-found; until ! kubectl get pathwaysjob "$JOB_NAME" >/dev/null 2>&1; do      echo "waiting for pathwaysjob $JOB_NAME to be removed...";     sleep 5; done; kubectl delete jobset "$JOB_NAME" --ignore-not-found; until ! kubectl get jobset "$JOB_NAME" >/dev/null 2>&1; do      echo "waiting for jobset $JOB_NAME to be removed...";     sleep 5; done; kubectl apply -f jobset.yaml; until kubectl get pod | grep "$JOB_NAME-pathways-head-0-0" | grep -q Running; do     echo "waiting for head pod...";        sleep 5; done;

python3 $GITHUB_PATH/experimental/pathways_dev/remote-ide.py -w "$JOB_NAME" -m "vscode" -b "$TEMP_BUCKET" -P $REMOTE_PW_PORT --check-active-session


#
