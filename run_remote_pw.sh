#!/bin/bash
# Usage:
#   CLUSTER_NAME=<cluster> JOB_NAME=<job> TPU_TYPE=<tpu> TOPOLOGY=<topo> ./run_remote_pw.sh
# Examples:
#   ./run_remote_pw.sh                    # uses defaults
#   CLUSTER_NAME=lance-v5p-16 JOB_NAME=lancewang-v5p-pw-3 \
#     TPU_TYPE=v5p TOPOLOGY=2x2x2 \
#     ./run_remote_pw.sh

CLUSTER_NAME=${CLUSTER_NAME:-tunix-v5p-16}

JOB_NAME=${JOB_NAME:-tunix-${USER}-${TPU_TYPE}-${TOPOLOGY}-pw-0}

# https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
TPU_TYPE=${TPU_TYPE:-v5p}
declare -A TPU_TYPE_MAP=(
  ["v5e"]="ct5lp-hightpu-1t"
  ["v5p"]="ct5p-hightpu-4t"
  ["v6e"]="ct6e-standard-1t"
  ["v7x"]="tpu7x-standard-4t"
)
TPU_TYPE=${TPU_TYPE_MAP[$TPU_TYPE]:-$TPU_TYPE}

TOPOLOGY=${TOPOLOGY:-2x2x2}

ZONE=${ZONE:-europe-west4-b}
REGION=${REGION:-europe-west4}
PROJECT=${PROJECT:-cloud-tpu-multipod-dev}

GITHUB_PATH=${GITHUB_PATH:-/github} # Specify your repos are in github folder
TEMP_BUCKET=${TEMP_BUCKET:-lancewang-dev-supercomputer-testing/tunix/pw}

if [ ! -d "${GITHUB_PATH}/experimental/.git" ]; then
  git clone sso://user/abhinavsing/experimental "${GITHUB_PATH}/experimental"
fi

# Get credentials and set up environment
gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials $CLUSTER_NAME --zone $REGION



cat > pathways-job.yaml <<EOF
apiVersion: pathways-job.pathways.domain/v1
kind: PathwaysJob
metadata:
  name: ${JOB_NAME}
  labels:
    kueue.x-k8s.io/queue-name: multislice-queue
    xpk.google.com/workload: ${JOB_NAME}
spec:
  maxRestarts: 0
  customComponents:
  workers:
  - type: ${TPU_TYPE}
    topology: ${TOPOLOGY}
    numSlices: 1
    maxSliceRestarts: 1
    terminationGracePeriodSeconds: 30
    priorityClassName: very-high
    nodeSelector:
  pathwaysDir: gs://cloud-pathways-staging/tmp
  controller:
    deploymentMode: default
    mainContainerName: jax-tpu
    elasticSlices: 0
    template:
      metadata:
      spec:
        containers:
        - name: jax-tpu
          image: gcr.io/cloud-tpu-multipod-dev/lance_deepswe:latest
          imagePullPolicy: Always
          env:
          ports:
          securityContext:
            privileged: true
          command:
          - bash
          - -c
          - |
            (while true; do echo "This loop runs forever"; done)
          resources:
            limits:
              cpu: "24"
              memory: 100G
          volumeMounts:
          - mountPath: /tmp
            name: shared-tmp
        nodeSelector:
          cloud.google.com/gke-nodepool: cpu-np
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        restartPolicy: Never
        volumes:
        - hostPath:
            path: /tmp
            type: DirectoryOrCreate
          name: shared-tmp
EOF

kubectl get pods; kubectl delete pathwaysjob "$JOB_NAME" --ignore-not-found; until ! kubectl get pathwaysjob "$JOB_NAME" >/dev/null 2>&1; do      echo "waiting for pathwaysjob $JOB_NAME to be removed...";     sleep 5; done; kubectl apply -f pathways-job.yaml; until kubectl get pod | grep "$JOB_NAME-pathways-head-0-0" | grep -q Running; do     echo "waiting for head pod...";        sleep 5; done; kubectl get pod; python3 $GITHUB_PATH/experimental/pathways_dev/remote-ide.py -w "$JOB_NAME" -m "vscode" -b lancewang-dev-supercomputer-testing/tunix/pw --check-active-session


