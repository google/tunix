
gcloud auth login

REGION=europe-west4
TPU_CLUSTER_NAME=tunix-v5p-16
PROJECT_ID=cloud-tpu-multipod-dev

# Use --region instead of --zone
gcloud container clusters get-credentials ${TPU_CLUSTER_NAME} \
    --region ${REGION} \
    --project ${PROJECT_ID}

conda activate deepswe
cd tunix
git fetch origin
git reset --hard origin/sizhi-deepswe-dev

# Go to R2EGym to
/opt/venv/bin/python3.12/site-packages/r2egym/agenthub/runtime/docker.py 
# update node selector as follows
"nodeSelector": {"cloud.google.com/gke-nodepool": "deepswe-cpu-pool"},


# use deepswe-cpu-pool for tunix-v5p-32

# CPU_CLUSTER_NAME=sizhi-cluster
# CPU_ZONE=us-south1-a
# PROJECT=tpu-prod-env-multipod
# gcloud container clusters get-credentials ${CPU_CLUSTER_NAME} \
#     --zone=${CPU_ZONE} \
#     --project=${CPU_PROJECT_ID}