#!/bin/bash
# Usage: ./v7x_256_config.sh <HF_TOKEN>

CLUSTER_NAME=bodaborg-tpu7x-spot-256-chip
ZONE=us-central1-c
REGION=us-central1
PROJECT=cloud-tpu-multipod-dev
TPU_TYPE=v7x-256
NUM_SLICES=1
export HF_TOKEN=$1



gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials $CLUSTER_NAME --zone $REGION
