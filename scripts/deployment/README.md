# Run a local RL workload

## Step 1: compile service proto

  ```shell
  $ pip install grpcio grpcio-tools
  $ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tunix/distributed/service/registry_service.proto
  ```

## Step 2: run components

1. Orchestrator

  ```shell
  $ bash scripts/deployment/launcher.sh --local --role=orchestrator
  ```

2. Rollout Worker

  ```shell
  $ bash scripts/deployment/launcher.sh --local --role=rollout
  ```

# Deploy a distributed RL workload

## Step 1: compile service proto

  ```shell
  $ pip install grpcio grpcio-tools
  $ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tunix/distributed/service/registry_service.proto
  ```

## Step 2: build and push Tunix docker image

  see build_docker.sh

## Step 3: launch components

1. Orchestrator

  ```shell
  $ bash scripts/deployment/launcher.sh --role=orchestrator
  ```

2. Rollout Worker

  ```shell
  $ bash scripts/deployment/launcher.sh --role=rollout
  ```
