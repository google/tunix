# Tunix Distributed Execution Scaffolding Examples

This directory provides hands-on examples demonstrating how to use the Tunix experimental distributed process runtime (`tunix.experimental.distributed.runtime`).

The Tunix distributed process runtime provides a **platform-agnostic execution framework** that allows Python worker processes to run either locally on a single machine or across distributed Kubernetes (K8s/GKE) pods without changing their application code. Workers discover peers dynamically via a built-in gRPC discovery service.

---

## Table of Contents
- [Prerequisites & Proto Compilation](#prerequisites--proto-compilation)
- [Example 1: Minimal Process](#example-1-minimal-process)
- [Example 2: Process with CLI Flags](#example-2-process-with-cli-flags)
- [Example 3: Peer Discovery and Inter-Process Communication](#example-3-peer-discovery-and-inter-process-communication)
- [Example 4: Simulated Distributed RL Workload (Local)](#example-4-simulated-distributed-rl-workload-local)
- [Example 5: Simulated Distributed RL Workload on Kubernetes](#example-5-simulated-distributed-rl-workload-on-kubernetes)

---

## Prerequisites & Proto Compilation

Before running any examples, generate the required protobuf Python stubs from the repository root:

```shell
# Compile the distributed process runtime discovery service proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
    tunix/experimental/distributed/runtime/discovery/discovery_service.proto

# Compile the RL simulation service proto (required for Examples 4 & 5)
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
    examples/distributed/rl/service.proto
```

---

## Example 1: Minimal Process

Every distributed process defines a standard entry point signature accepting command-line arguments (`argv`) and a platform-agnostic `ProcessContext`:

```python
from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv: list[str], context: ProcessContext | None) -> None:
  print("hello world")
```

### Run Locally

Execute the process via the distributed process runtime module `tunix.experimental.distributed.runtime.main`:

```shell
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --process_main=basics.basic.main
```

### Expected Output

```
hello world
```

---

## Example 2: Process with CLI Flags

Processes can parse arbitrary application flags passed after the runtime flags. The runtime automatically strips its own framework flags and forwards remaining arguments in `argv`.

```python
import argparse
from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv: list[str], context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="", help="Message to print")
  args = parser.parse_args(argv)

  print(args.message)
```

### Run Locally

```shell
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --process_main=basics.flag.main \
    --message="hello flag"
```

### Expected Output

```
hello flag
```

---

## Example 3: Peer Discovery and Inter-Process Communication

This example shows how two independent processes (`door` and `knocker`) discover each other dynamically and exchange metadata over gRPC.

### 1. The Server (`door.py`)
The `door` process starts a discovery server on port `12345` (`--discovery_port=12345`) and registers a callback to receive incoming peer registrations:

```python
def main(argv: list[str], context: ProcessContext | None) -> None:
  context.ipc.discovery.on_register(
      lambda hostname, _, metadata: (
          logging.info(f"{hostname} knocked and said: {pickle.loads(metadata)}")
      )
  )
```

### 2. The Client (`knocker.py`)
The `knocker` process connects to the `door` process at `door:12345` (`--discovery_addrs=door:12345`) and transmits serialized payload metadata:

```python
def main(argv: list[str], context: ProcessContext | None) -> None:
  context.ipc.discovery.register(metadata=pickle.dumps(args.say))
```

### Run Locally

In separate terminal windows (or sequentially):

```shell
# Terminal 1: Start the door service
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --process_main=basics.door.main \
    --discovery_id=door \
    --discovery_port=12345

# Terminal 2: Start the knocker process to connect to door
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --process_main=basics.knocker.main \
    --discovery_addrs=door:12345 \
    --say="open the door"
```

### Expected Output

#### Door Terminal
```
this is door!
discovery server started on port 12345
localhost knocked and said: open the door
discovery server stopped
```

#### Knocker Terminal
```
this is knocker!
registered to discovery server at localhost:12345
```

---

## Example 4: Simulated Distributed RL Workload (Local)

This example simulates a distributed reinforcement learning (RL) training workflow across **4 collaborating processes**:

- **1 Orchestrator** (`orchestrator.py`): Hosts the discovery server, registers worker endpoints, and drives the training iterations.
- **2 Rollout Workers** (`rollout.py`): Simulate generating completions from input prompts over gRPC.
- **1 Trainer Worker** (`trainer.py`): Simulates updating model weights based on completions and rewards.

### Workload Summary

- The workload learns to estimate the expected value of simple addition expressions (e.g., `2 + 3 = 5`).
- Small synthetic errors are introduced into completions randomly.
- Model weights are adjusted by +1% on correct outputs and +0.01% on errors. Over sufficient iterations, the weight converges toward `10.0`.

> [!NOTE]
> This simulation illustrates data flow and inter-process RPC orchestration across multiple worker roles rather than actual deep RL algorithms.

### Run Locally

Open separate terminal sessions for each role:

```shell
# 1. Start the orchestrator (acts as discovery hub on port 12345)
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --discovery_id=orchestrator \
    --discovery_port=12345 \
    --process_main=rl.orchestrator.main \
    --max_train_step=1000

# 2. Start Rollout Worker 0
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --discovery_addrs=orchestrator:12345 \
    --process_main=rl.rollout.main \
    --server_id=rollout-0 \
    --server_port=11111

# 3. Start Rollout Worker 1
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --discovery_addrs=orchestrator:12345 \
    --process_main=rl.rollout.main \
    --server_id=rollout-1 \
    --server_port=22222

# 4. Start Trainer Worker
PYTHONPATH=./examples/distributed \
    python -m tunix.experimental.distributed.runtime.main \
    --discovery_addrs=orchestrator:12345 \
    --process_main=rl.trainer.main \
    --server_id=trainer \
    --server_port=33333
```

---

## Example 5: Simulated Distributed RL Workload on Kubernetes

You can execute the exact same distributed RL simulation on a Kubernetes cluster using the `K8sExecutor` and JobSet deployment templates.

The helper launcher script (`launcher.sh`) generates Kubernetes deployment manifests using `yaml_generator.py` and deploys each role as a Kubernetes `JobSet`.

### Deploy on Kubernetes

```shell
# 1. Deploy the orchestrator JobSet
bash examples/distributed/rl/launcher.sh --role=orchestrator

# 2. Deploy the rollout worker pods
bash examples/distributed/rl/launcher.sh --role=rollout

# 3. Deploy the trainer worker pod
bash examples/distributed/rl/launcher.sh --role=trainer
```

