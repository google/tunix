# Tunix Worker Transport Example

This example demonstrates how to use the Tunix worker execution transport layer to seamlessly run worker tasks either locally (in-process) or remotely (over gRPC) using a unified actor handle API.

---

## 1. Overview & API

With Tunix worker transport, an orchestrator can submit tasks to local or remote workers through identical `ActorHandle` interfaces. Code written against an actor handle works transparently regardless of where the worker is running.

### Key API Functions (`transport.py`)

- **`transport.local(cls, *args, **kwargs)`**: Creates an actor handle for a local, in-process instance of target class `cls` initialized with `*args, **kwargs`. Ideal for debugging, testing, or zero-serialization execution.
- **`transport.remote(cls, address: str)`**: Creates an actor handle connecting to a remote worker daemon for class `cls` at network address `address`.

```python
from tunix.experimental.distributed.examples.worker_transport import transport
from tunix.experimental.distributed.examples.worker_transport.worker import Worker

# Local in-process worker handle
local_handle = transport.local(Worker, name="local")

# Remote gRPC worker handle
remote_handle = transport.remote(Worker, address="grpc://worker-host:12345")
```

---

## 2. Distributed Workflow

### Step 1: Define a Worker Class

Define a worker class containing the business logic or methods you wish to execute:

```python
class Worker:
  def __init__(self, name: str):
    self.name = name

  def ping(self, msg: str) -> str:
    return f"[{self.name}] ack: {msg}"
```

### Step 2: Choose a Worker Deployment Sub-workflow

Tunix supports three worker deployment models depending on your execution environment:

1. **Same-process Worker (In-process)**:
   - Co-locates the worker instance within the orchestrator process.
   - Ideal for debugging, testing, or zero-serialization execution.
   ```python
   handle = transport.local(Worker, name="same-process-worker")
   ```

2. **Same-host Worker (Separate local process)**:
   - Runs the worker inside an independent OS process on the same machine communicating via gRPC.
   - Useful for isolating memory/GIL or running standalone binary entry points locally.
   ```python
   # Connect to worker daemon running on local port 12345:
   handle = transport.remote(Worker, address="grpc://localhost:12345")
   ```

3. **Remote-host Worker (Distributed network process / K8s pod)**:
   - Deploys worker daemon processes across remote network hosts or Kubernetes pods.
   - Workers dynamically register their network endpoints using Tunix peer discovery (`context.ipc.discovery`), allowing the orchestrator to resolve and connect without hardcoded IP addresses.
   ```python
   # Address resolved dynamically via peer discovery callback
   handle = transport.remote(Worker, address=discovered_address)
   ```

### Step 3: Submit Tasks to Workers via Actor Handles

Regardless of whether the worker is running in the same process, on the same host, or on a remote host, the orchestrator interacts with worker handles using the identical asynchronous `asubmit` API:

```python
async def run_workflow(handle):
  # Submit method name and positional/keyword arguments
  result = await handle.asubmit("ping", msg="hello")
  print(result)  # Output: "[worker-name] ack: hello"
```

---

## 3. Directory Structure

```
worker_transport/
├── worker.py                  # Worker class definition
├── transport.py               # Generic transport.local(cls, ...) and transport.remote(cls, address) API functions
├── remote_worker_server.py    # Standalone remote worker daemon process entry point
├── worker_transport_test.py   # Automated end-to-end integration test
└── README.md                  # User guide (this file)
```
