import ray
from ray.util.state import list_nodes
import time
import os
import functools

# 1. Determine if we should bypass Ray
# Check for a custom environment variable, e.g., RUN_LOCAL=True
USE_REAL_RAY = os.environ.get("RUN_LOCAL", "false").lower() != "true"

def ray_remote_wrapper(*args, **kwargs):
    """
    A drop-in replacement for @ray.remote.
    If USE_REAL_RAY is True, it forwards everything to Ray.
    If False, it converts tasks and actors into standard Python objects.
    """
    if USE_REAL_RAY:
        import ray
        return ray.remote(*args, **kwargs)
    
    # --- NO-OP DECORATOR LOGIC ---
    def decorator(obj):
        if isinstance(obj, type):
            # It's a Class (Actor) -> Wrap it so .remote() instantiates the class normally
            class LocalActorMock:
                def __init__(self, *a, **kw):
                    self._instance = obj(*a, **kw)
                
                # Mock the .remote method for method calls
                def __getattr__(self, name):
                    attr = getattr(self._instance, name)
                    if callable(attr):
                        class RemoteMethodMock:
                            def __init__(self, func):
                                self.func = func
                            def __call__(self, *args, **kwargs):
                                return self.func(*args, **kwargs)
                            def remote(self, *args, **kwargs):
                                return self.func(*args, **kwargs)
                        return RemoteMethodMock(attr)
                    return attr
            
            # Attaching a mock .remote() to the class constructor itself
            obj.remote = lambda *a, **kw: LocalActorMock(*a, **kw)
            return obj
        else:
            # It's a Function (Task) -> Attach a mock .remote() that executes normally
            @functools.wraps(obj)
            def mock_remote_call(*args, **kwargs):
                return obj(*args, **kwargs)
            
            obj.remote = mock_remote_call
            return obj

    # Handle both @ray_remote_wrapper and @ray_remote_wrapper(num_cpus=2) syntax
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return decorator(args[0])
    return decorator

# --- Mocking ray.get() and ray.put() ---
def ray_get_mock(object_refs):
    """If running locally without Ray, data is already evaluated; just return it."""
    if USE_REAL_RAY:
        import ray
        return ray.get(object_refs)
    return object_refs

def ray_put_mock(value):
    """If running locally without Ray, return the value as-is."""
    if USE_REAL_RAY:
        import ray
        return ray.put(value)
    return value

# Monkey-patch ray module when not using real Ray
if not USE_REAL_RAY:
    ray.get = ray_get_mock
    ray.put = ray_put_mock

# 1. Define the Master Node as a stateful Actor holding the queue
@ray_remote_wrapper
class MasterNode:
    def __init__(self):
        self.queue = []
        print(f"Master process id: {os.getpid()}")

    def insert_string(self, generated_string: str):
        """Worker nodes call this method to push results into the queue."""
        self.queue.append(generated_string)
        print(f"[Master] Queued: '{generated_string}'. Total items: {len(self.queue)}")

    def get_queue_status(self):
        """Helper method to inspect the queue contents."""
        return self.queue


# 2. Define the Worker Task
# We pass the 'master_handle' so the worker knows where to send the data.
@ray_remote_wrapper
def generate_string_worker(master_handle, request_id: int, computation_time: float):
    print(f"[Worker] Processing request {request_id} (takes {computation_time}s)...")
    
    # Simulate a long-running generation task
    time.sleep(computation_time) 
    result_str = f"Data_Payload_#{request_id}"
    
    # Asynchronously push the finished string back to the Master's queue
    # This is an actor-to-actor / task-to-actor P2P call
    master_handle.insert_string.remote(result_str)
    print(f"[Worker] Request {request_id} complete! Sent to Master.")


# --- Orchestration Pipeline ---
if __name__ == "__main__":
    # Initialize Ray
    ray.init()
    print(f"Main process pid: {os.getpid()}")
    
    active_nodes = list_nodes()
    print(f"Ray nodes: {active_nodes}")
    
    free_tpus = ray.available_resources().get("TPU", 0)
    print(f"I can spawn {free_tpus} workers right now without queuing.")

    # Spin up the Master Node actor
    master = MasterNode.remote()

    print("[Master] Dispatching long-running work to workers asynchronously...")

    # We fire off 3 worker tasks with different processing times.
    # Because we do NOT wrap these in ray.get(), the master drops them 
    # into the cluster pipeline and immediately moves to the next line of code.
    generate_string_worker.remote(master, request_id=101, computation_time=3.0)
    generate_string_worker.remote(master, request_id=102, computation_time=1.0)
    generate_string_worker.remote(master, request_id=103, computation_time=4.5)

    print("[Master] All requests dispatched! Master is completely unblocked.")

    # Monitor the Master node's queue over time to witness it filling up asynchronously
    for second in range(6):
        time.sleep(1)
        # Check the queue state
        current_queue = ray.get(master.get_queue_status.remote())
        print(f"Timeline: {second + 1}s passed | Master Queue State: {current_queue}")