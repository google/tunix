import ray
import os

@ray.remote(num_cpus=2)
class JaxCPUWorker:
    def __init__(self, process_id: int, num_processes: int):
        self.process_id = process_id
        self.num_processes = num_processes

    def initialize_and_compute(self, coordinator_address: str):
        # 1. Blind JAX to hardware to enforce CPU execution
        os.environ["JAX_PLATFORMS"] = "cpu"
        
        import jax 
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec, NamedSharding
        
        # 2. Rendezvous and wire up the distributed gRPC mesh
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=self.num_processes,
            process_id=self.process_id
        )

        # 3. Define the Global Hardware Topology (The Mesh)
        # jax.devices() returns all 4 virtual CPU devices across the entire cluster.
        # We arrange them into a 1-dimensional mesh named 'batch'.
        global_devices = jax.devices()
        mesh = Mesh(global_devices, axis_names=('batch',))
        
        # We tell JAX to shard data along the 'batch' axis of the mesh
        sharding = NamedSharding(mesh, PartitionSpec('batch'))

        # 4. Generate the Input Data
        # We define a global array of 8 elements: [0, 1, 2, 3, 4, 5, 6, 7]
        global_data = jnp.arange(8, dtype=jnp.float32)

        # 5. Distribute the Data (The Magic)
        # When we apply the sharding spec, JAX slices the array into 4 pieces.
        # It automatically discards the data that doesn't belong to this specific process.
        sharded_data = jax.device_put(global_data, sharding)

        # 6. Define and Execute the SPMD Function
        # The @jax.jit decorator understands the sharding layout.
        @jax.jit
        def compute_function(x):
            return x * 10.0

        # This runs concurrently on all 4 processes. 
        # No cross-process network communication is needed for an element-wise operation!
        result = compute_function(sharded_data)

        # 7. Prove the data was sharded
        # Discover how many local CPU devices JAX actually spawned inside this process
        local_devices = jax.local_devices()
        
        local_shards = []
        for i in range(len(local_devices)):
            # Fetch the memory chunk for each local device
            shard_data = result.addressable_data(i)
            local_shards.append(shard_data.tolist())

        return (
            f"Process {self.process_id} | "
            f"Local Devices: {len(local_devices)} | "
            f"Computed Shards: {local_shards}"
        )


if __name__ == "__main__":
    ray.init()

    num_processes = 4
    workers = [
        JaxCPUWorker.remote(process_id=i, num_processes=num_processes) 
        for i in range(num_processes)
    ]

    coordinator_address = "127.0.0.1:1234"

    print("Dispatching sharded computation to JAX mesh...")
    futures = [
        worker.initialize_and_compute.remote(coordinator_address) 
        for worker in workers
    ]

    results = ray.get(futures)
    
    print("\nResults from the Ray Cluster:")
    # We sort the results to ensure they print in process_id order
    for res in sorted(results):
        print(res)