import ray
import os
import numpy as np

@ray.remote(num_cpus=2)
class JaxSPMDService:
    def __init__(self, process_id: int, num_processes: int):
        self.process_id = process_id
        self.num_processes = num_processes
        self.peers = []

    def set_peers(self, peers):
        """Allows Rank 0 to hold network references to the other worker processes."""
        self.peers = peers

    def initialize_mesh(self, coordinator_address: str):
        """Sets up the persistent JAX mesh. Run once during startup."""
        os.environ["JAX_PLATFORMS"] = "cpu"
        
        import jax 
        from jax.sharding import Mesh, PartitionSpec, NamedSharding
        
        # 1. Connect to the distributed mesh
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=self.num_processes,
            process_id=self.process_id
        )

        # 2. Define and cache the Mesh and Sharding topology
        self.mesh = Mesh(jax.devices(), axis_names=('batch',))
        self.sharding = NamedSharding(self.mesh, PartitionSpec('batch'))

        # 3. Define and cache the JIT compiled function
        @jax.jit
        def compute_function(x):
            return x * 10.0
            
        self.compute_function = compute_function

        return f"Process {self.process_id} initialized."

    def handle_incoming_request(self, global_data):
        """
        The Microservice Entry Point. 
        Only Rank 0 executes this method. It receives data, distributes it, and coordinates execution.
        """
        if self.process_id != 0:
            raise ValueError("External data should only be routed to Rank 0!")

        # 1. Share Data Efficiently
        # We put the incoming data into Ray's Object Store. 
        # This prevents Ray from serializing/copying the massive array 3 separate times.
        data_ref = ray.put(global_data)

        # 2. Trigger Peers Asynchronously
        # Rank 0 tells processes 1, 2, and 3 to start computing using the shared data reference.
        peer_futures = [peer._spmd_compute.remote(data_ref) for peer in self.peers]

        # 3. Trigger Local Computation
        # Rank 0 simultaneously enters the exact same SPMD function.
        my_result = self._spmd_compute(global_data)

        # 4. Synchronize and Return
        # Rank 0 waits for the peers to finish and aggregates the distributed logs.
        peer_results = ray.get(peer_futures)

        return {
            "Coordinator (Rank 0)": my_result,
            "Workers (Ranks 1-3)": peer_results
        }

    def _spmd_compute(self, global_data):
        """The core JAX execution block. All processes run this simultaneously."""
        import jax
        
        # 1. Shard the incoming data. 
        # JAX automatically slices the array and discards data not meant for this process.
        sharded_data = jax.device_put(global_data, self.sharding)

        # 2. Execute JIT function.
        # JAX's internal C++ backend handles the collective synchronization across processes here.
        result = self.compute_function(sharded_data)

        # 3. Extract the local memory chunks
        local_devices = jax.local_devices()
        local_shards = [result.addressable_data(i).tolist() for i in range(len(local_devices))]

        return f"Computed Shards: {local_shards}"


# ==========================================
# Driver Script (Orchestration & Client)
# ==========================================
if __name__ == "__main__":
    ray.init()

    num_processes = 4
    
    # 1. Spin up the processes
    workers = [
        JaxSPMDService.remote(process_id=i, num_processes=num_processes) 
        for i in range(num_processes)
    ]

    rank_0 = workers[0]
    peers = workers[1:]

    # 2. Wire Rank 0 to the peers
    ray.get(rank_0.set_peers.remote(peers))

    # 3. Initialize the Mesh
    coordinator_address = "127.0.0.1:1234"
    print("Initializing persistent JAX Mesh...")
    init_futures = [
        worker.initialize_mesh.remote(coordinator_address) 
        for worker in workers
    ]
    ray.get(init_futures)
    print("Mesh Ready!\n")

    # ------------------------------------------
    # The Microservice is now actively waiting.
    # We can send it data dynamically.
    # ------------------------------------------
    
    print("--- Simulating Incoming Request 1 ---")
    # Standard NumPy arrays are best for Ray serialization
    incoming_data_1 = np.arange(8, dtype=np.float32) 
    
    # Send data ONLY to Rank 0
    response_1 = ray.get(rank_0.handle_incoming_request.remote(incoming_data_1))
    
    print(f"Rank 0 Output: {response_1['Coordinator (Rank 0)']}")
    for peer_res in response_1['Workers (Ranks 1-3)']:
        print(f"Peer Output:   {peer_res}")


    print("\n--- Simulating Incoming Request 2 ---")
    incoming_data_2 = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.float32)
    
    response_2 = ray.get(rank_0.handle_incoming_request.remote(incoming_data_2))
    
    print(f"Rank 0 Output: {response_2['Coordinator (Rank 0)']}")
    for peer_res in response_2['Workers (Ranks 1-3)']:
        print(f"Peer Output:   {peer_res}")