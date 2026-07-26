import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import pjit
import pathwaysutils
try:
    pathwaysutils.initialize()
except Exception: pass

def main():
    devices = jax.devices()
    print(f"Total devices: {len(devices)}")
    if len(devices) < 64:
        print("Need at least 64 devices")
        return
    
    # Create submesh
    sub_devices = np.asarray(devices[:64]).reshape((8, 8))
    submesh = Mesh(sub_devices, ("fsdp", "tp"))
    
    print("Testing device_put to submesh directly...")
    host_array = np.ones((1024, 1024), dtype=np.float32)
    sharding = NamedSharding(submesh, P("fsdp", "tp"))
    
    try:
        # This is expected to crash based on our hypothesis
        device_array = jax.device_put(host_array, sharding)
        device_array.block_until_ready()
        print("SUCCESS: device_put to submesh worked!?")
    except Exception as e:
        print(f"FAILED: device_put to submesh crashed! Exception: {e}")
        
    print("Testing single-device put followed by jit scatter...")
    try:
        # Load onto a single device (e.g. CPU or TPU device 0)
        single_device_array = jax.device_put(host_array, devices[0])
        
        @jax.jit
        def scatter(x):
            return jax.lax.with_sharding_constraint(x, sharding)
            
        # scatter using JIT
        scattered_array = scatter(single_device_array)
        scattered_array.block_until_ready()
        print("SUCCESS: single-device put + jit scatter worked!")
    except Exception as e:
        print(f"FAILED: single-device + jit scatter crashed! {e}")

if __name__ == "__main__":
    main()
