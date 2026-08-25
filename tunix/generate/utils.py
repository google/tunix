import jax
import jax.numpy as jnp

def get_dtype_packing(dtype):
    """Returns the packing factor for the given dtype for TPU SRAM (4 bytes / itemsize)."""
    n_bytes = jnp.dtype(dtype).itemsize
    return 4 // n_bytes

def shard(x: jnp.ndarray, s: tuple[str | None, ...]):
  mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*s))
  )

def cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b
