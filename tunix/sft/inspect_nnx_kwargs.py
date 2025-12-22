
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Any

def inspect_nnx_kwargs_passing():
    # 1. Define a simple model
    class SimpleModel(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(2, 2, rngs=rngs)
            
        def __call__(self, x):
            return self.linear(x)

    model = SimpleModel(rngs=nnx.Rngs(0))
    
    # 2. Define a Dummy Optax Transform that checks for 'token_count' in kwargs
    def kwarg_update_fn(updates, state, params=None, **kwargs):
        print(f"\nInside transform update_fn.")
        print(f"Update structure match params? {type(updates)}")
        
        if 'token_count' in kwargs:
            print(f"SUCCESS: Received 'token_count' in kwargs: {kwargs['token_count']}")
        else:
            print(f"FAILURE: Did not receive 'token_count'. Kwargs keys: {list(kwargs.keys())}")
            
        return updates, state

    kwarg_transform = optax.GradientTransformation(
        lambda _: optax.EmptyState(),
        kwarg_update_fn
    )

    # 3. Wrap in nnx.Optimizer
    optimizer = nnx.Optimizer(model, kwarg_transform, wrt=nnx.Param)
    
    # 4. Try updating with kwarg
    grads = nnx.state(model, nnx.Param)
    token_count_val = 123.45
    
    try:
        print("Calling optimizer.update(model, grads, token_count=token_count_val)...")
        # Note: We are hoping nnx.Optimizer.update forwards kwargs to the optax transform
        optimizer.update(model, grads, token_count=token_count_val)
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    inspect_nnx_kwargs_passing()
