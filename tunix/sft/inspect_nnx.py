
import jax
import jax.numpy as jnp
from flax import nnx
import optax

def inspect_nnx_optimizer():
    # 1. Define a simple model
    class SimpleModel(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(2, 2, rngs=rngs)
            
        def __call__(self, x):
            return self.linear(x)

    model = SimpleModel(rngs=nnx.Rngs(0))
    
    # 2. Define a Dummy Optax Transform that expects a tuple
    def tuple_update_fn(updates, state, params=None):
        # Expect updates to be (grads, aux)
        print(f"Inside transform. shape of updates: {type(updates)}")
        if isinstance(updates, tuple):
            print("Updates is a tuple (as expected for weighted)")
            grads, aux = updates
            print(f"Aux data received: {aux}")
            return grads, state # Return just grads as 'updates' for the model
        else:
            print("Updates is NOT a tuple")
            return updates, state

    tuple_transform = optax.GradientTransformation(
        lambda _: optax.EmptyState(),
        tuple_update_fn
    )

    # 3. Wrap in nnx.Optimizer
    optimizer = nnx.Optimizer(model, tuple_transform, wrt=nnx.Param)
    
    # Check if .opt attribute exists
    if hasattr(optimizer, 'opt'):
        print(f"nnx.Optimizer has .opt attribute: {optimizer.opt}")
    elif hasattr(optimizer, 'tx'):
         print(f"nnx.Optimizer has .tx attribute: {optimizer.tx}")
    else:
        print("nnx.Optimizer DOES NOT expose inner transform via .opt or .tx easily found")
        print(dir(optimizer))

    # 4. Try modifying `update` call
    grads = nnx.state(model, nnx.Param) # Get gradients structure (just using params as dummy grads)
    aux = 100.0
    
    try:
        print("\nAttempting optimizer.update(model, (grads, aux))...")
        optimizer.update(model, (grads, aux))
        print("Success: nnx.Optimizer accepted tuple/aux data structure.")
    except Exception as e:
        print(f"\nCRASHED: {e}")

if __name__ == "__main__":
    inspect_nnx_optimizer()
