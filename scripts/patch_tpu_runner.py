import os

path = '/opt/venv/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py'
if os.path.exists(path):
    with open(path, 'r') as f:
        code = f.read()

    t1 = 'self.rng_key = jax.random.key(self.model_config.seed)'
    r1 = '''try:
            cpu = jax.devices("cpu")[0] if "cpu" in [d.platform for d in jax.devices()] else None
            if cpu is not None:
                with jax.default_device(cpu):
                    self.rng_key = jax.random.key(self.model_config.seed)
            else:
                self.rng_key = jax.random.key(self.model_config.seed)
        except Exception:
            self.rng_key = jax.random.key(self.model_config.seed)'''

    t2 = 'rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()'
    r2 = '''try:
            cpu = jax.devices("cpu")[0] if "cpu" in [d.platform for d in jax.devices()] else None
            if cpu is not None:
                with jax.default_device(cpu):
                    rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()
            else:
                rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()
        except Exception:
            rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()'''

    if t1 in code and t2 in code:
        code = code.replace(t1, r1, 1).replace(t2, r2, 1)
        with open(path, 'w') as f:
            f.write(code)
        print('Successfully patched tpu_runner.py!')
    else:
        print('Warning: Targets not found in tpu_runner.py, skipping patch.')
