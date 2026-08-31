import os

path = '/opt/venv/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py'
if os.path.exists(path):
    with open(path, 'r') as f:
        code = f.read()

    # Add top-level import
    import_target = 'import numpy as np'
    import_replacement = 'import numpy as np\nimport jax._src.random.core as rc'
    if import_target in code and 'import jax._src.random.core as rc' not in code:
        code = code.replace(import_target, import_replacement, 1)

    # Target 1: _init_random
    t1 = 'self.rng_key = jax.random.key(self.model_config.seed)'
    r1 = '''s = int(self.model_config.seed or 0)
        k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
        self.rng_key = rc.wrap_key_data(k_data, impl=rc.default_prng_impl())'''

    # Target 2: load_model rng_key creation
    t2 = 'rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()'
    r2 = '''s = int(self.model_config.seed or 0)
        k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
        rng_key = nnx.Rngs(rc.wrap_key_data(k_data, impl=rc.default_prng_impl())).params()'''

    if t1 in code and t2 in code:
        code = code.replace(t1, r1, 1).replace(t2, r2, 1)
        with open(path, 'w') as f:
            f.write(code)
        print('Successfully patched tpu_runner.py with clean top-level import!')
    else:
        print('Warning: Targets not found in tpu_runner.py, skipping patch.')
