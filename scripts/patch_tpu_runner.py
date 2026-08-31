import os

# 1. Patch tpu_runner.py
runner_path = '/opt/venv/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py'
if os.path.exists(runner_path):
    with open(runner_path, 'r') as f:
        code = f.read()

    # Add top-level import
    import_target = 'import numpy as np'
    import_replacement = 'import numpy as np\nimport jax._src.random.core as rc'
    if import_target in code and 'import jax._src.random.core as rc' not in code:
        code = code.replace(import_target, import_replacement, 1)

    # Target 1: _init_random
    t1 = 'self.rng_key = jax.random.key(self.model_config.seed)'
    r1 = '''s = int(self.model_config.seed or 0)
        cpu = jax.devices('cpu')[0] if 'cpu' in [d.platform for d in jax.devices()] else None
        k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
        base_arr = jax.device_put(k_data, cpu) if cpu is not None else k_data
        self.rng_key = rc.wrap_key_data(base_arr, impl=rc.default_prng_impl())'''

    # Target 2: load_model rng_key creation
    t2 = 'rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()'
    r2 = '''s = int(self.model_config.seed or 0)
        cpu = jax.devices('cpu')[0] if 'cpu' in [d.platform for d in jax.devices()] else None
        k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
        base_arr = jax.device_put(k_data, cpu) if cpu is not None else k_data
        rng_key = nnx.Rngs(rc.wrap_key_data(base_arr, impl=rc.default_prng_impl())).params()'''

    # Target 3: zero_array
    t3 = 'self.zero_array = jnp.array(0, dtype=jnp.int32)'
    r3 = 'self.zero_array = device_array(self.mesh, 0, sharding=NamedSharding(self.mesh, PartitionSpec()))'

    if t1 in code and t2 in code:
        code = code.replace(t1, r1, 1).replace(t2, r2, 1)
        if t3 in code:
            code = code.replace(t3, r3, 1)
        with open(runner_path, 'w') as f:
            f.write(code)
        print('Successfully patched tpu_runner.py with CPU device_put PRNGKeyArray!')
    else:
        print('Warning: Targets not found in tpu_runner.py, skipping patch.')

# 2. Patch block_table.py
bt_path = '/opt/venv/lib/python3.12/site-packages/tpu_inference/runner/block_table.py'
if os.path.exists(bt_path):
    with open(bt_path, 'r') as f:
        bt_code = f.read()

    target_bt = 'self.block_table = jnp.zeros(\n            (max_num_reqs, max_num_blocks_per_req),\n            dtype=jnp.int32,\n        )'
    replacement_bt = 'self.block_table = np.zeros(\n            (max_num_reqs, max_num_blocks_per_req),\n            dtype=np.int32,\n        )'

    if target_bt in bt_code:
        bt_code = bt_code.replace(target_bt, replacement_bt, 1)
        with open(bt_path, 'w') as f:
            f.write(bt_code)
        print('Successfully patched block_table.py!')
    else:
        print('Warning: Target not found in block_table.py')
