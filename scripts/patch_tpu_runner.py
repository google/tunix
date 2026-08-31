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
        cpu = jax.devices('cpu')[0]
        k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
        base_arr = jax.device_put(k_data, cpu)
        self.rng_key = rc.wrap_key_data(base_arr, impl=rc.default_prng_impl())'''

    # Target 2: load_model rng_key creation
    t2 = 'rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()'
    r2 = '''s = int(self.model_config.seed or 0)
        cpu = jax.devices('cpu')[0]
        k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
        base_arr = jax.device_put(k_data, cpu)
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
        print('Successfully patched tpu_runner.py!')
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

# 3. Patch maxtext_utils_nnx.py
nnx_path = '/opt/venv/lib/python3.12/site-packages/maxtext/utils/maxtext_utils_nnx.py'
if os.path.exists(nnx_path):
    with open(nnx_path, 'r') as f:
        nnx_code = f.read()

    # Add top-level import
    import_target_nnx = 'import numpy as np'
    import_replacement_nnx = 'import numpy as np\nimport jax._src.random.core as rc'
    if import_target_nnx in nnx_code and 'import jax._src.random.core as rc' not in nnx_code:
        nnx_code = nnx_code.replace(import_target_nnx, import_replacement_nnx, 1)

    t_rng = '  if rng_key is None:\n    rng_key = jax.random.PRNGKey(config.init_weights_seed)'
    r_rng = '''  if rng_key is None:
    cpu = jax.devices('cpu')[0]
    s = int(config.init_weights_seed or 0)
    k_data = np.array([np.uint32(s >> 32), np.uint32(s & 0xFFFFFFFF)], dtype=np.uint32)
    base_arr = jax.device_put(k_data, cpu)
    rng_key = rc.wrap_key_data(base_arr, impl=rc.default_prng_impl())'''

    if t_rng in nnx_code:
        nnx_code = nnx_code.replace(t_rng, r_rng, 1)
        with open(nnx_path, 'w') as f:
            f.write(nnx_code)
        print('Successfully patched maxtext_utils_nnx.py!')
    else:
        print('Warning: Target not found in maxtext_utils_nnx.py')
