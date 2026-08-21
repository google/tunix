# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance benchmark and XProf profiler for Qwen3 Attention kernels.

Profiles and collects XProf traces for:
1. Base Reference Attention (`use_flash_attention=False`)
2. JAX Splash Attention (`use_flash_attention=True, splash_attention_impl=JAX`)
3. Tokamax Splash Attention (`use_flash_attention=True, splash_attention_impl=TOKAMAX`)

Emits XProf trace viewer links (http://xprof/trace_viewer/<session_id>) and
prints latency metrics (ms/step) for head-to-head comparison.
"""

import copy
import time
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.qwen3 import model as model_lib

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.perftools.accelerators.xprof.api.python import xprof_session
except ImportError:
  xprof_session = None

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 2, "Batch size.")
_SEQ_LEN = flags.DEFINE_integer(
    "seq_len", 2048, "Sequence length (e.g. 1024, 2048, 4096, 8192)."
)
_NUM_ITERS = flags.DEFINE_integer(
    "num_iters", 10, "Number of profiling iterations."
)
_WARMUP_ITERS = flags.DEFINE_integer(
    "warmup_iters", 3, "Number of warmup iterations."
)
_BLOCK_SIZE = flags.DEFINE_integer(
    "block_size", 512, "Flash / Splash attention block size."
)
_NUM_KV_HEADS = flags.DEFINE_integer(
    "num_kv_heads", 8, "Number of KV heads (1 for MQA, 8 for GQA in Qwen3-0.6B)."
)
_DTYPE = flags.DEFINE_enum(
    "dtype",
    "bfloat16",
    ["bfloat16", "float32"],
    "Tensor computation dtype.",
)
_XPROF_PORT = flags.DEFINE_string("xprof_port", None, "XProf port for profiling.")
_JOB_NAME = flags.DEFINE_string("job_name", None, "Job name passed by XManager.")


def _setup_mesh() -> jax.sharding.Mesh:
  num_devices = len(jax.devices())
  mesh_config = [(num_devices, 1), ("fsdp", "tp")]
  return jax.make_mesh(
      *mesh_config,
      axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]),
  )


def profile_attention_variant(
    name: str,
    attn_module: model_lib.Attention,
    x: jax.Array,
    positions: jax.Array,
    attn_mask: jax.Array,
    mesh: jax.sharding.Mesh,
    num_iters: int,
    warmup_iters: int,
) -> tuple[float, str]:
  """Runs warm-up, measures execution time, and captures an XProf trace."""
  logging.info("=" * 60)
  logging.info("Benchmarking and profiling: [%s]", name)

  @nnx.jit
  def forward_step(m, x_in, pos_in, mask_in):
    with jax.named_scope(name):
      return m(x_in, pos_in, None, mask_in)

  # Warm-up to trigger JIT compilation before profiling
  logging.info("[%s] Running %d warm-up steps...", name, warmup_iters)
  with jax.set_mesh(mesh):
    for i in range(warmup_iters):
      with jax.profiler.TraceAnnotation(f"{name}_warmup_{i}"):
        _, out = forward_step(attn_module, x, positions, attn_mask)
        jax.block_until_ready(out)

  # Latency benchmark
  logging.info("[%s] Measuring latency across %d steps...", name, num_iters)
  step_times_ms = []
  with jax.set_mesh(mesh):
    for i in range(num_iters):
      with jax.profiler.TraceAnnotation(f"{name}_iter_{i}"):
        step_t0 = time.perf_counter()
        _, out = forward_step(attn_module, x, positions, attn_mask)
        jax.block_until_ready(out)
        step_t1 = time.perf_counter()
        step_times_ms.append((step_t1 - step_t0) * 1000.0)

  avg_ms = sum(step_times_ms) / len(step_times_ms)
  sorted_times = sorted(step_times_ms)
  median_ms = sorted_times[len(sorted_times) // 2]
  min_ms = min(step_times_ms)
  max_ms = max(step_times_ms)
  variance = sum((x - avg_ms) ** 2 for x in step_times_ms) / len(step_times_ms)
  std_ms = variance ** 0.5

  logging.info("[%s] Per-step timings (ms): %s", name, [f"{t:.3f}" for t in step_times_ms])
  logging.info(
      "[%s] Stats: Mean=%.3f ms | Median=%.3f ms | Min=%.3f ms | Max=%.3f ms | Std=%.3f ms",
      name,
      avg_ms,
      median_ms,
      min_ms,
      max_ms,
      std_ms,
  )

  # Capture XProf trace
  tag = f"qwen3_{name}_seq{_SEQ_LEN.value}"
  xprof_url = ""

  if xprof_session is not None:
    logging.info("[%s] Capturing programmatic XProf trace (tag=%s)...", name, tag)
    session = xprof_session.XprofSession()
    try:
      session.start_session(
          device_name="viperfish",
          enable_python_tracer=True,
          host_trace_level=3,
          host_cpu_profile=True,
          trace_mode="TRACE_COMPUTE_AND_SYNC",
      )
      with jax.set_mesh(mesh):
        for _ in range(num_iters):
          _, out = forward_step(attn_module, x, positions, attn_mask)
          jax.block_until_ready(out)
      raw_url = session.end_session_and_get_url(tag=tag, username="")
      xprof_url = raw_url.replace("?session_id=", "trace_viewer/")
      logging.info("[%s] XProf URL: %s", name, xprof_url)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info("[%s] Programmatic XProf session skipped: %s", name, e)
  else:
    log_dir = f"/tmp/xprof_traces/{tag}"
    logging.info("[%s] Capturing local JAX profiler trace to %s...", name, log_dir)
    jax.profiler.start_trace(log_dir)
    with jax.set_mesh(mesh):
      for _ in range(num_iters):
        _, out = forward_step(attn_module, x, positions, attn_mask)
        jax.block_until_ready(out)
    jax.profiler.stop_trace()
    xprof_url = f"file://{log_dir}"

  return avg_ms, xprof_url


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if jax.default_backend() != "tpu":
    logging.error(
        "Current JAX backend is '%s', but TPU hardware is required for Pallas"
        " splash attention. Please execute on a machine with TPU hardware or"
        " via XManager.",
        jax.default_backend(),
    )
    return

  dtype = jnp.bfloat16 if _DTYPE.value == "bfloat16" else jnp.float32

  logging.info(
      "Configuration: batch_size=%d, seq_len=%d, dtype=%s,"
      " block_size=%d, num_kv_heads=%d",
      _BATCH_SIZE.value,
      _SEQ_LEN.value,
      _DTYPE.value,
      _BLOCK_SIZE.value,
      _NUM_KV_HEADS.value,
  )

  # 1. Base Configuration
  config_base = model_lib.ModelConfig.qwen3_0p6b()
  config_base.dtype = dtype
  config_base.param_dtype = dtype
  config_base.num_kv_heads = _NUM_KV_HEADS.value
  config_base.use_flash_attention = False

  # 2. JAX Splash Attention Configuration
  config_jax = copy.deepcopy(config_base)
  config_jax.use_flash_attention = True
  config_jax.splash_attention_impl = model_lib.SplashAttentionImpl.JAX
  config_jax.flash_attention_block_size = _BLOCK_SIZE.value

  # 3. Tokamax Splash Attention Configuration
  config_tokamax = copy.deepcopy(config_base)
  config_tokamax.use_flash_attention = True
  config_tokamax.splash_attention_impl = model_lib.SplashAttentionImpl.TOKAMAX
  config_tokamax.flash_attention_block_size = _BLOCK_SIZE.value

  # Instantiate Attention modules and synchronize weights
  attn_base = model_lib.Attention(config_base, rngs=nnx.Rngs(0))
  attn_jax = model_lib.Attention(config_jax, rngs=nnx.Rngs(0))
  attn_tokamax = model_lib.Attention(config_tokamax, rngs=nnx.Rngs(0))

  _, base_state = nnx.split(attn_base)
  nnx.update(attn_jax, base_state)
  nnx.update(attn_tokamax, base_state)

  b, t, d = _BATCH_SIZE.value, _SEQ_LEN.value, config_base.embed_dim
  x = jax.random.normal(jax.random.PRNGKey(42), (b, t, d), dtype=dtype)
  attn_mask = jnp.tril(jnp.ones((b, t, t), dtype=jnp.bool_))
  positions = jnp.tile(jnp.arange(t)[None, :], (b, 1))

  mesh = _setup_mesh()

  results = {}
  variants = [
      ("base", attn_base),
      ("jax_splash", attn_jax),
      ("tokamax_splash", attn_tokamax),
  ]

  for name, module in variants:
    avg_ms, url = profile_attention_variant(
        name=name,
        attn_module=module,
        x=x,
        positions=positions,
        attn_mask=attn_mask,
        mesh=mesh,
        num_iters=_NUM_ITERS.value,
        warmup_iters=_WARMUP_ITERS.value,
    )
    results[name] = {"latency_ms": avg_ms, "xprof_url": url}

  # Summary Report
  logging.info("\n" + "=" * 70)
  logging.info("BENCHMARK & PROFILING SUMMARY (seq_len=%d, %s)", t, _DTYPE.value)
  logging.info("=" * 70)
  for name, data in results.items():
    speedup = results["base"]["latency_ms"] / data["latency_ms"]
    logging.info(
        "%-16s: %8.3f ms/step (%5.2fx vs base) | XProf: %s",
        name,
        data["latency_ms"],
        speedup,
        data["xprof_url"],
    )
  logging.info("=" * 70)


if __name__ == "__main__":
  app.run(main)
