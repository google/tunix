"""tasks/deepswe_4b_perf P4a: the P22 kernel stores its output once, at the last k step.

The pre-P4a body stored ``acc.astype(bf16)`` on every k step.  Only the last
store is ever written back, so the change removes VMEM traffic without touching
the f32 accumulation sequence or the final cast.  The gate rebuilds the
pre-P4a body inline and compares raw bf16 bits with the installed kernel in
interpret mode over 1/10/20/76 k-step shapes (contract-like tiles included).
Needs the pinned image's JAX (ShapeDtypeStruct manual_axis_type); skips on host.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[3]
SHIMS = ROOT / "canon-zero-tim" / "src" / "engine_shims"
TP4 = SHIMS / "models" / "qwen4b_tp4"

_SCRIPT = textwrap.dedent(
    """
    import os, sys
    import numpy as np
    sys.path[:0] = [os.environ["P4A_OVERLAY"], os.environ["P4A_SHIMS"]]
    import p22_pallas_matmul as mm
    jax, jnp, pl, pltpu = mm._imports()

    def legacy_matmul(x, y, block_m, block_n, block_k):
        m, k = x.shape
        _, n = y.shape

        def _kernel(x_ref, y_ref, out_ref, acc_ref):
            @pl.when(pl.program_id(2) == 0)
            def _init():
                acc_ref[...] = jnp.zeros_like(acc_ref)

            acc_ref[...] = acc_ref[...] + jnp.dot(
                x_ref[...], y_ref[...], preferred_element_type=jnp.float32
            )
            # pre-P4a body: stored on every k step
            out_ref[...] = acc_ref[...].astype(out_ref.dtype)

        return pl.pallas_call(
            _kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((block_m, block_k), lambda i, _j, q: (i, q)),
                    pl.BlockSpec((block_k, block_n), lambda _i, j, q: (q, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, _q: (i, j)),
                grid=(m // block_m, n // block_n, k // block_k),
                scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary")
            ),
            interpret=True,
            name="legacy_per_step_store",
        )(x, y)

    cases = 0
    for (m, k, n, bm, bn, bk) in (
        (256, 2560, 1280, 256, 1280, 128),   # 20 k steps, the 1a wide tile
        (256, 256, 256, 128, 256, 256),      # 1 k step
        (512, 1280, 512, 128, 512, 128),     # 10 k steps
        (128, 9728, 256, 128, 128, 128),     # 76 k steps (gate/up K at TP1)
    ):
        x = jnp.asarray((((np.arange(m * k) * 7919) % 1013) - 506).reshape(m, k) / 64, jnp.bfloat16)
        y = jnp.asarray((((np.arange(k * n) * 104729) % 977) - 488).reshape(k, n) / 128, jnp.bfloat16)
        out = mm.matmul(x, y, interpret=True, block_m=bm, block_n=bn, block_k=bk)
        ref = legacy_matmul(x, y, bm, bn, bk)
        a = np.asarray(out).view(np.uint16); b = np.asarray(ref).view(np.uint16)
        assert out.shape == (m, n), out.shape
        assert np.array_equal(a, b), (m, k, n, int((a != b).sum()))
        cases += 1
    print("P4A_BITWISE_OK cases=%d" % cases)
    """
)


def test_kernel_source_stores_only_at_the_last_k_step():
  source = (SHIMS / "p22_pallas_matmul.py").read_text()
  assert "pl.program_id(2) == pl.num_programs(2) - 1" in source
  body = source.split("def _kernel(x_ref, y_ref, out_ref, acc_ref):")[1].split("return pl.pallas_call(")[0]
  assert body.count("out_ref[...] = acc_ref[...].astype(out_ref.dtype)") == 1
  assert body.index("def _finalize():") < body.index("out_ref[...] = acc_ref[...].astype(out_ref.dtype)")


def test_last_step_store_is_bitwise_the_per_step_store_in_interpret_mode():
  import inspect
  import jax
  if "manual_axis_type" not in inspect.signature(jax.ShapeDtypeStruct.__init__).parameters:
    pytest.skip("the P22 kernel needs the pinned image's JAX (ShapeDtypeStruct manual_axis_type); run in-image")
  env = {
      **os.environ,
      "JAX_PLATFORMS": "cpu",
      "P4A_OVERLAY": str(TP4),
      "P4A_SHIMS": str(SHIMS),
      "CANON_PALLAS_MPAD": "1",
      "CANON_FIXED_AR": "1",
      "CANON_FIXED_AR_EMBED": "1",
      "CANON_PALLAS_ALL_PROJ": "1",
      "CANON_PALLAS_SWIGLU": "1",
      "CANON_QWEN3_TP_SIZE": "4",
  }
  for name in ("CANON_PALLAS_MATMUL", "CANON_PALLAS_MATERIALIZE", "CANON_CUT", "CANON_TAIL", "CANON_POSTRPA_M", "P16_NUM_LAYERS"):
    env.pop(name, None)
  result = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=1500)
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "P4A_BITWISE_OK cases=4" in result.stdout, result.stdout[-1500:]
