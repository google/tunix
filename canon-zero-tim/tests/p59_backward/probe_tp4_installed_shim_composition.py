#!/usr/bin/env python3
"""Execute the installed TP4 projection shim inside the P59 outer map."""

from __future__ import annotations

import os
import types


TP_SIZE = int(os.environ.get("P59_TEST_TP_SIZE", "4"))
if TP_SIZE == 4:
  model_contract = {
      "CANON_QWEN3_HIDDEN_SIZE": "2048",
      "CANON_QWEN3_INTERMEDIATE_SIZE": "6144",
      "CANON_QWEN3_NUM_ATTENTION_HEADS": "16",
      "CANON_QWEN3_NUM_KV_HEADS": "8",
      "CANON_QWEN3_HEAD_DIM": "128",
  }
elif TP_SIZE == 8:
  model_contract = {
      "CANON_QWEN3_HIDDEN_SIZE": "4096",
      "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
      "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
      "CANON_QWEN3_NUM_KV_HEADS": "8",
      "CANON_QWEN3_HEAD_DIM": "128",
  }
else:
  raise RuntimeError(f"installed-shim probe supports TP4/TP8, got TP{TP_SIZE}")

for name in (
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD",
    "CANON_PALLAS_CANONICAL_VJP",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_P59_RANK_PARALLEL_BACKWARD",
):
  os.environ[name] = "1"
os.environ.update(model_contract)
os.environ["CANON_QWEN3_TP_SIZE"] = str(TP_SIZE)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import linear_p22xk as linear  # noqa: E402
import numpy as np  # noqa: E402

from tunix.rl import canonical_qwen3_adapter  # noqa: E402


def main() -> None:
  device_count = 2 * TP_SIZE
  if len(jax.devices()) < device_count:
    raise RuntimeError(
        "P59 installed-shim composition requires "
        f"{device_count} devices for DP2xTP{TP_SIZE}"
    )
  xf = linear.P22XK_LINEAR_BASE.P22XI_XF_MODULE
  required = (
      "_column_parallel",
      "_p59_local_fused_pieces",
      "_p59_local_tp_context",
  )
  missing = tuple(name for name in required if not hasattr(xf, name))
  if missing or xf._column_parallel is not linear._column_parallel:
    raise RuntimeError(
        "installed P22.XF module changed: "
        f"name={xf.__name__} file={xf.__file__} missing={missing} "
        "live_column_identity="
        f"{xf._column_parallel is linear._column_parallel}"
    )

  devices = np.asarray(jax.devices()[:device_count])
  trainer_mesh = jax.sharding.Mesh(
      devices.reshape(2, TP_SIZE), ("dp", "tp")
  )
  engine_axes = (
      "data",
      "attn_dp",
      "attn_dp_expert",
      "expert",
      "model",
      "dcp",
  )
  engine_mesh = jax.sharding.Mesh(
      devices.reshape(2, 1, 1, 1, TP_SIZE, 1), engine_axes
  )
  xf.base._CANON_MESH = engine_mesh
  xf.base._CANON_TP_AXIS = "model"
  xf.pallas_matmul = lambda left, right, **_: jnp.matmul(left, right)
  site = types.SimpleNamespace(family="q_proj", contract_parallel=False)

  head_dim = 128
  local_heads = 4
  global_heads = local_heads * TP_SIZE
  global_width = global_heads * head_dim
  weight = jax.device_put(
      (
          jnp.arange(4 * global_width, dtype=jnp.float32).reshape(
              4, global_heads, head_dim
          )
          / 4096
      ).astype(jnp.bfloat16),
      jax.sharding.NamedSharding(
          trainer_mesh, jax.sharding.PartitionSpec(None, "tp", None)
      ),
  )
  hidden = jax.device_put(
      (jnp.arange(64, dtype=jnp.float32).reshape(16, 4) / 64).astype(
          jnp.bfloat16
      ),
      jax.sharding.NamedSharding(
          trainer_mesh, jax.sharding.PartitionSpec("dp", None)
      ),
  )
  cotangent = jax.device_put(
      jnp.ones((16, global_heads, head_dim), jnp.bfloat16),
      jax.sharding.NamedSharding(
          trainer_mesh, jax.sharding.PartitionSpec("dp", "tp", None)
      ),
  )
  segmented = object.__new__(
      canonical_qwen3_adapter._P28SegmentedEngineForward
  )
  segmented._engine_mesh = engine_mesh

  def local_pullback(local_weight, local_hidden, local_cotangent):
    def forward(weight_arg, hidden_arg):
      output = xf._column_parallel(
          site,
          "TD,DNH->TNH",
          hidden_arg,
          weight_arg,
          "model.layers.0.self_attn.q_proj",
      )
      pieces = xf._p59_local_fused_pieces(
          output,
          (head_dim,),
          1,
          "model.layers.0.self_attn.q_proj",
          expected_local_width=local_heads * head_dim,
          tp_sharded_last_dim=False,
          site_family="q_proj",
      )
      if pieces is None:
        raise RuntimeError("P59 local q_proj layout was not selected")
      q_output = jnp.concatenate(pieces, axis=-1)
      if q_output.shape != output.shape:
        raise RuntimeError(
            f"P59 local q_proj layout changed shape: {q_output.shape}"
        )

      # Reproduce the Attempt-4 gate/up boundary exactly: the engine retains
      # one logical layout shard and a global declared width, while the P59
      # outer TP map has already produced the 1536-wide physical slice.
      flat = output.reshape(output.shape[0], -1)
      intermediate_width = 6144 if TP_SIZE == 4 else 12288
      local_intermediate = intermediate_width // TP_SIZE
      gate_output = jnp.ones(
          (flat.shape[0], local_intermediate), dtype=flat.dtype
      )
      for family in ("gate_proj", "up_proj"):
        fused = xf._p59_local_fused_pieces(
            gate_output,
            (intermediate_width,),
            1,
            f"model.layers.0.mlp.{family}",
            expected_local_width=local_intermediate,
            tp_sharded_last_dim=True,
            site_family=family,
        )
        if (
            fused is None
            or jnp.concatenate(fused, axis=-1).shape != gate_output.shape
        ):
          raise RuntimeError(f"P59 local {family} layout was not selected")
      try:
        xf._p59_local_fused_pieces(
            output,
            (head_dim + 1,),
            1,
            "negative.q_proj",
            expected_local_width=local_heads * head_dim,
            tp_sharded_last_dim=False,
            site_family="q_proj",
        )
      except RuntimeError as error:
        if "last-dimension mismatch" not in str(error):
          raise
      else:
        raise RuntimeError("P59 local q_proj wrong-width negative did not fire")
      try:
        xf._p59_local_fused_pieces(
            gate_output[..., :-1],
            (intermediate_width,),
            1,
            "negative.gate_proj",
            expected_local_width=local_intermediate,
            tp_sharded_last_dim=True,
            site_family="gate_proj",
        )
      except RuntimeError as error:
        if "feature width mismatch" not in str(error):
          raise
      else:
        raise RuntimeError(
            "P59 local gate_proj wrong-feature-width negative did not fire"
        )
      return q_output

    _, pullback = jax.vjp(forward, local_weight, local_hidden)
    dweight, dhidden = pullback(local_cotangent)
    return jnp.expand_dims(dweight, 0), dhidden

  parallel = segmented._p59_parallel_map(
      local_pullback,
      (weight, hidden, cotangent),
      lambda data_axis, axis_size, aligned, manual_axes: (
          canonical_qwen3_adapter._rank_staged_specs(
              aligned[0], data_axis, manual_axes
          ),
          canonical_qwen3_adapter._rank_local_leading_specs(
              aligned[1],
              data_axis,
              axis_size,
              "installed projection hidden cotangent",
              manual_axes,
          ),
      ),
      rank_local_arg_indices=(1, 2),
      module_name="zt_tr_dp_parallel_installed_projection",
      scope_name="zt/tr/dp_parallel/installed_projection",
  )
  staged, dhidden = parallel(weight, hidden, cotangent)

  _, ordinary_pullback = jax.vjp(
      lambda weight_arg, hidden_arg: jnp.einsum(
          "td,dnh->tnh", hidden_arg, weight_arg
      ),
      weight,
      hidden,
  )
  expected_rows = []
  for rank in range(2):
    row_mask = jnp.arange(cotangent.shape[0], dtype=jnp.int32) // 8 == rank
    isolated = jnp.where(
        row_mask[:, None, None], cotangent, jnp.zeros_like(cotangent)
    )
    expected_rows.append(ordinary_pullback(isolated)[0])
  expected_staged = jnp.stack(expected_rows)
  _, expected_dhidden = ordinary_pullback(cotangent)
  if not np.array_equal(np.asarray(staged), np.asarray(expected_staged)):
    raise AssertionError("installed projection staged weight gradient changed")
  if not np.array_equal(np.asarray(dhidden), np.asarray(expected_dhidden)):
    actual_host = np.asarray(dhidden, dtype=np.float32)
    serial_host = np.asarray(expected_dhidden, dtype=np.float32)
    oracle_host = np.einsum(
        "tnh,dnh->td",
        np.asarray(cotangent, dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
    )
    different = np.argwhere(actual_host != serial_host)
    first = tuple(map(int, different[0]))
    raise AssertionError(
        "installed projection TP input reduction changed: "
        f"mismatch={different.shape[0]}/{actual_host.size} first={first} "
        f"parallel={actual_host[first]} serial={serial_host[first]} "
        "parallel_fp64_max_abs="
        f"{np.max(np.abs(actual_host - oracle_host))} "
        "serial_fp64_max_abs="
        f"{np.max(np.abs(serial_host - oracle_host))}"
    )
  if staged.sharding.spec != jax.sharding.PartitionSpec("dp", None, "tp"):
    raise AssertionError(f"installed projection staged sharding changed: {staged.sharding}")
  if dhidden.sharding.spec != jax.sharding.PartitionSpec("dp"):
    raise AssertionError(f"installed projection hidden sharding changed: {dhidden.sharding}")

  # Flag presence alone must not select either local boundary outside P59's
  # outer manual data/model context.  Exercise the installed global projection
  # path and compare its physical TP placement, not just its logical shape.
  engine_hidden = jax.device_put(
      hidden,
      jax.sharding.NamedSharding(
          engine_mesh, jax.sharding.PartitionSpec(None, None)
      ),
  )
  engine_weight = jax.device_put(
      weight,
      jax.sharding.NamedSharding(
          engine_mesh, jax.sharding.PartitionSpec(None, "model", None)
      ),
  )
  ordinary_global = xf._column_parallel(
      site,
      "TD,DNH->TNH",
      engine_hidden,
      engine_weight,
      "ordinary.serving.q_proj",
  )
  expected_ordinary = jnp.einsum(
      "td,dnh->tnh", engine_hidden, engine_weight
  )
  if ordinary_global.shape != (16, global_heads, head_dim):
    raise AssertionError(
        f"ordinary projection output boundary changed: {ordinary_global.shape}"
    )
  if not np.array_equal(
      np.asarray(ordinary_global), np.asarray(expected_ordinary)
  ):
    raise AssertionError("ordinary projection values changed with P59 flag")
  expected_ordinary_sharding = jax.sharding.NamedSharding(
      engine_mesh, jax.sharding.PartitionSpec(None, "model", None)
  )
  if ordinary_global.sharding.devices_indices_map(ordinary_global.shape) != (
      expected_ordinary_sharding.devices_indices_map(ordinary_global.shape)
  ):
    raise AssertionError(
        "ordinary projection TP device-index map changed with P59 flag"
    )
  if xf._p59_local_fused_pieces(
      jnp.ones((16, global_heads, head_dim), jnp.bfloat16),
      (head_dim,),
      1,
      "ordinary.serving.q_proj",
      expected_local_width=global_heads * head_dim,
      tp_sharded_last_dim=False,
      site_family="q_proj",
  ) is not None:
    raise AssertionError("ordinary serving selected the P59 local split")

  poisoned = np.asarray(staged).copy()
  poisoned[0, 0, 0] = np.float32(poisoned[0, 0, 0]) + np.float32(1)
  if np.array_equal(poisoned, np.asarray(staged)):
    raise AssertionError("installed projection negative control did not fire")
  print(
      f"P59_TP{TP_SIZE}_INSTALLED_PROJECTION_PASS "
      f"topology=DP2xTP{TP_SIZE} q_proj_layout_shards=1 "
      "gate_up_layout_shards_one=2 wrong_width_negative=2 ordinary_global=1 "
      "serial_parallel=exact optimizer_commits=0",
      flush=True,
  )


if __name__ == "__main__":
  main()
