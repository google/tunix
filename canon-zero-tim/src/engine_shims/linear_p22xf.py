"""Additive P22.XF shim: fixed-tile Pallas for all seven layer projections."""

from __future__ import annotations

import importlib.util
import os

from p22xf_contract import BK, BN, match_site, preflight


BASE_PATH = __import__("canon_shim_root").resolve('linear_patched.py')
value = os.environ.get("CANON_PALLAS_ALL_PROJ", "")
if value not in ("", "1"):
    raise RuntimeError(f"CANON_PALLAS_ALL_PROJ must be unset or 1, got {value!r}")

spec = importlib.util.spec_from_file_location("_canon_linear_p22xf_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load canonical linear module from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_einsum_call = base.JaxEinsum.__call__


if value == "1":
    from p22_pallas_matmul import matmul as pallas_matmul

    preflight(require_enabled=True)


def _column_parallel(site, equation, inputs, weight, prefix):
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    mesh = base._CANON_MESH

    if site.family in ("gate_proj", "up_proj"):
        in_specs = (P(None, None), P(None, base._CANON_TP_AXIS))
        out_specs = P(None, base._CANON_TP_AXIS)

        def local(a_local, w_local):
            out = pallas_matmul(a_local, w_local, block_n=BN, block_k=BK)
            print(
                f"[PATHTRACE] CANON_PALLAS_ALL_PROJ=1 site={site.family} prefix={prefix} "
                f"M={a_local.shape[0]} Klocal={a_local.shape[1]} Nlocal={w_local.shape[1]}",
                flush=True,
            )
            return out
    else:
        # q/k/v: output-head axis is TP sharded.  NDH is supported for the
        # optional q layout by transposing the local weight before flattening.
        if equation == "TD,NDH->TNH":
            weight_spec = P(base._CANON_TP_AXIS, None, None)
        else:
            weight_spec = P(None, base._CANON_TP_AXIS, None)
        in_specs = (P(None, None), weight_spec)
        out_specs = P(None, base._CANON_TP_AXIS, None)

        def local(a_local, w_local):
            if equation == "TD,NDH->TNH":
                w_local = w_local.transpose(1, 0, 2)
            k_local, n_heads, head_dim = w_local.shape
            w2 = w_local.reshape(k_local, n_heads * head_dim)
            out2 = pallas_matmul(a_local, w2, block_n=BN, block_k=BK)
            out = out2.reshape(a_local.shape[0], n_heads, head_dim)
            print(
                f"[PATHTRACE] CANON_PALLAS_ALL_PROJ=1 site={site.family} prefix={prefix} "
                f"M={a_local.shape[0]} Klocal={k_local} Nlocal={n_heads * head_dim}",
                flush=True,
            )
            return out

    try:
        mapped = shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                           check_vma=False)
    except TypeError:
        mapped = shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                           check_rep=False)
    return mapped(inputs, weight)


def _contract_parallel(site, equation, inputs, weight, prefix):
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    mesh = base._CANON_MESH
    count = int(mesh.shape[base._CANON_TP_AXIS])
    in_spec, weight_spec = base._CANON_FIXED_SPECS[equation]

    def local(a_local, w_local):
        a2 = a_local.reshape(a_local.shape[0], -1)
        w2 = w_local.reshape(a2.shape[1], -1)
        partial = pallas_matmul(a2, w2, block_n=BN, block_k=BK)[None]
        print(
            f"[PATHTRACE] CANON_PALLAS_ALL_PROJ=1 site={site.family} prefix={prefix} "
            f"M={a2.shape[0]} Klocal={a2.shape[1]} Nlocal={w2.shape[1]}",
            flush=True,
        )
        parts = [partial]
        current = partial
        for _ in range(count - 1):
            current = base.jax.lax.ppermute(
                current,
                base._CANON_TP_AXIS,
                [(i, (i + 1) % count) for i in range(count)],
            )
            parts.append(current)
        index = base.jax.lax.axis_index(base._CANON_TP_AXIS)
        ordered = base.jnp.stack(parts)[
            base.jnp.argsort((index - base.jnp.arange(count)) % count)
        ]
        acc = ordered[0]
        for part_index in range(1, count):
            acc = acc + ordered[part_index]
        return acc[0]

    print(
        f"[PATHTRACE] CANON_FIXED_AR=1 fixed-order tree at {prefix} "
        f"({equation}, tp={count})",
        flush=True,
    )
    try:
        mapped = shard_map(
            local,
            mesh=mesh,
            in_specs=(P(*in_spec), P(*weight_spec)),
            out_specs=P(None, None),
            check_vma=False,
        )
    except TypeError:
        mapped = shard_map(
            local,
            mesh=mesh,
            in_specs=(P(*in_spec), P(*weight_spec)),
            out_specs=P(None, None),
            check_rep=False,
        )
    return mapped(inputs, weight)


def _p22xf_einsum_call(self, inputs):
    if base._CANON_MESH is not _CANON_MESH:
        base._CANON_MESH = _CANON_MESH
    if value != "1":
        return original_einsum_call(self, inputs)
    site = match_site(self.prefix, self.einsum_str)
    if site is None:
        return original_einsum_call(self, inputs)
    # The frozen bf16 checkpoint uses the package's *unquantized* method
    # wrapper even though a quant_method object is present.  Preserve its
    # post-einsum slice/concat exactly; reject every genuinely quantized or
    # merged method rather than silently bypassing its semantics.
    from tpu_inference.layers.common.utils import slice_sharded_tensor_for_concatenation
    from tpu_inference.layers.jax.quantization.unquantized import UnquantizedLinearMethod

    method = self.quant_method
    if type(method) is not UnquantizedLinearMethod:
        raise RuntimeError(
            f"P22.XF requires exact UnquantizedLinearMethod at {self.prefix}, "
            f"got {type(method).__module__}.{type(method).__name__}"
        )
    config = method.linear_config
    if not config.fuse_matmuls or config.defer_all_reduce:
        raise RuntimeError(
            f"P22.XF unsupported unquantized config at {self.prefix}: "
            f"fuse_matmuls={config.fuse_matmuls} defer_all_reduce={config.defer_all_reduce}"
        )
    if site.contract_parallel:
        output = _contract_parallel(site, self.einsum_str, inputs, self.weight.value, self.prefix)
    else:
        output = _column_parallel(site, self.einsum_str, inputs, self.weight.value, self.prefix)
    if self.bias is not None:
        output += self.bias
    if not site.contract_parallel:
        pieces = slice_sharded_tensor_for_concatenation(
            output, config.output_sizes, config.n_shards
        )
        output = base.jnp.concatenate(pieces, axis=-1)
    return output


base.JaxEinsum.__call__ = _p22xf_einsum_call

for name, obj in vars(base).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
        globals()[name] = obj
