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
    from p56_pallas_norm_matmul import continue_decode_norm_matmul
    from p56_pallas_norm_matmul import norm_matmul as pallas_norm_matmul

    preflight(require_enabled=True)


def _column_parallel(site, equation, inputs, weight, prefix, norm=None):
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    mesh = base._CANON_MESH
    # norm: optional (gamma, epsilon) from a P56.4.6 deferred rmsnorm; the
    # local body then runs the verbatim normalize as the matmul prologue
    # in one custom call instead of consuming a pre-normalized tensor.
    norm_epsilon = norm[1] if norm is not None else None

    def _project(a_local, w2, gamma_local):
        if gamma_local is None:
            return pallas_matmul(a_local, w2, block_n=BN, block_k=BK)
        # The custom-vjp coat keeps the fused Pallas primal and supplies
        # the composed canonical-replica backward, so admission gates
        # that differentiate through the engine layer stay analytic
        # instead of hitting pallas AD (the r13 crash).
        from p22xk_vjp_ops import norm_matmul as coated_norm_matmul

        if (
            os.environ.get("CANON_CONTINUE_DECODE", "")
            and int(a_local.shape[0]) % 128
        ):
            return continue_decode_norm_matmul(
                a_local,
                gamma_local,
                w2,
                epsilon=norm_epsilon,
                block_n=BN,
                block_k=BK,
            )

        def _fused_forward(a, g, b):
            return pallas_norm_matmul(
                a, g, b, epsilon=norm_epsilon, block_n=BN, block_k=BK
            )

        return coated_norm_matmul(
            a_local, gamma_local, w2, epsilon=norm_epsilon,
            forward=_fused_forward,
        )

    if site.family in ("gate_proj", "up_proj"):
        in_specs = (P(None, None), P(None, base._CANON_TP_AXIS))
        out_specs = P(None, base._CANON_TP_AXIS)

        def local(a_local, w_local, gamma_local=None):
            out = _project(a_local, w_local, gamma_local)
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

        def local(a_local, w_local, gamma_local=None):
            if equation == "TD,NDH->TNH":
                w_local = w_local.transpose(1, 0, 2)
            k_local, n_heads, head_dim = w_local.shape
            w2 = w_local.reshape(k_local, n_heads * head_dim)
            out2 = _project(a_local, w2, gamma_local)
            out = out2.reshape(a_local.shape[0], n_heads, head_dim)
            print(
                f"[PATHTRACE] CANON_PALLAS_ALL_PROJ=1 site={site.family} prefix={prefix} "
                f"M={a_local.shape[0]} Klocal={k_local} Nlocal={n_heads * head_dim}",
                flush=True,
            )
            return out

    if norm is not None:
        in_specs = in_specs + (P(None),)
        map_args = (inputs, weight, norm[0])
    else:
        map_args = (inputs, weight)
    try:
        mapped = shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                           check_vma=False)
    except TypeError:
        mapped = shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                           check_rep=False)
    return mapped(*map_args)


def _contract_parallel(site, equation, inputs, weight, prefix):
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    mesh = base._CANON_MESH
    count = int(mesh.shape[base._CANON_TP_AXIS])
    in_spec, weight_spec = base._CANON_FIXED_SPECS[equation]

    gather_mode = os.environ.get("CANON_FIXED_AR_GATHER", "")
    if gather_mode not in ("", "0", "1"):
        raise RuntimeError(
            f"CANON_FIXED_AR_GATHER must be unset/0/1, got {gather_mode!r}"
        )
    gather_mode = gather_mode == "1"

    def local(a_local, w_local):
        a2 = a_local.reshape(a_local.shape[0], -1)
        w2 = w_local.reshape(a2.shape[1], -1)
        if gather_mode:
            # P56.4.5a: one all_gather delivers every rank's partial in
            # rank order (all_gather concatenates by axis index), and the
            # sum below adds them in that same rank order -- the identical
            # operand values in the identical association order as the
            # ppermute ring, so the committed activations are bit-equal
            # while the per-call collective count drops from three
            # sequential hops to one and the stack/argsort glue vanishes.
            partial = pallas_matmul(a2, w2, block_n=BN, block_k=BK)
            print(
                f"[PATHTRACE] CANON_PALLAS_ALL_PROJ=1 site={site.family} prefix={prefix} "
                f"M={a2.shape[0]} Klocal={a2.shape[1]} Nlocal={w2.shape[1]}",
                flush=True,
            )
            gathered = base.jax.lax.all_gather(
                partial, base._CANON_TP_AXIS, axis=0, tiled=False
            )
            acc = gathered[0]
            for part_index in range(1, count):
                acc = acc + gathered[part_index]
            return acc
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
        f"[PATHTRACE] CANON_FIXED_AR=1 "
        f"{'gather-ordered-sum' if gather_mode else 'fixed-order tree'} "
        f"at {prefix} ({equation}, tp={count})",
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
    # A P56.4.6 deferred rmsnorm may arrive instead of a tensor; unwrap it
    # BEFORE any early return so it can never silently reach stock code.
    norm = None
    if getattr(inputs, "_p22xh_deferred", False):
        if value != "1":
            raise RuntimeError(
                "P56.4.6 deferred norm requires CANON_PALLAS_ALL_PROJ=1"
            )
        norm = (inputs.weight, inputs.epsilon)
        inputs = inputs.x
    if value != "1":
        return original_einsum_call(self, inputs)
    site = match_site(self.prefix, self.einsum_str)
    if site is None:
        if norm is not None:
            raise RuntimeError(
                f"P56.4.6 deferred norm reached an unmatched einsum at "
                f"{self.prefix}"
            )
        return original_einsum_call(self, inputs)
    if norm is not None and site.contract_parallel:
        raise RuntimeError(
            f"P56.4.6 deferred norm reached a contract-parallel site at "
            f"{self.prefix}"
        )
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
        output = _column_parallel(
            site, self.einsum_str, inputs, self.weight.value, self.prefix,
            norm=norm,
        )
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
