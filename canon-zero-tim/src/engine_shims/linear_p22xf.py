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


def _p59_local_tp_context() -> bool:
    """Return whether P59 already maps the live engine over DP and TP."""
    if os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "") != "1":
        return False
    import jax

    context = jax.sharding.get_abstract_mesh()
    if tuple(context.axis_names) != ("data", "model"):
        return False
    axis_types = dict(zip(context.axis_names, context.axis_types))
    if (
        axis_types.get("data") is not jax.sharding.AxisType.Manual
        or axis_types.get("model") is not jax.sharding.AxisType.Manual
    ):
        return False
    mesh = base._CANON_MESH
    if mesh is None or base._CANON_TP_AXIS != "model":
        raise RuntimeError(
            "P59 local projection requires the live engine model axis"
        )
    if "data" not in mesh.shape or "model" not in mesh.shape:
        raise RuntimeError(
            "P59 local projection requires engine data/model axes"
        )
    if (
        int(context.shape["data"]) <= 1
        or int(context.shape["model"]) <= 1
        or int(context.shape["data"]) != int(mesh.shape["data"])
        or int(context.shape["model"]) != int(mesh.shape["model"])
    ):
        raise RuntimeError(
            "P59 local projection context and engine topology differ"
        )
    return True


def _p59_fixed_order_tp_sum(value):
    """Sum one replicated-input cotangent in explicit TP rank order.

    The per-shard cotangent has the BF16 primal dtype.  Accumulating eight TP
    partials in BF16 introduced multiple avoidable roundings and was measurably
    farther from the FP64 oracle.  Match the canonical matmul accumulator
    contract by adding rank-ordered partials in FP32, then cast once at the
    replicated-input boundary.  Barriers on both operands preserve each
    source-level association instead of leaving the FP32 chain open to XLA
    reassociation.
    """
    count = int(base._CANON_MESH.shape[base._CANON_TP_AXIS])
    gathered = base.jax.lax.all_gather(
        value.astype(base.jnp.float32),
        base._CANON_TP_AXIS,
        axis=0,
        tiled=False,
    )
    total = gathered[0]
    for rank in range(1, count):
        total = (
            base.jax.lax.optimization_barrier(total)
            + base.jax.lax.optimization_barrier(gathered[rank])
        )
    return total.astype(value.dtype)


def _p59_local_fused_pieces(
    output,
    output_sizes,
    n_shards,
    prefix,
    *,
    expected_local_width,
    tp_sharded_last_dim,
    site_family,
):
    """Apply the engine's local concat layout inside the P59 TP map.

    ``n_shards`` belongs to the fused-output layout; it is not the live mesh TP
    size.  A non-fused q/k/v projection legitimately reports one layout shard
    even on TP4/TP8 and is already TP-local at this boundary.  Conversely,
    gate/up keep their global logical ``output_sizes`` while the enclosing P59
    map has already reduced the last dimension to one physical TP slice.  Use
    the live TP size only for that last-dimension layout and validate the
    resulting feature width against the model-exact projection contract.
    """
    if not _p59_local_tp_context():
        return None
    n_shards = int(n_shards)
    output_sizes = tuple(map(int, output_sizes))
    expected_local_width = int(expected_local_width)
    tp_size = int(base._CANON_MESH.shape[base._CANON_TP_AXIS])
    if n_shards not in (1, tp_size):
        raise RuntimeError(
            f"P59 local fused-linear layout shard mismatch at {prefix}: "
            f"layout_shards={n_shards} live_tp={tp_size}"
        )
    actual_local_width = 1
    for dimension in output.shape[1:]:
        actual_local_width *= int(dimension)
    if actual_local_width != expected_local_width:
        raise RuntimeError(
            "P59 local projection feature width mismatch at "
            f"{prefix}: {actual_local_width} != {expected_local_width}"
        )
    if tp_sharded_last_dim and output.ndim != 2:
        raise RuntimeError(
            f"P59 local {site_family} expected rank-2 TP-last output at "
            f"{prefix}, got shape={output.shape}"
        )
    split_divisor = tp_size if tp_sharded_last_dim else n_shards
    if any(size % split_divisor for size in output_sizes):
        raise RuntimeError(
            f"P59 local fused-linear split is not divisible at {prefix}: "
            f"sizes={output_sizes} divisor={split_divisor}"
        )
    local_sizes = tuple(size // split_divisor for size in output_sizes)
    expected_local_last_dim = sum(local_sizes)
    if int(output.shape[-1]) != expected_local_last_dim:
        raise RuntimeError(
            "P59 local fused-linear last-dimension mismatch at "
            f"{prefix}: {output.shape[-1]} != {expected_local_last_dim}"
        )
    pieces = []
    start = 0
    for local_size in local_sizes:
        pieces.append(output[..., start:start + local_size])
        start += local_size
    if start != int(output.shape[-1]):
        raise RuntimeError(
            f"P59 local fused-linear split did not consume output at "
            f"{prefix}: {start} != {output.shape[-1]}"
        )
    if tp_sharded_last_dim:
        print(
            "[PATHTRACE] P59_LOCAL_FUSED_LINEAR_READY "
            f"tp={tp_size} site={site_family} "
            f"local_width={expected_local_width} "
            f"declared_width={sum(output_sizes)} "
            f"layout_shards={n_shards} pieces={len(pieces)}",
            flush=True,
        )
    return tuple(pieces)


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
    if _p59_local_tp_context():
        import jax

        if norm is None:

            @jax.custom_vjp
            def p59_column(a_local, w_local):
                return local(a_local, w_local)

            def p59_column_fwd(a_local, w_local):
                output_local = local(a_local, w_local)
                return output_local, (a_local, w_local)

            def p59_column_bwd(residual, cotangent):
                a_local, w_local = residual
                _, pullback = jax.vjp(local, a_local, w_local)
                da_local, dw_local = pullback(cotangent)
                return _p59_fixed_order_tp_sum(da_local), dw_local

            p59_column.defvjp(p59_column_fwd, p59_column_bwd)
            return p59_column(*map_args)

        @jax.custom_vjp
        def p59_norm_column(a_local, w_local, gamma_local):
            return local(a_local, w_local, gamma_local)

        def p59_norm_column_fwd(a_local, w_local, gamma_local):
            output_local = local(a_local, w_local, gamma_local)
            return output_local, (a_local, w_local, gamma_local)

        def p59_norm_column_bwd(residual, cotangent):
            a_local, w_local, gamma_local = residual
            _, pullback = jax.vjp(
                local, a_local, w_local, gamma_local
            )
            da_local, dw_local, dgamma_local = pullback(cotangent)
            return (
                _p59_fixed_order_tp_sum(da_local),
                dw_local,
                _p59_fixed_order_tp_sum(dgamma_local),
            )

        p59_norm_column.defvjp(
            p59_norm_column_fwd, p59_norm_column_bwd
        )
        return p59_norm_column(*map_args)
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
    if _p59_local_tp_context():
        return local(inputs, weight)
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
        pieces = _p59_local_fused_pieces(
            output,
            config.output_sizes,
            config.n_shards,
            self.prefix,
            expected_local_width=site.n_local,
            tp_sharded_last_dim=site.family in ("gate_proj", "up_proj"),
            site_family=site.family,
        )
        if pieces is None:
            pieces = slice_sharded_tensor_for_concatenation(
                output, config.output_sizes, config.n_shards
            )
        output = base.jnp.concatenate(pieces, axis=-1)
    return output


base.JaxEinsum.__call__ = _p22xf_einsum_call

for name, obj in vars(base).items():
    if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
        globals()[name] = obj
