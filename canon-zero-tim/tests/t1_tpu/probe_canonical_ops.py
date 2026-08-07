"""Hard admission gate for the promoted canonical Qwen MLP operator chain.

The generic way-count probe is intentionally synthetic.  This gate instead calls the exact
P22.XK operators installed into ``tpu_inference``: promoted RMSNorm, column-parallel gate/up
projections, promoted SwiGLU, contract-parallel down projection, and the production fixed-order
TP reduction.  It compares a standalone forward with the primal returned by a weight-gradient
program on the full ``(replica, model)`` slice.

The model dimensions come from the installed model-specific ``p22xf_contract``.  A green result
therefore applies only to that installed canonical chain and topology.  Missing promotion
sentinels, incomplete measurements, dead gradients, or any primal byte difference are hard
failures.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


REPLICA_AXIS = "replica"
TP_AXIS = "model"
M = 256
DEFAULT_DEPTHS = (1, 2, 4, 8)


@dataclass(frozen=True)
class CanonicalModules:
    linear: ModuleType
    qwen2: ModuleType
    qwen3: ModuleType
    contract: ModuleType


def _envd(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _parse_depths(raw: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("CANON_CANONICAL_DEPTHS must contain comma-separated integers") from exc
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("CANON_CANONICAL_DEPTHS must contain positive integers")
    if len(depths) != len(set(depths)):
        raise ValueError("CANON_CANONICAL_DEPTHS must not contain duplicates")
    return depths


def _attest_promoted_modules(
    linear: ModuleType,
    qwen2: ModuleType,
    qwen3: ModuleType,
    contract: ModuleType,
) -> None:
    required_true = (
        (linear, "P22XK_MATMUL_ACTIVE"),
        (qwen2, "P22XK_SWIGLU_ACTIVE"),
        (qwen3, "P22XK_RMSNORM_ACTIVE"),
    )
    for module, name in required_true:
        if getattr(module, name, None) is not True:
            raise RuntimeError(
                f"canonical promotion sentinel is not true: {module.__name__}.{name}"
            )

    required_callables = (
        (linear, "_column_parallel"),
        (linear, "_contract_parallel"),
        (linear, "traced_canonical_vjp_matmul"),
        (qwen2, "traced_canonical_vjp_swiglu"),
        (qwen3, "traced_canonical_vjp_rmsnorm"),
        (contract, "match_site"),
        (contract, "preflight"),
    )
    for module, name in required_callables:
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"canonical callable is absent: {module.__name__}.{name}")

    for name in ("HIDDEN_SIZE", "INTERMEDIATE_SIZE", "TP_SIZE"):
        value = getattr(contract, name, None)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"invalid model contract value: {name}={value!r}")

    xf_module = linear.P22XK_LINEAR_BASE.P22XI_XF_MODULE
    if xf_module.pallas_matmul is not linear.traced_canonical_vjp_matmul:
        raise RuntimeError("the live projection chain does not terminate at P22.XK matmul")
    for name in ("_column_parallel", "_contract_parallel"):
        operation = getattr(linear, name)
        if operation.__globals__.get("pallas_matmul") is not linear.traced_canonical_vjp_matmul:
            raise RuntimeError(f"{name} does not resolve the live P22.XK matmul")
    live_swiglu = qwen2.P22XK_QWEN2_BASE.P22XJ_XG_MODULE.pallas_swiglu
    if live_swiglu is not qwen2.traced_canonical_vjp_swiglu:
        raise RuntimeError("the live MLP chain does not terminate at P22.XK SwiGLU")
    if qwen3.P22XK_QWEN3_BASE.pallas_rmsnorm is not qwen3.traced_canonical_vjp_rmsnorm:
        raise RuntimeError("the live norm chain does not terminate at P22.XK RMSNorm")


def _load_canonical_modules(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> CanonicalModules:
    modules = CanonicalModules(
        linear=importer("tpu_inference.layers.jax.linear"),
        qwen2=importer("tpu_inference.models.jax.qwen2"),
        qwen3=importer("tpu_inference.models.jax.qwen3"),
        contract=importer("p22xf_contract"),
    )
    _attest_promoted_modules(
        modules.linear, modules.qwen2, modules.qwen3, modules.contract
    )
    modules.contract.preflight(require_enabled=True)
    return modules


def _create_full_slice_mesh(devices: Sequence[object], tp: int) -> Mesh:
    if tp <= 1 or len(devices) % tp:
        raise ValueError(f"TP width {tp} must divide {len(devices)} visible devices")
    shape = (len(devices) // tp, tp)
    try:
        arranged = mesh_utils.create_device_mesh(
            shape, devices, allow_split_physical_axes=True
        )
    except TypeError as exc:
        raise RuntimeError(
            "this JAX build lacks allow_split_physical_axes; refusing an unverified reshape"
        ) from exc
    array = np.asarray(arranged, dtype=object)
    source_ids = [int(device.id) for device in devices]
    arranged_ids = [int(device.id) for device in array.flat]
    if array.shape != shape:
        raise RuntimeError(f"canonical mesh shape {array.shape} does not equal {shape}")
    if len(set(arranged_ids)) != len(arranged_ids):
        raise RuntimeError("canonical mesh repeats at least one device")
    if set(arranged_ids) != set(source_ids):
        raise RuntimeError("canonical mesh does not cover the full visible slice")
    return Mesh(array, (REPLICA_AXIS, TP_AXIS))


def _bind_linear_mesh(linear: ModuleType, mesh: Mesh) -> None:
    xi_module = linear.P22XK_LINEAR_BASE
    xf_module = xi_module.P22XI_XF_MODULE
    modules = (linear, xi_module, xf_module, xf_module.base)
    for module in modules:
        module._CANON_MESH = mesh
        module._CANON_TP_AXIS = TP_AXIS
    for module in modules:
        if module._CANON_MESH is not mesh or module._CANON_TP_AXIS != TP_AXIS:
            raise RuntimeError(f"failed to bind canonical mesh on {module.__name__}")


def _host_case(hidden: int, intermediate: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260807)

    def normal(shape: tuple[int, ...], scale: float) -> np.ndarray:
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)

    x = normal((M, hidden), 0.25)
    weights = {
        "norm": normal((hidden,), 0.02) + np.float32(1.0),
        "gate": normal((hidden, intermediate), 0.02),
        "up": normal((hidden, intermediate), 0.02),
        "down": normal((intermediate, hidden), 0.02),
    }
    return x, weights


def _put_case(
    mesh: Mesh,
    x_host: np.ndarray,
    weights_host: Mapping[str, np.ndarray],
):
    def put(value: np.ndarray, spec: P):
        return jax.device_put(
            jnp.asarray(value, dtype=jnp.bfloat16), NamedSharding(mesh, spec)
        )

    x = put(x_host, P(None, None))
    weights = {
        "norm": put(weights_host["norm"], P(None)),
        "gate": put(weights_host["gate"], P(None, TP_AXIS)),
        "up": put(weights_host["up"], P(None, TP_AXIS)),
        "down": put(weights_host["down"], P(TP_AXIS, None)),
    }
    return x, weights


def _make_layer(mesh: Mesh, modules: CanonicalModules):
    linear, qwen2, qwen3, contract = (
        modules.linear,
        modules.qwen2,
        modules.qwen3,
        modules.contract,
    )
    _bind_linear_mesh(linear, mesh)
    gate_site = contract.match_site("model.layers.0.mlp.gate_proj", "mn,np->mp")
    up_site = contract.match_site("model.layers.0.mlp.up_proj", "mn,np->mp")
    down_site = contract.match_site("model.layers.0.mlp.down_proj", "mn,np->mp")
    if gate_site is None or up_site is None or down_site is None:
        raise RuntimeError("the model contract did not resolve all three MLP projection sites")

    def local_norm(x_local, weight_local):
        return qwen3.traced_canonical_vjp_rmsnorm(
            x_local, weight_local, epsilon=1.0e-6
        )

    def local_swiglu(gate_local, up_local):
        return qwen2.traced_canonical_vjp_swiglu(gate_local, up_local)

    try:
        norm = shard_map(
            local_norm,
            mesh=mesh,
            in_specs=(P(None, None), P(None)),
            out_specs=P(None, None),
            check_vma=False,
        )
        swiglu = shard_map(
            local_swiglu,
            mesh=mesh,
            in_specs=(P(None, TP_AXIS), P(None, TP_AXIS)),
            out_specs=P(None, TP_AXIS),
            check_vma=False,
        )
    except TypeError:
        norm = shard_map(
            local_norm,
            mesh=mesh,
            in_specs=(P(None, None), P(None)),
            out_specs=P(None, None),
            check_rep=False,
        )
        swiglu = shard_map(
            local_swiglu,
            mesh=mesh,
            in_specs=(P(None, TP_AXIS), P(None, TP_AXIS)),
            out_specs=P(None, TP_AXIS),
            check_rep=False,
        )

    def layer(x, weights):
        hidden = norm(x, weights["norm"])
        gate = linear._column_parallel(
            gate_site, "mn,np->mp", hidden, weights["gate"],
            "model.layers.0.mlp.gate_proj",
        )
        up = linear._column_parallel(
            up_site, "mn,np->mp", hidden, weights["up"],
            "model.layers.0.mlp.up_proj",
        )
        activation = swiglu(gate, up)
        projected = linear._contract_parallel(
            down_site, "mn,np->mp", activation, weights["down"],
            "model.layers.0.mlp.down_proj",
        )
        return (x + projected).astype(jnp.bfloat16)

    return layer


def _differing_bytes(left: np.ndarray, right: np.ndarray) -> int:
    return int(
        np.count_nonzero(
            np.ascontiguousarray(left).view(np.uint8)
            != np.ascontiguousarray(right).view(np.uint8)
        )
    )


def _gradient_health(gradient) -> tuple[bool, int]:
    leaves = jax.tree_util.tree_leaves(gradient)
    if not leaves:
        return False, 0
    finite = all(
        bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(leaf)))))
        for leaf in leaves
    )
    nonzero = sum(
        int(np.asarray(jax.device_get(jnp.count_nonzero(leaf)))) for leaf in leaves
    )
    return finite, nonzero


def _measure(layer, x, weights, depth: int):
    def forward(x_arg, weights_arg):
        value = x_arg
        for _ in range(depth):
            value = layer(value, weights_arg)
        return value

    plain = jax.jit(forward)(x, weights)

    def loss(x_arg, weights_arg):
        output = forward(x_arg, weights_arg)
        return jnp.sum(output.astype(jnp.float32)), output

    (_, differentiated_primal), gradient = jax.jit(
        jax.value_and_grad(loss, argnums=1, has_aux=True)
    )(x, weights)
    jax.block_until_ready((plain, differentiated_primal, gradient))
    plain_host = np.asarray(jax.device_get(plain))
    differentiated_host = np.asarray(jax.device_get(differentiated_primal))
    finite, nonzero = _gradient_health(gradient)
    return plain_host, differentiated_host, finite, nonzero


def _admit_rows(rows: Sequence[Mapping[str, object]], expected: int) -> bool:
    return bool(
        len(rows) == expected
        and all(int(row["differing_bytes"]) == 0 for row in rows)
        and all(bool(row["gradient_finite"]) for row in rows)
        and all(int(row["gradient_nonzero"]) > 0 for row in rows)
    )


def main() -> int:
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_allow_excess_precision=false" not in flags:
        print(
            "[canonical-op] REFUSING: XLA_FLAGS lacks "
            "--xla_allow_excess_precision=false",
            file=sys.stderr,
        )
        return 2

    try:
        modules = _load_canonical_modules()
        depth_default = ",".join(str(value) for value in DEFAULT_DEPTHS)
        depths = _parse_depths(_envd("CANON_CANONICAL_DEPTHS", depth_default))
        devices = list(jax.devices())
        tp = int(modules.contract.TP_SIZE)
        configured_tp = int(_envd("CANON_TP_SIZE", str(tp)))
        if configured_tp != tp:
            raise ValueError(
                f"configured TP width {configured_tp} does not match model contract {tp}"
            )
        mesh = _create_full_slice_mesh(devices, tp)
        _bind_linear_mesh(modules.linear, mesh)
        x_host, weights_host = _host_case(
            int(modules.contract.HIDDEN_SIZE), int(modules.contract.INTERMEDIATE_SIZE)
        )
        x, weights = _put_case(mesh, x_host, weights_host)
        layer = _make_layer(mesh, modules)
    except (RuntimeError, ValueError, SystemExit) as exc:
        print(f"[canonical-op] REFUSING: {exc}", file=sys.stderr)
        return 2

    ids = tuple(int(device.id) for device in mesh.devices.flat)
    print(
        f"[canonical-op] model_hidden={modules.contract.HIDDEN_SIZE} "
        f"model_intermediate={modules.contract.INTERMEDIATE_SIZE} "
        f"devices={len(devices)} mesh_shape={mesh.devices.shape} tp={tp} "
        f"depths={list(depths)} full_slice=1",
        flush=True,
    )
    print(f"[canonical-op.mesh] ids={ids}", flush=True)

    rows: list[dict[str, object]] = []
    for depth in depths:
        plain, differentiated, finite, nonzero = _measure(
            layer, x, weights, depth
        )
        byte_count = _differing_bytes(plain, differentiated)
        row = {
            "depth": depth,
            "differing_bytes": byte_count,
            "gradient_finite": finite,
            "gradient_nonzero": nonzero,
        }
        rows.append(row)
        print(
            f"[canonical-op] depth={depth:2d} "
            f"differing_bytes={byte_count}/{plain.nbytes} "
            f"gradient_finite={int(finite)} gradient_nonzero={nonzero} "
            f"{'SAME' if byte_count == 0 else 'DIFFERS'}",
            flush=True,
        )

    expected = len(depths)
    print(f"[canonical-op] measurements={len(rows)} expected={expected}", flush=True)
    if not _admit_rows(rows, expected):
        print("[canonical-op] VERDICT: FAIL", flush=True)
        return 1
    print("[canonical-op] VERDICT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
