"""Measure third-program drift on full-slice DP-by-TP meshes.

The probe intentionally uses every visible device.  A prefix such as ``devices[:4]`` is not a
valid TP4 experiment on a multi-host Pathways slice: it may cut across host boundaries, and it
does not represent the production DP-by-TP mesh.  For TP width ``w`` this probe constructs the
topology-aware logical mesh ``(num_devices // w, w)`` with axes ``(replica, m)``.

Each (width, depth) point reuses one host-generated input and one set of weights across three
arms:

* ``replicated``: no tensor-parallel reduction;
* ``stock-ar``: XLA's reduction over the TP axis;
* ``f4-fixed``: the same TP partials summed in a fixed global-rank order.

The replicated arm is essential on Pathways.  If it drifts too, a reduction is not necessary
for the observed third-program difference, so stock-versus-F4 byte counts cannot isolate the
cause.  ``differing_bytes`` is only a bitwise yes/no gate; rel-L2, one-minus-cosine, and max-abs
quantify magnitude without relying on a saturating byte count.

Environment:
    CANON_WAYCOUNT_WIDTHS  comma-separated TP widths (default: supported members of 2,4,8)
    CANON_WAYCOUNT_DEPTHS  comma-separated stack depths (default: 8,15,24)

XLA_FLAGS must contain ``--xla_allow_excess_precision=false``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

D, F, T = 512, 2048, 256
REPLICA_AXIS = "replica"
TP_AXIS = "m"
ARM_NAMES = ("replicated", "stock-ar", "f4-fixed")


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    """Parse a nonempty comma-separated positive-integer list."""

    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _default_widths(num_devices: int) -> list[int]:
    """Return the historically relevant widths that exactly divide the full slice."""

    return [width for width in (2, 4, 8) if num_devices % width == 0]


def _validate_schedule(num_devices: int, widths: Sequence[int], depths: Sequence[int]) -> None:
    """Reject any schedule that cannot form full-slice replica-by-TP meshes."""

    if num_devices < 2:
        raise ValueError("at least two visible devices are required")
    if not widths:
        raise ValueError("no TP width is available")
    if not depths:
        raise ValueError("no stack depth is configured")
    for width in widths:
        if width < 2:
            raise ValueError(f"TP width must be at least 2, got {width}")
        if width > num_devices or num_devices % width:
            raise ValueError(
                f"TP width {width} must exactly divide {num_devices} visible devices"
            )


def _device_id(device: object) -> int:
    return int(getattr(device, "id"))


def _device_coords(device: object) -> object:
    return getattr(device, "coords", "-")


def _attest_full_slice(
    built: np.ndarray, devices: Sequence[object], width: int
) -> list[list[int]]:
    """Return TP groups after proving exact full-slice coverage and shape."""

    expected_shape = (len(devices) // width, width)
    array = np.asarray(built, dtype=object)
    if array.shape != expected_shape:
        raise ValueError(f"mesh shape {array.shape} does not equal {expected_shape}")

    source_ids = [_device_id(device) for device in devices]
    built_ids = [_device_id(device) for device in array.flat]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("visible device ids are not unique")
    if len(set(built_ids)) != len(built_ids):
        raise ValueError("built mesh repeats at least one device")
    if set(built_ids) != set(source_ids):
        missing = sorted(set(source_ids) - set(built_ids))
        extra = sorted(set(built_ids) - set(source_ids))
        raise ValueError(f"built mesh is not full-slice: missing={missing} extra={extra}")
    return [[_device_id(device) for device in row] for row in array]


def _create_full_slice_mesh(devices: Sequence[object], width: int) -> np.ndarray:
    """Build a topology-aware (replica, TP) mesh without unsafe reshape fallback."""

    shape = (len(devices) // width, width)
    try:
        built = mesh_utils.create_device_mesh(
            shape, devices, allow_split_physical_axes=True
        )
    except TypeError as exc:
        raise RuntimeError(
            "this JAX build lacks allow_split_physical_axes; refusing a plain reshape "
            "because it would not validate Pathways topology placement"
        ) from exc
    _attest_full_slice(np.asarray(built, dtype=object), devices, width)
    return np.asarray(built, dtype=object)


def _rms(x, gain):
    xf = x.astype(jnp.float32)
    return (
        xf
        * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6)
        * gain.astype(jnp.float32)
    ).astype(jnp.bfloat16)


def _make_fwd(mesh: Mesh, width: int, *, fixed: bool):
    """Return a transformer-like MLP stack reduced only over the TP axis."""

    def _fixed_tree(partial):
        # Every TP rank observes all partials, then sums them in global TP-rank order.
        parts = [partial]
        current = partial
        permutations = [(rank, (rank + 1) % width) for rank in range(width)]
        for _ in range(width - 1):
            current = jax.lax.ppermute(current, TP_AXIS, permutations)
            parts.append(current)
        rank = jax.lax.axis_index(TP_AXIS)
        ordered = jnp.stack(parts)[
            jnp.argsort((rank - jnp.arange(width)) % width)
        ]
        result = ordered[0]
        for index in range(1, width):
            result = result + ordered[index]
        return result

    def fwd(x, weights):
        for params in weights:
            hidden = _rms(x, params["g"])
            gate = jax.nn.silu(
                jnp.einsum("td,df->tf", hidden, params["Wg"]).astype(jnp.float32)
            ).astype(jnp.bfloat16)
            activation = gate * jnp.einsum(
                "td,df->tf", hidden, params["Wu"]
            )
            if fixed:

                def local_reduce(local_activation, local_down):
                    return _fixed_tree((local_activation @ local_down)[None])[0]

                try:
                    mapped = shard_map(
                        local_reduce,
                        mesh=mesh,
                        in_specs=(P(None, TP_AXIS), P(TP_AXIS, None)),
                        out_specs=P(None, None),
                        check_vma=False,
                    )
                except TypeError:
                    mapped = shard_map(
                        local_reduce,
                        mesh=mesh,
                        in_specs=(P(None, TP_AXIS), P(TP_AXIS, None)),
                        out_specs=P(None, None),
                        check_rep=False,
                    )
                projected = mapped(activation, params["Wd"])
            else:
                projected = jnp.einsum(
                    "tf,fd->td", activation, params["Wd"]
                )
            x = x + projected
        return x

    return fwd


def _differing_bytes(left: np.ndarray, right: np.ndarray) -> int:
    return int(
        (
            np.ascontiguousarray(left).view(np.uint8)
            != np.ascontiguousarray(right).view(np.uint8)
        ).sum()
    )


def _error_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    """Return rel-L2, one-minus-cosine, and max-absolute error in float64."""

    if np.array_equal(left, right):
        return 0.0, 0.0, 0.0
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    delta = a - b
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    rel_l2 = float(np.linalg.norm(delta)) / max(norm_a, np.finfo(np.float64).tiny)
    denominator = norm_a * norm_b
    one_minus_cosine = (
        float("nan") if denominator == 0.0 else 1.0 - float(np.dot(a, b) / denominator)
    )
    max_abs = float(np.max(np.abs(delta)))
    return rel_l2, one_minus_cosine, max_abs


def _host_case(rng: np.random.Generator, depth: int):
    """Generate one host case that all three arms must reuse exactly."""

    weights = []
    for _ in range(depth):
        weights.append(
            {
                "g": (rng.normal(size=(D,)) * 0.1 + 1.0).astype(np.float32),
                "Wg": (rng.normal(size=(D, F)) * 0.02).astype(np.float32),
                "Wu": (rng.normal(size=(D, F)) * 0.02).astype(np.float32),
                "Wd": (rng.normal(size=(F, D)) * 0.02).astype(np.float32),
            }
        )
    x = (rng.normal(size=(T, D)) * 0.5).astype(np.float32)
    return x, weights


def _put_case(mesh: Mesh, x_host, weights_host, *, tp_sharded: bool):
    """Place one host case with either replicated or TP-sharded parameters."""

    def put(array, spec):
        return jax.device_put(
            jnp.asarray(array, jnp.bfloat16), NamedSharding(mesh, spec)
        )

    x = put(x_host, P(None, None))
    if tp_sharded:
        specs = {
            "g": P(None),
            "Wg": P(None, TP_AXIS),
            "Wu": P(None, TP_AXIS),
            "Wd": P(TP_AXIS, None),
        }
    else:
        specs = {name: P(None, None) for name in ("Wg", "Wu", "Wd")}
        specs["g"] = P(None)
    weights = [
        {name: put(array, specs[name]) for name, array in params.items()}
        for params in weights_host
    ]
    return x, weights


def _measure(fwd, x, weights) -> tuple[np.ndarray, np.ndarray]:
    plain = np.asarray(jax.device_get(jax.jit(fwd)(x, weights)))

    def loss(x_arg, weights_arg):
        output = fwd(x_arg, weights_arg)
        return jnp.sum(output.astype(jnp.float32)), output

    (_, differentiated_primal), _ = jax.jit(
        jax.value_and_grad(loss, argnums=0, has_aux=True)
    )(x, weights)
    differentiated = np.asarray(jax.device_get(differentiated_primal))
    return plain, differentiated


def _expected_measurements(widths: Iterable[int], depths: Iterable[int]) -> int:
    return len(tuple(widths)) * len(tuple(depths)) * len(ARM_NAMES)


def _measurements_complete(completed: int, expected: int) -> bool:
    """Return the fail-closed row-count verdict."""

    return completed == expected


def main() -> int:
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_allow_excess_precision=false" not in flags:
        print(
            "[waycount] REFUSING: XLA_FLAGS lacks "
            "--xla_allow_excess_precision=false; this would measure another program family.",
            file=sys.stderr,
        )
        return 2

    devices = jax.devices()
    num_devices = len(devices)
    try:
        widths_raw = os.environ.get("CANON_WAYCOUNT_WIDTHS", "").strip()
        widths = (
            _parse_int_list(widths_raw, name="CANON_WAYCOUNT_WIDTHS")
            if widths_raw
            else _default_widths(num_devices)
        )
        depths = _parse_int_list(
            os.environ.get("CANON_WAYCOUNT_DEPTHS", "8,15,24"),
            name="CANON_WAYCOUNT_DEPTHS",
        )
        _validate_schedule(num_devices, widths, depths)
    except ValueError as exc:
        print(f"[waycount] REFUSING: {exc}", file=sys.stderr)
        return 2

    print(
        f"[waycount] devices={num_devices} kind={devices[0].device_kind} "
        f"widths={widths} depths={depths} arms={list(ARM_NAMES)}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    completed = 0
    expected = _expected_measurements(widths, depths)
    for width in widths:
        built = _create_full_slice_mesh(devices, width)
        groups = _attest_full_slice(built, devices, width)
        print(
            f"[waycount.mesh] width={width} shape={built.shape} "
            f"devices={built.size} unique={len(set(device.id for device in built.flat))} "
            "full_slice=1",
            flush=True,
        )
        for group_index, row in enumerate(np.asarray(built, dtype=object)):
            ids = [_device_id(device) for device in row]
            coords = [_device_coords(device) for device in row]
            print(
                f"[waycount.mesh] width={width} group={group_index:02d} "
                f"ids={ids} coords={coords}",
                flush=True,
            )
        if groups != [[_device_id(device) for device in row] for row in built]:
            raise AssertionError("mesh attestation changed between validation and reporting")

        mesh = Mesh(built, (REPLICA_AXIS, TP_AXIS))
        for depth in depths:
            x_host, weights_host = _host_case(rng, depth)
            replicated_case = _put_case(
                mesh, x_host, weights_host, tp_sharded=False
            )
            tp_case = _put_case(mesh, x_host, weights_host, tp_sharded=True)
            arms = (
                ("replicated", _make_fwd(mesh, width, fixed=False), replicated_case),
                ("stock-ar", _make_fwd(mesh, width, fixed=False), tp_case),
                ("f4-fixed", _make_fwd(mesh, width, fixed=True), tp_case),
            )
            for arm, fwd, (x, weights) in arms:
                plain, differentiated = _measure(fwd, x, weights)
                byte_count = _differing_bytes(plain, differentiated)
                rel_l2, one_minus_cosine, max_abs = _error_metrics(
                    plain, differentiated
                )
                print(
                    f"[waycount] width={width:2d} replicas={num_devices // width:2d} "
                    f"depth={depth:3d} arm={arm:10s} "
                    f"differing_bytes={byte_count:7d}/{plain.nbytes} "
                    f"rel_l2={rel_l2:.6e} one_minus_cos={one_minus_cosine:.6e} "
                    f"max_abs={max_abs:.6e} "
                    f"{'SAME' if byte_count == 0 else 'DIFFERS'}",
                    flush=True,
                )
                completed += 1

    print(
        f"[waycount] measurements={completed} expected={expected}", flush=True
    )
    if not _measurements_complete(completed, expected):
        print(
            "[waycount] VERDICT: INCONCLUSIVE -- measurement count mismatch",
            flush=True,
        )
        return 1
    print("[waycount] VERDICT: COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
