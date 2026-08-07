#!/usr/bin/env python3
"""Fail fast when the Pathways service cannot deserialize client Mosaic IR.

Pathways releases are coupled to JAX versions. The generic P1 graph does not enter Mosaic, so it
can complete even when production Pallas kernels cannot compile. This probe invokes the exact
promoted Qwen RMSNorm on a minimal full-slice input and records the client package versions before
P1b allocates the complete canonical MLP case.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import sys
from collections.abc import Mapping, Sequence

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

import probe_canonical_ops as canonical


ROWS = 8
VERSION_PATTERN = re.compile(
    r"Unsupported version:\s*expected <=\s*(\d+)\s*but got\s*(\d+)"
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _version_mismatch(message: str) -> tuple[int, int] | None:
    match = VERSION_PATTERN.search(message)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _compact_error(exc: BaseException) -> str:
    mismatch = _version_mismatch(str(exc))
    if mismatch is not None:
        server_max, client_module = mismatch
        return (
            "stable-mosaic-version "
            f"server_max={server_max} client_module={client_module}"
        )
    first_line = next(
        (line.strip() for line in str(exc).splitlines() if line.strip()),
        type(exc).__name__,
    )
    return re.sub(r"\s+", " ", first_line)[:240]


def _admitted(shape: Sequence[int], hidden: int, finite: bool) -> bool:
    return tuple(shape) == (ROWS, hidden) and finite


def _version_contract(
    versions: Mapping[str, str], expected_jax: str, release: str
) -> tuple[bool, str]:
    client_matches = (
        bool(expected_jax)
        and versions.get("jax") == expected_jax
        and versions.get("jaxlib") == expected_jax
    )
    if not client_matches:
        return False, "client-version-contract"
    if not release or not release.endswith(f"jax_{expected_jax}"):
        return False, "pathways-release-contract"
    return True, "ok"


def _mapped_rmsnorm(mesh, modules):
    def local_norm(x_local, weight_local):
        return modules.qwen3.traced_canonical_vjp_rmsnorm(
            x_local, weight_local, epsilon=1.0e-6
        )

    try:
        return shard_map(
            local_norm,
            mesh=mesh,
            in_specs=(P(None, None), P(None)),
            out_specs=P(None, None),
            check_vma=False,
        )
    except TypeError:
        return shard_map(
            local_norm,
            mesh=mesh,
            in_specs=(P(None, None), P(None)),
            out_specs=P(None, None),
            check_rep=False,
        )


def main() -> int:
    versions = {
        "jax": getattr(jax, "__version__", "unknown"),
        "jaxlib": _package_version("jaxlib"),
        "pathwaysutils": _package_version("pathwaysutils"),
    }
    print(
        "[mosaic.compat] VERSIONS "
        + " ".join(f"{name}={value}" for name, value in versions.items()),
        flush=True,
    )
    expected_jax = os.environ.get("CANON_EXPECT_JAX_VERSION", "").strip()
    release = os.environ.get("CANON_EXPECT_PATHWAYS_RELEASE", "").strip()
    print(
        f"[mosaic.compat] CONTRACT expected_jax={expected_jax or 'unset'} "
        f"pathways_release={release or 'unset'}",
        flush=True,
    )
    version_ok, version_reason = _version_contract(
        versions, expected_jax, release
    )
    if not version_ok:
        print(
            f"[mosaic.compat] VERDICT: FAIL {version_reason}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    try:
        modules = canonical._load_canonical_modules()
        hidden = int(modules.contract.HIDDEN_SIZE)
        tp = int(modules.contract.TP_SIZE)
        devices = list(jax.devices())
        mesh = canonical._create_full_slice_mesh(devices, tp)
        canonical._bind_linear_mesh(modules.linear, mesh)
        operation = _mapped_rmsnorm(mesh, modules)
        x_host = np.linspace(-0.5, 0.5, ROWS * hidden, dtype=np.float32).reshape(
            ROWS, hidden
        )
        weight_host = np.linspace(0.9, 1.1, hidden, dtype=np.float32)
        x = jax.device_put(
            jnp.asarray(x_host, dtype=jnp.bfloat16),
            NamedSharding(mesh, P(None, None)),
        )
        weight = jax.device_put(
            jnp.asarray(weight_host, dtype=jnp.bfloat16),
            NamedSharding(mesh, P(None)),
        )
        output = jax.jit(operation)(x, weight)
        jax.block_until_ready(output)
        finite = bool(
            np.asarray(jax.device_get(jnp.all(jnp.isfinite(output))))
        )
        shape = tuple(int(value) for value in output.shape)
    except Exception as exc:
        print(
            f"[mosaic.compat] COMPILE FAIL reason={_compact_error(exc)}",
            file=sys.stderr,
            flush=True,
        )
        print("[mosaic.compat] VERDICT: FAIL", flush=True)
        return 1

    print(
        f"[mosaic.compat] COMPILE PASS shape={shape} finite={int(finite)} "
        f"devices={len(devices)} mesh_shape={mesh.devices.shape} tp={tp}",
        flush=True,
    )
    if not _admitted(shape, hidden, finite):
        print("[mosaic.compat] VERDICT: FAIL", flush=True)
        return 1
    print("[mosaic.compat] VERDICT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
