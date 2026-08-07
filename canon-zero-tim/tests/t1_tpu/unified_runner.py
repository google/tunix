"""Fail-stop single-session runner for the T1 topology admission gates.

Pathways initialization is intentionally shared, but failure state is not.  The first probe
that raises or exits nonzero stops the sequence and emits a machine-readable ``SKIP_TAINTED``
marker.  Results produced after a JAX runtime error in the same client session are not release
evidence.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax


T2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "t2_dp"))
if T2_DIR not in sys.path:
    sys.path.insert(0, T2_DIR)


@dataclass(frozen=True)
class Probe:
    name: str
    title: str
    module: str
    call_main: bool = True
    max_devices: int | None = None


PROBES = (
    Probe("P0", "Pathways/JAX registration", "probe_devices"),
    Probe("P2", "mesh order / slice", "probe_mesh_order"),
    Probe("P3", "bucket contract", "probe_bucket_contract"),
    Probe("P4", "F4 cost model", "probe_f4_cost"),
    Probe("P1", "generic full-slice way-count diagnostic", "probe_waycount"),
    Probe("P1b", "canonical Qwen operator admission", "probe_canonical_ops"),
    # These three historical scripts deliberately form device prefixes and tiny 2x2 meshes.
    # They remain useful on the original directly attached <=4-device host, but are not valid
    # Pathways full-slice probes.  P1 supersedes their topology questions on larger slices.
    Probe("H1", "legacy minrepro: F4 tree", "p19_minrepro_f4", False, 4),
    Probe("H2", "legacy minrepro: third program", "p19_minrepro_thirdprog", False),
    Probe("H3", "legacy minrepro: device topology", "p19_minrepro_topo", False, 4),
    Probe("H4", "legacy minrepro: mesh geometry", "p19_minrepro_mesh2d", False, 4),
)

T2_PROBE = Probe("T2", "same-session DP gradient/update admission", "probe_dp_update")

OVERLAY_CHECKS = (
    ("tpu_inference.layers.jax.linear", "P22XK_MATMUL_ACTIVE", True),
    ("tpu_inference.layers.jax.linear", "P22XK_LINEAR_BASE", None),
    ("tpu_inference.layers.jax.embed", "_CANON_F4E_ANNOUNCED", None),
    ("tpu_inference.models.jax.qwen3", "P22XK_RMSNORM_ACTIVE", True),
    ("tpu_inference.models.jax.qwen2", "P22XK_SWIGLU_ACTIVE", True),
)


def _tainted_marker(after: str, remaining: Sequence[Probe]) -> str:
    skipped = ",".join(probe.name for probe in remaining) or "none"
    return f"[t1.unified] SKIP_TAINTED after={after} skipped={skipped}"


def _configured_probes(environ: Mapping[str, str] | None = None) -> tuple[Probe, ...]:
    env = os.environ if environ is None else environ
    value = env.get("CANON_RUN_T2_DP", "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"CANON_RUN_T2_DP must be 0 or 1, got {value!r}")
    if value == "0":
        return PROBES
    insertion = next(
        index + 1 for index, probe in enumerate(PROBES) if probe.name == "P1b"
    )
    return PROBES[:insertion] + (T2_PROBE,) + PROBES[insertion:]


def _verify_overlay(
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    emit: Callable[[str], None] = print,
    emit_error: Callable[[str], None] | None = None,
) -> bool:
    """Verify every promoted module and return one aggregate fail-closed result."""

    if emit_error is None:
        emit_error = lambda message: print(message, file=sys.stderr)
    ok = True
    for module_name, attribute, expected in OVERLAY_CHECKS:
        try:
            module = importer(module_name)
            if not hasattr(module, attribute):
                emit_error(f"[verify] FAIL {module_name}.{attribute} absent")
                ok = False
                continue
            actual = getattr(module, attribute)
            if expected is not None and actual is not expected and actual != expected:
                emit_error(
                    f"[verify] FAIL {module_name}.{attribute}={actual!r}, "
                    f"expected {expected!r}"
                )
                ok = False
                continue
            suffix = f"={actual!r}" if expected is not None else ""
            emit(f"[verify] OK {module_name}.{attribute}{suffix}")
        except Exception as exc:
            emit_error(
                f"[verify] FAIL {module_name}: {type(exc).__name__}: {exc}"
            )
            ok = False
    return ok


def _run_probe_sequence(
    probes: Sequence[Probe],
    *,
    num_devices: int,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    emit: Callable[[str], None] = print,
    emit_error: Callable[[str], None] | None = None,
    print_traceback: Callable[[], None] = traceback.print_exc,
) -> int:
    """Run applicable probes and stop at the first failure or exception."""

    if emit_error is None:
        emit_error = lambda message: print(message, file=sys.stderr)

    for index, probe in enumerate(probes):
        if probe.max_devices is not None and num_devices > probe.max_devices:
            emit(
                f"[t1.unified] SKIP_NOT_APPLICABLE probe={probe.name} "
                f"visible_devices={num_devices} max_devices={probe.max_devices} "
                "reason=legacy-subset-mesh"
            )
            continue

        emit(f"\n== {probe.name}  {probe.title} ==")
        try:
            module = importer(probe.module)
            if probe.call_main:
                return_code = module.main()
                if return_code != 0:
                    emit_error(
                        f"[t1.unified] FAIL probe={probe.name} exit={return_code}"
                    )
                    emit(_tainted_marker(probe.name, probes[index + 1 :]))
                    return 1
        except Exception as exc:
            print_traceback()
            emit_error(
                f"[t1.unified] FAIL probe={probe.name} "
                f"exception={type(exc).__name__}"
            )
            emit(_tainted_marker(probe.name, probes[index + 1 :]))
            return 1
    return 0


def run_all_probes() -> int:
    print(
        "[t1.unified] Pathways initialized in one fail-stop session. Starting probes...",
        flush=True,
    )
    try:
        probes = _configured_probes()
    except ValueError as exc:
        print(f"[t1.unified] REFUSING: {exc}", file=sys.stderr, flush=True)
        return 2
    print("\n== Overlay promotion verification ==", flush=True)
    if not _verify_overlay():
        print(_tainted_marker("overlay", probes), flush=True)
        print("===== T1 FAIL -- overlay verification failed =====", flush=True)
        return 1

    return_code = _run_probe_sequence(
        probes, num_devices=len(jax.devices()), emit=lambda line: print(line, flush=True)
    )
    print("\n" + ("=" * 60), flush=True)
    if return_code == 0:
        print(
            "===== T1 COMPLETE -- all applicable probes produced measurements =====",
            flush=True,
        )
        print(
            "NOTE: COMPLETE proves execution coverage only. Admission still requires "
            "interpreting the paired-arm metrics in CLUSTER_ADMISSION.md.",
            flush=True,
        )
    else:
        print(
            "===== T1 FAIL -- the first failing probe tainted all later probes =====",
            flush=True,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(run_all_probes())
