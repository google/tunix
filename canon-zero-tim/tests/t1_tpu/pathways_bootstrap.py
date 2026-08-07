"""Initialize the Pathways proxy backend before JAX is imported.

Pathways registration is optional for a directly attached TPU, but mandatory when the
environment selects JAX's ``proxy`` platform.  In that mandatory case an import or
initialization failure must be loud: silently falling back would make every topology result
belong to a different runtime than the one the operator intended to admit.

This module deliberately does not import JAX.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence
from types import ModuleType


_SUBSLICE_FLAGS = (
    "--FLAGS_pathways_enforce_subset_devices_form_subslice=false",
    "--pathways_enforce_subset_devices_form_subslice=false",
)
_SUBSLICE_ENV = (
    "FLAGS_pathways_enforce_subset_devices_form_subslice",
    "PATHWAYS_ENFORCE_SUBSET_DEVICES_FORM_SUBSLICE",
)
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def pathways_required(environ: Mapping[str, str]) -> bool:
    """Return whether failure to initialize Pathways must abort the probe."""

    override = environ.get("CANON_REQUIRE_PATHWAYS", "").strip().lower()
    if override:
        if override in _TRUE:
            return True
        if override in _FALSE:
            return False
        raise ValueError(
            "CANON_REQUIRE_PATHWAYS must be one of 0/1/false/true/no/yes/off/on"
        )

    platforms = {
        item.strip().lower()
        for item in environ.get("JAX_PLATFORMS", "").split(",")
        if item.strip()
    }
    return bool(
        "proxy" in platforms
        or environ.get("JAX_BACKEND_TARGET", "").strip()
        or environ.get("PATHWAYS_HEAD", "").strip()
    )


_PATHWAYS_INITIALIZED = False


def initialize_pathways(
    *,
    environ: MutableMapping[str, str] | None = None,
    argv: MutableSequence[str] | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    emit: Callable[[str], None] = print,
) -> bool:
    """Initialize Pathways and emit one machine-checkable status marker.

    Returns ``True`` after successful initialization.  A directly attached local run may
    return ``False`` when Pathways is unavailable.  A proxy/Pathways run raises instead.
    Exception messages are intentionally omitted from the marker so credentials or endpoint
    strings cannot leak into logs.
    """
    global _PATHWAYS_INITIALIZED
    if _PATHWAYS_INITIALIZED:
        return True

    env = os.environ if environ is None else environ
    args = sys.argv if argv is None else argv
    required = pathways_required(env)

    for key in _SUBSLICE_ENV:
        env[key] = "false"
    for flag in _SUBSLICE_FLAGS:
        if flag not in args:
            args.append(flag)

    try:
        module = importer("pathwaysutils")
    except Exception as exc:
        emit(
            f"[T1.PATHWAYS] required={int(required)} initialized=0 "
            f"status=import-{type(exc).__name__}"
        )
        if required:
            raise RuntimeError(
                "Pathways is required but pathwaysutils could not be imported"
            ) from exc
        return False

    try:
        from absl import flags as _absl_flags
        try:
            _absl_flags.FLAGS(["prog", "--pathways_enforce_subset_devices_form_subslice=false", "--FLAGS_pathways_enforce_subset_devices_form_subslice=false"], known_only=True)
        except Exception:
            try:
                _absl_flags.FLAGS(["prog"])
            except Exception:
                pass
        if hasattr(_absl_flags.FLAGS, "pathways_enforce_subset_devices_form_subslice"):
            _absl_flags.FLAGS.set_default("pathways_enforce_subset_devices_form_subslice", False)
            _absl_flags.FLAGS.pathways_enforce_subset_devices_form_subslice = False
    except Exception:
        pass

    try:
        module.initialize()
    except Exception as exc:
        emit(
            f"[T1.PATHWAYS] required={int(required)} initialized=0 "
            f"status=initialize-{type(exc).__name__}"
        )
        if required:
            raise RuntimeError("Pathways is required but initialization failed") from exc
        return False

    _PATHWAYS_INITIALIZED = True
    emit(f"[T1.PATHWAYS] required={int(required)} initialized=1 status=ok")
    return True
