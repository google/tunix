#!/usr/bin/env python3
"""CPU gates for the opt-in XProf name-stack helper."""

import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src" / "engine_shims" / "xprof_labels.py"
SPEC = importlib.util.spec_from_file_location("canon_xprof_labels", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  OK   {name}")


original = os.environ.get(MODULE.ENV)
try:
    os.environ.pop(MODULE.ENV, None)
    check("unset is disabled", not MODULE.enabled())
    with MODULE.scope("zt/ro/model"):
        pass

    os.environ[MODULE.ENV] = "0"
    check("zero is disabled", not MODULE.enabled())

    os.environ[MODULE.ENV] = "1"
    check("one is enabled", MODULE.enabled())
    check("layer prefix maps to l07", MODULE.layer_tag("model.layers.7.mlp") == "l07")
    with MODULE.scope("zt/ro/model/l07/mlp"):
        pass

    os.environ[MODULE.ENV] = "pretty"
    try:
        MODULE.enabled()
    except RuntimeError as error:
        check("unknown mode fails closed", "must be unset/0/1" in str(error))
    else:
        raise AssertionError("unknown mode did not fail closed")
finally:
    if original is None:
        os.environ.pop(MODULE.ENV, None)
    else:
        os.environ[MODULE.ENV] = original

print("XPROF_LABELS_CPU_PASS cases=5")
sys.exit(0)
