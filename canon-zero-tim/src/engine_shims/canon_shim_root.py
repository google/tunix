"""Resolve shim-chain sibling paths without hardcoding a deployment directory.

The release chain loaded each next layer through an absolute host path baked into
every shim's BASE_PATH.  On any other host that import fails silently, the engine
falls back to the stock module, and the run goes green without ever exercising the
intervention.  Each BASE_PATH now calls resolve() instead; that substitution is the
ONLY line changed in each shim file.

Resolution order:
  1. $CANON_SHIM_ROOT if set (explicit deployment override)
  2. the directory containing this file (the default: chain members are siblings)
"""
import os


def resolve(name: str) -> str:
    root = os.environ.get("CANON_SHIM_ROOT") or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, name)
