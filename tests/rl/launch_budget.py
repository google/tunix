"""Counts device program launches per adapter source line (CPU, test support).

Every jitted call in JAX ends in a C++ ``PjitFunction``; wrapping the
factory that creates them (``jax._src.pjit._cpp_pjit``) and the jitted
helpers that ``jax.numpy`` created before this module was imported makes
every launch visible to Python, including cache hits on the C++ fast path.
Each launch is attributed to the innermost ``canonical_qwen3_adapter``
frame on the stack, so a test can pin how many programs a loop issues.

Import this module before building an adapter (it patches at import time).
"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib
import sys

from jax._src import pjit as _pjit

_ADAPTER_SUFFIX = "tunix/rl/canonical_qwen3_adapter.py"
_records: list[tuple[str, str]] = []
_active = [False]


def _site():
  frame = sys._getframe(2)  # pylint: disable=protected-access
  while frame is not None:
    if frame.f_code.co_filename.endswith(_ADAPTER_SUFFIX):
      return f"{frame.f_code.co_name}:{frame.f_lineno}"
    frame = frame.f_back
  return "<outside adapter>"


def _record(name):
  if _active[0]:
    _records.append((_site(), name))


_original_cpp_pjit = _pjit._cpp_pjit  # pylint: disable=protected-access


class _CountingJit:
  """Calls the C++ PjitFunction and records the launch; everything else
  (``_cache_size``, ``lower``, ``trace``, ...) is delegated unchanged."""

  def __init__(self, cpp, name):
    self._cpp = cpp
    self._name = name
    functools.update_wrapper(self, cpp, updated=())

  def __call__(self, *args, **kwargs):
    _record(self._name)
    return self._cpp(*args, **kwargs)

  def __getattr__(self, attr):
    return getattr(self._cpp, attr)


def _counting_cpp_pjit(fun, jit_info):
  return _CountingJit(_original_cpp_pjit(fun, jit_info), getattr(fun, "__name__", "jit"))


_pjit._cpp_pjit = _counting_cpp_pjit  # pylint: disable=protected-access

_HELPER_MODULES = (
    "jax._src.numpy.lax_numpy",
    "jax._src.numpy.array_methods",
    "jax._src.numpy.indexing",
    "jax._src.numpy.util",
    "jax._src.numpy.reductions",
    "jax._src.numpy.array_creation",
    "jax._src.numpy.ufuncs",
    "jax._src.lax.lax",
    "jax._src.lax.slicing",
)


def _wrap_existing_helpers():
  for module_name in _HELPER_MODULES:
    try:
      module = importlib.import_module(module_name)
    except ImportError:
      continue
    for attr, value in list(vars(module).items()):
      if type(value).__name__ != "PjitFunction":
        continue

      def make(original, label):
        def wrapper(*args, **kwargs):
          _record(label)
          return original(*args, **kwargs)

        wrapper.__name__ = label
        return wrapper

      setattr(module, attr, make(value, attr))


_wrap_existing_helpers()


@contextlib.contextmanager
def counting():
  """Collects (adapter site, program name) for every launch inside."""
  del _records[:]
  _active[0] = True
  try:
    yield _records
  finally:
    _active[0] = False


@contextlib.contextmanager
def paused():
  """Suspends counting (e.g. around a test fixture's own eager stub)."""
  previous = _active[0]
  _active[0] = False
  try:
    yield
  finally:
    _active[0] = previous


def by_site(records):
  return collections.Counter(site for site, _ in records)


def by_program(records):
  return collections.Counter(name for _, name in records)


def in_function(records, function_name):
  """Launches attributed to lines of one adapter function."""
  return [(site, name) for site, name in records if site.split(":")[0] == function_name]


def named_programs(records, prefixes=("fwd_", "bwd_", "local_pullback", "rebuild", "start", "add", "compute", "<lambda>")):
  """Launches of the adapter's own named programs (not eager glue)."""
  return [(site, name) for site, name in records if name.startswith(prefixes)]
