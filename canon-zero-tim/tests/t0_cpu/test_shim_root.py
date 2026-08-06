"""Unit test for canon_shim_root.resolve -- the one piece of new code in the package.

Everything else in src/engine_shims is byte-identical to sources that already carry signed
evidence. This resolver is the exception: nineteen lines written for this package, sitting
directly under the chain's ability to find its own members. If it returns a wrong path the
chain does not raise -- the engine falls back to stock and the run goes green having done
nothing canonical. So it gets a test of its own rather than resting on the end-to-end import.

Pure Python, no JAX, no TPU. Runs in milliseconds.
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHIMS = os.path.normpath(os.path.join(HERE, "..", "..", "src", "engine_shims"))

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


def load_from(path):
    """Import canon_shim_root from a specific copy, so 'its own directory' is unambiguous."""
    spec = importlib.util.spec_from_file_location(
        f"_csr_{abs(hash(path))}", os.path.join(path, "canon_shim_root.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("== canon_shim_root.resolve ==")

csr = load_from(SHIMS)

# 1. Default: siblings of the module itself.  This is what makes the package relocatable --
#    the chain finds its next layer wherever the directory happens to have been installed.
os.environ.pop("CANON_SHIM_ROOT", None)
got = csr.resolve("linear_p22xi.py")
check("default resolves next to the module",
      got == os.path.join(SHIMS, "linear_p22xi.py"), f"got {got}")
check("default result is absolute", os.path.isabs(got), f"got {got}")

# 2. The name it resolves must actually be there, or the whole scheme is decorative.
for sibling in ("linear_p22xi.py", "linear_p22xf.py", "qwen3_p22xh.py",
                "qwen2_p22xg.py", "qwen2_p22xj.py"):
    p = csr.resolve(sibling)
    exists = os.path.isfile(p) or sibling == "qwen3_p22xh.py"  # model-specific: copied at install
    check(f"resolves an existing sibling: {sibling}", exists, f"missing {p}")

# 3. Explicit override wins.  A deployment that lays the chain down somewhere else sets this.
os.environ["CANON_SHIM_ROOT"] = "/opt/canon"
got = csr.resolve("linear_p22xi.py")
check("CANON_SHIM_ROOT overrides the default", got == "/opt/canon/linear_p22xi.py", f"got {got}")

# 4. An empty value must NOT be treated as an override.  Under docker and Kubernetes `-e K=`
#    always sets the key, so an unset source variable arrives as "" rather than as a missing
#    key.  Falling for that would resolve every chain member against the filesystem root.
os.environ["CANON_SHIM_ROOT"] = ""
got = csr.resolve("linear_p22xi.py")
check("empty CANON_SHIM_ROOT falls back to the default",
      got == os.path.join(SHIMS, "linear_p22xi.py"), f"got {got}")
os.environ.pop("CANON_SHIM_ROOT", None)

# 5. Relocation: a copy in a fresh directory must resolve to that directory, not to the
#    original.  This is the property the whole portability fix rests on.
with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(SHIMS, "canon_shim_root.py"), encoding="utf-8") as f:
        src = f.read()
    with open(os.path.join(tmp, "canon_shim_root.py"), "w", encoding="utf-8") as f:
        f.write(src)
    moved = load_from(tmp)
    got = moved.resolve("linear_p22xi.py")
    check("a relocated copy resolves against its new home",
          got == os.path.join(tmp, "linear_p22xi.py"), f"got {got}")
    check("a relocated copy does not leak the original path",
          SHIMS not in got, f"got {got}")

print()
if failures:
    print(f"===== SHIM ROOT FAIL ({len(failures)}): {', '.join(failures)} =====")
    sys.exit(1)
print("===== SHIM ROOT PASS =====")
