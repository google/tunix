# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assert that the installed TPU stack is the one this run expects.

`vllm-tpu==X` requires `tpu-inference==X`, which pins jax, jaxlib and libtpu
exactly. Installing jax separately on top is what silently swapped libtpu out
from under vLLM and produced C++ ABI crashes, so the check here is not "is jax
importable" but "is every member of the chain the version this vllm-tpu asked
for, and is there exactly one vllm".

Three call sites, three depths:
  * image build   -- versions only; a build machine has no TPU.
  * post-build    -- `--devices N`, which initialises the TPU. This is the only
                     check that can catch an ABI mismatch, because the symbols
                     are resolved when libtpu loads, not when jax imports.
  * run wrapper   -- versions again, to fail fast with a clear message when a
                     run is pointed at the wrong image.
"""

import argparse
import importlib.metadata as md
import re
import sys

# vllm-tpu -> {package: version}, read off the PyPI metadata of
# tpu-inference==<same version>. Extend when a version is added.
#
# flax is here on purpose: every tpu-inference pins flax==0.12.4 while this
# repo's pyproject asks for >=0.12.5, so the TPU stack deliberately wins that
# one package. Asserting it keeps that a recorded decision rather than a drift
# nobody notices.
EXPECTED = {
    "0.25.0": {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.42.1",
               "flax": "0.12.4"},
    "0.24.0": {"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.42.1",
               "flax": "0.12.4"},
    "0.23.0": {"jax": "0.10.1", "jaxlib": "0.10.1", "libtpu": "0.0.41",
               "flax": "0.12.4"},
    "0.22.0": {"jax": "0.10.0", "jaxlib": "0.10.0", "libtpu": "0.0.40",
               "flax": "0.12.4"},
    "0.21.0": {"jax": "0.9.2", "jaxlib": "0.9.2", "libtpu": "0.0.39",
               "flax": "0.12.4"},
}

REPORT = ("vllm-tpu", "tpu-inference", "jax", "jaxlib", "libtpu", "torch",
          "flax", "google-tunix", "qwix")

# tunix asks for this; the TPU stack drags in an older one, so the image
# reinstates it afterwards the way scripts/install_tunix_vllm_requirement.sh:39
# does. Checked as a floor, not an exact pin.
QWIX_MIN = (0, 1, 6)


def canon(name):
  """PEP 503 normalisation: tpu_inference and tpu-inference are one package."""
  return re.sub(r"[-_.]+", "-", name or "").lower()


def installed():
  out = {}
  for d in md.distributions():
    out.setdefault(canon(d.metadata["Name"]), d.version)
  return out


def as_tuple(version):
  return tuple(int(p) for p in re.findall(r"\d+", version or "")[:3])


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--vllm-tpu", required=True,
                 help="The vllm-tpu version this image was built for.")
  p.add_argument("--devices", type=int, default=0,
                 help="If >0, initialise the TPU and require this many chips."
                      " Only meaningful where TPU is attached.")
  args = p.parse_args()

  dist = installed()
  print("installed stack:")
  for k in REPORT:
    print(f"  {k:<15} {dist.get(k)}")

  problems = []

  # A wrong image is a normal outcome to report, not a crash to raise: the run
  # wrapper calls this to fail fast with something a human can act on.
  try:
    import vllm  # pylint: disable=g-import-not-at-top
    print(f"  {'vllm module':<15} {vllm.__version__}  @ {vllm.__file__}")
  except ImportError as exc:
    problems.append(f"vllm is not importable ({exc})")

  # `vllm` and `vllm-tpu` are different distributions that both provide the
  # `vllm` import name; two of them means files overwrote each other.
  owners = sorted(n for n in ("vllm", "vllm-tpu") if n in dist)
  if owners != ["vllm-tpu"]:
    problems.append(
        f"expected only vllm-tpu to provide the vllm package, found {owners}")

  if dist.get("vllm-tpu") != args.vllm_tpu:
    problems.append(f"vllm-tpu {dist.get('vllm-tpu')} != {args.vllm_tpu}")
  if dist.get("tpu-inference") != args.vllm_tpu:
    problems.append(
        f"tpu-inference {dist.get('tpu-inference')} != {args.vllm_tpu}"
        " (vllm-tpu pins it exactly)")

  want = EXPECTED.get(args.vllm_tpu)
  if want is None:
    print(f"  NOTE: {args.vllm_tpu} not in EXPECTED; only structural checks ran")
  else:
    for pkg, want_v in want.items():
      if dist.get(pkg) != want_v:
        problems.append(f"{pkg} {dist.get(pkg)} != {want_v}")

  if as_tuple(dist.get("qwix")) < QWIX_MIN:
    problems.append(
        f"qwix {dist.get('qwix')} < {'.'.join(map(str, QWIX_MIN))}"
        " (the TPU stack downgrades it; reinstate with --no-deps)")

  if args.devices:
    # Loading libtpu is what resolves the native symbols, so an ABI mismatch
    # surfaces here and nowhere earlier.
    import jax  # pylint: disable=g-import-not-at-top
    devices = jax.devices()
    print(f"  {'jax.devices()':<15} {devices}")
    tpus = [d for d in devices if d.platform == "tpu"]
    if len(tpus) != args.devices:
      problems.append(
          f"expected {args.devices} TPU devices, got {len(tpus)}: {devices}")

  if problems:
    print("\nSTACK CHECK FAILED:")
    for pr in problems:
      print(f"  - {pr}")
    sys.exit(1)
  print(f"\nstack OK (vllm-tpu {args.vllm_tpu}"
        + (f", {args.devices} TPU chips)" if args.devices else ")"))


if __name__ == "__main__":
  main()
