#!/usr/bin/env python3
"""Qwen3-4B TP4 projection contract for the additive P22.XK stack."""

from __future__ import annotations

import os
from dataclasses import dataclass


ENV = "CANON_PALLAS_ALL_PROJ"
CONFLICTS = (
    "CANON_PALLAS_MATMUL",
    "CANON_PALLAS_MATERIALIZE",
    "CANON_POSTRPA_M",
    "CANON_CUT",
    "CANON_TAIL",
)
BM = 128
BN = 128
BK = 128
MATMUL_K_PADDING = {2432: 2560}
MATMUL_N_PADDING = {2432: 2560, 37984: 38144}
SWIGLU_FEATURE_PADDING = {2432: 2560}

HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 9728
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
TP_SIZE = 4

_MODEL_ENV = {
    "CANON_QWEN3_HIDDEN_SIZE": str(HIDDEN_SIZE),
    "CANON_QWEN3_INTERMEDIATE_SIZE": str(INTERMEDIATE_SIZE),
    "CANON_QWEN3_NUM_ATTENTION_HEADS": str(NUM_ATTENTION_HEADS),
    "CANON_QWEN3_NUM_KV_HEADS": str(NUM_KV_HEADS),
    "CANON_QWEN3_HEAD_DIM": str(HEAD_DIM),
    "CANON_QWEN3_TP_SIZE": str(TP_SIZE),
}


@dataclass(frozen=True)
class Site:
  suffix: str
  family: str
  equations: tuple[str, ...]
  k_local: int
  n_local: int
  contract_parallel: bool


SITES = (
    Site(".q_proj", "q_proj", ("TD,DNH->TNH", "TD,NDH->TNH"), 2560, 1024, False),
    Site(".k_proj", "k_proj", ("TD,DKH->TKH",), 2560, 256, False),
    Site(".v_proj", "v_proj", ("TD,DKH->TKH",), 2560, 256, False),
    Site(".o_proj", "o_proj", ("TNH,NHD->TD",), 1024, 2560, True),
    Site(".gate_proj", "gate_proj", ("mn,np->mp",), 2560, 2432, False),
    Site(".up_proj", "up_proj", ("mn,np->mp",), 2560, 2432, False),
    Site(".down_proj", "down_proj", ("mn,np->mp",), 2432, 2560, True),
)


def validate_model_env() -> None:
  wrong = [
      f"{name}={os.environ.get(name)!r}"
      for name, expected in _MODEL_ENV.items()
      if os.environ.get(name, "") != expected
  ]
  if wrong:
    raise RuntimeError(
        "Qwen3-4B TP4 model contract mismatch: " + ", ".join(wrong)
    )


# Historical compatibility name imported by older overlay call sites.
validate_qwen8b_env = validate_model_env


def validate_manifest(sites) -> None:
  expected = {
      "q_proj": (2560, 1024),
      "k_proj": (2560, 256),
      "v_proj": (2560, 256),
      "o_proj": (1024, 2560),
      "gate_proj": (2560, 2432),
      "up_proj": (2560, 2432),
      "down_proj": (2432, 2560),
  }
  if len(sites) != 7 or len({site.family for site in sites}) != 7:
    raise ValueError("expected seven unique Qwen3-4B TP4 projection sites")
  if sum(site.contract_parallel for site in sites) != 2:
    raise ValueError("expected exactly o/down contract-parallel sites")
  for site in sites:
    if (site.k_local, site.n_local) != expected[site.family]:
      raise ValueError(
          f"{site.family} shape mismatch: {(site.k_local, site.n_local)}"
      )
    k_padded = MATMUL_K_PADDING.get(site.k_local, site.k_local)
    n_padded = MATMUL_N_PADDING.get(site.n_local, site.n_local)
    if k_padded % BK or n_padded % BN:
      raise ValueError(
          f"{site.family} local shape {(site.k_local, site.n_local)} "
          f"does not admit padded BK/BN={BK}/{BN} geometry"
      )
  if MATMUL_K_PADDING != {2432: 2560}:
    raise ValueError("Qwen3-4B TP4 matmul K padding must be 2432->2560")
  if MATMUL_N_PADDING != {2432: 2560, 37984: 38144}:
    raise ValueError(
        "Qwen3-4B TP4 matmul N padding must cover MLP 2432->2560 "
        "and tied-head 37984->38144"
    )
  local_feature = INTERMEDIATE_SIZE // TP_SIZE
  if SWIGLU_FEATURE_PADDING != {local_feature: 2560}:
    raise ValueError(
        "Qwen3-4B TP4 SwiGLU padding must be exactly 2432->2560"
    )
  if local_feature % 256 == 0 or SWIGLU_FEATURE_PADDING[local_feature] % 256:
    raise ValueError("Qwen3-4B TP4 SwiGLU padding must resolve BF256 remainder")


def preflight(*, require_enabled: bool) -> None:
  value = os.environ.get(ENV, "")
  if value not in ("", "1"):
    raise RuntimeError(f"{ENV} must be unset or 1, got {value!r}")
  if require_enabled and value != "1":
    raise RuntimeError(f"{ENV}=1 required")
  if value == "1":
    validate_model_env()
    if os.environ.get("CANON_FIXED_AR", "") != "1":
      raise RuntimeError("CANON_FIXED_AR=1 required")
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
      raise RuntimeError("conflicting diagnostics: " + ",".join(active))
  validate_manifest(SITES)


def match_site(prefix: str, equation: str) -> Site | None:
  matches = [site for site in SITES if prefix.endswith(site.suffix)]
  if not matches:
    return None
  if len(matches) != 1:
    raise RuntimeError(f"ambiguous Qwen3-4B TP4 site for {prefix!r}")
  site = matches[0]
  if equation not in site.equations:
    raise RuntimeError(f"equation mismatch at {prefix}: {equation!r}")
  return site


def self_test() -> None:
  old = {name: os.environ.get(name) for name in _MODEL_ENV}
  old_enabled = os.environ.get(ENV)
  old_fixed_ar = os.environ.get("CANON_FIXED_AR")
  try:
    os.environ.update(_MODEL_ENV)
    os.environ[ENV] = "1"
    os.environ["CANON_FIXED_AR"] = "1"
    preflight(require_enabled=True)
    assert match_site(
        "model.layers.0.self_attn.q_proj", "TD,DNH->TNH"
    ).n_local == 1024
    assert match_site(
        "model.layers.0.mlp.down_proj", "mn,np->mp"
    ).k_local == 2432
    assert (BM, BN, BK) == (128, 128, 128)
    assert MATMUL_K_PADDING == {2432: 2560}
    assert MATMUL_N_PADDING == {2432: 2560, 37984: 38144}
    assert SWIGLU_FEATURE_PADDING == {2432: 2560}
    os.environ["CANON_QWEN3_TP_SIZE"] = "8"
    try:
      preflight(require_enabled=True)
    except RuntimeError:
      pass
    else:
      raise AssertionError("wrong TP width was accepted")
  finally:
    for name, previous in old.items():
      if previous is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = previous
    for name, previous in ((ENV, old_enabled), ("CANON_FIXED_AR", old_fixed_ar)):
      if previous is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = previous
  print("P58_QWEN4B_TP4_CONTRACT_SELFTEST_PASS cases=5/5")


if __name__ == "__main__":
  self_test()
