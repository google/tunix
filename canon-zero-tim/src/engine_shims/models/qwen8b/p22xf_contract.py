#!/usr/bin/env python3
"""Qwen3-8B projection contract for the additive P22.XK stack.

This module intentionally shadows the 32B ``p22xf_contract`` only when the
FrozenLake Qwen3-8B runner prepends this directory to ``PYTHONPATH``.  The
admitted 32B module and artifacts remain untouched.
"""

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
BN = 256
BK = 256

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
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
    Site(".q_proj", "q_proj", ("TD,DNH->TNH", "TD,NDH->TNH"), 4096, 1024, False),
    Site(".k_proj", "k_proj", ("TD,DKH->TKH",), 4096, 256, False),
    Site(".v_proj", "v_proj", ("TD,DKH->TKH",), 4096, 256, False),
    Site(".o_proj", "o_proj", ("TNH,NHD->TD",), 1024, 4096, True),
    Site(".gate_proj", "gate_proj", ("mn,np->mp",), 4096, 3072, False),
    Site(".up_proj", "up_proj", ("mn,np->mp",), 4096, 3072, False),
    Site(".down_proj", "down_proj", ("mn,np->mp",), 3072, 4096, True),
)


def validate_qwen8b_env() -> None:
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, expected in _MODEL_ENV.items()
        if os.environ.get(name, "") != expected
    ]
    if wrong:
        raise RuntimeError(
            "Qwen3-8B P22.XK model contract mismatch: " + ", ".join(wrong)
        )


def validate_manifest(sites) -> None:
    if len(sites) != 7:
        raise ValueError(f"expected 7 projection sites, got {len(sites)}")
    suffixes = [site.suffix for site in sites]
    families = [site.family for site in sites]
    if len(set(suffixes)) != 7 or len(set(families)) != 7:
        raise ValueError("projection suffixes/families must be unique")
    if sum(site.contract_parallel for site in sites) != 2:
        raise ValueError("expected exactly o/down contract-parallel sites")
    expected_shapes = {
        "q_proj": (4096, 1024),
        "k_proj": (4096, 256),
        "v_proj": (4096, 256),
        "o_proj": (1024, 4096),
        "gate_proj": (4096, 3072),
        "up_proj": (4096, 3072),
        "down_proj": (3072, 4096),
    }
    for site in sites:
        if (site.k_local, site.n_local) != expected_shapes[site.family]:
            raise ValueError(
                f"{site.family} shape {(site.k_local, site.n_local)} does not "
                f"match Qwen3-8B TP4 {expected_shapes[site.family]}"
            )
        if site.k_local % BK or site.n_local % BN:
            raise ValueError(
                f"{site.family} local shape {(site.k_local, site.n_local)} "
                f"does not divide BK/BN={BK}/{BN}"
            )


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise RuntimeError(f"{ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise RuntimeError(f"{ENV}=1 required")
    if value == "1":
        validate_qwen8b_env()
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
        raise RuntimeError(f"ambiguous Qwen3-8B P22.XK site for {prefix!r}")
    site = matches[0]
    if equation not in site.equations:
        raise RuntimeError(
            f"P22.XF equation mismatch at {prefix}: {equation!r} not in "
            f"{site.equations!r}"
        )
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
        assert len(SITES) == 7
        assert match_site("model.layers.0.self_attn.q_proj", "TD,DNH->TNH").n_local == 1024
        assert match_site("model.layers.0.mlp.down_proj", "mn,np->mp").k_local == 3072
        os.environ["CANON_QWEN3_HIDDEN_SIZE"] = "5120"
        try:
            preflight(require_enabled=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong hidden width was accepted")
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if old_enabled is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old_enabled
        if old_fixed_ar is None:
            os.environ.pop("CANON_FIXED_AR", None)
        else:
            os.environ["CANON_FIXED_AR"] = old_fixed_ar
    print("P21_QWEN8B_P22XF_CONTRACT_SELFTEST_PASS cases=4/4")


if __name__ == "__main__":
    self_test()
