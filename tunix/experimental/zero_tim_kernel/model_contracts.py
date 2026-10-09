# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-model projection contracts for the Zero-TIM kernels ("P22.XF").

Every geometry is an immutable ``ModelContract`` value in ``CONTRACTS`` and the
kernels that need a contract take it as an explicit argument.  The seven site
shapes are derived from the model dimensions and ``ModelContract.validate``
fails closed on any width that neither divides its tile nor is admitted by a
padding map.

A contract pins, per model and TP degree:

* the matmul tiles (BM, BN, BK).  BK is the per-element accumulation order of
  every projection, so it must be identical in decode, prefill and training;
* which TP-local K/N widths may be zero-padded to a tile multiple (and to
  what), for matmuls and for the SwiGLU feature width.  Zero padding appends
  exact zeros to the contraction and extra output columns that are sliced off,
  so the real columns keep their contraction order bit for bit;
* the seven projection sites of a Qwen decoder layer, their einsum equations,
  TP-local shapes, and whether they are column- or contract(row)-parallel.
"""

from __future__ import annotations

import dataclasses

# Enable flag for the P22.XF projection stack.
ENV = "CANON_PALLAS_ALL_PROJ"
VOCAB_SIZE = 151936
# Feature tile of the SwiGLU kernel (pallas_swiglu.BF).
SWIGLU_BF = 256

_Q_EQUATIONS = ("TD,DNH->TNH", "TD,NDH->TNH")
_KV_EQUATIONS = ("TD,DKH->TKH",)
_O_EQUATIONS = ("TNH,NHD->TD",)
_MLP_EQUATIONS = ("mn,np->mp",)


@dataclasses.dataclass(frozen=True)
class ProjectionSite:
  """One projection einsum of a decoder layer, as seen by one TP rank."""

  suffix: str
  family: str
  equations: tuple[str, ...]
  k_local: int
  n_local: int
  contract_parallel: bool


@dataclasses.dataclass(frozen=True)
class ModelContract:
  """The fixed-tile / padding contract of one model at one TP degree."""

  name: str
  block_m: int
  block_n: int
  block_k: int
  matmul_k_padding: dict[int, int]
  matmul_n_padding: dict[int, int]
  swiglu_feature_padding: dict[int, int]
  hidden_size: int
  intermediate_size: int
  num_attention_heads: int
  num_kv_heads: int
  head_dim: int
  tp_size: int

  @property
  def sites(self) -> tuple[ProjectionSite, ...]:
    """The seven TP-local projection sites (q/k/v/o/gate/up/down)."""
    q_width = self.num_attention_heads * self.head_dim // self.tp_size
    kv_width = self.num_kv_heads * self.head_dim // self.tp_size
    mlp_width = self.intermediate_size // self.tp_size
    hidden = self.hidden_size
    return (
        ProjectionSite(
            ".q_proj", "q_proj", _Q_EQUATIONS, hidden, q_width, False
        ),
        ProjectionSite(
            ".k_proj", "k_proj", _KV_EQUATIONS, hidden, kv_width, False
        ),
        ProjectionSite(
            ".v_proj", "v_proj", _KV_EQUATIONS, hidden, kv_width, False
        ),
        ProjectionSite(
            ".o_proj", "o_proj", _O_EQUATIONS, q_width, hidden, True
        ),
        ProjectionSite(
            ".gate_proj", "gate_proj", _MLP_EQUATIONS, hidden, mlp_width, False
        ),
        ProjectionSite(
            ".up_proj", "up_proj", _MLP_EQUATIONS, hidden, mlp_width, False
        ),
        ProjectionSite(
            ".down_proj", "down_proj", _MLP_EQUATIONS, mlp_width, hidden, True
        ),
    )

  @property
  def tiles(self) -> tuple[int, int, int]:
    return (self.block_m, self.block_n, self.block_k)

  @property
  def local_vocab(self) -> int:
    return VOCAB_SIZE // self.tp_size

  def model_env(self) -> dict[str, str]:
    """The ``CANON_QWEN3_*`` environment variables pinned for this model."""
    return {
        "CANON_QWEN3_HIDDEN_SIZE": str(self.hidden_size),
        "CANON_QWEN3_INTERMEDIATE_SIZE": str(self.intermediate_size),
        "CANON_QWEN3_NUM_ATTENTION_HEADS": str(self.num_attention_heads),
        "CANON_QWEN3_NUM_KV_HEADS": str(self.num_kv_heads),
        "CANON_QWEN3_HEAD_DIM": str(self.head_dim),
        "CANON_QWEN3_TP_SIZE": str(self.tp_size),
    }

  def match_site(self, prefix: str, equation: str) -> ProjectionSite | None:
    """Returns the site whose suffix ends ``prefix``; checks the equation."""
    matches = [site for site in self.sites if prefix.endswith(site.suffix)]
    if not matches:
      return None
    if len(matches) != 1:
      raise RuntimeError(f"ambiguous {self.name} P22.XK site for {prefix!r}")
    site = matches[0]
    if equation not in site.equations:
      raise RuntimeError(
          f"P22.XF equation mismatch at {prefix}: {equation!r} not in "
          f"{site.equations!r}"
      )
    return site

  def validate(self) -> None:
    """Checks the contract's internal consistency (fail closed)."""
    sites = self.sites
    if len({site.suffix for site in sites}) != 7:
      raise ValueError(f"{self.name}: projection suffixes must be unique")
    if sum(site.contract_parallel for site in sites) != 2:
      raise ValueError(
          f"{self.name}: expected exactly o/down contract-parallel"
      )
    for site in sites:
      _padded(
          site.k_local,
          self.block_k,
          self.matmul_k_padding,
          f"{self.name} {site.family} K",
      )
      _padded(
          site.n_local,
          self.block_n,
          self.matmul_n_padding,
          f"{self.name} {site.family} N",
      )
    _padded(
        self.local_vocab,
        self.block_n,
        self.matmul_n_padding,
        f"{self.name} lm_head N",
    )
    _padded(
        self.intermediate_size // self.tp_size,
        SWIGLU_BF,
        self.swiglu_feature_padding,
        f"{self.name} SwiGLU F",
    )
    for label, mapping in (
        ("K", self.matmul_k_padding),
        ("N", self.matmul_n_padding),
        ("SwiGLU", self.swiglu_feature_padding),
    ):
      for width, padded in mapping.items():
        if padded <= width:
          raise ValueError(
              f"{self.name}: {label} padding {width}->{padded} must grow"
          )


def _padded(width: int, tile: int, mapping: dict[int, int], label: str) -> int:
  if width % tile == 0:
    return width
  padded = mapping.get(width)
  if padded is None or padded <= width or padded % tile:
    raise ValueError(
        f"{label}={width} does not divide tile {tile} and is not admitted by "
        f"the padding contract {mapping!r}"
    )
  return padded


def _qwen(name, *, tiles, k_pad, n_pad, sw_pad, hidden, inter, heads, tp):
  block_m, block_n, block_k = tiles
  return ModelContract(
      name=name,
      block_m=block_m,
      block_n=block_n,
      block_k=block_k,
      matmul_k_padding=dict(k_pad),
      matmul_n_padding=dict(n_pad),
      swiglu_feature_padding=dict(sw_pad),
      hidden_size=hidden,
      intermediate_size=inter,
      num_attention_heads=heads,
      num_kv_heads=8,
      head_dim=128,
      tp_size=tp,
  )


_T256 = (128, 256, 256)
_T128 = (128, 128, 128)

CONTRACTS: dict[str, ModelContract] = {
    c.name: c
    for c in (
        _qwen(
            "qwen1p7b",
            tiles=_T256,
            k_pad={},
            n_pad={37984: 38144},
            sw_pad={},
            hidden=2048,
            inter=6144,
            heads=16,
            tp=4,
        ),
        _qwen(
            "qwen1p7b_tp1",
            tiles=_T256,
            k_pad={},
            n_pad={151936: 152064},
            sw_pad={},
            hidden=2048,
            inter=6144,
            heads=16,
            tp=1,
        ),
        _qwen(
            "qwen1p7b_tp2",
            tiles=_T256,
            k_pad={},
            n_pad={75968: 76032},
            sw_pad={},
            hidden=2048,
            inter=6144,
            heads=16,
            tp=2,
        ),
        _qwen(
            "qwen4b",
            tiles=_T128,
            k_pad={1216: 1280},
            n_pad={1216: 1280, 18992: 19200},
            sw_pad={1216: 1280},
            hidden=2560,
            inter=9728,
            heads=32,
            tp=8,
        ),
        _qwen(
            "qwen4b_tp4",
            tiles=_T128,
            k_pad={2432: 2560},
            n_pad={2432: 2560, 37984: 38144},
            sw_pad={2432: 2560},
            hidden=2560,
            inter=9728,
            heads=32,
            tp=4,
        ),
        _qwen(
            "qwen8b",
            tiles=_T256,
            k_pad={},
            n_pad={37984: 38144},
            sw_pad={},
            hidden=4096,
            inter=12288,
            heads=32,
            tp=4,
        ),
        _qwen(
            "qwen8b_tp1",
            tiles=_T256,
            k_pad={},
            n_pad={151936: 152064},
            sw_pad={},
            hidden=4096,
            inter=12288,
            heads=32,
            tp=1,
        ),
        _qwen(
            "qwen8b_tp2",
            tiles=_T256,
            k_pad={},
            n_pad={75968: 76032},
            sw_pad={},
            hidden=4096,
            inter=12288,
            heads=32,
            tp=2,
        ),
        _qwen(
            "qwen8b_tp8",
            tiles=_T128,
            k_pad={},
            n_pad={18992: 19200},
            sw_pad={},
            hidden=4096,
            inter=12288,
            heads=32,
            tp=8,
        ),
        _qwen(
            "qwen32b",
            tiles=_T128,
            k_pad={},
            n_pad={18992: 19200},
            sw_pad={3200: 3328},
            hidden=5120,
            inter=25600,
            heads=64,
            tp=8,
        ),
    )
}


def get_contract(name: str) -> ModelContract:
  """Returns the registered contract ``name`` (e.g. ``"qwen8b"``)."""
  try:
    return CONTRACTS[name]
  except KeyError:
    raise KeyError(
        f"unknown Zero-TIM model contract {name!r}; known: {sorted(CONTRACTS)}"
    ) from None


def find_contract(*, hidden_size: int, tp_size: int) -> ModelContract:
  """Returns the unique contract for a (hidden size, TP degree) geometry."""
  matches = [
      c
      for c in CONTRACTS.values()
      if c.hidden_size == hidden_size and c.tp_size == tp_size
  ]
  if len(matches) != 1:
    raise KeyError(
        f"no unique Zero-TIM contract for hidden={hidden_size} tp={tp_size}"
    )
  return matches[0]
