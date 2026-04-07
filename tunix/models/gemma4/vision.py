"""Minimal vision.py for Gemma 4."""
import dataclasses

@dataclasses.dataclass(slots=True, frozen=True)
class SigLIPShardingConfig:
  @staticmethod
  def get_default_sharding(is_sampling=False):
    return SigLIPShardingConfig()

@dataclasses.dataclass(slots=True, kw_only=True)
class SigLIPConfig:
  pass

class SigLiP:
  pass
