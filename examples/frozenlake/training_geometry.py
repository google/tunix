"""Closed FrozenLake full-training geometry identities (host-only imports)."""

from dataclasses import dataclass
from typing import Mapping


SELECTOR = "CANON_P57_TRAIN_GEOMETRY"
LEGACY = "dp8-tp8-b256"
SMALL = "dp4-tp8-b128"
CHOICES = (LEGACY, SMALL)


@dataclass(frozen=True)
class TrainingGeometry:
  name: str
  dp: int
  prompts: int
  topology: str
  tp: int = 8
  generations: int = 8
  local_m: int = 256

  @property
  def devices(self) -> int:
    return self.dp * self.tp

  @property
  def trajectories(self) -> int:
    return self.prompts * self.generations

  @property
  def workload(self) -> str:
    return f"frozenlake-dp{self.dp}-tp8"

  @property
  def profile(self) -> str:
    return f"qwen3-8b-dp{self.dp}-tp8-frozenlake-v1-hp"

  @property
  def profile_file(self) -> str:
    return f"cluster/profiles/{self.profile}.env"

  @property
  def global_m(self) -> int:
    return self.dp * self.local_m

  def environment(self) -> dict[str, str]:
    return {
        "CANON_P32_WORKLOAD": self.workload,
        "CANON_PROFILE_FILE": self.profile_file,
        "CANON_PROFILE": self.profile,
        "CANON_DP_SIZE": str(self.dp),
        "CANON_TP_SIZE": str(self.tp),
        "CANON_TOTAL_DEVICES": str(self.devices),
        "CANON_ENGINE_DP_SIZE": str(self.dp),
        "CANON_GLOBAL_PROMPTS": str(self.prompts),
        "CANON_LOCAL_PROMPTS": "4",
        "CANON_NUM_GENERATIONS": "8",
        "CANON_LOCAL_TRAJECTORIES": "32",
        "CANON_GLOBAL_TRAJECTORIES": str(self.trajectories),
        "CANON_DP_PROBE_LOCAL_SAMPLES": "32",
        "CANON_LOGPROB_M": "256",
        "CANON_TARGET_M": "256",
        "CANON_MAX_BATCHED": "256",
        "MIN_TOKEN_BUCKET": str(self.global_m),
        "CANON_P33_SHARED_MESH": f"{self.dp},8",
        "FL_SHARED_MESH": f"{self.dp},8",
    }


def geometry(name: str = LEGACY) -> TrainingGeometry:
  if name == LEGACY:
    return TrainingGeometry(name, 8, 32, "4x4x4")
  if name == SMALL:
    return TrainingGeometry(name, 4, 16, "2x4x4")
  raise ValueError(f"unregistered FrozenLake train geometry: {name!r}")


def from_env(env: Mapping[str, str]) -> TrainingGeometry:
  """Absence preserves legacy; a present selector has exactly one new value.

  This selects expected values, not admission. Callers must still validate
  the full profile/workload/horizon/numerical contract against those values.
  """
  if SELECTOR in env:
    if env[SELECTOR] != SMALL:
      raise ValueError(f"{SELECTOR} must be absent or {SMALL}")
    return geometry(SMALL)
  return geometry()


def full_expected(expected: Mapping[str, str], env: Mapping[str, str]) -> dict[str, str]:
  """Adapt only fields actually checked by a full-training contract.

  Never use this helper for diagnostics/native/one-host identities. It does
  not rewrite the caller's environment or pretend DP4 is a DP8 workload.
  """
  selected = from_env(env)
  result = dict(expected)
  if selected.name == SMALL:
    for key, value in selected.environment().items():
      if key in result:
        result[key] = value
  return result


def validate_selected_full(env: Mapping[str, str]) -> TrainingGeometry:
  selected = from_env(env)
  if selected.name == SMALL:
    required = {
        **selected.environment(), "CANON_V1_HP_FULL": "1",
        "CANON_P57_RUN_KIND": "train", "CANON_P57_TIM_ARM": "zero",
        "CANON_P57_EXPECTED_UPDATES": "300", "CANON_P57_STOP_AFTER_STEP": "300",
        "CANON_P33_RUN_STAGE": "full", "CANON_P33_NO_COMMIT": "0",
        "CANON_P33_ENABLE_EVAL": "0", "CANON_P33_DISABLE_EVAL": "1",
        "CANON_P31_ENABLE_EVAL": "0", "CANON_FROZENLAKE_CKPT_MODE": "disabled",
        "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1", "CANON_P59_CHECKED_VMA": "1",
        "CANON_P67_P66_VMA_P59_ONLY": "1", "CANON_V1_HP_FIRST_UPDATE_GATE": "1",
    }
    wrong = [key for key, value in required.items() if env.get(key) != value]
    if (env.get("CANON_P57_WORKLOAD_CANDIDATE", ""),
        env.get("CANON_P57_DATA_SPLIT", "")) not in (("", ""), ("m15", "main")):
      wrong.append("candidate/split")
    if wrong:
      raise ValueError(f"DP4xTP8/B128 full identity drifted at {wrong}")
  elif env.get("CANON_P32_WORKLOAD") == geometry(SMALL).workload:
    raise ValueError("DP4xTP8/B128 workload requires its explicit selector")
  return selected
