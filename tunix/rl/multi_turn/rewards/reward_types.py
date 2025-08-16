# reward_types.py
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class RewardOutput:
    reward: float                         # Scalar score
    metadata: Dict[str, Any] = field(default_factory=dict)  # Debugging information
