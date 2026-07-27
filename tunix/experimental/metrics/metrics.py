from typing import Any, Callable, Dict
import flax
import jax
import jax.typing


@flax.struct.dataclass
class WeightedMetric:
  """A metric that requires weighted reduction.

  Attributes:
    unreduced_sum: The sum of the metric values. Should be a scalar ArrayLike.
    denominator: The weight of the metric. Should be a scalar ArrayLike.
    eps: Optional epsilon added to denominator for numerical stability.
    min_denom: Optional minimum bound for the denominator.
  """

  unreduced_sum: jax.typing.ArrayLike
  denominator: jax.typing.ArrayLike
  eps: float | None = flax.struct.field(default=None, pytree_node=False)
  min_denom: float | None = flax.struct.field(default=None, pytree_node=False)

  def compute_scale(self) -> jax.typing.ArrayLike:
    """Computes the scale factor (1 / denominator) with optional bounds (eps, min_denom)."""
    raise NotImplementedError()

  def compute(self) -> jax.typing.ArrayLike:
    """Computes the reduced value of the metric (unreduced_sum * compute_scale())."""
    raise NotImplementedError()


@flax.struct.dataclass
class MetricsBuffer:
  """A buffer for storing and aggregating unreduced metrics.

  Keeps metrics unreduced (e.g., on-device to prevent sync-overhead),
  and serves as a payload container when metrics are requested on the host.
  """

  id: jax.typing.ArrayLike | int | str
  weighted_metrics: Dict[str, WeightedMetric] = flax.struct.field(
      default_factory=dict
  )
  scalar_metrics: Dict[str, jax.typing.ArrayLike] = flax.struct.field(
      default_factory=dict
  )
  # Host-side aggregation functions for specific metrics (not traced by JAX)
  aggregation_fns: Dict[str, Callable[[jax.typing.ArrayLike], Any]] = (
      flax.struct.field(default_factory=dict, pytree_node=False)
  )
  mode: str = flax.struct.field(default="train", pytree_node=False)
