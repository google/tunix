 class AbstractTrainer(abc.ABC):
  """The pure ML algorithmic core. 
  
  Knows nothing about Orchestrators, RPCs, or networking.
  Operates entirely on local JAX/Flax data structures.
  """

  @abc.abstractmethod
  def init_state(self) -> Any:
    """Initializes the model weights, optimizer state, and mesh configurations."""
    pass

  @abc.abstractmethod
  def train_step(self, payload: TrainerPayload, *, loss_fn: Any, apply_gradients: bool, **kwargs) -> MetricsBuffer:
    """Executes a single mathematical gradient update. Updates internal state."""
    pass

  @abc.abstractmethod
  def eval_step(self, payload: TrainerPayload, *, loss_fn: Any, **kwargs) -> MetricsBuffer:
    """Executes a single evaluation step without modifying state."""
    pass

  @abc.abstractmethod
  def save_checkpoint(self, path: Optional[str] = None, **kwargs) -> str:
    """Serializes the current model and optimizer state to disk."""
    pass

  @abc.abstractmethod
  def restore_checkpoint(self, path: str, **kwargs) -> int:
    """Restores state from disk. Returns the restored global step."""
    pass

  @abc.abstractmethod
  def get_weights(self, **kwargs) -> Any:
    """Returns the current model weights (e.g., for Raiden weight syncing)."""
    pass
