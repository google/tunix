# Tunix Programming Guide: Custom Logging Backends

Tunix provides a flexible, protocol-based logging system that allows you to integrate any logging service or library by creating a custom backend. This guide explains how to create a backend that conforms to the Metrax `LoggingBackend` protocol and how to use it with the Tunix `MetricsLogger`.

## 1. The Metrax `LoggingBackend` Protocol

Create a custom backend that **conforms** to the Metrax `LoggingBackend` protocol.

**Note:** You do not need to explicitly inherit from `LoggingBackend`. Because Metrax uses Python's structural typing (duck typing), your class just needs to implement the required methods described below.

* **`log_scalar(self, event: str, value: float | np.ndarray, **kwargs)`**: This method is called whenever a metric is logged. It receives the metric name (`event`), its value, and optional keyword arguments (like `step`).

* **`close(self)`**: This method is called when logging is finished, allowing the backend to flush any buffered data and release resources.

```python
from typing import Protocol
import numpy as np

class LoggingBackend(Protocol):
  """Defines the interface for a pluggable logging backend."""

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    """Logs a scalar value.

    Args:
      event: The name of the metric/event (e.g., "train/loss").
      value: The scalar value of the metric.
      **kwargs: Additional arguments, typically including 'step' (int).
    """
    ...

  def close(self):
    """Closes the logger and flushes any pending data."""
    ...
```

## 2. Creating a Custom Backend

To create a custom backend, define a class that implements these two methods.

### Example: Creating a CLU Backend

Let's create a backend for Google's [Common Loop Utils (CLU)](https://github.com/google/CommonLoopUtils) metric writers.

```python
from clu import metric_writers
import jax
import numpy as np

class CluBackend:
  """A custom backend for CLU metric writers."""

  def __init__(self):
    # Only initialize the writer on the main process
    if jax.process_index() == 0:
      self._writer = metric_writers.create_default_writer(logdir="custom_path")
    else:
      self._writer = None

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    # If we are not on the main process, do nothing.
    if self._writer is None:
      return

    # Extract the step, defaulting to 0 if it's not provided.
    step = int(kwargs.get("step", 0))

    # CLU's write_scalars takes a step and a dictionary of {name: value}.
    self._writer.write_scalars(step, {event: value})

  def close(self):
    if self._writer:
      self._writer.close()
```

#### Step 1: Define the Class and `__init__`

Initialize your specific logger. For CLU, this involves creating a `MetricWriter`. It's best practice to only initialize writers on the main process (process index 0) to avoid duplicate logging in multi-process environments.

#### Step 2: Implement `log_scalar`

Map the generic `log_scalar` call to your logger's specific API.

#### Step 3: Implement `close`

Ensure your logger is properly closed to flush data to disk.

## 3. Using Your Custom Backend

To ensure your `MetricsLoggerOptions` configuration is safe to copy (required for advanced workflows like RL), Tunix requires you to pass **factories** (callables that return a backend instance) rather than live instances.

### Case A: Simple Backend (No Arguments)

If your backend class requires no arguments in its `__init__`, you can simply pass the class itself.

```python
class SimplePrintBackend:
    def log_scalar(self, event, value, **kwargs):
        print(f"{event}: {value}")
    def close(self): pass

# 1. Pass the class directly. It acts as its own factory.
options = MetricsLoggerOptions(
    log_dir="/tmp/logs",
    backend_factories=[SimplePrintBackend]
)

# 2. Initialize logger. It will instantiate SimplePrintBackend() for you.
logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
logger.log("train/loss", 0.5, mode="train", step=1)
logger.close()
```

### Case B: Backend with Arguments (Using `lambda`)

If your backend requires arguments (like our `CluBackend` needing `logdir`), use a `lambda` function to create a simple factory.

```python
from tunix.sft import metrics_logger

# 1. Create a factory using a lambda.
#    This defers the creation of the backend until the logger is initialized.
my_clu_factory = lambda: CluBackend(logdir="/tmp/my_experiment_logs")

# 2. Create options and include your factory in the 'backends' list.
options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/default_logs",
    backend_factories=[my_clu_factory]
)

# 3. Initialize the logger. It will call your factory to create the live backend.
logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
logger.log("train/loss", 0.5, mode="train", step=1)
logger.close()
```
