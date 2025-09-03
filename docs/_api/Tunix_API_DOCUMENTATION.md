# Tunix API Reference

Tunix is a comprehensive, JAX-based library for fine-tuning and serving Large Language Models (LLMs). Built upon Flax NNX, its architecture is designed for modularity, performance, and scalability, supporting a wide range of training paradigms from Supervised Fine-Tuning (SFT) to advanced Reinforcement Learning from Human Feedback (RLHF). This document provides a complete technical overview and API reference for the library.

## Contents

- [Technical Architecture Overview](#technical-architecture-overview)
- [API Reference](#api-reference)
  - [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
    - [PEFT Trainer](#peft-trainer)
    - [Training Hooks](#training-hooks)
  - [Distillation](#distillation)
    - [Distillation Trainer](#distillation-trainer)
    - [Base Distillation Strategy](#base-distillation-strategy)
    - [Logit Distillation Strategy](#logit-distillation-strategy)
    - [Feature Pooling Strategy](#feature-pooling-strategy)
    - [Feature Projection Strategy](#feature-projection-strategy)
    - [Attention Distillation Strategies](#attention-distillation-strategies)
  - [RL (Reinforcement Learning)](#rl-reinforcement-learning)
    - [DPO Trainer](#dpo-trainer)
    - [GRPO Learner](#grpo-learner)
    - [RL Cluster](#rl-cluster)
  - [Models](#models)
    - [Gemma](#gemma)
    - [Gemma3](#gemma3)
    - [Llama3](#llama3)
    - [Qwen3](#qwen3)
  - [Generation](#generation)
    - [JAX Sampler](#jax-sampler)
    - [VLLM Sampler](#vllm-sampler)

## Technical Architecture Overview

As a Principal Software Architect, here is a high-level technical overview of the Tunix library's software architecture, based on the provided source code.

### Tunix: High-Level Technical Overview

Tunix is a comprehensive, JAX-based library for fine-tuning and serving Large Language Models (LLMs). Built upon Flax NNX, its architecture is designed for modularity, performance, and scalability, supporting a wide range of training paradigms from Supervised Fine-Tuning (SFT) to advanced Reinforcement Learning from Human Feedback (RLHF).

#### Core Architectural Principles

The Tunix architecture is guided by several key principles:

1.  **Modularity and Decoupling**: Components are designed to be independent and interchangeable. This is evident in the separation of training logic from model implementations, the use of strategy patterns for loss functions (e.g., `DistillationTrainer`), and pluggable backends for inference (`VanillaSampler` vs. `VllmSampler`). This design allows researchers and engineers to easily customize or replace parts of the system without affecting others.

2.  **Performance and Scalability**: The library is built for high-performance, distributed environments.
    *   **JAX-native**: Core operations are JIT-compiled for maximum performance on accelerators.
    *   **Distributed Training**: Integrated support for data and model parallelism (FSDP, Tensor Parallelism) is a first-class citizen, managed through explicit `ShardingConfig` objects in models and `jax.sharding.Mesh` configurations in the training cluster.
    *   **Efficient Inference**: The architecture includes optimized KV caching for autoregressive decoding and offers an integration with the high-throughput vLLM engine.

3.  **Extensibility**: The use of abstract base classes (e.g., `BaseSampler`) and inheritance (e.g., `DpoTrainer` extending `PeftTrainer`) provides clear extension points for adding new models, samplers, or training algorithms.

4.  **Separation of Concerns**: The architecture clearly delineates different stages of the MLOps lifecycle:
    *   **Model Definition (`tunix.models`)**: Pure architectural definitions of LLMs.
    *   **Checkpoint Handling (`tunix.models.*.params`)**: Logic for loading and converting weights from various external formats (SafeTensors, Orbax) is isolated from the model code.
    *   **Generation (`tunix.generate`)**: A dedicated module for inference with multiple backend options.
    *   **Training (`tunix.sft`, `tunix.rl`, etc.)**: A flexible framework for implementing diverse training algorithms.

#### Key Architectural Components

The library is organized into several major components that work in concert.

##### 1. Model Implementations (`tunix.models`)

This module contains the core definitions for various open-source LLMs (Gemma, Gemma3, Llama3, Qwen3).

*   **Standardized Structure**: Models are implemented using Flax NNX and follow a consistent Transformer architecture (`Embedder`, `Attention`, `MLP`, `RMSNorm`, composed into `Block`s).
*   **Sharding Configuration**: Each model includes a `ShardingConfig` dataclass, which explicitly defines how model parameters and activations are to be distributed across a device `Mesh`. This is fundamental to the library's scalability.
*   **Decoupled Weight Loading**: Each model has a corresponding `params.py` file responsible for loading pretrained checkpoints. This module handles the complex logic of mapping and transforming weights from external formats (e.g., Hugging Face SafeTensors) to the internal NNX state structure, keeping the model definitions clean.
*   **Advanced Features**: The implementations support modern LLM features like Grouped-Query Attention (GQA), Mixture-of-Experts (MoE in Qwen3), and advanced Rotational Position Embeddings (RoPE in Gemma3).

##### 2. Generation and Inference (`tunix.generate`)

This component provides the functionality for autoregressive text generation.

*   **`BaseSampler` Interface**: Defines a common contract for all samplers, ensuring they can be used interchangeably.
*   **Pluggable Backends**:
    *   **`Sampler`**: A "vanilla" JAX-native implementation. It is highly optimized with JIT-compiled prefill and decode steps, making it ideal for research and customized generation logic. It supports greedy, top-p, and beam search sampling.
    *   **`VllmSampler`**: A powerful alternative that integrates with the vLLM inference engine. This provides a production-ready, high-throughput serving option. It includes logic for mapping and synchronizing Tunix-trained weights to the vLLM engine.
*   **`TokenizerAdapter`**: A classic adapter pattern that provides a unified interface over different tokenizer libraries (SentencePiece, Hugging Face Transformers), decoupling the core code from specific tokenizer dependencies.

##### 3. Training Framework

Tunix provides a layered and highly extensible training framework that supports multiple paradigms.

*   **`PeftTrainer` (`tunix.sft`)**: This is the foundational training class. It manages the core training loop, evaluation, checkpointing (`CheckpointManager`), metrics logging, and profiling. It is designed for Parameter-Efficient Fine-Tuning (PEFT), with built-in support for updating only LoRA parameters.

*   **Specialized Trainers**: These classes inherit from `PeftTrainer` to implement specific algorithms:
    *   **`DistillationTrainer`**: Implements knowledge distillation. It introduces a `teacher_model` and uses a `Strategy` pattern to encapsulate the logic for calculating the distillation loss, making the trainer independent of the specific distillation technique used.
    *   **`DpoTrainer`**: Implements Direct Preference Optimization. It encapsulates the DPO-specific loss function and data preparation logic, taking a policy model and a reference model.

*   **Reinforcement Learning Framework (`tunix.rl`)**: This is the most sophisticated part of the architecture, designed for complex RLHF workflows.
    *   **`RLCluster`**: A central orchestrator that manages all components of an RLHF system. It defines distinct `Role`s (Actor, Critic, Reference, Rollout, Reward) and allows each role to be mapped to a different hardware `Mesh`. This disaggregated design is critical for scaling RLHF, as it allows, for example, a fleet of sampler devices to be used for rollout while a separate, more powerful pod is used for training.
    *   **`GrpoLearner`**: Implements the Group Relative Policy Optimization (GRPO) algorithm. It orchestrates the entire RL loop: it uses the `RLCluster` to generate responses, computes rewards with user-provided functions, calculates advantages, and then updates the actor policy using the underlying `PeftTrainer`. It features an asynchronous data pipeline to hide the latency of generation behind trainer computation.

#### Data Flow

Data flows through the system via structured dataclasses (e.g., `TrainingInput`, `TrainExample`). A typical flow is:
1.  Raw data (text, preference pairs) is loaded.
2.  A data processing pipeline (e.g., `tunix.models.gemma.data` using `grain`, or `process_dpo_record`) tokenizes and formats the data into a `TrainingInput` object.
3.  The `Trainer`'s `_prepare_inputs` method further processes this input for the specific training algorithm (e.g., computing reference model log-probabilities in DPO).
4.  The sharded data is fed into the JIT-compiled training step, which computes the loss and updates the model weights.

#### Conclusion

The Tunix library exhibits a mature and robust software architecture. By emphasizing modularity, performance, and a clear separation of concerns, it provides a powerful and flexible platform for both research and production-scale LLM fine-tuning. Its standout features are the scalable, disaggregated `RLCluster` for advanced RLHF and the pluggable inference architecture that supports both a native JAX sampler and the production-grade vLLM engine.

## API Reference

### SFT (Supervised Fine-Tuning)

#### PEFT Trainer
| Class/Function | Description |
|---|---|
| `PeftTrainer(...)` | PEFT trainer for LoRA. Only LoRA parameters are updated. |
| `TrainingConfig()` | Configuration for the trainer. |
| `TrainingInput()` | Dataclass representing the input for a single training step. |
| `is_lora_enabled(model)` | Checks if the model has LoRA layers enabled. |
| `time_measure([context])` | A context manager for measuring the execution time of a code block. |

---

##### `PeftTrainer`
*class* `tunix.sft.peft_trainer.PeftTrainer`(*model: nnx.Module*, *optimizer: optax.GradientTransformation*, *training_config: TrainingConfig*)

PEFT trainer for LoRA. Only LoRA parameters are updated.

**Attributes:**
- **model**: The model to train.
- **config**: The training config.
- **optimizer**: The optimizer to use. To monitor the learning rate at each step, use `optax.inject_hyperparams` to inject learning rate as a hyperparameter. For example: `optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)`
- **loss_fn**: The loss function to use.
- **eval_loss_fn**: The loss function to use for evaluation.
- **gen_model_input_fn**: The function to generate model input from training input.
- **checkpoint_manager**: The checkpoint manager to use.
- **metrics_logger**: The metrics logger to use.
- **is_managed_externally**: Whether the trainer is managed externally.
- **training_hooks**: The training hooks to use.
- **data_hooks**: The data hooks to use.

**Methods:**

`with_training_hooks(training_hooks: hooks.TrainingHooks) -> PeftTrainer`
:   Sets the training hooks for the trainer.
  - **Parameters:**
    - `training_hooks`: An object implementing the `TrainingHooks` protocol.
  - **Returns:** The trainer instance with the hooks configured.

`with_data_hooks(data_hooks: hooks.DataHooks) -> PeftTrainer`
:   Sets the data hooks for the trainer.
  - **Parameters:**
      - `data_hooks`: An object implementing the `DataHooks` protocol.
  - **Returns:** The trainer instance with the hooks configured.

`clear_jit_cache()`
:   Clears the JIT cache of the train and eval step functions. This function should be called when the trainer is being reused after overiding the training related states, for example, the loss function.

`with_loss_fn(loss_fn: Callable[...], has_aux: bool = False) -> PeftTrainer`
:   Sets the loss function for the trainer.
  - **Parameters:**
      - `loss_fn`: The loss function. It should take the model as the first argument, followed by the model inputs. It can optionally return auxiliary data.
      - `has_aux`: Whether the loss function returns auxiliary data as a second element in a tuple.
  - **Returns:** The trainer instance with the loss function configured.

`with_gen_model_input_fn(gen_model_input_fn: Callable[...]) -> PeftTrainer`
:   Generates model input from training input. NB: output of this function will be passed to the loss function, so the args should match what loss function expects.
  - **Parameters:**
      - `gen_model_input_fn`: A function that generates model input from training input.
  - **Returns:** The trainer instance with the input generation function configured.

`create_train_step_fn() -> Callable[...]`
:   Creates the train step function.
  - **Returns:** A callable that executes one training step, taking the model, optimizer state, and a batch of data as input, and returning the updated model, optimizer state, and metrics.

`create_eval_step_fn() -> Callable[...]`
:   Creates the eval step function.
  - **Returns:** A callable that executes one evaluation step, taking the model and a batch of data as input, and returning the evaluation metrics.

`jit_train_and_eval_step(skip_jit: bool = False) -> Tuple[Callable, Callable]`
:   Creates and returns the train and eval step functions. This function will return the cached ones if available.
  - **Parameters:**
      - `skip_jit`: If True, the train and eval step functions will not be JITed.
  - **Returns:** A tuple of train and eval step functions.

`train(train_ds: Iterable[Any], eval_ds: Iterable[Any] | None = None, skip_jit: bool = False)`
:   Training loop.
  - **Parameters:**
      - `train_ds`: An iterable providing training batches.
      - `eval_ds`: An optional iterable providing evaluation batches.
      - `skip_jit`: If True, the train and eval step functions will not be JITed.

`train_steps() -> int`
:   Returns the number of train steps taken.
  - **Returns:** The total number of training steps completed.

`close()`
:   Closes resources used by the trainer, such as the metrics logger.

---
##### `TrainingConfig`
*class* `tunix.sft.peft_trainer.TrainingConfig`

Configuration for the trainer. This dataclass holds all hyperparameters and settings for a training run, such as the number of epochs, batch size, logging frequency, and checkpointing options.

**Methods:**

`get_with_default(key: str, default: Any) -> Any`
:   Gets a configuration value by key, returning a default if not set.
  - **Parameters:**
      - `key`: The name of the configuration attribute.
      - `default`: The value to return if the attribute is not found.
  - **Returns:** The value of the configuration attribute or the default value.

---
##### `TrainingInput`
*class* `tunix.sft.peft_trainer.TrainingInput`

A dataclass representing the input for a single training step. This class serves as a container for the data that will be processed by the `gen_model_input_fn` and ultimately fed into the model and loss function.

---
##### `is_lora_enabled`
`tunix.sft.peft_trainer.is_lora_enabled`(model: nnx.Module) -> bool

Checks if the model has LoRA layers enabled. This utility function inspects the model's parameters to determine if any LoRA-specific parameters exist, which indicates that LoRA is active.
- **Parameters:**
  - `model`: The `nnx.Module` to inspect.
- **Returns:** `True` if LoRA parameters are found, `False` otherwise.

---
##### `time_measure`
`tunix.sft.peft_trainer.time_measure`(context: str = '') -> ContextManager

A context manager for measuring the execution time of a code block. It logs the elapsed time upon exiting the context, prefixed with the provided context string.
- **Parameters:**
  - `context`: An optional string to identify the timed code block in logs.
- **Returns:** A context manager that yields nothing.
- **Example:**
  ```python
  >>> with tunix.sft.peft_trainer.time_measure('data loading'):
  ...     # some long-running operation
  ...     time.sleep(1)
  # Logs: "[data loading] took 1.00s"
  ```

---
#### Training Hooks
| Class | Description |
|---|---|
| `TrainingHooks()` | Hooks to inject custom logic at various points in the training lifecycle. |
| `DataHooks()` | Hooks to wire in external data loader and processing logic. |

---

##### `TrainingHooks`
*class* `tunix.sft.hooks.TrainingHooks`

Hooks to be used for training. This class defines a set of callbacks that can be implemented to inject custom logic at various points in the training and evaluation lifecycle. It is a key component of the extensibility system for the `PeftTrainer`, allowing users to add functionality like custom logging, metric calculation, or other side effects without modifying the core training loop.

Users can create a subclass of `TrainingHooks` and override the methods corresponding to the events they want to handle. An instance of this subclass is then passed to the `PeftTrainer`.

**Methods:**

`on_train_start(train_ctx: 'PeftTrainer.PeftTrainer')`
:   Called at the beginning of training. This hook is executed once, after the trainer has been initialized but before the first training step begins.
  - **Parameters:** `train_ctx` – The training context object, providing access to the trainer’s state, model, and configuration.

`on_train_end(train_ctx: 'PeftTrainer.PeftTrainer')`
:   Called at the end of training. This hook is executed once after the training loop has completed.
  - **Parameters:** `train_ctx` – The training context object.

`on_train_step_start(train_ctx: 'PeftTrainer.PeftTrainer')`
:   Called at the beginning of a training step. This hook is executed before the forward and backward pass of each training step.
  - **Parameters:** `train_ctx` – The training context object.

`on_train_step_end(train_ctx: 'PeftTrainer.PeftTrainer', train_loss: float)`
:   Called at the end of a training step. This hook is executed after the model parameters have been updated. It receives the loss computed for the current step.
  - **Parameters:**
      - `train_ctx`: The training context object.
      - `train_loss`: The loss value for the completed training step.

`on_eval_step_start(train_ctx: 'PeftTrainer.PeftTrainer')`
:   Called at the beginning of an evaluation step. This hook is executed before a batch of evaluation data is processed.
  - **Parameters:** `train_ctx` – The training context object.

`on_eval_step_end(train_ctx: 'PeftTrainer.PeftTrainer', eval_loss: float)`
:   Called at the end of an evaluation step. This hook is executed after an evaluation batch has been processed.
  - **Parameters:**
      - `train_ctx`: The training context object.
      - `eval_loss`: The loss value for the completed evaluation step.

---
##### `DataHooks`
*class* `tunix.sft.hooks.DataHooks`

Hooks to wire in external data loader and processing logic. This class provides an interface for abstracting the data loading process from the `PeftTrainer`. By implementing these hooks, users can integrate their own custom data pipelines, iterators, or pre-processing logic into the training loop.

**Methods:**

`load_next_train_batch(train_ctx: 'PeftTrainer.PeftTrainer') -> Any`
:   Loads the next batch of data for training. This method is called by the trainer at the beginning of each training step.
  - **Parameters:** `train_ctx` – The training context object.
  - **Returns:** A batch of training data.

`load_next_eval_batch(train_ctx: 'PeftTrainer.PeftTrainer') -> Any`
:   Loads the next batch of data for evaluation. This method is called by the trainer during the evaluation phase.
  - **Parameters:** `train_ctx` – The training context object.
  - **Returns:** A batch of evaluation data.

### Distillation

#### Distillation Trainer
| Class | Description |
|---|---|
| `DistillationTrainer(...)`| Orchestrates knowledge distillation from a teacher to a student model. |
| `TrainingConfig()` | Dataclass for configuring distillation training parameters. |
| `TrainingInput()` | Dataclass for providing inputs to the distillation process. |

---
##### `DistillationTrainer`
*class* `tunix.distillation.distillation_trainer.DistillationTrainer`(*student_model: nnx.Module*, *teacher_model: nnx.Module*, *strategy: strategies.BaseStrategy*, *optimizer: optax.GradientTransformation*, *training_config: TrainingConfig*)

Orchestrates the knowledge distillation process from a larger teacher model to a smaller student model. This trainer is a specialized subclass of `PeftTrainer` that manages the complete distillation workflow. It leverages a pluggable strategy pattern, where a `BaseStrategy` object defines the specific distillation method (e.g., logit matching, feature matching).

**Parameters:**
- `student_model`: The student model (`nnx.Module`) to be trained. This is typically a smaller, more efficient model.
- `teacher_model`: The pretrained teacher model (`nnx.Module`) that provides supervisory signals. Its weights are frozen during training.
- `strategy`: An instance of a `tunix.distillation.strategies.BaseStrategy` subclass. This object encapsulates the logic for computing the distillation loss.
- `optimizer`: An `optax.GradientTransformation` used to update the parameters of the student model.
- `training_config`: A `TrainingConfig` object containing hyperparameters and configuration for the training session.

**Methods:**

`with_gen_model_input_fn(gen_model_input_fn: Callable[[Any], dict[str, ArrayLike]]) -> DistillationTrainer`
:   Sets a custom function to generate model inputs from a data batch. This allows for flexible data preprocessing.
  - **Parameters:** `gen_model_input_fn` – A callable that takes a raw data batch and returns a dictionary of model inputs.
  - **Returns:** The trainer instance, allowing for method chaining.

`with_loss_fn(loss_fn: Callable[..., ArrayLike | Tuple[ArrayLike, Any]], has_aux: bool = False) -> DistillationTrainer`
:   Overrides the default loss computation with a custom function.
  - **Parameters:**
      - `loss_fn`: A callable that computes the loss. It will receive the student model, teacher output, and inputs as arguments.
      - `has_aux`: If `True`, `loss_fn` is expected to return a tuple `(loss, auxiliary_data)`.
  - **Returns:** The trainer instance, allowing for method chaining.

`get_train_loss(model: nnx.Module, teacher_output: Any, inputs: dict[str, ArrayLike]) -> ArrayLike | Tuple[ArrayLike, Any]`
:   Computes the distillation loss for a training step.
  - **Parameters:**
      - `model`: The student model with updated parameters for the current step.
      - `teacher_output`: The pre-computed output from the teacher model.
      - `inputs`: A dictionary of input tensors for the models.
  - **Returns:** The computed loss scalar, or `(loss, auxiliary_data)`.

`get_eval_loss(model: nnx.Module, teacher_output: Any, inputs: dict[str, ArrayLike]) -> ArrayLike | Tuple[ArrayLike, Any]`
:   Computes the distillation loss for an evaluation step.
  - **Parameters:**
      - `model`: The student model being evaluated.
      - `teacher_output`: The pre-computed output from the teacher model.
      - `inputs`: A dictionary of input tensors for the models.
  - **Returns:** The computed loss scalar, or `(loss, auxiliary_data)`.

`close() -> None`
:   Closes resources used by the trainer, such as loggers and profilers.

---
##### `TrainingConfig`
*class* `tunix.distillation.distillation_trainer.TrainingConfig`

Dataclass for configuring distillation training parameters. This class holds hyperparameters and settings for the training loop, such as the number of training steps, batch size, logging frequency, and checkpointing configuration.

---
##### `TrainingInput`
*class* `tunix.distillation.distillation_trainer.TrainingInput`

Dataclass for providing inputs to the distillation process. This class serves as a container for the data batches passed to the trainer's `train_step` or `eval_step` methods.

---
#### Base Distillation Strategy
| Class/Alias | Description |
|---|---|
| `BaseStrategy(...)` | Abstract Base Class for all distillation strategies. |
| `ModelForwardCallable` | A callable that executes a model's forward pass. |

---
##### `BaseStrategy`
*class* `tunix.distillation.strategies.base_strategy.BaseStrategy`(*student_forward_fn: ModelForwardCallable[Any]*, *teacher_forward_fn: ModelForwardCallable[Any]*, *labels_fn: Callable[..., jax.Array]*)

Abstract Base Class for all distillation strategies. Defines the common interface for computing the distillation loss. Concrete strategies must implement the `compute_loss` method.

**Parameters:**
- `student_forward_fn`: A callable that executes the student model's forward pass.
- `teacher_forward_fn`: A callable that executes the teacher model's forward pass.
- `labels_fn`: A callable that extracts labels from a batch of inputs.

**Methods:**

`compute_loss(student_output: Any, teacher_output: Any, labels: jax.Array)`
:   Computes the distillation loss based on model outputs and labels.

`compute_eval_loss(student_output: Any, labels: jax.Array)`
:   Computes the distillation loss based on model outputs and labels for evaluation.

`pre_process_models(student_model: nnx.Module, teacher_model: nnx.Module)`
:   Pre-processes the models to prepare for distillation. Can be used to modify models.

`post_process_models(student_model: nnx.Module, teacher_model: nnx.Module)`
:   Post-processes the models after distillation, usually reverting changes from `pre_process_models`.

`get_teacher_outputs(teacher_model: nnx.Module, inputs: dict[str, jax.Array])`
:   Computes the teacher model outputs.

`get_student_outputs(student_model: nnx.Module, inputs: dict[str, jax.Array])`
:   Computes the student model outputs.

`get_train_loss(student_model: nnx.Module, teacher_output: Any, inputs: dict[str, jax.Array])`
:   Computes the distillation loss for training.

`get_eval_loss(student_model: nnx.Module, inputs: dict[str, jax.Array])`
:   Computes the task loss based on student model forward pass and labels for evaluation.

---
##### `ModelForwardCallable`
`tunix.distillation.strategies.base_strategy.ModelForwardCallable` is an alias for `Callable[..., Any]`.

---
#### Logit Distillation Strategy
| Class | Description |
|---|---|
| `LogitStrategy(...)` | Implements Logit Distillation by combining a standard task loss with a KL divergence loss on the student and teacher logits. |

---
##### `LogitStrategy`
*class* `tunix.distillation.strategies.logit.LogitStrategy`(*student_forward_fn: ModelForwardCallable[jax.Array]*, *teacher_forward_fn: ModelForwardCallable[jax.Array]*, *labels_fn: Callable[..., jax.Array]*, *temperature: float = 2.0*, *alpha: float = 0.5*)

Implements Logit Distillation. This strategy minimizes the KL divergence between the student and teacher logits. The final loss is a weighted combination of this distillation loss and a standard task loss (e.g., softmax cross-entropy on the ground truth labels). The final loss is calculated as: `loss = alpha * task_loss + (1 - alpha) * distillation_loss`.

**Parameters:**
- `student_forward_fn`: A callable that executes the forward pass of the student model.
- `teacher_forward_fn`: A callable that executes the forward pass of the teacher model.
- `labels_fn`: A callable that extracts ground truth labels from a batch.
- `temperature`: The temperature for softening the probability distributions. Higher values create softer distributions. Defaults to 2.0.
- `alpha`: The weight for the task loss. The distillation loss is weighted by `1 - alpha`. Defaults to 0.5.

**Methods:**

`compute_eval_loss(student_output: jax.Array, labels: jax.Array)`
:   Computes the task loss for evaluation.

`compute_loss(student_output: jax.Array, teacher_output: jax.Array, labels: jax.Array)`
:   Computes the combined distillation and task loss.

---
#### Feature Pooling Strategy
| Class | Description |
|---|---|
| `FeaturePoolingStrategy(...)` | Implements Feature Pooling distillation. |

---
##### `FeaturePoolingStrategy`
*class* `tunix.distillation.strategies.feature_pooling.FeaturePoolingStrategy`(*student_forward_fn: ModelForwardCallable[jax.Array]*, *teacher_forward_fn: ModelForwardCallable[jax.Array]*, *labels_fn: Callable[..., jax.Array]*, *feature_layer: type[nnx.Module]*, *alpha: float = 0.75*, *feature_loss_fn: Callable[...] | None = None*, *cosine_distance_axis: int | tuple[int, ...] = -1*)

Implements Feature Pooling distillation. This strategy captures the feature maps and computes loss (typically cosine distance) between student and teacher feature maps from selected feature layers. It combines this feature loss with a standard task loss.

**Parameters:**
- `student_forward_fn`: The forward pass function for the student model.
- `teacher_forward_fn`: The forward pass function for the teacher model.
- `labels_fn`: A function to extract ground truth labels from the input batch.
- `feature_layer`: The class of the layer (e.g., `nnx.Linear`) from which to extract feature maps.
- `alpha`: The weighting factor. Final loss is `alpha * feature_loss + (1 - alpha) * task_loss`. Defaults to 0.75.
- `feature_loss_fn`: The loss function to apply to the feature maps. If `None`, defaults to cosine distance.
- `cosine_distance_axis`: The axis for cosine distance if `feature_loss_fn` is not provided. Defaults to -1.

**Methods:**

`pre_process_models(...)`, `post_process_models(...)`, `get_teacher_outputs(...)`, `get_student_outputs(...)`
:   Standard strategy methods for model processing and output retrieval.

`compute_eval_loss(student_output: dict[str, jax.Array], labels: jax.Array)`
:   Computes the task loss.

`compute_loss(student_output: dict[str, jax.Array], teacher_output: jax.Array, labels: jax.Array)`
:   Computes the combined feature pooling and task loss.

---
#### Feature Projection Strategy
| Class | Description |
|---|---|
| `FeatureProjectionStrategy(...)` | Implements Feature Projection distillation. |

---
##### `FeatureProjectionStrategy`
*class* `tunix.distillation.strategies.feature_projection.FeatureProjectionStrategy`(*student_forward_fn: ModelForwardCallable[jax.Array]*, *teacher_forward_fn: ModelForwardCallable[jax.Array]*, *labels_fn: Callable[..., jax.Array]*, *feature_layer: type[nnx.Module]*, *dummy_input: dict[str, jax.Array]*, *rngs: nnx.Rngs*, *alpha: float = 0.75*, *feature_loss_fn: Callable[...] | None = None*)

Implements Feature Projection distillation. This strategy computes loss (typically MSE) between projected student and teacher feature maps. It combines this feature loss with a standard task loss.

**Parameters:**
- `student_forward_fn`, `teacher_forward_fn`, `labels_fn`: Standard strategy callables.
- `feature_layer`: The class type of the layer from which to extract intermediate features.
- `dummy_input`: A sample input dictionary required to trace the model graph and instrument the feature layers.
- `rngs`: An `nnx.Rngs` object for stochastic operations.
- `alpha`: Weighting factor for the feature loss. Defaults to 0.75.
- `feature_loss_fn`: The loss function to compare feature maps. If `None`, defaults to Mean Squared Error.

**Methods:**

`pre_process_models(...)`
:   Instruments models to intercept feature layer outputs and adds a projection layer to the student.

`post_process_models(...)`
:   Removes instrumentation and auxiliary layers.

`get_teacher_outputs(...)`, `get_student_outputs(...)`
:   Retrieve feature maps (and logits for student) from the instrumented models.

`compute_eval_loss(...)`, `compute_loss(...)`
:   Compute evaluation and combined training loss respectively.

---
#### Attention Distillation Strategies
| Class | Description |
|---|---|
| `AttentionTransferStrategy(...)` | Implements Attention Transfer distillation by matching attention maps. |
| `AttentionProjectionStrategy(...)` | Implements Attention Projection distillation using learned projection layers. |

---
##### `AttentionTransferStrategy`
*class* `tunix.distillation.strategies.attention.AttentionTransferStrategy`(*student_forward_fn: ModelForwardCallable[jax.Array]*, *teacher_forward_fn: ModelForwardCallable[jax.Array]*, *labels_fn: Callable[..., jax.Array]*, *attention_layer: type[nnx.Module]*, *alpha: float = 0.75*, *attention_loss_fn: Callable[...] | None = None*)

Implements Attention Transfer distillation. This strategy minimizes the difference (typically Cosine Distance) between student and teacher attention maps.

**Parameters:**
- `student_forward_fn`, `teacher_forward_fn`, `labels_fn`: Standard strategy callables.
- `attention_layer`: The class type of the attention modules (e.g., `LlamaAttention`) to extract attention maps from.
- `alpha`: Interpolation weight. `alpha * attention_loss + (1 - alpha) * task_loss`. Defaults to 0.75.
- `attention_loss_fn`: Loss function for attention maps. If `None`, defaults to Cosine Distance.

---
##### `AttentionProjectionStrategy`
*class* `tunix.distillation.strategies.attention.AttentionProjectionStrategy`(*student_forward_fn: ModelForwardCallable[jax.Array]*, *teacher_forward_fn: ModelForwardCallable[jax.Array]*, *labels_fn: Callable[..., jax.Array]*, *attention_layer: type[nnx.Module]*, *dummy_input: dict[str, jax.Array]*, *rngs: nnx.Rngs*, *alpha: float = 0.75*, *attention_loss_fn: Callable[...] | None = None*)

Implements Attention Projection distillation. This strategy minimizes the difference (typically MSE) between projected student and teacher attention maps.

**Parameters:**
- `student_forward_fn`, `teacher_forward_fn`, `labels_fn`: Standard strategy callables.
- `attention_layer`: The class type of the attention modules to extract attention maps from.
- `dummy_input`: A sample input batch to initialize learnable projection layers.
- `rngs`: JAX NNX random number generators for initializing projection layers.
- `alpha`: Interpolation weight. Defaults to 0.75.
- `attention_loss_fn`: Loss function for projected attention maps. If `None`, defaults to Mean Squared Error (MSE).

### RL (Reinforcement Learning)

#### DPO Trainer
| Class/Function | Description |
|---|---|
| `DpoTrainer(...)` | The central trainer for Direct Preference Optimization (DPO). |
| `DpoTrainingConfig(...)` | Dataclass for DPO-specific training configurations. |
| `TrainingInput(...)` | Dataclass holding tokenized prompt, chosen, and rejected responses. |
| `TrainExample(...)` | Dataclass representing a single, processed example for the DPO loss function. |
| `dpo_loss_fn(...)` | Computes the Direct Preference Optimization (DPO) loss. |
| `compute_logps(...)` | Computes the log probabilities for tokens in a sequence. |
| `process_dpo_record(...)` | Processes and tokenizes a raw data record for DPO training. |

---
##### `DpoTrainer`
*class* `tunix.rl.dpo.dpo_trainer.DpoTrainer`(*model: nnx.Module*, *ref_model: nnx.Module*, *optimizer: optax.GradientTransformation*, *training_config: DpoTrainingConfig*)

The central trainer for Direct Preference Optimization (DPO). This class extends `PeftTrainer` to implement the DPO algorithm. It orchestrates the training loop, managing a policy model (`model`) and a fixed reference model (`ref_model`).

**Parameters:**
- `model`: The policy model to be trained (typically a `LoRA` model).
- `ref_model`: The fixed reference model. Its parameters are not updated.
- `optimizer`: The Optax `GradientTransformation` for updating the model.
- `training_config`: A `DpoTrainingConfig` object.

---
##### `DpoTrainingConfig`
*class* `tunix.rl.dpo.dpo_trainer.DpoTrainingConfig`(*beta: float = 0.1*, *label_smoothing: float = 0.0*, *max_seq_length: int = 2048*)

Dataclass for DPO-specific training configurations.

**Parameters:**
- `beta`: The temperature parameter that controls the strength of the preference penalty. Defaults to 0.1.
- `label_smoothing`: The amount of label smoothing to apply. Defaults to 0.0.
- `max_seq_length`: The maximum sequence length for tokenized inputs. Defaults to 2048.

---
##### `dpo_loss_fn`
`tunix.rl.dpo.dpo_trainer.dpo_loss_fn`(*model: nnx.Module*, *train_example: TrainExample*, *beta: float*, *label_smoothing: float*)

Computes the Direct Preference Optimization loss. It takes a batch of paired chosen and rejected examples, calculates their log probabilities under both the policy and reference models, and then computes the final loss.

**Parameters:**
- `model`: The policy model (`nnx.Module`) being trained.
- `train_example`: A `TrainExample` object containing the batched and processed data.
- `beta`: The temperature parameter for the DPO loss.
- `label_smoothing`: The amount of label smoothing to apply.
- **Returns:** The computed DPO loss for the batch.

---
##### `process_dpo_record`
`tunix.rl.dpo.dpo_trainer.process_dpo_record`(*record: dict[str, Any]*, *tokenizer: Any*, *max_seq_length: int*)

Processes and tokenizes a single record for DPO training. It takes a dictionary with 'prompt', 'chosen', and 'rejected' keys and tokenizes them.

**Parameters:**
- `record`: A dictionary containing the training data.
- `tokenizer`: The tokenizer to use.
- `max_seq_length`: The maximum length for the tokenized sequences.
- **Returns:** A `TrainingInput` object containing the tokenized data.

---
#### GRPO Learner
| Class/Function | Description |
|---|---|
| `GrpoConfig()` | Configuration for the GRPO algorithm. |
| `GrpoLearner(...)` | GRPO (Group Relative Policy Optimization) learner. |
| `grpo_loss_fn(...)` | GRPO loss function. |
| `RepeatIterable(...)` | A simple wrapper on top of one example to repeat it N times. |
| `TrainExample()` | A dataclass holding a single training example for the GRPO learner. |

---
##### `GrpoLearner`
*class* `tunix.rl.grpo.grpo_learner.GrpoLearner`(*rl_cluster: rl_cluster_lib.RLCluster*, *reward_fns: RewardFn | List[RewardFn]*, *grpo_config: GrpoConfig*)

GRPO (Group Relative Policy Optimization) learner. GRPO is a variant of PPO that reduces memory usage by eliminating the need for a separate value function model. It works by generating multiple responses, evaluating them with a reward model, and calculating a relative advantage to update the policy.

**Methods:**

`prepare_dataset(...)`
:   Prepares the dataset for training by generating rollouts, computing rewards, and queueing examples.

`train(train_ds: Iterable[...], eval_ds: Iterable[...] | None = None, skip_jit: bool = False)`
:   The main GRPO training loop, which alternates between sampling from the policy and updating it based on the GRPO objective.

---
##### `GrpoConfig`
*class* `tunix.rl.grpo.grpo_learner.GrpoConfig`

Configuration for the GRPO algorithm.

**Attributes:**
- `num_generations`: The number of responses to generate per prompt ('G' in the paper).
- `num_iterations`: The number of update iterations per batch (μ).
- `beta`: The KL divergence penalty coefficient (𝛽).
- `epsilon`: The clipping value for the policy update (𝜀).
- `loss_algo`: The loss algorithm to use (`grpo` or `gspo-token`).

---
##### `grpo_loss_fn`
`tunix.rl.grpo.grpo_learner.grpo_loss_fn`(*model*, *train_example*, *beta*, *epsilon*, *loss_algo*)

The GRPO loss function. It aims to maximize the expected advantage of actions while constraining the policy update to stay close to the reference policy.

**Parameters:**
- `model`: The policy model to be trained.
- `train_example`: A `TrainExample` instance with processed input data.
- `beta`: The KL divergence penalty coefficient.
- `epsilon`: The clipping value.
- `loss_algo`: The loss algorithm to use.
- **Returns:** A tuple containing the loss and an auxiliary dictionary.

---
#### RL Cluster
| Class | Description |
|---|---|
| `Role` | Defines the computational role of a model in the cluster. |
| `RLTrainingConfig` | Configuration for the RL training process, including optimizers. |
| `ClusterConfig` | Defines the hardware and software configuration for the entire RL cluster. |
| `RLCluster(...)` | Manages a distributed fleet of models for complex RL algorithms. |

---
##### `RLCluster`
*class* `tunix.rl.rl_cluster.RLCluster`(*actor: ModelOrPath*, *critic: ModelOrPath | None*, *reference: ModelOrPath | None*, *reward: ModelOrPath | None*, *tokenizer: Any | None*, *cluster_config: ClusterConfig*)

A pivotal component that manages a distributed fleet of models, each with a specific `Role` (e.g., `ACTOR`, `REFERENCE`). It is configured via a `ClusterConfig` that maps these roles to hardware meshes, seamlessly supporting both simple co-located and complex, disaggregated multi-host/multi-pod setups.

**Parameters:**
- `actor`, `critic`, `reference`, `reward`: The models or paths to their weights.
- `tokenizer`: The tokenizer for all models in the cluster.
- `cluster_config`: A `ClusterConfig` object defining the cluster's architecture.

**Methods:**

`rollout()`
:   Executes the policy rollout phase to generate experience data.

`actor_trainer()`, `critic_trainer()`
:   Returns the underlying `PeftTrainer` instance for the actor/critic model.

`update_actor(...)`, `update_critic(...)`
:   Performs a training step to update the actor/critic model's weights.

`generate(prompts: list[str])`
:   Generates text completions from prompts using the actor model.

`get_ref_per_token_logps(...)`, `get_old_per_token_logps(...)`
:   Computes per-token log probabilities using the reference model or the pre-update actor model.

`sync_weights()`
:   Syncs weights between the training model and the inference sampler. This is crucial for ensuring the rollout sampler has the latest policy weights.

---
##### `ClusterConfig`
*class* `tunix.rl.rl_cluster.ClusterConfig`

Defines the hardware and software configuration for the entire RL cluster.

**Attributes:**
- `role_to_mesh`: Mapping from model `Role` to a JAX `Mesh`. This determines the hardware layout.
- `rollout_engine`: Rollout engine to use (e.g., `"vanilla"`, `"vllm"`).
- `offload_to_cpu`: Whether to offload models to CPU to save device memory.
- `training_config`: An `RLTrainingConfig` object.
- `rollout_config`: Configuration for the rollout generation process.

### Models

#### Gemma
##### Gemma: Model
| Class/Function | Description |
|---|---|
| `Transformer` | The main Gemma transformer model implementation. |
| `TransformerConfig` | Configuration dataclass for the Gemma model. |
| `ShardingConfig` | Defines sharding for model parameters and activations. |
| `Block` | A single transformer block. |
| `Attention` | The multi-head/grouped-query attention module. |
| `FeedForward` | The feed-forward (MLP) layer. |
| `Embedder` | Handles input token embedding and output logit projection. |
| `RMSNorm` | Implements Root Mean Square Normalization. |

---
##### Gemma: Parameter Utilities
| Function | Description |
|---|---|
| `load_and_format_params(path)` | Loads pretrained Gemma weights and formats them for the Tunix NNX model. |
| `load_metadata(path)` | Loads model configuration metadata from a Gemma checkpoint. |

---
##### Gemma: Data Utilities
| Class/Function | Description |
|---|---|
| `GemmaTokenizer([model_path])` | Tokenizer for encoding/decoding text using Sentencepiece. |
| `create_datasets(...)` | Creates train and eval data iterators from a dataset. |

---
##### Gemma: Sampler
| Class/Function | Description |
|---|---|
| `Sampler(transformer, vocab, ...)` | JAX-native sampler for Gemma transformer. |
| `SamplerOutput()` | Output of the sampler. |
| `sample_best(logits)` | Samples the single best token (greedy decoding). |
| `sample_top_p(logits, key, ...)` | Samples from logits using temperature, top-p, and top-k filtering. |

---
#### Gemma3
##### Gemma3: Model
| Class/Function | Description |
|---|---|
| `Gemma3` | The main Gemma3 transformer model implementation. |
| `Gemma3Config` | Configuration dataclass for the Gemma3 model. |
| `ShardingConfig` | Defines sharding for model parameters and activations. |
| `Block` | A single transformer block. |
| `Attention` | The multi-head/grouped-query attention module. |
| `FeedForward` | The feed-forward (MLP) layer. |
| `Embedder` | Handles input token embedding and output logit projection. |
| `RMSNorm` | Implements Root Mean Square Normalization. |

---
##### Gemma3: Parameter Utilities
| Function | Description |
|---|---|
| `create_model_from_checkpoint(...)` | Loads a Gemma3 model from a checkpoint and initializes it with pretrained weights. |
| `create_tokenizer([path])` | Creates and returns a tokenizer for the Gemma3 model. |

---
#### Llama3
##### Llama3: Model
| Class/Function | Description |
|---|---|
| `Llama3` | The complete Llama3 transformer model implementation. |
| `ModelConfig` | Configuration for the Llama3 model. |
| `ShardingConfig` | Sharding configuration for Llama3 model. |
| `DecoderLayer` | A single transformer decoder layer. |
| `Attention` | The multi-head/grouped-query attention module. |
| `MLP` | The SwiGLU-based MLP module. |
| `Embedder` | The token embedding and decoding module. |
| `RMSNorm` | An RMSNorm normalization layer. |

---
##### Llama3: Parameter Utilities
| Function | Description |
|---|---|
| `create_model_from_safe_tensors(...)` | Loads pretrained Llama3 weights from safetensors and instantiates a Tunix NNX model. |

---
#### Qwen3
##### Qwen3: Model
| Class/Function | Description |
|---|---|
| `Qwen3` | The complete Qwen3 language model. |
| `ModelConfig` | Configuration for the Qwen3 model. |
| `ShardingConfig` | Sharding configuration for Qwen3 model. |
| `DecoderLayer` | A single transformer decoder layer. |
| `MoELayer` | Implements the Mixture-of-Experts (MoE) layer. |
| `Attention` | Implements the Grouped-Query Attention mechanism. |
| `MLP` | Implements the MLP (feed-forward) layer. |
| `Embedder` | Embedder module for token-to-vector conversion. |
| `RMSNorm` | Implements the Root Mean Square Normalization layer. |

---
##### Qwen3: Parameter Utilities
| Function | Description |
|---|---|
| `create_model_from_safe_tensors(...)` | Loads pretrained weights from safetensors and constructs a Qwen3 model. |

### Generation

#### JAX Sampler
| Class/Function | Description |
|---|---|
| `Sampler(transformer, tokenizer, ...)` | A JAX-native sampler for autoregressive text generation. |
| `CacheConfig()` | Configuration for the KV cache. |
| `sample_best(logits)` | Selects the token with the highest probability (greedy decoding). |
| `sample_top_p(logits, key, ...)` | Samples from the smallest set of tokens whose cumulative probability exceeds `top_p`. |

---
##### `Sampler`
*class* `tunix.generate.sampler.Sampler`(*transformer: nnx.Module*, *tokenizer: Any*, *cache_config: CacheConfig*)

A JAX-native sampler for transformer models. This class orchestrates autoregressive text generation. It manages the model's state, the KV cache, and the sampling loop. It is designed to be JIT-compiled for performance.

**Parameters:**
- `transformer`: An NNX module representing the language model.
- `tokenizer`: The tokenizer instance used for encoding and decoding text.
- `cache_config`: A `CacheConfig` object specifying the properties of the KV cache.

**Methods:**

`init_sample_state(...)`
:   Initializes the sampling state given input prompts, preparing all necessary components for a generation run.

`tokenize(input_string: str)`
:   Tokenizes the input string using the provided tokenizer.

---
##### `CacheConfig`
*class* `tunix.generate.sampler.CacheConfig`

Configuration for the KV cache. This dataclass holds all the static configuration required to initialize the KV cache for a given model architecture and generation task.

**Parameters:**
- `max_sequence_length`: The maximum number of tokens the cache can hold.
- `max_batch_size`: The maximum number of sequences to process in parallel.
- `dtype`: The JAX dtype for the cache tensors (e.g., `jnp.bfloat16`).
- `num_layers`: The number of transformer layers in the model.
- `num_heads`: The number of attention heads per layer.
- `head_dim`: The dimension of each attention head.

---
#### VLLM Sampler
| Class | Description |
|---|---|
| `VllmSampler(...)` | A sampler for vLLM-style autoregressive decoding using JAX and NNX models. |
| `MappingConfig()` | A configuration object that defines rules for mapping model parameters. |

---
##### `VllmSampler`
*class* `tunix.generate.vllm_sampler.VllmSampler`(*tokenizer: Any*, *mesh: jax.sharding.Mesh*, *max_model_len: int*, *model_version: str*, *mapping_config: MappingConfig*, *hbm_utilization: Optional[float] = 0.3*)

A sampler that wraps the high-performance vLLM inference engine. It provides the same interface as the native JAX sampler but delegates generation to vLLM's optimized backend.

**Parameters:**
- `tokenizer`: The tokenizer instance.
- `mesh`: The JAX device mesh on which the model weights are sharded.
- `max_model_len`: The maximum sequence length the model can handle.
- `model_version`: A string identifier for the model architecture (e.g., 'gemma-7b').
- `mapping_config`: A `MappingConfig` object specifying how to map weights to the vLLM format.
- `hbm_utilization`: The target fraction of HBM to use for the KV cache. Defaults to 0.3.

**Methods:**

`update_params(updated_weights: jaxtyping.PyTree)`
:   Dynamically updates the model parameters in the vLLM engine, useful for applying LoRA adapters without a full reload.

`load_checkpoint(path_or_weights: str | jaxtyping.PyTree)`
:   Loads a full set of model weights into the vLLM engine from a checkpoint path or a PyTree.

`tokenize(input_string: str)`
:   Tokenizes an input string.

`detokenize(input_strings: List[str], request_outputs: List[Any])`
:   Converts generated token sequences back into human-readable text.

---
##### `MappingConfig`
*class* `tunix.generate.vllm_sampler.MappingConfig`

A configuration object that defines rules for mapping model parameters between the Tunix NNX format and the format expected by the vLLM engine. This is essential for loading Tunix-trained checkpoints or applying fine-tuned updates to the vLLM backend.
