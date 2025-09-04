# Tunix: A JAX-native LLM Post-Training Library

**Tunix(Tune-in-JAX)** is a JAX based library designed to streamline the
 post-training of Large Language Models. It provides efficient and scalable
 supports for:

* **Supervised Fine-Tuning**
* **Reinforcement Learning (RL)**
* **Knowledge Distillation**

Tunix leverages the power of JAX for accelerated computation and seamless
integration with JAX-based modeling framework
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

**Current Status: Early Development**

Tunix is in early development. We're actively working to expand its
capabilities, usability and improve its performance. Stay tuned for upcoming
updates and new features!

## Key Features

Tunix is still under development, here's a glimpse of the current features:

* **Supervised Fine-Tuning:**
    * Full Weights Fine-Tuning
    * Parameter-Efficient Fine-Tuning (PEFT) with LoRA Layers
* **Reinforcement Learning (RL):**
    * Group Relative Policy Optimization (GRPO)
    * Direct Preference Optimization (DPO)
    * Proximal Policy Optimization (PPO)
    * token-level Group Sequence Policy Optimization (GSPO-token)
* **Knowledge Distillation:**
    * Logit Strategy: A classic approach where the student learns to match the teacher's output probability distribution.
    * Attention Transfer & Projection Strategies: Methods to align the attention mechanisms between the student and teacher models.
    * Feature Pooling & Projection Strategies: General techniques for matching intermediate feature representations, even between models of different architectures.
* **Modularity:**
    * Components are designed to be reusable and composable
    * Easy to customize and extend
* **Efficiency:**
    * Native support of common model sharding strategies such as DP, FSDP and TP
    * Designed for distributed training on accelerators (TPU)

## Upcoming

* **Advanced Algorithms:**
    * Addtional state-of-the-art RL and distillation algorithms
* **Scalability:**
    * Distributed training for large models
    * Efficient inference support for RL workflow
* **Accelerator:**
    * Efficient execution on GPU.
* **User Guides:**
    * Comprehensive onboarding materials and example notebooks

## Installation

Tunix doesn't have a PyPI package yet. To use Tunix, you need to install from
GitHub directly.

```sh
pip install git+https://github.com/google/tunix
```

## Getting Started

To get started, we have a bunch of detailed examples and tutorials.

- [PEFT Gemma with QLoRA](https://github.com/google/tunix/blob/main/examples/qlora_demo.ipynb)
- [Training Gemma on grade school Math problems using GRPO](https://github.com/google/tunix/blob/main/examples/grpo_demo.ipynb)
- [Logit Distillation using Gemma models](https://github.com/google/tunix/blob/main/examples/logit_distillation.ipynb)

To setup Jupyter notebook on single host GCP TPU VM, please refer to the [setup script](https://github.com/google/tunix/blob/main/scripts/setup_notebook_tpu_single_host.sh).

We plan to provide clear, concise documentation and more examples in the near
future.

## Contributing and Feedbacks

We welcome contributions! As Tunix is in early development, the contribution
process is still being formalized. In the meantime, you can make feature
requests, report issues and ask questions in our [Tunix GitHub discussion
forum](https://github.com/google/tunix/discussions).

## Stay Tuned!

Thank you for your interest in Tunix. We're working hard to bring you a powerful
and efficient library for LLM post-training. Please follow our progress and
check back for updates!
