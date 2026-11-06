# Examples and Guides

This section provides a high-level overview of the Colab notebooks, scripts, and
example directories.

All examples are located in this
[directory](https://github.com/google/tunix/tree/main/examples).

<table>
  <thead>
    <tr>
      <th align="center">Category</th>
      <th align="center">Name/Path</th>
      <th align="center">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" align="center" valign="middle" style="text-align: center; vertical-align: middle;"><b>Colab Notebook</b></td>
      <td><a href="https://github.com/google/tunix/tree/main/examples/qlora_gemma.ipynb"><code>qlora_gemma.ipynb</code></a></td>
      <td>End-to-end tutorial on fine-tuning (SFT) Gemma 270M model for English-French translation using parameter-efficient LoRA and QLoRA techniques.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/grpo_gemma.ipynb"><code>grpo_gemma.ipynb</code></a></td>
      <td>Reinforcement learning tutorial using Group Relative Policy Optimization (GRPO) to train the Gemma 3 1B IT model for math reasoning on the GSM8K benchmark.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/dpo_gemma.ipynb"><code>dpo_gemma.ipynb</code></a></td>
      <td>Preference tuning using Direct Preference Optimization (DPO) to tune the Gemma 3 1B-IT model on the GSM8K dataset.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/logit_distillation.ipynb"><code>logit_distillation.ipynb</code></a></td>
      <td>Demonstrates knowledge distillation from a Gemma 7B-IT teacher to a Gemma 2B-IT student for translation task.</td>
    </tr>
    <tr>
      <td rowspan="6" align="center" valign="middle" style="text-align: center; vertical-align: middle;"><b>Script</b></td>
      <td><a href="https://github.com/google/tunix/tree/main/examples/rl/grpo/gsm8k/"><code>rl/grpo/gsm8k/</code></a></td>
      <td>Bash scripts for fine-tuning different models and presets (Gemma, Llama, etc.) on the GSM8K mathematical reasoning task using GRPO.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/rl/grpo/gsm8k/verl_compatible/"><code>rl/grpo/gsm8k/verl_compatible/</code></a></td>
      <td>Bash scripts for GRPO-training on the GSM8K dataset to train with a verl-compatible setup.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/deepscaler/"><code>deepscaler/</code></a></td>
      <td>Scripts and notebooks for reproducing the <a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Deepscaler experiment</a> (<code>train_deepscaler_nb.py</code>) and math evaluation.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/sft/mtnt/"><code>sft/mtnt/</code></a></td>
      <td>Bash scripts for SFT examples on the MTNT translation task for Gemma, Llama, and Qwen models.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/model_load/"><code>model_load/</code></a></td>
      <td>Examples for loading Gemma2 and Gemma3 models from safetensors format.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/agentic/"><code>agentic/</code></a></td>
      <td>Examples and scripts for agentic workflows, with async rollout.</td>
    </tr>
  </tbody>
</table>

## Choosing a GRPO entry point

The [GSM8K shell recipes](https://github.com/google/tunix/tree/main/examples/rl/grpo/gsm8k/)
and the [Python demo](https://github.com/google/tunix/blob/main/scripts/grpo_demo_llama3_qwen2.py)
both use Tunix's GRPO learner. They differ in how they configure and assemble the
training workflow.

- Use the shell recipes to configure repeatable training jobs without editing
  Python. They launch `python3 -m tunix.cli.grpo_main` with a base YAML file,
  model-specific settings, and `key=value` overrides. For example,
  [run_llama3.2_1b.sh](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_llama3.2_1b.sh)
  loads `tunix/cli/base_config.yaml`, applies a model preset through
  `override_config_file`, and then applies command-line overrides. The
  configuration controls model loading, tokenization, data, reward modules,
  optimizer settings, device meshes, logging, and checkpointing. See
  [Launching jobs](launching.md) for CLI setup and usage.
- Use `scripts/grpo_demo_llama3_qwen2.py` to study or modify the workflow directly
  in Python. It constructs the models, optimizer, rollout configuration, and
  learner explicitly, and defines its reward functions in the script. It uses
  flags such as `--root-dir` and `--model-version`, together with constants in
  the source, rather than the CLI YAML configuration. Its model table includes
  selected Llama 3, Qwen 2.5, and Qwen 3 models. It also evaluates the model before
  and after training. The demo is marked WIP and clears its configured training
  checkpoint directory at startup; review its paths and defaults before running.

Both entry points support `vanilla`, `vllm`, and `sglang_jax` rollout engines.
The CLI selects the engine with `rollout_engine`; the demo uses
`--rollout-engine`. SGLang-JAX is therefore available through the CLI as well as
the demo. Backend dependencies, model weight mappings, and device requirements
still apply, and defaults differ between the two entry points. See the
[rollout guide](rollout.md) for backend setup and configuration.
