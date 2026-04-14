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
