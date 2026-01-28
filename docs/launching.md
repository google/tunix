

# Launching Jobs

Tunix supports several ways to launch training jobs, depending on your workflow:

*   **[Tunix CLI](#tunix-cli)**: The default choice. A simple CLI launching tool with comprehensive configuration options.
*   **[Interactive and Custom Launch](#interactive-and-custom-launch)**: While the Tunix CLI is preferred for simplicity, this approach provides full control. This is ideal for:

    *   **Early experimentation**: Getting familiar with the framework.
    *   **Advanced customization**: Complex cases requiring flexibility beyond the CLI.

## Tunix CLI

Tunix offers a configurable CLI to launch SFT and RL job directly from the command line. There a number of knobs and parameters allowing full customization of the job (described below), as well as a number of pre-defined examples to get you started.

### Configuration Hierarchy

You can tune CLI parameters in one of three ways. Configurations are merged from these sources in the following order of precedence (later sources override earlier ones):

1.  **Base Config File** (Lowest Priority):
    The default settings found in `base_config.yaml`.

2.  **Config File Override**:
    An optional secondary config file specified via the `override_config_file` argument (e.g., `override_config_file=/path/to/override.yaml`). Values here override the base config.
    *(See an [example script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/configs/gemma2_2b.yaml)).*

3.  **CLI Arguments** (Highest Priority):
    Individual `key=value` pairs provided as command-line arguments. These override values from both the base config and the override file.


Here is the updated section with the formatting improved and the GRPO table added. I have cleaned up the links to point directly to the files.


### Example CLI Scripts

This collection includes command-line interface (CLI) scripts designed to handle various tasks.

> NOTE: 🔑 Required Credentials
>
> Before running the scripts, please ensure you have the following environment variables configured to access the necessary model repositories:
>
> *   **Hugging Face Access:**
>     *   `HF_TOKEN`: Required to authenticate and download models from Hugging Face.
>
> *   **Kaggle Access:**
>     *   `KAGGLE_USERNAME`: Your Kaggle username.
>     *   `KAGGLE_KEY`: Your Kaggle API key.

<section class="zippy">

#### Guide to Setup Credential

 TL;DR: This is a guide that covers how to get your keys, authorize them for restricted access, and save them permanently in a `.env` file.

##### Hugging Face (Token & Access)

**Step A: Generate the Token**

1.  Log in to [Hugging Face](https://huggingface.co/).
2.  Click your **Profile Picture** > **Settings** > **Access Tokens**.
3.  Click **Create new token**.
    *   **Name:** `CLI-Access`
    *   **Permissions:** **Read** (sufficient for downloading).
4.  **Copy** the token string (starts with `hf_`).

**Step B: Authorize Restricted Models (Critical)**

*Your token will fail if you skip this.*

1.  Go to the specific model page (e.g., [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)).
2.  Find the **"Access this model"** banner at the top.
3.  Review the license and click **Agree and access repository**.
4.  Wait for permissions to sync to your token.



##### Kaggle (Key & Access)

**Step A: Generate the API Key**

1. Log in to [Kaggle](https://www.kaggle.com/).
2. Click your **Profile Picture** > **Settings**.
3. Scroll to the **API** section and click **Create New Token**.
4. Open the downloaded `kaggle.json` file to find your username and key.

**Step B: Authorize Access (Critical)**

1. Go to the specific **Models** page on Kaggle (e.g., [`gemma model family`](https://www.kaggle.com/models/google/gemma)).
2. Click the **Request Access** tab.
3. Review and Sign **Consent Form**.


##### Create the `.env` File

Instead of typing `export` every time, we will save these credentials in a file that sits in your project folder.

1. Create a new file in your project root named `.env` (no filename, just the extension).
2. Paste the following content into it, replacing the placeholders with your actual keys:

  ```bash
  # .env

  # Hugging Face Access
  HF_TOKEN=hf_12345exampletokenstring

  # Kaggle Access
  KAGGLE_USERNAME=your_kaggle_username
  KAGGLE_KEY=example_kaggle_key

  ```

>**Security Warning:** If you are using Git, you **must** add `.env` to your `.gitignore` file immediately. This prevents you from accidentally uploading your passwords to GitHub.

  </section>

*   **Supervised Fine Tuning**

    *Refer to the [hardware requirement](https://github.com/google/tunix/blob/main/examples/sft/mtnt/README.md) before proceeding.*

    *   **Peft Training on MTNT dataset** ([Source Folder](https://github.com/google/tunix/blob/main/examples/sft/mtnt))

        | Model Variant     | Script Name             | Link                                                                                                                             |
        | :---------------- | :---------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
        | **Gemma 2B**      | `run_gemma_2b.sh`       | [View Script](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_gemma_2b.sh)       |
        | **Gemma 2 2B**    | `run_gemma2_2b.sh`      | [View Script](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_gemma2_2b.sh)      |
        | **Gemma 3 4B**    | `run_gemma3_4b.sh`      | [View Script](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_gemma3_4b.sh)      |
        | **Llama 3.2 3B**  | `run_llama3.2_3b.sh`    | [View Script](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_llama3.2_3b.sh)    |
        | **Qwen 2.5 0.5B** | `run_qwen2.5_0.5b.sh`   | [View Script](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_qwen2.5_0.5b.sh)   |


*   **Reinforcement Learning**

    *Refer to the [hardware requirement](https://github.com/google/tunix/blob/main/examples/rl/README.md) before proceeding.*

    *   **GRPO Training on GSM8K dataset** ([Source Folder](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/))

        | Model Variant     | Script Name             | Link                                                                                                                             |
        | :---------------- | :---------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
        | **Gemma 7B**      | `run_gemma_7b.sh`       | [View Script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_gemma_7b.sh)       |
        | **Gemma 2 2B**    | `run_gemma2_2b.sh`      | [View Script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_gemma2_2b.sh)      |
        | **Gemma 3 1B**    | `run_gemma3_1b.sh`      | [View Script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_gemma3_1b.sh)      |
        | **Gemma 3 4B**    | `run_gemma3_4b.sh`      | [View Script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_gemma3_4b.sh)      |
        | **Llama 3.1 8B**  | `run_llama3.1_8b.sh`    | [View Script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_llama3.1_8b.sh)    |
        | **Llama 3.2 1B**  | `run_llama3.2_1b.sh`    | [View Script](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/run_llama3.2_1b.sh)    |





### CLI Scripts Overview

CLI Core code and CLI example launch scripts reside separately. This section provides a high level overview of the CLI structure.

* [CLI Core](https://github.com/google/tunix/blob/main/cli)
  * `base_config.yaml`: Define all the configurations that could be tuned when launching a job.

  * `peft_main.py`: Main entry point to trigger Parameter-Efficient Fine-Tuning (PEFT) Trainer from the CLI configs.

  * `grpo_main.py`: Main entry point to trigger Group Relative Policy Optimization (GRPO) Trainer from the CLI configs.

  * `config.py`: Logic to read and process the config passed by command line or environment variable.

  * `reward_fn/...`: Predefined reward functions for reinforcement learning jobs. You could modify your shell scripts to use your own reward function.

  * `gsm8k.py`: Predefined reward functions running on [gsm8k dataset](https://www.tensorflow.org/datasets/catalog/gsm8k).

  * `gsm8k_verl.py`: Predefined reward functions on gsm8k dataset compatible with verl, refer to [this](https://github.com/google/tunix/blob/main/examples/rl/grpo/gsm8k/verl_compatible/README.md) for details.


* [CLI Examples](https://github.com/google/tunix/blob/main/examples)

  * `SFT/..`: All available SFT CLI shell scripts.

  * `RL/.. `: All available RL CLI shell scripts.


### Usage


#### Setup Cloud VM Environment

**TL;DR:** To automate the setup process for a single host, run the provided shell script matching your hardware accelerator.

**Automated Setup:**

*   **For TPU:**

    ```shell
    source scripts/setup_cli_tpu_single_host.sh
    ```

*   **For GPU:**

    ```shell
    source scripts/setup_cli_gpu_single_host.sh
    ```

<section class="zippy">

**Manual Setup (Detailed Breakdown)**

The following steps explain the environment setup process performed by the scripts above. You may skip this if you ran one of the automated scripts.

*   **1. Create a project specific environment.**

    ```shell
    python3 -m venv .venv
    # Or simply `python -m venv .venv` depending on your system configuration.
    ```

*   **2. Activate the Environment**

    ```shell
    source .venv/bin/activate
    ```

*   **3. Install Tunix dependency**

    Make sure you have an updated pip version installed:

    ```shell
    pip install --upgrade pip
    ```

    *   **Option A: TPU Only**
        If you only require TPU, install with `[prod]` extra:

        ```shell
        pip install -e .[prod]
        ```

    *   **Option B: Other Accelerators (e.g., GPU)**
        First, install the core Tunix dependency:

        ```shell
        pip install -e .
        ```

        Then, install your accelerator-specific dependency separately, for example:

        ```shell
        # Example for GPU
        pip install jax[gpu]
        ```

</section>

### Config Explanation

This section provides a detailed explanation of the configuration parameters available in `base_config.yaml`. These parameters allow you to customize model selection, training dynamics, hardware utilization (mesh), and reinforcement learning specific settings.



#### Model Configuration (`model_config`)

These parameters define the base model, where to download it from, and how to shard it across TPUs/GPUs. Note that `actor_model_config`, `reference_model_config`, and `rollout_model_config` typically inherit from this base configuration. 

* **`model_name`**: The unique full name identifier of the model. This
    corresponds to the full name and should match exactly with the model name
    used in Hugging Face or Kaggle. It is typically all lowercase and formatted
    as `<model-family>-<model-version>`.
    *   *Example*: `gemma-2b`, `llama-3.1-8b`, `gemma2-2b-it`.
    Refer to [models documentation](models.md#naming-conventions) for model naming.

* **`model_source`**: The source repository for the model. Options: `"huggingface"`, `"kaggle"`, `"gcs"`, or empty string `""` for local paths.
* **`model_id`**: The exact repository ID (case sensitive) as it appears on Hugging Face or Kaggle (e.g., `"meta-llama/Llama-3.1-8B"`).
  * *Example for Hugging Face*: `meta-llama/Llama-3.1-8B` is extracted as shown belows
  {: width="75%"}
  * *Example for Kaggle*: `google/gemma-2/flax/gemma2-2b-it` is extracted as shown belows
  {: width="75%"}


* **`model_path`**: Used if `model_source` is GCS or local. Specifies the direct file path to the model.
* **`model_download_path`**: Local directory where downloaded checkpoints will be cached (e.g., `"/tmp/models"`).
* **`rng_seed`**: Integer seed for initializing the `nnx.Rngs` state to manage randomness (e.g., `0`).
* **`model_display`**: Boolean flag. If set to `true`, prints the model structure/summary.
* **`intermediate_ckpt_dir`**: Directory for temporary storage when converting specific formats like Kaggle Gemma/Gemma2 to NNX (e.g., `"/tmp/intermediate_ckpt/"`).
* **`lora_config`**: Configuration for Low-Rank Adaptation (LoRA).
  * `module_path`: Regex identifying layers to adapt (e.g., `".*q_einsum|.*kv_einsum|.*gate_proj..."`). Refer to [Lora](performance.md#peft-with-lora) for more details.
  * `rank`: The rank of the low-rank approximation (e.g., `16`).
  * `alpha`: Scaling factor for LoRA weights (e.g., `2.0`).
  * `weight_qtype`: Quantization type for the weights (e.g., `"nf4"`).
  * `tile_size`: Tile size for efficient computation (e.g., `256`).


* **`mesh`**: Defines the hardware mesh layout for distributed training.
  * `shape`: Tuple string defining mesh dimensions (e.g., `"(2,2)"` for a 2x2 grid).
  * `axis_names`: Names for mesh axes, often used for parallelism strategies (e.g., `"('fsdp','tp')"` for Fully Sharded Data Parallelism and Tensor Parallelism).


#### Tokenizer Configuration (`tokenizer_config`)

* **`tokenizer_path`**: Path or ID of the tokenizer. Usually matches `model_id`.

* **`tokenizer_type`**: The library to use for tokenization. Options: `"huggingface"`, `"sentencepiece"`.

* **`add_bos`** / **`add_eos`**: Boolean. Whether to automatically prepend Beginning of Sentence (BOS) or append End of Sentence (EOS) tokens.


#### Dataset Configuration

* **`dataset_name`**: The identifier for the dataset (e.g., `"Helsinki-NLP/opus-100"` for SFT or `"gsm8k"` for RL).

* **`batch_size`**: Global batch size per training step.

* **`max_target_length`**: Maximum length of the target sequence (in tokens).

* **`num_train_epochs`**: Number of complete passes through the training dataset.

* **`tfds_download`**: Boolean. Controls download behavior for TensorFlow Datasets.


#### Optimizer Configuration (`optimizer_config`)

Controls the gradient descent algorithm and learning rate scheduling. Tunix uses [optax](https://optax.readthedocs.io/en/latest/getting_started.html) to for optimizer.

* **`opt_type`**: The optimizer algorithm to use (e.g., `"adamw"`). Refer to [optax.optimizer](https://optax.readthedocs.io/en/latest/api/optimizers.html#optimizers) for available optimizer.

* **`learning_rate`**: A global scaling factor, either fixed or evolving along iterations with a scheduler if `schedule_type` is set.

* **`schedule_type`**: The learning rate schedule function (e.g., `"warmup_cosine_decay_schedule"`). Refer to [optax.schedule](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#) for available schedulers.

* **`warmup_steps`**: Number of steps to linearly increase the learning rate from `init_value` to `peak_value`.

* **`decay_steps`**: Number of steps for the decay phase of the schedule.

* **`max_grad_norm`**: Gradient clipping threshold. Essential for preventing exploding gradients, especially in RL.




#### Training Configuration (`training_config`)

General settings for the training loop, logging, and checkpointing.

* **`max_steps`**: Total number of training steps to run.

* **`eval_every_n_steps`**: Frequency of running evaluation steps.

* **`gradient_accumulation_steps`**: Number of steps to accumulate gradients 
before performing a parameter update (simulates larger batch sizes).

* **`checkpointing_options`**:
  * `max_to_keep`: Number of recent checkpoints to retain.
  * `save_interval_steps`: How often to save a checkpoint.


* **`metrics_logging_options`**: Settings for logging. Includes project name, run name, and flush frequency.

* **`data_sharding_axis`**: Specifies which mesh axis is used for data sharding (e.g., `["fsdp"]`).



#### RL & GRPO Configuration (`grpo_config`, `rollout_config`)

Specific parameters for Reinforcement Learning jobs, particularly Group Relative Policy Optimization (GRPO).

* **`num_generations`**: (GRPO specific) The number of responses generated per prompt in a single step (corresponds to $\varepsilon$ in the paper).

* **`beta`**: Coefficient for the KL divergence penalty. Keeps the trained model close to the reference model.

* **`epsilon`**: Clipping parameter for the loss function (similar to PPO) to ensure stable updates.

* **`temperature`**: Sampling temperature for rollouts. Higher values (e.g., 0.9) encourage diversity, which is critical for GRPO.

* **`total_generation_steps`**: Maximum tokens to generate during the rollout phase.

* **`reward_functions`**: List of python file paths containing the reward logic (e.g., checking math answers for GSM8K).

## Interactive and Custom Launch

For interactive development or custom cluster setup, refer to our notebooks, examples, and guides, which demonstrate how to build the cluster and launch jobs programmatically:

*   **[Quick Start Guides](quickstart.md)**: Step-by-step guides for SFT, RL, Agentic workflows, and more.
*   **[Examples & Colabs](examples.md)**: A comprehensive list of interactive notebooks and scripts.

These resources are an excellent starting point for learning Tunix core concepts. Additionally, they demonstrate how to use Tunix in more complex scenarios that require full control. However, note that the **[Tunix CLI](#tunix-cli)** is recommended option for most standard workflows due to its simplicity.
