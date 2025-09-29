# Tunix CLI 
This directory, `tunix/cli`, contains the core command-line interface scripts for running post-training tasks with the Tunix framework.

Overview
The Python scripts in this folder are the entry points for initiating training and inference jobs. They are designed to be configurable and can be launched directly from the command line. However, for ease of use and to demonstrate best practices, we recommend using the example shell scripts provided in the /examples directory.

## Scripts
- `base_config.yaml`: Define all the configurations that could be tuned from CLI launched.

- `peft_main.py`: Main entry point to trigger Parameter-Efficient Fine-Tuning (PEFT) Trainer from CLI configs

- `grpo_main.py`: Main entry point to trigger Group Relative Policy Optimization (GRPO) Trainer from CLI configs


## Usage
While you can run these scripts directly, the intended workflow is to use the wrapper scripts in the examples folder. These examples show how to pass the correct arguments and configurations for various use cases.

For sft, we provide scripts running on mtnt translation dataset. See available [scripts](examples/sft/mtnt)

For rl, we provide scripts running grpo on gsm8k math dataset. See available [scripts](examples/rl/gsm8k)

For launching shell scripts from examples, you would navigate to the examples directory and execute a script like this:

### Setup Environment 
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```
### Make sure you are in the root of the tunix repository
```
cd examples/
./sft/mtnt/run_gemma_2b.sh
```
This shell script will, in turn, call the appropriate Python script from the `tunix/cli` directory with the necessary parameters.

Contributing
(Optional: Add guidelines here if you are open to contributions for these scripts.)

Please refer to the main project README for more detailed information on the overall project structure and setup.
