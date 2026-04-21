import wandb
import pandas as pd

# 1. Setup API and source run
api = wandb.Api()
source_run_path = "tunix/rhh72dzt" # Replace with your run path
run = api.run(source_run_path)

# 2. Retrieve metrics (pandas=True makes slicing very easy)
# Note: history() by default samples. For full data, use .scan()
history = run.history(pandas=True) 

# Determine the "halfway" point (by index or by a specific step)
index = 41
filtered_history = history.iloc[:index]

# 3. Initialize a new "cleaned" run
original_config = run.config
original_config["learning_rate"] = 5e-7
new_run = wandb.init(
    project="tunix",
    name=f"vllm-on-head-bs64-lr5e-7-continue-40-16k",
    config=original_config # Optional: copy the original config
)

# 4. Manually re-log the metrics
for _, row in filtered_history.iterrows():
    # Filter out wandb internal metrics (starting with _)
    metrics = {k: v for k, v in row.items() if not k.startswith('_') and pd.notna(v)}
    
    # Use the original step to keep the X-axis consistent
    step = int(row["_step"])
    wandb.log(metrics, step=step)

new_run.finish()
print("Successfully created a partial session!")