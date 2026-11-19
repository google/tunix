# Distributed vs. Agentic RL Efficiency Benchmark

`benchmark.py` runs one GRPO workload on the distributed stack
(`tunix/dist_rl/launch/`) or the agentic stack (`tunix.cli.grpo_main`), then
compares their recorded timings.

## Files

- `benchmark.py`: the harness. `run` writes a run manifest and, with
  `--execute`, trains and summarizes; `report` compares finished runs. Planning
  and reporting need only the Python standard library.
- `runtime.py`: opt-in timing hooks that only the benchmark entry points
  install.
- `frozenlake_workload.py`: the workload, the Gemma4 E2B FrozenLake recipe. For
  the distributed stack, it runs
  `../../examples/recipes/benchmark_frozenlake_gemma4_e2b.sh` with
  `section.key=value` overrides, including the benchmark's own vLLM limits
  (`max_num_seqs=32`, `max_num_batched_tokens=8192`); for the agentic stack, it
  builds the matching `grpo_main` config. `frozenlake_workload_test.py` checks
  them against `examples/frozenlake_dist/frozenlake.yaml` and the recipe.

Both stacks use BF16 compute, FP32 loading, exact token continuity, separate
two-chip actor and rollout meshes, and one prompt group (8 trajectories) per
microbatch. The benchmark injects timing hooks through
`cluster_config.trainer_main`, `cluster_config.rollout_main`, and
`cluster_config.orchestrator_main`, which wrap the node mains in
`launch/nodes/`.

## Running

Use one dedicated four-chip TPU host with a pre-downloaded model. Keep code,
dependencies, and flags fixed across the runs you compare, and use a new
`--output` for every run. Without `--execute`, `run` only writes the plan.

```bash
BENCH=(python3 tunix/dist_rl/benchmarks/rl_efficiency/benchmark.py)
TIMING=stages  # Module timings; use pipeline for end-to-end step time.
runs=()
for repeat in 1 2 3; do
  order=(agentic dist); (( repeat % 2 )) || order=(dist agentic)
  for stack in "${order[@]}"; do
    out="$PWD/artifacts/rl_efficiency/$TIMING-$repeat-$stack"
    "${BENCH[@]}" run --mode "$stack" --timing-mode "$TIMING" --output "$out" \
      --model-dir /path/to/gemma-4-E2B-it --cache-root "$PWD/artifacts/cache" \
      --hardware v5p-4 --execute
    runs+=("$out")
  done
done
"${BENCH[@]}" report "${runs[@]}" > "$TIMING-comparison.json"
```

Defaults run 150 steps (5 warmup) with 64 prompt groups × 8 generations;
`run --help` lists all flags. `stages` waits for device completion after
training, log-probs, and weight sync (compare
`api_calls.<module>.host_time_per_step_seconds`). `pipeline` keeps intra-step
overlap (compare `step_time_seconds.mean`). Do not mix modes in one report.

The report gives, for every numeric summary field, each stack's median across
runs, `dist_minus_agentic`, and `dist_relative_to_agentic_percent`. Positive
time differences mean distributed is slower. It rejects failed or incomplete
runs, timed-out trajectories, missing worker hooks, and mismatched workload,
hardware, model, dataset, dtype, timing mode, or per-step training work.
