"""
FrozenLake Dataset Generator

This script generates training and test datasets for the FrozenLake environment.
Each dataset entry contains environment configuration parameters (seed, size, p)
that can be used to create FrozenLake environment instances.

The generated datasets are saved as Parquet files and can be used for training
reinforcement learning agents on various FrozenLake configurations.

Usage:
    python recipes/frozenlake/data.py --train_size 10000 --test_size 100

The script generates:
- Training dataset: Random FrozenLake configurations for agent training
- Test dataset: Separate set of configurations for evaluation
"""

import argparse
import os

import numpy as np
import pandas as pd


DEFAULT_DIR = os.getcwd()


def get_frozenlake_dict(seed: int, size: int, p: float) -> dict:
    """
    Create a dictionary with FrozenLake environment configuration parameters.

    Args:
        seed: Random seed for environment generation
        size: Grid size (size x size grid)
        p: Probability of moving in the intended direction (1-p = slip probability)

    Returns:
        Dictionary containing environment configuration with keys: seed, size, p
    """
    return {"env_name": "frozenlake", "seed": int(seed), "size": int(size), "p": float(p)}


def generate_dataset_parameters(size: int, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random parameters for FrozenLake environments.

    This function creates diverse environment configurations by sampling:
    - Random seeds for environment generation
    - Grid sizes ranging from 2x2 to 9x9
    - Slip probabilities between 0.15-0.4 (p values 0.6-0.85)

    Args:
        size: Number of environment configurations to generate
        random_seed: Random seed for reproducible parameter generation

    Returns:
        Tuple of (seeds, sizes, p_values) numpy arrays
    """
    np.random.seed(random_seed)
    seeds = np.random.randint(0, 100000, size=size)
    sizes = np.random.randint(2, 10, size=size)  # Grid sizes from 2x2 to 9x9
    p_values = np.random.uniform(0.6, 0.85, size=size)  # Slip probability between 0.15-0.4

    return seeds, sizes, p_values


def save_dataset(data: list[dict], filepath: str) -> None:
    """
    Save dataset to Parquet file format.

    Converts the list of environment configuration dictionaries to a pandas
    DataFrame and saves it as a Parquet file for efficient storage and loading.

    Args:
        data: List of environment configuration dictionaries
        filepath: Full path where to save the Parquet file
    """
    df = pd.DataFrame(data)
    df.to_parquet(filepath)
    print(f"Saved {len(data)} entries to {filepath}")


def main():
    """
    Main function to generate and save FrozenLake datasets.

    Parses command line arguments, generates training and test datasets with
    different random seeds for diversity, and saves them as Parquet files.
    """
    parser = argparse.ArgumentParser(
        description="Generate FrozenLake environment configuration datasets for training and testing."
    )
    parser.add_argument(
        "--local_dir", default=os.path.join(DEFAULT_DIR, "data/frozenlake"), help="Local directory to save the datasets"
    )
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy datasets to (optional)")
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of training environment configurations to generate (default: 10000)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of test environment configurations to generate (default: 100)",
    )

    args = parser.parse_args()

    # Create local directory
    local_dir = os.path.expanduser(args.local_dir)
    print(f"Using local directory: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    print(f"Using local directory: {local_dir}")

    # Generate training dataset parameters
    train_seeds, train_sizes, train_ps = generate_dataset_parameters(args.train_size, random_seed=42)
    train_data = [get_frozenlake_dict(seed, train_sizes[idx], train_ps[idx]) for idx, seed in enumerate(train_seeds)]

    # Generate test dataset parameters (different random seed for diversity)
    test_seeds, test_sizes, test_ps = generate_dataset_parameters(args.test_size, random_seed=123)
    test_data = [get_frozenlake_dict(seed, test_sizes[idx], test_ps[idx]) for idx, seed in enumerate(test_seeds)]

    # Save datasets as Parquet files
    save_dataset(train_data, os.path.join(local_dir, "train.parquet"))
    save_dataset(test_data, os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    main()