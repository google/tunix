# Copyright 2026 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import random
import numpy as np

try:
    from examples.frozenlake.env import FrozenLakeEnv
except ImportError:
    from env import FrozenLakeEnv

def stress_test(num_steps=1000000):
    """Stress test for FrozenLakeEnv._step_impl."""
    print(f"Starting stress test with {num_steps} steps...")
    
    # Initialize environment
    entry = {"size": np.array([8]), "p": np.array([0.8]), "seed": np.array([42])}
    env = FrozenLakeEnv(entry, group_id=1, pair_index=1, max_steps=1, is_slippery=True)
    env.reset()
    
    start_time = time.time()
    
    success_count = 0
    hole_count = 0
    invalid_count = 0
    effective_count = 0
    
    for i in range(1, num_steps + 1):
        # Valid actions are 1, 2, 3, 4. We can also test invalid actions like 0 or 5.
        # Let's mostly use valid actions, but occasionally throw in an invalid one.
        action = random.choice([1, 2, 3, 4])
            
        result = env._step_impl(action)
        
        if action in [0, 5]:
            invalid_count += 1
            assert not result.info["action_is_effective"], "Invalid action should not be effective"
        elif result.info["action_is_effective"]:
            effective_count += 1
            
        if result.done:
            if env.success():
                success_count += 1
            else:
                hole_count += 1
            env.reset()
            
        if i % 100000 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            sps = i / elapsed
            print(f"Step {i}/{num_steps} - Elapsed: {elapsed:.2f}s - SPS: {sps:.2f}")
            
    end_time = time.time()
    duration = end_time - start_time
    steps_per_second = num_steps / duration
    
    print("\nStress Test Results:")
    print(f"Total steps: {num_steps}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Successes (reached G): {success_count}")
    print(f"Failures (reached H): {hole_count}")
    print(f"Invalid actions tested: {invalid_count}")
    print(f"Effective steps: {effective_count}")
    print("Environment remains stable after stress testing.")

if __name__ == "__main__":
    # Run with fewer steps if you want a quick check, e.g., 100,000
    stress_test(num_steps=1000000)
