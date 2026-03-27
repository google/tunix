# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import multiprocessing
import os
import re
from typing import Callable, Optional

import numpy as np
from math_verify.errors import TimeoutException

from typing import Any, Callable, Dict, List

from tunix.utils import math_utils
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.rewards import reward_types


def silent_worker_init():
    """
    This runs inside each worker process IMMEDIATELY after it spawns.
    It hides accelerators so workers don't try to 'steal' the TPU/GPU.
    """
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Hide TPUs
    os.environ["TPU_VISIBLE_DEVICES"] = ""
    # Hide GPUs (just in case)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Prevent JAX from pre-allocating memory if it does get imported
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def compute_reward(ground_truth: str, response: str):
  from math_verify.metric import math_metric
  from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
  math_rewards_verify = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
      ExprExtractionConfig(),
      LatexExtractionConfig(),
    ),
  )
  try:
    result = math_rewards_verify([ground_truth], [response])
    return result[0] if isinstance(result, (list, tuple)) else result
  except Exception as e:
    print(f"Error occurred while computing math reward: {e}")
    return 0.0

def math_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs):
  """
  A reward function for math tasks that implements the RewardFunction protocol.
  Args:
    task: The task dictionary containing data_source, ground_truth and other metadata
    action: The agent's response/solution

  Returns:
    float: The calculated reward value based on math evaluation
  """
  # Extract information from task_info
  ground_truth = [""]  * len(answer)
  for i in range(len(answer)):
    # Process the ground truth(s)
    ground_truths = answer[i]
    if ground_truths is None:
      # return RewardOutput(0.0, {"is_correct": False})
      rewards.append(0.0)
      continue

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, str | float | int):
      ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
      truth = str(truth)
      if "\\boxed" in truth:
        processed_truth = math_utils.extract_answer(truth)
        if processed_truth is not None:
          processed_ground_truths.append(processed_truth)
      else:
        processed_ground_truths.append(truth)
    if processed_ground_truths:
      ground_truth[i] = processed_ground_truths[0]
    else:
        logging.warning(f"Could not process any valid ground truth for index {i}.")
        ground_truth[i] = ""

  for i in range(len(ground_truth)):
    ground_truth[i] =  "\\boxed{" + ground_truth[i] + "}"
    print(f"Ground truth: {ground_truth[i]}, last 30 chars of completions: {completions[i][-30:]}")
  args_list = [(ground_truth[i], completions[i]) for i in range(len(completions))]
  
  ctx = multiprocessing.get_context('spawn')
  num_procs = min(len(completions), ctx.cpu_count())
  timeout_per_item = 5  # Total time allowed for the BATCH to complete
  results = []
  pool = ctx.Pool(processes=num_procs, initializer=silent_worker_init)
  try:
    jobs = [pool.apply_async(compute_reward, a) for a in args_list]
    for i, job in enumerate(jobs):
      try:
        # Wait up to 15 seconds for this specific result
        res = job.get(timeout=15)
        results.append(res)
      except mp.TimeoutError:
        print(f"⚠️ Reward worker {i} timed out. Assigning 0.0 reward.")
      results.append(0.0) 
      except Exception as e:
        print(f"❌ Reward worker {i} failed with error: {e}. Assigning 0.0.")
        results.append(0.0)
  finally:
    pool.close() # Prevents new tasks
    pool.terminate()
    pool.join()  # Waits for workers to exit and cleans up semaphores
  

  print(f"Final rewards: {results}")
  return results
