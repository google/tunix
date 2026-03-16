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
import re
from typing import Callable, Optional

import numpy as np
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from typing import Any, Callable, Dict, List

from tunix.utils import math_utils
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.rewards import reward_types

math_rewards_verify = math_metric(
  gold_extraction_target=(LatexExtractionConfig(),),
  pred_extraction_target=(
    ExprExtractionConfig(),
    LatexExtractionConfig(),
  ),
)


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

  results = []
  timeout_seconds=2
  ctx = multiprocessing.get_context('spawn')
  for i in range(len(completions)):
    response = completions[i]
    truth = ground_truth[i]
    q = ctx.Queue()
    ground_truth_parsable = "\\boxed{" + truth + "}"
    print(f"Normalized response: {response}, Processed ground truth: {ground_truth_parsable}")
    proc = ctx.Process(target=lambda q, response, ground_truth_parsable: q.put(math_rewards_verify([ground_truth_parsable], [response], None)), args=(queue, response, ground_truth_parsable))
    proc.start()
    proc.join(timeout=timeout_seconds)
    
    if proc.is_alive():
      print(f"Hard timeout reached for prediction! Terminating...")
      proc.terminate()
      proc.join()
      results.append(0.0) # Assign 0 reward on timeout
    else:
      try:
        # Adding a small timeout to get() prevents hanging if the process died
        res, _ = q.get(timeout=1) 
        results.append(res[0])
      except Exception:
        results.append(0.0)
    q.close()          # Tells the queue no more data is coming
    q.join_thread()    # Ensures the background feeder thread exits cleanly
  print(f"Final rewards: {results}")
  return results
