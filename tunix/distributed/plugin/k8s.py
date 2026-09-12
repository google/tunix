# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubernetes utils."""

import os


def get_jobset_hostname() -> str:
  required_envs = [
      "JOBSET_NAME",
      "REPLICATED_JOB_NAME",
      "JOB_INDEX",
      "POD_INDEX",
  ]
  missing_envs = [env for env in required_envs if env not in os.environ]
  if missing_envs:
    raise ValueError(
        f"Missing required environment variable(s): {', '.join(missing_envs)}"
    )

  jobset_name = os.environ["JOBSET_NAME"]
  replicated_job = os.environ["REPLICATED_JOB_NAME"]
  job_index = os.environ["JOB_INDEX"]
  pod_index = os.environ["POD_INDEX"]

  # Constructing a fully qualified domain name (FQDN) manually if needed:
  # Format: <pod-hostname>.<headless-service-name>.<namespace>.svc.cluster.local
  fqdn = f"{jobset_name}-{replicated_job}-{job_index}-{pod_index}.{jobset_name}-{replicated_job}.default.svc.cluster.local"
  return fqdn
