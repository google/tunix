#!/usr/bin/env python3
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
"""Map TPU VFIO IDs sorted by PCI address.

Usage:
    python get_sorted_tpu_ids.py [-H] [--begin BEGIN] [--count COUNT]

Options:
    -H, --human           Output in human-readable table format (default: comma-separated sorted IDs)
    --begin BEGIN         Starting index for subrange (default: 0)
    --count COUNT         Number of elements to include in subrange (default: all)
    -h, --help            Show this help message and exit
"""

import argparse
import os
import sys


def get_sorted_tpu_mappings(
    vfio_dir: str = "/dev/vfio",
    iommu_base: str = "/sys/kernel/iommu_groups",
) -> list[tuple[str, int]]:
  """Scans /dev/vfio and maps VFIO IDs to their PCI addresses, sorted by PCI address."""
  if not os.path.exists(vfio_dir):
    return []

  mappings = []
  for entry in os.listdir(vfio_dir):
    if entry.isdigit():
      vfio_id = int(entry)
      devices_path = os.path.join(iommu_base, str(vfio_id), "devices")
      if os.path.exists(devices_path):
        for dev in os.listdir(devices_path):
          symlink_path = os.path.join(devices_path, dev)
          if os.path.islink(symlink_path):
            pci_addr = os.path.basename(os.readlink(symlink_path))
          else:
            pci_addr = dev
          mappings.append((pci_addr, vfio_id))

  mappings.sort(key=lambda x: x[0])
  return mappings


def main():
  parser = argparse.ArgumentParser(
      description="Map TPU VFIO IDs sorted by PCI address."
  )
  parser.add_argument(
      "-H",
      "--human",
      action="store_true",
      help="Output in human-readable table format",
  )
  parser.add_argument(
      "--begin", type=int, default=0, help="Starting index for subrange"
  )
  parser.add_argument(
      "--count",
      type=int,
      default=None,
      help="Number of elements to include in subrange",
  )
  args = parser.parse_args()

  mappings = get_sorted_tpu_mappings()
  if not mappings:
    if args.human:
      print("No TPU VFIO devices found.")
    sys.exit(0 if args.human else 1)

  sorted_ids = [m[1] for m in mappings]

  # Apply subrange (begin and count)
  begin = max(0, args.begin)
  if args.count is not None:
    end = begin + max(0, args.count)
    sorted_ids = sorted_ids[begin:end]
    mappings = mappings[begin:end]
  else:
    sorted_ids = sorted_ids[begin:]
    mappings = mappings[begin:]

  if args.human:
    print(f"{'PCI Address':<18} -> {'VFIO ID':<10}")
    print("-" * 32)
    for pci_addr, vfio_id in mappings:
      print(f"{pci_addr:<18} -> {vfio_id:<10}")

    print("\nSorted VFIO IDs (by PCI order):")
    print(sorted_ids)
  else:
    print(",".join(map(str, sorted_ids)))


if __name__ == "__main__":
  main()
