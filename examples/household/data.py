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

"""Procedural task generator and dataset loader for Household text games."""

import json
import os
import random
from typing import Any

import datasets as datasets_lib
import grain.python as grain
import pandas as pd

ROOM_TYPES = [
    ("kitchen", "a modern kitchen with clean countertops and tiled floor"),
    ("living_room", "a spacious living room with an armchair and coffee table"),
    ("bedroom", "a quiet bedroom with a large bed and soft rug"),
    ("hallway", "a well-lit hallway connecting the rooms"),
    ("study", "a quiet study lined with wooden bookshelves and a study desk"),
    ("bathroom", "a clean tiled bathroom with a sink and mirror"),
]


def generate_task(seed: int = 42) -> dict[str, Any]:
  """Deterministically generate a household task specification."""
  rng = random.Random(seed)

  # Choose 3 to 5 rooms
  num_rooms = rng.randint(3, 5)
  selected_rooms = rng.sample(ROOM_TYPES, num_rooms)
  room_names = [r[0] for r in selected_rooms]

  # Generate connected graph of exits
  rooms: dict[str, Any] = {}
  for rname, rdesc in selected_rooms:
    rooms[rname] = {
        "description": rdesc,
        "exits": {},
        "containers": {},
        "floor_items": [],
    }

  directions = [
      ("north", "south"),
      ("east", "west"),
      ("northeast", "southwest"),
  ]

  for i in range(len(room_names) - 1):
    u = room_names[i]
    v = room_names[i + 1]
    fwd, bwd = directions[i % len(directions)]
    rooms[u]["exits"][fwd] = v
    rooms[v]["exits"][bwd] = u

  # Select a locked container and target item
  key_name = rng.choice(["silver_key", "brass_key", "golden_key"])
  chest_name = rng.choice(["wooden_chest", "desk_cabinet", "safe"])
  target_item = rng.choice(["golden_trophy", "ancient_scroll", "family_album"])

  # Place locked chest in one room
  chest_room = rng.choice(room_names)
  rooms[chest_room]["containers"][chest_name] = {
      "is_open": False,
      "is_locked": True,
      "key_name": key_name,
      "items": [target_item],
  }

  # Place key in another room (in a drawer or on floor)
  candidate_rooms = [r for r in room_names if r != chest_room]
  if not candidate_rooms:
    candidate_rooms = room_names
  key_room = rng.choice(candidate_rooms)

  if rng.random() > 0.5:
    drawer_name = f"{key_room}_drawer"
    rooms[key_room]["containers"][drawer_name] = {
        "is_open": False,
        "is_locked": False,
        "items": [key_name],
    }
    key_loc_desc = f"in the {drawer_name.replace('_', ' ')} in the {key_room}"
  else:
    rooms[key_room]["floor_items"].append(key_name)
    key_loc_desc = f"on the floor in the {key_room}"

  # Start room
  start_room = rng.choice(room_names)

  instruction = (
      f"Find the {key_name.replace('_', ' ')} {key_loc_desc}, unlock the"
      f" {chest_name.replace('_', ' ')} in the {chest_room}, and take the"
      f" {target_item.replace('_', ' ')}."
  )

  return {
      "env_name": "household",
      "seed": seed,
      "instruction": instruction,
      "start_room": start_room,
      "initial_inventory": [],
      "rooms": rooms,
      "goal": {
          "type": "take",
          "item": target_item,
      },
      "prompts": "",
  }


def create_dataset(
    split: str = "train",
    data_dir: str = "/tmp/data/household",
    seed: int = 42,
    train_size: int = 1000,
    test_size: int = 50,
    **kwargs,
) -> grain.MapDataset:
  """Generate or load the dataset and return a grain MapDataset."""
  del kwargs
  os.makedirs(data_dir, exist_ok=True)
  filepath = os.path.join(data_dir, f"{split}.parquet")

  size = train_size if split == "train" else test_size
  if not os.path.exists(filepath):
    data = []
    base_seed = seed if split == "train" else seed + 100000
    for i in range(size):
      task = generate_task(seed=base_seed + i)
      # Store serialized task for parquet compatibility
      data.append({
          "env_name": "household",
          "seed": task["seed"],
          "task_json": json.dumps(task),
          "prompts": "",
      })
    df = pd.DataFrame(data)
    df.to_parquet(filepath)

  df = pd.read_parquet(filepath)
  hf_ds = datasets_lib.Dataset.from_pandas(df)

  def process_item(item: dict[str, Any]) -> dict[str, Any]:
    item["prompts"] = ""
    return item

  return grain.MapDataset.source(hf_ds).map(process_item)  # pyrefly: ignore[bad-argument-type]
