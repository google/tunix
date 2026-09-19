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

"""Household text-based interactive environment (AlfWorld / TextWorld style).

This environment provides a multi-room, stateful text world designed for
multi-turn agentic reinforcement learning. The agent navigates between rooms,
inspects furniture, opens/closes containers, unlocks chests with keys, and
manipulates objects to complete household task goals.
"""

from collections.abc import Callable
import copy
import dataclasses
import json
import re
from typing import Any

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv
from tunix.rl.agentic.environments.base_environment import EnvStepResult


@dataclasses.dataclass
class Container:
  """Represents a container (e.g. drawer, refrigerator, chest)."""

  name: str
  is_open: bool = False
  is_locked: bool = False
  key_name: str | None = None
  items: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Room:
  """Represents a room in the house graph."""

  name: str
  description: str
  exits: dict[str, str] = dataclasses.field(default_factory=dict)
  containers: dict[str, Container] = dataclasses.field(default_factory=dict)
  floor_items: list[str] = dataclasses.field(default_factory=list)


class HouseholdEnv(BaseTaskEnv):
  """Multi-turn household text game environment for Agentic RL.

  Subclasses BaseTaskEnv to manage step counters and truncation.
  """

  INVALID_ACTION_PENALTY: float = -0.05
  STEP_PENALTY: float = -0.02
  MILESTONE_REWARD: float = 0.2
  GOAL_REWARD: float = 1.0

  def __init__(
      self,
      task: dict[str, Any] | None = None,
      *,
      reward_fn: Callable[..., Any] | None = None,
      max_steps: int = 25,
      **kwargs,
  ):
    """Initialize HouseholdEnv.

    Args:
      task: Dictionary describing initial world layout, starting room, and goal.
      reward_fn: Optional custom reward function.
      max_steps: Maximum interaction steps before forced truncation.
      **kwargs: Extra parameters passed to BaseTaskEnv.
    """
    super().__init__(
        task=task or {}, reward_fn=reward_fn, max_steps=max_steps, **kwargs
    )
    self._rooms: dict[str, Room] = {}
    self._current_room: str = ""
    self._inventory: list[str] = []
    self._achieved_milestones: set[str] = set()
    self._init_world_from_task()

  def _init_world_from_task(self) -> None:
    """Initialize or reset the world state from the task definition."""
    task_dict = self.task
    if isinstance(task_dict, dict):
      if "task" in task_dict and isinstance(task_dict["task"], dict):
        task_dict = task_dict["task"]
      elif "task_json" in task_dict:
        raw_json = task_dict["task_json"]
        if isinstance(raw_json, (list, tuple)):
          raw_json = raw_json[0]
        if hasattr(raw_json, "item"):
          raw_json = raw_json.item()
        task_dict = json.loads(raw_json)
    self._task_data: dict[str, Any] = task_dict if isinstance(task_dict, dict) else {}

    self._inventory = copy.deepcopy(
        self._task_data.get("initial_inventory", [])
    )
    self._achieved_milestones = set()
    self._rooms = {}

    rooms_data = self._task_data.get("rooms", {})
    if not rooms_data:
      # Default fallback room if no task provided.
      self._rooms["living_room"] = Room(
          name="living_room",
          description="a cozy living room with a sofa and coffee table",
          exits={},
          containers={},
          floor_items=[],
      )
      self._current_room = "living_room"
      return

    for room_name, rdata in rooms_data.items():
      containers = {}
      for cname, cdata in rdata.get("containers", {}).items():
        containers[cname] = Container(
            name=cname,
            is_open=cdata.get("is_open", False),
            is_locked=cdata.get("is_locked", False),
            key_name=cdata.get("key_name", None),
            items=list(cdata.get("items", [])),
        )
      self._rooms[room_name] = Room(
          name=room_name,
          description=rdata.get("description", f"the {room_name}"),
          exits=dict(rdata.get("exits", {})),
          containers=containers,
          floor_items=list(rdata.get("floor_items", [])),
      )

    self._current_room = self._task_data.get(
        "start_room", next(iter(self._rooms.keys()))
    )

  def _describe_room(self) -> str:
    """Generate text description of current room, contents, and exits."""
    room = self._rooms[self._current_room]
    lines = [f"You are in the {room.name} ({room.description})."]

    # Exits
    if room.exits:
      exit_desc = ", ".join(
          f"{direction} to {neighbor}"
          for direction, neighbor in sorted(room.exits.items())
      )
      lines.append(f"Exits: {exit_desc}.")
    else:
      lines.append("There are no obvious exits.")

    # Containers
    if room.containers:
      c_descs = []
      for cname, c in sorted(room.containers.items()):
        status = (
            "locked" if c.is_locked else ("open" if c.is_open else "closed")
        )
        c_descs.append(f"a {cname.replace('_', ' ')} ({status})")
      lines.append(f"You see: {', '.join(c_descs)}.")

    # Floor items
    if room.floor_items:
      items_str = ", ".join(
          item.replace("_", " ") for item in sorted(room.floor_items)
      )
      lines.append(f"On the floor: {items_str}.")

    return "\n".join(lines)

  def _initial_observation(self) -> str:
    """Return initial observation for episode start."""
    self._init_world_from_task()
    instruction = self._task_data.get(
        "instruction", "Explore the house and inspect items."
    )
    obs = f"Goal: {instruction}\n\n{self._describe_room()}"
    return obs

  def _normalize_name(self, text: str) -> str:
    """Convert natural language names to internal underscores."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
    return cleaned.replace(" ", "_")

  def _step_impl(self, action: Any) -> EnvStepResult:
    """Execute a single player action and return transition result."""
    if not isinstance(action, str):
      action = str(action)

    clean_action = action.strip()
    # Strip markdown quotes if wrapped
    if clean_action.startswith("```") and clean_action.endswith("```"):
      clean_action = clean_action.strip("`").strip()

    cmd = clean_action.lower()
    room = self._rooms[self._current_room]
    reward = self.STEP_PENALTY
    info: dict[str, Any] = {"action": clean_action, "valid": True}
    feedback = ""

    # 1. Navigation: go <direction> or go to <room>
    go_match = re.match(r"^go(?:\s+to)?\s+(?:the\s+)?([a-z_\s]+)$", cmd)
    if go_match:
      target = self._normalize_name(go_match.group(1))
      # Check directional exit
      if target in room.exits:
        self._current_room = room.exits[target]
        feedback = f"You move {target}.\n\n{self._describe_room()}"
      elif target in [r for r in room.exits.values()]:
        # Target room name specified directly
        self._current_room = target
        feedback = f"You move to the {target}.\n\n{self._describe_room()}"
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = f"You cannot go '{target}' from here."

    # 2. Look
    elif cmd in ("look", "look around"):
      feedback = self._describe_room()

    # 3. Inventory
    elif cmd in ("inventory", "inv", "i"):
      if self._inventory:
        inv_str = ", ".join(
            item.replace("_", " ") for item in sorted(self._inventory)
        )
        feedback = f"You are carrying: {inv_str}."
      else:
        feedback = "Your inventory is empty."

    # 4. Open <container>
    elif cmd.startswith("open "):
      cname = self._normalize_name(cmd[5:])
      if cname in room.containers:
        c = room.containers[cname]
        if c.is_locked:
          reward += self.INVALID_ACTION_PENALTY
          info["valid"] = False
          feedback = f"The {cname.replace('_', ' ')} is locked."
        elif c.is_open:
          feedback = f"The {cname.replace('_', ' ')} is already open."
        else:
          c.is_open = True
          items_str = (
              ", ".join(i.replace("_", " ") for i in c.items)
              if c.items
              else "nothing"
          )
          feedback = (
              f"You open the {cname.replace('_', ' ')}. Inside you see:"
              f" {items_str}."
          )
          milestone_key = f"opened_{cname}"
          if milestone_key not in self._achieved_milestones:
            self._achieved_milestones.add(milestone_key)
            reward += self.MILESTONE_REWARD
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = f"There is no {cname.replace('_', ' ')} here."

    # 5. Close <container>
    elif cmd.startswith("close "):
      cname = self._normalize_name(cmd[6:])
      if cname in room.containers:
        c = room.containers[cname]
        if not c.is_open:
          feedback = f"The {cname.replace('_', ' ')} is already closed."
        else:
          c.is_open = False
          feedback = f"You close the {cname.replace('_', ' ')}."
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = f"There is no {cname.replace('_', ' ')} here."

    # 6. Unlock <container> with <key>
    elif cmd.startswith("unlock "):
      unlock_match = re.match(
          r"^unlock\s+([a-z_\s]+)\s+with\s+([a-z_\s]+)$", cmd
      )
      if unlock_match:
        cname = self._normalize_name(unlock_match.group(1))
        key = self._normalize_name(unlock_match.group(2))
        if cname not in room.containers:
          reward += self.INVALID_ACTION_PENALTY
          info["valid"] = False
          feedback = f"There is no {cname.replace('_', ' ')} here."
        else:
          c = room.containers[cname]
          if not c.is_locked:
            feedback = f"The {cname.replace('_', ' ')} is not locked."
          elif key not in self._inventory:
            reward += self.INVALID_ACTION_PENALTY
            info["valid"] = False
            feedback = f"You do not have the {key.replace('_', ' ')}."
          elif c.key_name and c.key_name != key:
            reward += self.INVALID_ACTION_PENALTY
            info["valid"] = False
            feedback = (
                f"The {key.replace('_', ' ')} does not unlock the"
                f" {cname.replace('_', ' ')}."
            )
          else:
            c.is_locked = False
            feedback = (
                f"You unlock the {cname.replace('_', ' ')} with the"
                f" {key.replace('_', ' ')}."
            )
            milestone_key = f"unlocked_{cname}"
            if milestone_key not in self._achieved_milestones:
              self._achieved_milestones.add(milestone_key)
              reward += self.MILESTONE_REWARD
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = "Command format: unlock <container> with <key>"

    # 7. Take <item> [from <container>]
    elif cmd.startswith("take "):
      take_match = re.match(
          r"^take\s+([a-z_\s]+?)(?:\s+from\s+([a-z_\s]+))?$", cmd
      )
      if take_match:
        item = self._normalize_name(take_match.group(1))
        cname = (
            self._normalize_name(take_match.group(2))
            if take_match.group(2)
            else None
        )

        if cname:
          # Take from specified container
          if cname not in room.containers:
            reward += self.INVALID_ACTION_PENALTY
            info["valid"] = False
            feedback = f"There is no {cname.replace('_', ' ')} here."
          elif not room.containers[cname].is_open:
            reward += self.INVALID_ACTION_PENALTY
            info["valid"] = False
            feedback = f"The {cname.replace('_', ' ')} is closed."
          elif item not in room.containers[cname].items:
            reward += self.INVALID_ACTION_PENALTY
            info["valid"] = False
            feedback = (
                f"The {cname.replace('_', ' ')} does not contain"
                f" {item.replace('_', ' ')}."
            )
          else:
            room.containers[cname].items.remove(item)
            self._inventory.append(item)
            feedback = (
                f"You take the {item.replace('_', ' ')} from the"
                f" {cname.replace('_', ' ')}."
            )
            milestone_key = f"took_{item}"
            if milestone_key not in self._achieved_milestones:
              self._achieved_milestones.add(milestone_key)
              reward += self.MILESTONE_REWARD
        else:
          # Check floor first, then open containers in room
          if item in room.floor_items:
            room.floor_items.remove(item)
            self._inventory.append(item)
            feedback = f"You take the {item.replace('_', ' ')} from the floor."
            milestone_key = f"took_{item}"
            if milestone_key not in self._achieved_milestones:
              self._achieved_milestones.add(milestone_key)
              reward += self.MILESTONE_REWARD
          else:
            found_in_c = None
            for c in room.containers.values():
              if c.is_open and item in c.items:
                found_in_c = c
                break
            if found_in_c:
              found_in_c.items.remove(item)
              self._inventory.append(item)
              feedback = (
                  f"You take the {item.replace('_', ' ')} from the"
                  f" {found_in_c.name.replace('_', ' ')}."
              )
              milestone_key = f"took_{item}"
              if milestone_key not in self._achieved_milestones:
                self._achieved_milestones.add(milestone_key)
                reward += self.MILESTONE_REWARD
            else:
              reward += self.INVALID_ACTION_PENALTY
              info["valid"] = False
              feedback = f"Cannot find '{item.replace('_', ' ')}' to take."
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = "Command format: take <item> or take <item> from <container>"

    # 8. Put <item> in/on <container>
    elif cmd.startswith("put "):
      put_match = re.match(
          r"^put\s+([a-z_\s]+?)\s+(?:in|into|on)\s+([a-z_\s]+)$", cmd
      )
      if put_match:
        item = self._normalize_name(put_match.group(1))
        cname = self._normalize_name(put_match.group(2))
        if item not in self._inventory:
          reward += self.INVALID_ACTION_PENALTY
          info["valid"] = False
          feedback = f"You are not carrying {item.replace('_', ' ')}."
        elif cname not in room.containers:
          reward += self.INVALID_ACTION_PENALTY
          info["valid"] = False
          feedback = f"There is no {cname.replace('_', ' ')} here."
        elif not room.containers[cname].is_open:
          reward += self.INVALID_ACTION_PENALTY
          info["valid"] = False
          feedback = f"The {cname.replace('_', ' ')} is closed."
        else:
          self._inventory.remove(item)
          room.containers[cname].items.append(item)
          feedback = (
              f"You put the {item.replace('_', ' ')} into the"
              f" {cname.replace('_', ' ')}."
          )
          milestone_key = f"put_{item}_in_{cname}"
          if milestone_key not in self._achieved_milestones:
            self._achieved_milestones.add(milestone_key)
            reward += self.MILESTONE_REWARD
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = "Command format: put <item> in <container>"

    # 9. Examine <object>
    elif cmd.startswith("examine ") or cmd.startswith("x "):
      obj = self._normalize_name(cmd.split(" ", 1)[1])
      if obj in room.containers:
        c = room.containers[obj]
        lock_status = "locked" if c.is_locked else "unlocked"
        open_status = "open" if c.is_open else "closed"
        content_desc = (
            f"It contains: {', '.join(c.items)}."
            if c.is_open and c.items
            else ""
        )
        feedback = (
            f"The {obj.replace('_', ' ')} is {lock_status} and {open_status}."
            f" {content_desc}".strip()
        )
      elif obj in self._inventory:
        feedback = f"A {obj.replace('_', ' ')} in your inventory."
      elif obj in room.floor_items:
        feedback = f"A {obj.replace('_', ' ')} lying on the floor."
      else:
        reward += self.INVALID_ACTION_PENALTY
        info["valid"] = False
        feedback = f"You do not see {obj.replace('_', ' ')} here."

    else:
      # Unrecognized command
      reward += self.INVALID_ACTION_PENALTY
      info["valid"] = False
      feedback = (
          f"Unrecognized command '{clean_action}'. Valid verbs: go <dir>, look,"
          " inventory, open <container>, close <container>, unlock <container>"
          " with <key>, take <item>, put <item> in <container>, examine <item>."
      )

    # Check goal condition
    done = self._check_goal()
    if done:
      reward += self.GOAL_REWARD
      feedback += (
          "\nCongratulations! Task accomplished:"
          f" {self._task_data.get('instruction', '')}"
      )

    info["achieved_milestones"] = list(self._achieved_milestones)
    info["inventory"] = list(self._inventory)
    info["current_room"] = self._current_room

    return EnvStepResult(
        observation=feedback,
        reward=reward,
        done=done,
        info=info,
    )

  def _check_goal(self) -> bool:
    """Evaluate if goal condition in task is satisfied."""
    task_dict = getattr(self, "_task_data", self.task)
    goal = task_dict.get("goal", {})
    if not goal:
      return False

    goal_type = goal.get("type")
    if goal_type == "take":
      target_item = goal.get("item")
      return target_item in self._inventory
    elif goal_type == "put":
      target_item = goal.get("item")
      target_container = goal.get("container")
      for room in self._rooms.values():
        if target_container in room.containers:
          if target_item in room.containers[target_container].items:
            return True
      return False
    elif goal_type == "open":
      target_container = goal.get("container")
      for room in self._rooms.values():
        if target_container in room.containers:
          return room.containers[target_container].is_open
      return False
    return False

  @classmethod
  def from_dict(cls, env_args: dict[str, Any]) -> "HouseholdEnv":
    """Create HouseholdEnv from deserialized arguments dictionary."""
    task = env_args.get("task") or env_args.get("entry") or env_args
    max_steps = env_args.get("max_steps", 25)
    return cls(task=task, max_steps=max_steps)
