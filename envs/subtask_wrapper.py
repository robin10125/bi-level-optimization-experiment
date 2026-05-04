"""
Gymnasium wrapper that injects shaped rewards for a selected subtask type.

Subtask types:
  none -  no shaping (transparent pass-through)
  pickup_key - one-time bonus when the agent first picks up a key
  open_door - one-time bonus when the agent first opens a door
  explore - small bonus for each previously-unvisited cell
                (scaled so the total max bonus across the grid ≈ SHAPING_COEF)

Episode state (picked-up flags, visited set, open-door set) is reset on
every call to reset(), so the wrapper is safe across episode boundaries.
"""

from __future__ import annotations

import gymnasium
from minigrid.core.world_object import Door, Key

import config
from curriculum.subtask_selector import SUBTASK_TYPES


class SubtaskWrapper(gymnasium.Wrapper):
    """Wraps any MiniGrid env and injects per-step shaped rewards."""

    def __init__(self, env: gymnasium.Env, subtask: str):
        assert subtask in SUBTASK_TYPES, f"Unknown subtask: {subtask!r}"
        super().__init__(env)
        self.subtask = subtask
        self._reset_episode_state()

    #########################################################################
    # Gymnasium API
    #########################################################################

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode_state()
        self._open_door_positions = self._current_open_doors()
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self._shaped_reward()
        return obs, reward, terminated, truncated, info

    #########################################################################
    # Shaped reward dispatch
    #########################################################################
    def _shaped_reward(self) -> float:
        if self.subtask == "none":
            return 0.0
        elif self.subtask == "pickup_key":
            return self._key_bonus()
        elif self.subtask == "open_door":
            return self._door_bonus()
        elif self.subtask == "explore":
            return self._explore_bonus()
        return 0.0

    def _key_bonus(self) -> float:
        if self._picked_up_key:
            return 0.0
        if isinstance(self.unwrapped.carrying, Key):
            self._picked_up_key = True
            return config.SUBTASK_SHAPING_COEF
        return 0.0

    def _door_bonus(self) -> float:
        now_open  = self._current_open_doors()
        new_opens = now_open - self._open_door_positions
        self._open_door_positions = now_open
        return config.SUBTASK_SHAPING_COEF if new_opens else 0.0

    def _explore_bonus(self) -> float:
        env = self.unwrapped
        pos = tuple(env.agent_pos)
        if pos not in self._visited_cells:
            self._visited_cells.add(pos)
            n_interior = (env.width - 2) * (env.height - 2)
            return config.SUBTASK_SHAPING_COEF / n_interior
        return 0.0

    #########################################################################
    # Helpers
    #########################################################################

    def _reset_episode_state(self) -> None:
        self._picked_up_key       = False
        self._visited_cells: set  = set()
        self._open_door_positions: set = set()

    def _current_open_doors(self) -> set[tuple[int, int]]:
        env = self.unwrapped
        result = set()
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if isinstance(obj, Door) and obj.is_open:
                    result.add((x, y))
        return result
