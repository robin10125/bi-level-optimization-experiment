"""
MiniGrid environment constructed from a parameter dict.

Template space defines the axes the curriculum generator can vary.
Each params dict must contain all keys from TEMPLATE_SPACE.
"""

from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv


# Defines the discrete axes and their ordered values.
# CurriculumGenerator indexes into these lists.
TEMPLATE_SPACE: dict[str, list] = {
    "grid_size": [5, 6, 7, 8, 9],
    "num_rooms": [1, 2, 4],
    "num_keys":  [0, 1, 2],
    "num_doors": [0, 1, 2],
    "num_lava":  [0, 1, 3, 5],
    "goal_type": ["reach_goal", "pickup_ball"],
}


def _usable_cells(params: dict) -> int:
    """
    Return the number of interior cells available for agent/object placement.

    Room-divider walls and placed doors consume interior cells, so small layouts
    can become overfilled even if the raw grid area looks large enough.
    """
    size = params["grid_size"]
    usable = (size - 2) * (size - 2)
    num_rooms = params["num_rooms"]

    if num_rooms == 2:
        # Vertical wall fills all interior rows except one opening. A door placed
        # in that opening still consumes the remaining free cell.
        usable -= (size - 3) + min(params["num_doors"], 1)
    elif num_rooms == 4:
        # Cross wall blocks (n - 4) cells vertically and (n - 4) horizontally,
        # with the center cell shared by both lines.
        usable -= (2 * size - 9)
        usable -= min(params["num_doors"], 4)

    return usable


def is_valid_task(params: dict) -> bool:
    """
    Return whether a template leaves enough free cells for all placed objects
    plus the agent spawn.
    """
    required = 1 + params["num_keys"] + params["num_lava"] + 1
    return required <= _usable_cells(params)


class TemplateEnv(MiniGridEnv):
    """
    A MiniGrid environment fully specified by a params dict.
    Supports single-room, two-room, and four-room layouts with configurable
    keys, locked doors, lava, and two goal types.
    """

    def __init__(self, params: dict, render_mode: str | None = None):
        self.params = params
        self._target_ball_color: str | None = None

        size = params["grid_size"]

        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "complete the task"),
            width=size,
            height=size,
            max_steps=4 * size * size,
            render_mode=render_mode,
        )

    #########################################################################
    # Grid generation
    #########################################################################

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        num_rooms = self.params["num_rooms"]
        if num_rooms == 1:
            self._build_single_room(width, height)
        elif num_rooms == 2:
            self._build_two_rooms(width, height)
        else:
            self._build_four_rooms(width, height)

        # Place goal object last so lava/keys don't block it
        if self.params["goal_type"] == "reach_goal":
            self.place_obj(Goal(), max_tries=100)
        else:
            color = self._rand_elem(COLOR_NAMES)
            self._target_ball_color = color
            self.place_obj(Ball(color), max_tries=100)

        for _ in range(self.params["num_lava"]):
            try:
                self.place_obj(Lava(), max_tries=100)
            except RecursionError:
                break  # grid is full; skip remaining lava tiles

        self.place_agent()
        if self.params["goal_type"] == "reach_goal":
            self.mission = "reach the green goal square"
        else:
            self.mission = "pick up the ball"

    def _build_single_room(self, width: int, height: int) -> None:
        """Open room. Keys placed as collectibles (no locked doors)."""
        for _ in range(self.params["num_keys"]):
            self.place_obj(Key(self._rand_elem(COLOR_NAMES)), max_tries=100)

    def _build_two_rooms(self, width: int, height: int) -> None:
        """Two rooms separated by a vertical wall. Up to one door."""
        mid = width // 2

        # Choose opening position before drawing wall
        opening_y = self._rand_int(1, height - 1)

        for y in range(1, height - 1):
            if y != opening_y:
                self.grid.set(mid, y, Wall())

        num_doors = min(self.params["num_doors"], 1)
        num_keys  = self.params["num_keys"]

        if num_doors > 0:
            locked = num_keys > 0
            color  = self._rand_elem(COLOR_NAMES)
            self.grid.set(mid, opening_y, Door(color, is_locked=locked))
            if locked:
                self.place_obj(Key(color), max_tries=100)
                for _ in range(num_keys - 1):
                    self.place_obj(Key(self._rand_elem(COLOR_NAMES)), max_tries=100)
        else:
            for _ in range(num_keys):
                self.place_obj(Key(self._rand_elem(COLOR_NAMES)), max_tries=100)

    def _build_four_rooms(self, width: int, height: int) -> None:
        """Four rooms with a cross wall. Openings in each wall segment."""
        mid_x, mid_y = width // 2, height // 2

        # One opening per wall segment, chosen before drawing walls
        openings: set[tuple[int, int]] = {
            (mid_x, self._rand_int(1, mid_y)),
            (mid_x, self._rand_int(mid_y + 1, height - 1)),
            (self._rand_int(1, mid_x), mid_y),
            (self._rand_int(mid_x + 1, width - 1), mid_y),
        }

        for x in range(1, width - 1):
            if (x, mid_y) not in openings:
                self.grid.set(x, mid_y, Wall())

        for y in range(1, height - 1):
            if (mid_x, y) not in openings:
                self.grid.set(mid_x, y, Wall())

        num_doors = min(self.params["num_doors"], len(openings))
        num_keys  = self.params["num_keys"]
        locked    = num_keys > 0
        door_color = self._rand_elem(COLOR_NAMES)

        for i, (ox, oy) in enumerate(sorted(openings)):
            if i < num_doors:
                self.grid.set(ox, oy, Door(door_color, is_locked=locked))
            # else: leave as open passage

        if locked and num_doors > 0:
            for _ in range(num_keys):
                self.place_obj(Key(door_color), max_tries=100)

    #########################################################################
    # Step override for pickup_ball goal type
    #########################################################################

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        if (
            self.params["goal_type"] == "pickup_ball"
            and not terminated
            and self.carrying is not None
            and isinstance(self.carrying, Ball)
            and self.carrying.color == self._target_ball_color
        ):
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
