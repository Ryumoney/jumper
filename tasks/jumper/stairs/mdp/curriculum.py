"""Stair-height curriculum: promote a robot that reached the landing.

The shared `TerrainLevels` promotes on velocity-tracking error and waits for the
command ladder to finish first. Neither fits here: this task commands a narrow
forward range with no command ladder, and on a staircase the question is simply
whether the robot got to the top. A hexapod can shuffle on flat ground without
handling it -- the objection `TerrainLevels` answers -- but it cannot gain the
landing's x without climbing every step in between, so forward progress along
the stairs is an honest bar here.
"""

from __future__ import annotations

from typing import Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def stairs_levels(
    env: Any,
    env_ids: torch.Tensor,
    promote_distance: float,
    demote_fraction: float = 0.4,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Move each finished episode's robot one row up or down the difficulty ladder.

    Args:
        promote_distance: forward progress (m, world x from the spawn origin) that
            counts as having reached the landing.
        demote_fraction: below this fraction of the distance it was asked for --
            `promote_distance`, or less if its forward command over the episode
            asked for less -- the robot is moved to lower stairs.
        command_name: the velocity command whose forward axis sets that ask.
    """
    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]
    progress = asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = progress > promote_distance
    command = env.command_manager.get_command(command_name)
    asked = (command[env_ids, 0].clamp(min=0.0) * env.max_episode_length_s).clamp(
        max=promote_distance
    )
    move_down = (progress < demote_fraction * asked) & ~move_up
    # The first reset happens before any step: positions are spawn poses, not
    # outcomes, and must not move anyone.
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)
    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    return {"mean_level": levels.mean(), "max_level": levels.max()}
