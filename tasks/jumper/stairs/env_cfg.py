"""Environment config for jumper.stairs.

Built on the same velocity skeleton and tripod gait prior as `jumper.tripod`, with
three changes, each of which is a decision of this task:

1. **Ground.** A terrain generator of `StraightStairsTerrainCfg` patches, one
   difficulty per row, instead of the plane. `terrain_scan=True` keeps the
   critic's height scan (the actor stays proprioceptive, as on the robot).
2. **Command.** Forward only, at a walking pace, with no command ladder: the
   staircase runs along world +x and the robot spawns facing roughly along it.
3. **Curriculum.** `stairs_levels`: a robot that reaches the landing moves to
   higher steps, one that covers less than 40% of the way moves down.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.mdp.rewards import stance_foot_ground_gap, standing_foot_load
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg
from ..tripod.mdp.phase import GAIT_FREQ_HZ, phase_clock
from ..tripod.mdp.rewards import TRIPOD_A, TRIPOD_B, stance_foot_load, tripod_gait
from .mdp.curriculum import stairs_levels
from .terrain import StraightStairsTerrainCfg

PHASE_OBS_NAME = "gait_phase"

# ── The staircase ──────────────────────────────────────────────────────────
# Rise per step from row 0 to the last row. 0.005 m keeps row 0 nearly flat so
# the gait is learned before any obstacle. The top was 0.06 m (56% of standing
# height) and saturated: iteration 2200 of that run (Colab, 4096 envs) reached
# the landing in 3 of 3 tries at 0.06 m and 28 of 30 across all ten rows (0.25
# m/s forward, training observation noise, MacBook native replay, 2026-11-23).
# 0.10 m is 94% of standing height (STAND_Z), so it may lie past what the body
# can clear; the row where the curriculum settles is then the measured limit.
STEP_HEIGHT_RANGE = (0.005, 0.10)
TREAD = 0.30          # m; the robot's fore-aft stance is 0.331 m
# Six steps, not eight: with the heading held up the stairs, a perfect walker
# under this task's commands reaches the landing in 71% of 20 s episodes at six
# steps against 49% at eight (3000 simulated episodes of ideal command tracking,
# MacBook, 2026-11-15). Promotion is the curriculum's only way up, so an
# unreachable landing keeps the run on the flattest rows.
NUM_STEPS = 6
RUN_UP = 1.0          # m of flat floor before the first riser; spawn at its middle
PATCH = (8.0, 2.0)    # m (length along the stairs, width)
# 16 rungs keep the rise between neighbouring rows at 0.63 cm over the wider
# range -- the 0.61 cm the 0.06 m run climbed through; 10 would make it 1.06 cm.
NUM_ROWS = 16         # difficulty rungs
NUM_COLS = 8          # patches per rung; more columns = more robots per rung
BORDER = 2.0          # m of flat border around the grid; the replay garden dresses it

# Promotion: the base has passed the last riser, i.e. stands on the landing.
PROMOTE_DISTANCE = RUN_UP / 2.0 + NUM_STEPS * TREAD

# ── The command ────────────────────────────────────────────────────────────
# Forward up to 0.30 m/s: over a 20 s episode up to 6 m of asked-for travel,
# against the 2.3 m needed to reach the landing. The range has to include 0 --
# the operator's centred stick rests there (`operator.spans` raises otherwise) --
# so slow commands are judged against what they asked for, in `stairs_levels`.
LIN_VEL_X = (0.0, 0.30)
LIN_VEL_Y = (-0.05, 0.05)
ANG_VEL_Z = (-0.2, 0.2)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
    """
    cfg = velocity_env_cfg(
        controls=Path(__file__).parent / "controls.yaml",
        asset=asset,
        play=play,
        gait_reward=tripod_gait,
        gait_name="tripod",
        action_rate_weight=-0.1,
        pose_weight=1.0,
        # Tripod's 0.025. A 1500-iteration Colab run (T4, 2026-11-15) at 0.05 learned
        # to stand: commanded 0.10 m/s forward it moved -0.0009 m/s. Both foot
        # terms charge only a foot that travels or lands, so a higher target taxes
        # every step and leaves standing still free. Walk first, then raise it.
        foot_target_height=0.025,
        command_lin_ceiling=0.5,
        command_ang_ceiling=0.75,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        terrain_scan=True,
    )

    # ── Tripod gait prior, as in jumper.tripod ───────────────────────────
    for group in ("actor", "critic"):
        cfg.observations[group].terms[PHASE_OBS_NAME] = ObservationTermCfg(
            func=phase_clock,
            history_length=0,
            params={
                "freq_hz": GAIT_FREQ_HZ,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        )
    cfg.rewards["tripod_gait"].params["freq_hz"] = GAIT_FREQ_HZ
    cfg.rewards["stance_load"] = RewardTermCfg(
        func=stance_foot_load,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "force_threshold": 5.0,
            "command_threshold": 0.05,
            "freq_hz": GAIT_FREQ_HZ,
        },
    )
    cfg.rewards["foot_clearance"].weight = -4.0
    cfg.rewards["foot_swing_height"].weight = -3.0
    _pose_std = cfg.rewards["pose"].params["std_standing"]
    cfg.rewards["pose"].params["std_standing"] = {k: 2.0 * v for k, v in _pose_std.items()}
    cfg.rewards["stance_load_stand"] = RewardTermCfg(
        func=standing_foot_load,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "force_threshold": 3.0,
            "command_threshold": 0.05,
        },
    )
    cfg.rewards["stance_ground_gap"] = RewardTermCfg(
        func=stance_foot_ground_gap,
        weight=-0.5,
        params={
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "groups": (TRIPOD_A, TRIPOD_B),
            "freq_hz": GAIT_FREQ_HZ,
            "max_gap": 0.05,
            "command_threshold": 0.05,
        },
    )

    # ── 1. Ground: the generated staircase ──────────────────────────────
    terrain = cfg.scene.terrain
    assert terrain is not None
    terrain.terrain_type = "generator"
    terrain.terrain_generator = TerrainGeneratorCfg(
        seed=0,
        curriculum=True,
        size=PATCH,
        border_width=BORDER,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        color_scheme="none",
        sub_terrains={
            "stairs": StraightStairsTerrainCfg(
                size=PATCH,
                step_height_range=STEP_HEIGHT_RANGE,
                tread=TREAD,
                num_steps=NUM_STEPS,
                run_up=RUN_UP,
                # The garden look is for watching only: ray sensors read its
                # decoration as ground (see terrain.py), so training keeps the
                # bare staircase.
                garden=play,
                garden_margin=BORDER,
            ),
        },
    )
    # Everyone starts on the flattest row; replay spreads across all of them.
    terrain.max_init_terrain_level = NUM_ROWS - 1 if play else 0

    # Spawn near the patch's origin, facing up the stairs.
    reset = cfg.events["reset_base"].params["pose_range"]
    reset.update({"x": (-0.1, 0.1), "y": (-0.4, 0.4), "yaw": (-0.3, 0.3)})

    # ── 2. Command: forward, no ladder ──────────────────────────────────
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = LIN_VEL_X
    twist.ranges.lin_vel_y = LIN_VEL_Y
    twist.ranges.ang_vel_z = ANG_VEL_Z
    twist.rel_standing_envs = 0.05
    # Every robot steers back to facing up the stairs (world +x). The inherited
    # setting aimed 30% of robots at a random heading in +/-pi and gave the rest a
    # random yaw rate, so a perfect walker reached the landing in only 16.5% of
    # episodes (same simulation as `NUM_STEPS`). The operator still overrides yaw
    # in replay: its command is written after the heading controller's.
    twist.heading_command = True
    twist.ranges.heading = (0.0, 0.0)
    twist.rel_heading_envs = 1.0

    # ── 3. Curriculum: stair height only ────────────────────────────────
    cfg.curriculum = {
        "stairs": CurriculumTermCfg(
            func=stairs_levels,
            params={"promote_distance": PROMOTE_DISTANCE},
        ),
    }

    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)
    return cfg
