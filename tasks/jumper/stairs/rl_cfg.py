"""Hyper-parameters for jumper.stairs: jumper.tripod's, with a shorter default run.

The gait and network are tripod's, so its numbers are the starting point; only the
iteration budget differs, since the staircase curriculum is the whole run here.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.stairs/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.stairs",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        # **Watch this one on rough terrain.** Measured on a resumed rough run,
        # `Policy/mean_std` climbed 0.239 -> 0.429 over ~300 iterations and
        # settled there, and everything that fell with it was a precision term:
        #
        #     tripod_gait          0.716 -> 0.567   -21%
        #     action_rate penalty -0.464 -> -0.821  +77%
        #     track_linear         1.552 -> 1.549    -0.2%
        #     upright              0.962 -> 0.978    +1.7%
        #     episode length         972 ->   997    +2.5%
        #
        # The coarse objectives held or improved while the fine ones collapsed,
        # and `action_rate_l2` -- which is literally the squared step-to-step
        # action difference, i.e. a noise meter -- grew by very nearly the factor
        # sigma^2 grew by (1.77 against 1.93). That is injected action noise, not
        # a locomotion failure, and 0.01 is its coefficient.
        #
        # It was **not** lowered, because `Train/mean_reward` is collected under
        # that noise and so falls mechanically whether or not the deployed
        # deterministic policy got worse. Establish that first -- replay two
        # checkpoints with `play.py`, which is deterministic -- then tune.
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=24,
        max_iterations=3_000,
    )


def runner_cls() -> type:
    """mjlab's velocity runner, which logs extra velocity tracking metrics, with
    the curriculum levels added to its checkpoints (`common/runner.py`)."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner

def play_status():
    """The line `scripts/play.py` prints while replaying: achieved speed against
    commanded. Shared by the four velocity tasks; see `common/play.py`."""
    from ..common.play import velocity_status

    return velocity_status
