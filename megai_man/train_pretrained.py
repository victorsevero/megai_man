from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

import wandb
from megai_man.callbacks import (
    CurriculumNoEnemiesCallback,
    MaskableEvalCallback,
    StageLoggingCallback,
    TrainingStatsLoggerCallback,
)
from megai_man.env import make_venv


def train():
    n_envs = 8
    dmg = 0.05
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        "render_mode": None,
        "no_enemies": False,
        "damage_punishment": dmg,
        "forward_factor": 0.05,
        "backward_factor": 0.055,
        "time_punishment_factor": 0,
        "distance_only_on_ground": True,
    }
    venv = make_venv(**env_kwargs)

    model_name = [
        "cutman_pretrained",
        "_noTermBackScreen",
        f"_dmg{dmg}" if dmg != 0.05 else "",
        "_gamma95",
        "_10spikepunish",
        "_enemies_curriculum",
    ]
    model_name = "".join(model_name)
    tensorboard_log = "logs/cutman"

    if Path(f"models/{model_name}.zip").exists():
        model = MaskablePPO.load(
            f"models/{model_name}",
            env=venv,
            device="cuda",
            tensorboard_log=tensorboard_log,
        )
        reset_num_timesteps = False
    else:
        model = MaskablePPO.load(
            f"models/no_enemies_complete",
            env=venv,
            device="cuda",
            tensorboard_log=tensorboard_log,
            gamma=0.95,
        )
        reset_num_timesteps = True

    total_timesteps = 10_000_000
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path="checkpoints/",
        name_prefix=model_name,
    )
    eval_venv = make_venv(
        **{
            **env_kwargs,
            "n_envs": 1,
            "screen": None,
            "render_mode": None,
        }
    )

    first_enemy_screen = None
    # first_enemy_screen = 0
    # venv.env_method("set_screen_with_enemies", first_enemy_screen)
    # eval_venv.env_method("set_screen_with_enemies", first_enemy_screen)

    eval_callback = MaskableEvalCallback(
        # same env, just replacing n_envs with 1
        eval_venv,
        callback_after_eval=CurriculumNoEnemiesCallback(first_enemy_screen),
        n_eval_episodes=1,
        eval_freq=100_000 // n_envs,
        best_model_save_path=f"models/{model_name}_best",
        verbose=0,
    )
    wandb.init(project="mega-man-1", sync_tensorboard=True, save_code=True)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            StageLoggingCallback(),
            TrainingStatsLoggerCallback(),
            checkpoint_callback,
            eval_callback,
            WandbCallback(),
        ],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
