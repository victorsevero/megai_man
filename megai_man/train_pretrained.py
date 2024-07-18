from pathlib import Path

import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

from megai_man.callbacks import (
    StageLoggingCallback,
    TrainingStatsLoggerCallback,
)
from megai_man.env import make_venv


def train():
    n_envs = 8
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        "render_mode": None,
        "no_enemies": False,
        "damage_punishment": 0.05,
        "forward_factor": 0.05,
        "backward_factor": 0.055,
        "time_punishment_factor": 0,
        "distance_only_on_ground": True,
    }
    venv = make_venv(**env_kwargs)

    model_name = "cutman_pretrained"
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
        )
        reset_num_timesteps = True

    total_timesteps = 20_000_000
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
    eval_callback = MaskableEvalCallback(
        # same env, just replacing n_envs with 1
        eval_venv,
        n_eval_episodes=1,
        eval_freq=250_000 // n_envs,
        best_model_save_path=f"models/{model_name}_best",
        verbose=0,
    )
    wandb.init(project="mega-man-1")
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
