from pathlib import Path

import numpy as np
from callbacks import StageLoggingCallback
from env import make_venv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from wandb.integration.sb3 import WandbCallback

import wandb


def train():
    AlgoClass = MaskablePPO
    n_envs = 8
    dmg = 0.05
    no_boss = False
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan-boss",
        "render_mode": None,
        "damage_punishment": dmg,
        "no_boss": no_boss,
    }
    venv = make_venv(**env_kwargs)
    n_steps = 1024
    batch_size = 128
    model_kwargs = {
        "learning_rate": 2.5e-4,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 4,
        # future_horizon = frame_skip * frame_time / (1 - gamma) =
        # 4 * (1 / 60) / (1 - 0.995) = 13.3 seconds in real game time
        # that's the future horizon that our agent is capable of planning for
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 1e-4,
        "vf_coef": 1,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"share_features_extractor": False},
    }

    model_name = "sevs_boss"
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = AlgoClass.load(
            f"models/{model_name}",
            env=venv,
            device="cuda",
            tensorboard_log=tensorboard_log,
        )
    else:
        model = AlgoClass(
            policy="CnnPolicy",
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_kwargs,
        )
    total_timesteps = 20_000_000
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
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
    callback_on_best = StopTrainingOnRewardThreshold(
        # stop when boss killed without losing HP. 3 dmg per hit, boss has 28HP
        reward_threshold=dmg * np.ceil(28 / 3),
        verbose=1,
    )
    eval_callback = MaskableEvalCallback(
        # same env, just replacing n_envs with 1
        eval_venv,
        n_eval_episodes=1,
        eval_freq=100_000 // n_envs,
        callback_on_new_best=callback_on_best,
        best_model_save_path=f"models/{model_name}_best",
        verbose=0,
    )
    wandb.init(project="mega-man-1", sync_tensorboard=True, save_code=True)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            StageLoggingCallback(no_boss),
            checkpoint_callback,
            eval_callback,
            WandbCallback(),
        ],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
