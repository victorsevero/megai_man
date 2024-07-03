from pathlib import Path

from env import make_venv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from torch import nn

from megai_man.callbacks import (
    StageLoggingCallback,
    TrainingStatsLoggerCallback,
)


def train():
    n_envs = 8
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        "screen": None,
        "frameskip": 4,
        "truncate_if_no_improvement": True,
        "obs_space": "ram",
        "action_space": "multi_discrete",
        "render_mode": None,
        "damage_terminate": False,
        "fixed_damage_punishment": 0.05,
        "forward_factor": 0.05,
        "backward_factor": 0.055,
        "distance_only_on_ground": True,
        "term_back_screen": True,
        "_enforce_subproc": False,
    }
    venv = make_venv(**env_kwargs)
    n_steps = 512
    batch_size = 64
    model_kwargs = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "clip_range": 0.2,
        "vf_coef": 1,
        "ent_coef": 1e-4,
        "n_epochs": 10,
        "policy_kwargs": {
            "net_arch": [256, 256],
            "share_features_extractor": False,
            "activation_fn": nn.ReLU,
        },
    }
    model_name = (
        "ram2.3"
        f"_steps{n_steps}"
        f"_batch{batch_size}"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        f"_epochs{model_kwargs['n_epochs']}"
        f"_clip{model_kwargs['clip_range']}"
        f"_ecoef{model_kwargs['ent_coef']:.0e}"
        "_nn256x2"
        "_relu"
        f"{'_twoFEs' if not model_kwargs['policy_kwargs']['share_features_extractor'] else ''}"
        "_"
        "_rews0.05+screen1_dmg"
        f"{env_kwargs['fixed_damage_punishment']}"
        "_groundrew"
        "_termbackscreen2"
    )
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = PPO.load(
            f"models/{model_name}",
            env=venv,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_kwargs,
        )
    total_timesteps = 20_000_000
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // n_envs // 10,
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
    eval_callback = EvalCallback(
        eval_venv,
        n_eval_episodes=1,
        eval_freq=250_000 // n_envs,
        best_model_save_path=f"models/{model_name}_best",
        verbose=0,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            StageLoggingCallback(),
            TrainingStatsLoggerCallback(),
            checkpoint_callback,
            eval_callback,
        ],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
