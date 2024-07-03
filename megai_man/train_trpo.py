from pathlib import Path

from callbacks import StageLoggingCallback, TrainingStatsLoggerCallback
from env import make_venv
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def train():
    multi_input = True
    n_envs = 8
    frameskip = 4
    frame_stack = 3
    time_punishment_factor = 0
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        "screen": None,
        "frameskip": frameskip,
        "frame_stack": frame_stack,
        "truncate_if_no_improvement": True,
        "obs_space": "screen",
        "action_space": "multi_discrete",
        "crop_img": False,
        "invincible": False,
        "no_enemies": False,
        "render_mode": None,
        "damage_terminate": False,
        "fixed_damage_punishment": 0.05,
        "forward_factor": 0.05,
        "backward_factor": 0.055,
        "time_punishment_factor": time_punishment_factor,
        "multi_input": multi_input,
        "curriculum": False,
        "screen_rewards": False,
        "score_reward": 0,
        "distance_only_on_ground": True,
        "term_back_screen": True,
        "_enforce_subproc": False,
    }
    venv = make_venv(**env_kwargs)

    model_name = "trpo"
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = TRPO.load(
            f"models/{model_name}",
            env=venv,
            device="cuda",
            tensorboard_log=tensorboard_log,
        )
    else:
        model = TRPO(
            policy="MultiInputPolicy",
            env=venv,
            n_steps=512,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
        )
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
    eval_callback = EvalCallback(
        # same env, just replacing n_envs with 1
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
