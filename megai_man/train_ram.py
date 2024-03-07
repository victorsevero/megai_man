from pathlib import Path

from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def train():
    n_envs = 8
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        frameskip=1,
        truncate_if_no_improvement=True,
        obs_space="ram",
        action_space="multi_discrete",
        render_mode=False,
        damage_terminate=False,
        fixed_damage_punishment=1,
        forward_factor=0.25,
        backward_factor=0.3,
    )
    n_steps = 2048
    mini = 1
    # model_kwargs = {
    #     "n_steps": n_steps,
    #     "batch_size": n_steps * n_envs // mini,
    #     "learning_rate": 1e-4,
    #     "clip_range": 0.2,
    #     "vf_coef": 0.5,
    #     "ent_coef": 1e-2,
    #     "n_epochs": 10,
    # }
    model_name = "ram2_smaller_rewards"
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
            # **model_kwargs,
        )
    total_timesteps = 10_000_000
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // n_envs // 10,
        save_path="checkpoints/",
        name_prefix=model_name,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[MinDistanceCallback(), checkpoint_callback],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
