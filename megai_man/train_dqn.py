from pathlib import Path

from callbacks import MinDistanceCallback
from env import make_venv
from policy import FineTunedArch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def train():
    n_envs = 1
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        sticky_prob=0.0,
        frameskip=4,
        damage_terminate=False,
        damage_factor=1 / 10,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="discrete",
        render_mode=None,
    )
    model_kwargs = {
        "buffer_size": 100_000,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "learning_starts": 100_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 1e-2,
        # "policy_kwargs": {"features_extractor_class": FineTunedArch},
    }
    model_name = "dqn_zoo_fs4"
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = DQN.load(
            f"models/{model_name}",
            env=venv,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = DQN(
            policy="CnnPolicy",
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_kwargs,
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
