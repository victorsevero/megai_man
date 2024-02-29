from pathlib import Path

from callbacks import MinDistanceCallback
from env import make_venv
from policy import FineTunedArch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def train():
    n_envs = 8
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        sticky_prob=0.0,
        frameskip=4,
        damage_terminate=False,
        damage_factor=1 / 10,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="multi_discrete",
        render_mode=None,
    )
    # use this as a guide of max n_steps possible:
    # 3-frames stacking; 84x84 warped frames; float32 = 4 bytes
    # obs size is n_envs * 3 * 84 * 84 * 4 = 677376
    # approximately 677kB per venv observation
    n = 8
    zoo_steps = 128
    mini = 1
    model_kwargs = {
        "n_steps": n * zoo_steps,
        # "batch_size": n * zoo_steps * n_envs // mini,
        "batch_size": 64,
        "learning_rate": 2.5e-4,
        "gamma": 0.999,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 1e-2,
        "n_epochs": 4,
        "policy_kwargs": {"features_extractor_class": FineTunedArch},
    }

    model_name = "finetuned_arch_epochs4"
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = PPO.load(
            f"models/{model_name}",
            env=venv,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = PPO(
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
