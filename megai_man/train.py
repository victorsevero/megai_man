from pathlib import Path

from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn


def train():
    n_envs = 8
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        frameskip=8,
        frame_stack=3,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="multi_discrete",
        crop_img=True,
        render_mode=None,
        damage_terminate=False,
        fixed_damage_punishment=2,
        forward_factor=0.5,
        backward_factor=0.6,
    )
    model_kwargs = {
        "clip_range": 0.25,
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
            "share_features_extractor": True,
        },
        "gae_lambda": 0.9,
        "n_steps": 2048,
        "batch_size": n_envs * 2048,
        "n_epochs": 8,
        # future_horizon = frame_skip * frame_time / (1 - gamma) =
        # 4 * (1 / 60) / (1 - 0.99) = 6.67 seconds in real game time
        # that's the future horizon that our agent is capable of planning for
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "ent_coef": 1e-2,
        # "normalize_advantage": False,
    }

    model_name = "andrychowicz_1minibatch_share_fe_nepochs8_ecoef1e-2_relu_small_rewards"
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = PPO.load(
            f"models/{model_name}",
            env=venv,
            device="cuda",
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
