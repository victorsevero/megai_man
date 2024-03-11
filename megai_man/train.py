from pathlib import Path

from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def train():
    n_envs = 8
    frameskip = 4
    frame_stack = 2
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        frameskip=frameskip,
        frame_stack=frame_stack,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="multi_discrete",
        crop_img=True,
        invincible=True,
        render_mode=None,
        damage_terminate=False,
        fixed_damage_punishment=1,
        forward_factor=0.1,
        backward_factor=0.11,
    )
    model_kwargs = {
        "learning_rate": 2.5e-4,
        "n_steps": 64,
        "batch_size": 64,
        "n_epochs": 1,
        # future_horizon = frame_skip * frame_time / (1 - gamma) =
        # 4 * (1 / 60) / (1 - 0.995) = 13.3 seconds in real game time
        # that's the future horizon that our agent is capable of planning for
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 1e-2,
    }

    model_name = (
        "sevs"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        f"_epochs{model_kwargs['n_epochs']}"
        f"_gamma{model_kwargs['gamma']}"
        f"_gae{model_kwargs['gae_lambda']}"
        f"_clip{model_kwargs['clip_range']}"
        f"_norm{'yes' if model_kwargs['normalize_advantage'] else 'no'}"
        f"_ecoef{model_kwargs['ent_coef']:.0e}"
        "_"  # for separating env parameters
        f"_fs{frameskip}"
        f"_stack{frame_stack}"
        "_crop224"
        # "_death2"
        "_smallest_rewards"
        # "_trunc6min"
        "_trunc1minnoprog"
        "_spikefix2"
        "_INVINCIBLE"
    )
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
        save_freq=1_000_000 // n_envs,
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
