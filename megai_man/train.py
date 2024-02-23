import torch
from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO


def train():
    n_envs = 1
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        sticky_prob=0.0,
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
        "batch_size": n * zoo_steps * n_envs // mini,
        "learning_rate": 2.5e-4,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 1e-2,
        "n_epochs": 4,
    }
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        # tensorboard_log="logs/cutman",
        policy_kwargs={"optimizer_class": torch.optim.RMSprop},
        verbose=0,
        seed=666,
        device="cuda",
        **model_kwargs,
    )
    model_name = f"dummy"
    # model = PPO.load(
    #     f"models/{model_name}",
    #     env=venv,
    #     tensorboard_log="logs/cutman",
    # )
    model.learn(
        total_timesteps=n * zoo_steps * n_envs,
        callback=[MinDistanceCallback()],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
