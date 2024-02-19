from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO


def train():
    n_envs = 8
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        damage_factor=1 / 10,
        truncate_if_no_improvement=True,
        obs_space="ram",
        render_mode=None,
    )
    n_steps = 2048
    mini = 1
    model_kwargs = {
        "n_steps": n_steps,
        "batch_size": n_steps * n_envs // mini,
        "learning_rate": 1e-4,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 1e-2,
        "n_epochs": 10,
    }
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=venv,
    #     tensorboard_log="logs/cutman",
    #     verbose=0,
    #     seed=666,
    #     device="cuda",
    #     **model_kwargs,
    # )
    model_name = f"envfix4_ram_entropy"
    model = PPO.load(
        f"models/{model_name}",
        env=venv,
        tensorboard_log="logs/cutman",
    )
    model.learn(
        total_timesteps=10_000_000,
        callback=[MinDistanceCallback()],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
