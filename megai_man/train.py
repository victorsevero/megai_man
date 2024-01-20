from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO


def train():
    venv = make_venv(
        n_envs=8,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        damage_factor=1 / 28,
        truncate_if_no_improvement=True,
        render_mode=None,
    )
    model_kwargs = {
        "n_steps": 128,
        "n_epochs": 4,
        "batch_size": 256,
        "learning_rate": 2.5e-4,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 1e-3,
    }
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        tensorboard_log="logs/cutman",
        verbose=0,
        seed=666,
        device="cuda",
        **model_kwargs,
    )
    model_name = "zoo_ecoef0.001_wo_sticky_wo_dmgterm_df1to28"
    # model = PPO.load(
    #     f"models/{model_name}",
    #     env=venv,
    #     tensorboard_log="logs/cutman",
    # )
    model.learn(
        total_timesteps=1_000_000,
        callback=[MinDistanceCallback()],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}")


if __name__ == "__main__":
    train()
