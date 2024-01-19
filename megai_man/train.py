from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO


def train():
    venv = make_venv(
        n_envs=8,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=True,
        damage_factor=1,
        render_mode=None,
    )
    # model = PPO(
    #     policy="CnnPolicy",
    #     env=venv,
    #     learning_rate=lambda f: f * 2.5e-4,
    #     n_steps=128,
    #     batch_size=1024,
    #     n_epochs=4,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.3,
    #     ent_coef=0.01,
    #     tensorboard_log="logs/cutman",
    #     policy_kwargs={"optimizer_kwargs": {"eps": 1e-7}},
    #     verbose=0,
    #     seed=666,
    #     device="cuda",
    # )
    # model_kwargs = {
    #     "n_steps": 512,
    #     "batch_size": 4096,
    #     "learning_rate": 5e-4,
    #     "ent_coef": 1e-4,
    #     "clip_range": 0.2,
    #     "n_epochs": 10,
    #     "gae_lambda": 1.0,
    #     "max_grad_norm": 0.6,
    #     "vf_coef": 0.5,
    # }
    model_kwargs = {
        "n_steps": 128,
        "n_epochs": 4,
        "batch_size": 256,
        "learning_rate": 2.5e-4,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 1e-2,
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
    model_name = "dmg_factor_zoo_wo_sticky"
    # model = PPO.load(
    #     f"models/{model_name}",
    #     env=venv,
    #     # tensorboard_log="logs/cutman",
    # )
    model.learn(
        total_timesteps=2_000_000,
        callback=[MinDistanceCallback()],
        log_interval=1,
        tb_log_name=model_name,
        # reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}")


if __name__ == "__main__":
    train()
