from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import PPO


def model_name_from(kwargs: dict):
    kwargs = kwargs.copy()
    kwargs["features_dim"] = kwargs["policy_kwargs"][
        "features_extractor_kwargs"
    ]["features_dim"]
    del kwargs["policy_kwargs"]

    string = ""
    for k, v in kwargs.items():
        if k == "n_steps":
            continue
        if string:
            string += "_"
        prefix = "".join([x[0] for x in k.split("_")])
        if k in [
            "learning_rate",
            "vf_coef",
            "ent_coef",
            "gae_lambda",
            "max_grad_norm",
        ]:
            value = f"{v:.2e}"
        else:
            value = v
        string += f"{prefix}{value}"

    return string


def train():
    n_envs = 8
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        damage_factor=1 / 10,
        truncate_if_no_improvement=True,
        render_mode=None,
    )
    # use this as a guide of max n_steps possible:
    # 3-frames stacking; 84x84 warped frames; float32 = 4 bytes
    # obs size is n_envs * 3 * 84 * 84 * 4 = 677376
    # approximately 677kB per venv observation
    model_kwargs = {
        "n_steps": 8192,
        "batch_size": 256,
        "learning_rate": 2.5e-4,
        "clip_range": 0.1,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "gamma": 0.99,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "features_extractor_kwargs": {
                "features_dim": 512,
            },
        },
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
    model_name = "optimized_nenvs16_framescaling"
    # model = PPO.load(
    #     f"models/{model_name}",
    #     env=venv,
    #     tensorboard_log="logs/cutman",
    # )
    model.learn(
        total_timesteps=10_000_000,
        callback=[MinDistanceCallback()],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    # model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
