from callbacks import MinDistanceCallback
from env import make_venv
from stable_baselines3 import DQN


def train():
    n_envs = 8
    venv = make_venv(
        n_envs=n_envs,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        damage_factor=1 / 10,
        truncate_if_no_improvement=True,
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
    }
    # model = DQN(
    #     policy="CnnPolicy",
    #     env=venv,
    #     tensorboard_log="logs/cutman",
    #     verbose=0,
    #     seed=666,
    #     device="cuda",
    #     **model_kwargs,
    # )
    model_name = "dqn_zoo_envfix4"
    model = DQN.load(
        f"models/{model_name}",
        env=venv,
        tensorboard_log="logs/cutman",
    )
    model.learn(
        total_timesteps=5_000_000,
        callback=[MinDistanceCallback()],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
