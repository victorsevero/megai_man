from env import make_venv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def test():
    venv = make_venv(
        n_envs=1,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        render_mode="human",
        # record=".",
    )

    model_name = "full_cutman"
    model = PPO.load(f"models/{model_name}", env=venv)

    rewards, lengths = evaluate_policy(
        model=model,
        env=venv,
        n_eval_episodes=1,
        deterministic=True,
        return_episode_rewards=True,
    )
    print(f"Episode length: {lengths[0]}; Episode reward: {rewards[0]}")


if __name__ == "__main__":
    test()
