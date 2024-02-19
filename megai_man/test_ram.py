from env import make_venv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_policy_details(model, env, deterministic=True):
    obs = env.reset()
    done = False
    current_length = 0
    while not done:
        actions, _ = model.predict(
            obs,
            state=None,
            deterministic=deterministic,
        )
        action = env.unwrapped.envs[0].unwrapped.get_action_meaning(actions[0])
        obs, rewards, dones, infos = env.step(actions)
        done = dones[0]
        current_length += 1
        print(f"Action: {action}")
        print(f"Reward: {rewards[0]}")
        if "terminal_observation" in infos[0]:
            del infos[0]["terminal_observation"]
        print(f"Infos: {infos[0]}\n")
    print()


def test():
    venv = make_venv(
        n_envs=1,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        truncate_if_no_improvement=False,
        obs_space="ram",
        render_mode="human",
        record=".",
    )

    model_name = "envfix4_ram_entropy"
    model = PPO.load(f"models/{model_name}", env=venv)

    deterministic = True

    # rewards, lengths = evaluate_policy(
    #     model=model,
    #     env=venv,
    #     n_eval_episodes=1,
    #     deterministic=True,
    #     return_episode_rewards=deterministic,
    # )
    # print(f"Episode length: {lengths[0]}; Episode reward: {rewards[0]}")

    evaluate_policy_details(model, venv, deterministic=deterministic)


if __name__ == "__main__":
    test()