from env import make_venv
from stable_baselines3 import DQN


def evaluate_policy_details(model, env):
    obs = env.reset()
    done = False
    current_length = 0
    while not done:
        actions, _ = model.predict(obs, state=None, deterministic=True)
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
        truncate_if_no_improvement=True,
        action_space="discrete",
        render_mode="human",
        record=".",
    )

    model_name = "dqn_zoo_envfix4"
    model = DQN.load(f"models/{model_name}", env=venv)

    evaluate_policy_details(model, venv)


if __name__ == "__main__":
    test()
