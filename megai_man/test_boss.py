from env import make_venv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


def evaluate_policy_details(model, env, deterministic=True):
    obs = env.reset()
    dones = [False]
    current_length = 0
    while not dones[0]:
        if isinstance(model, MaskablePPO):
            action_masks = get_action_masks(env)
            actions, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
        else:
            actions, _ = model.predict(obs, deterministic=deterministic)
        action = env.unwrapped.envs[0].unwrapped.get_action_meaning(actions[0])
        obs, rewards, dones, infos = env.step(actions)
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
        state="CutMan-boss",
        render_mode="human",
        # record=".",
        damage_punishment=0.05,
        no_boss=False,
    )
    model_name = "models/sevs_boss_best/best_model"
    model = MaskablePPO.load(model_name, env=venv)
    evaluate_policy_details(model, venv, deterministic=True)


if __name__ == "__main__":
    test()
