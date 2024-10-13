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
        state="CutMan",
        render_mode="human",
        # record=".",
        no_enemies=False,
        damage_punishment=0.05,
        forward_factor=0.05,
        backward_factor=0.055,
        distance_only_on_ground=True,
    )
    venv.env_method("set_screen_with_enemies", 0)
    # model_name = (
    #     # fmt: off
    #     "checkpoints/"
    #     "cutman_pretrained_noTermBackScreen_gamma95_10spikepunish_enemies_curriculum"
    #     "_27004864_steps"
    #     # fmt: on
    # )
    model_name = (
        # fmt: off
        "models/"
        "cutman_pretrained_noTermBackScreen_gamma95_10spikepunish_enemies_curriculum"
        "_best/best_model"
        # fmt: on
    )
    # model_name = (
    #     # fmt: off
    #     "models/"
    #     "cutman_pretrained_noTermBackScreen_gamma95_10spikepunish_enemies_curriculum"
    #     ".zip"
    #     # fmt: on
    # )
    model = MaskablePPO.load(model_name, env=venv)
    evaluate_policy_details(model, venv, deterministic=True)


if __name__ == "__main__":
    test()
