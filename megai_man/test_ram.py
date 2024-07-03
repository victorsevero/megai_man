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
        frameskip=4,
        frame_stack=4,
        truncate_if_no_improvement=True,
        obs_space="ram",
        action_space="multi_discrete",
        render_mode="human",
        # record=".",
        damage_terminate=False,
        fixed_damage_punishment=0.05,
        forward_factor=0.05,
        backward_factor=0.055,
        multi_input=False,
        distance_only_on_ground=True,
        term_back_screen=True,
    )

    # model_name = (
    #     "checkpoints/"
    #     "ram_bignet__rews0.05+screen1_dmg0.05"
    #     "_1000000_steps"
    # )
    model_name = (
        # fmt: off
        "models/"
        "ram_steps512_batch64_bignet__rews0.05+screen1_dmg0.05_groundrew_termbackscreen2"
        "_best/best_model"
        # fmt: on
    )
    # model_name = (
    #     "models/"
    #     "sevs_all_steps512_batch128_lr2.5e-04_epochs4_clip0.2_ecoef1e-03_gamma0.99_vf1__fs4_stack1rews0.05+screen1_dmg0.12_time_punishment0_groundonly_termbackscreen2_trunc60snoprog_spikefix6_scen3_actionskipB_recurrent"
    #     ".zip"
    # )

    model = PPO.load(model_name, env=venv)
    evaluate_policy_details(model, venv, deterministic=True)


if __name__ == "__main__":
    test()
