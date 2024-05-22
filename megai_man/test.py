import numpy as np
from env import make_venv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO


def evaluate_policy_details(model: RecurrentPPO, env, deterministic=True):
    obs = env.reset()
    states = None
    dones = [False]
    episode_starts = np.ones((1,), dtype=bool)
    current_length = 0
    while not dones[0]:
        actions, states = model.predict(
            obs,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        action = env.unwrapped.envs[0].unwrapped.get_action_meaning(actions[0])
        obs, rewards, dones, infos = env.step(actions)
        episode_starts = dones
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
        state="NightmarePit",
        screen=None,
        frameskip=4,
        frame_stack=1,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="multi_discrete",
        crop_img=True,
        invincible=False,
        render_mode="human",
        record="non_determ",
        damage_terminate=False,
        fixed_damage_punishment=1,
        forward_factor=0.5,
        backward_factor=0.55,
        multi_input=True,
    )
    # model_name = (
    #     "checkpoints/"
    #     "sevs_all_steps512_batch4096_lr2.5e-04_epochs4_clip0.2_ecoef1e-02_gamma0.99__fs4_stack1_crop224small_rewards3_time_punishment0_trunc60snoprog_spikefix6_scen3_actionskipB_multinput_recurrent"
    #     "_3000000_steps"
    # )
    model_name = (
        "models/"
        "sevs_NIGHTMAREPIT_all_steps512_batch128_lr1.0e-04_epochs4_clip0.1_ecoef1e-02_gamma0.99__fs4_stack1_crop224common_rews_time_punishment0_trunc60snoprog_spikefix6_scen3_actionskipB_multinput2_recurrent_NIGHTMAREREW"
        ".zip"
    )
    model = RecurrentPPO.load(model_name, env=venv)

    evaluate_policy_details(model, venv, deterministic=True)


if __name__ == "__main__":
    test()
