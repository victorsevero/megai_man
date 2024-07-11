import numpy as np
from env import make_venv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import PPO


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
        screen=None,
        frameskip=4,
        frame_stack=3,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="multi_discrete",
        crop_img=False,
        render_mode="human",
        # record=".",
        damage_terminate=False,
        fixed_damage_punishment=1,
        forward_factor=0.05,
        backward_factor=0.055,
        multi_input=False,
        no_enemies=True,
        distance_only_on_ground=True,
        term_back_screen=True,
    )
    # model_name = (
    #     "checkpoints/"
    #     "sevs_steps512_batch128_lr2.5e-04_epochs4_clip0.2_ecoef1e-03_gamma0.99_vf0.5_maxgrad0.5_twoFEs__fs4_stack3_rews0.05+scrn1_scorerew0_dmg0.05_ground_termbackscrn_spikefix6_Vscrnfix2_scen5mult_skipB_multin5_def_mask_NO_ENEM2_vsbl"
    #     "_30254864_steps"
    # )
    model_name = (
        "models/"
        "sevs_steps1024_batch128_lr2.5e-04_epochs4_clip0.2_ecoef1e-04_gamma0.99_vf1_maxgrad0.5_twoFEs__fs4_stack3_rews0.05+scrn1_scorerew0_dmg0.05_ground_termbackscrn_spikefix7_Vscrnfix2_scen5multnoB_mask_NO_ENEM2_vsbl"
        "_best/best_model"
    )
    # model_name = (
    #     "models/"
    #     "sevs_all_steps512_batch128_lr2.5e-04_epochs4_clip0.2_ecoef1e-03_gamma0.99_vf1__fs4_stack1rews0.05+screen1_dmg0.12_time_punishment0_groundonly_termbackscreen2_trunc60snoprog_spikefix6_scen3_actionskipB_recurrent"
    #     ".zip"
    # )
    model = MaskablePPO.load(model_name, env=venv)
    evaluate_policy(model, venv, n_eval_episodes=1, deterministic=True)

    # evaluate_policy_details(model, venv, deterministic=True)


if __name__ == "__main__":
    test()
