from env import make_venv
from stable_baselines3 import PPO


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
        frameskip=4,
        frame_stack=2,
        truncate_if_no_improvement=True,
        obs_space="screen",
        action_space="multi_discrete",
        crop_img=True,
        invincible=True,
        render_mode="human",
        record=".",
        damage_terminate=False,
        fixed_damage_punishment=2,
        forward_factor=0.5,
        backward_factor=0.6,
    )
    model_name = (
        "checkpoints/"
        "sevs_lr2.5e-04_epochs1_gamma0.995_gae0.9_clip0.2_normyes_ecoef1e-03__fs4_stack2_crop224_death10_smallest_rewards_trunc1minnoprog_INVINCIBLE"
        "_2000000_steps"
    )
    model = PPO.load(model_name, env=venv)

    evaluate_policy_details(model, venv)


if __name__ == "__main__":
    test()
