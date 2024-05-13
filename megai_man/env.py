from pathlib import Path

import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)
from wrappers import (
    ActionSkipWrapper,
    FrameskipWrapper,
    MultiInputWrapper,
    StageWrapper,
    TargetScreenWrapper,
    VecRemoveVectorStacks,
    WarpFrame,
)

retro.data.Integrations.add_custom_path(
    str(Path(__file__).parent / "custom_integrations")
)


def make_venv(
    n_envs=8,
    state=retro.State.DEFAULT,
    frameskip=1,
    frame_stack=3,
    truncate_if_no_improvement=True,
    obs_space="screen",
    action_space="multi_discrete",
    crop_img=True,
    invincible=False,
    render_mode="human",
    record=False,
    multi_input=False,
    curriculum=False,
    _enforce_subproc=False,
    **stage_wrapper_kwargs,
):
    def env_fn():
        return make_env(
            state=state,
            frameskip=frameskip,
            truncate_if_no_improvement=truncate_if_no_improvement,
            obs_space=obs_space,
            action_space=action_space,
            crop_img=crop_img,
            invincible=invincible,
            render_mode=render_mode,
            record=record,
            multi_input=multi_input,
            curriculum=curriculum,
            **stage_wrapper_kwargs,
        )

    if (n_envs == 1) and not _enforce_subproc:
        venv = DummyVecEnv([env_fn])
    else:
        venv = SubprocVecEnv([env_fn] * n_envs)
    if obs_space == "screen":
        venv = VecFrameStack(venv, n_stack=frame_stack)
        venv = VecTransposeImage(venv)
        if multi_input:
            venv = VecRemoveVectorStacks(venv)
    venv = VecMonitor(
        venv,
        info_keywords=("distance", "min_distance", "max_screen", "hp"),
    )
    return venv


def make_env(
    state=retro.State.DEFAULT,
    frameskip=1,
    truncate_if_no_improvement=True,
    obs_space="screen",
    action_space="multi_discrete",
    crop_img=True,
    invincible=False,
    render_mode="human",
    record=False,
    multi_input=False,
    curriculum=False,
    **stage_wrapper_kwargs,
):
    if obs_space == "screen":
        obs_type = retro.Observations.IMAGE
    elif obs_space == "ram":
        obs_type = retro.Observations.RAM
    else:
        raise ValueError(f"Invalid observation space `{obs_space}`")

    if action_space == "multi_discrete":
        use_restricted_actions = retro.Actions.MULTI_DISCRETE
    elif action_space == "filtered":
        use_restricted_actions = retro.Actions.FILTERED
    elif action_space == "discrete":
        use_restricted_actions = retro.Actions.DISCRETE
    else:
        raise ValueError(f"Invalid action space `{action_space}`")
    env = retro.make(
        game="MegaMan-v2-Nes",
        state=state,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=use_restricted_actions,
        render_mode=render_mode,
        record=record,
        obs_type=obs_type,
    )
    env = ActionSkipWrapper(env)
    if invincible:
        env.unwrapped.em.add_cheat("VVXXAPSZ")
    if frameskip > 1:
        env = FrameskipWrapper(env, skip=frameskip)
    if not truncate_if_no_improvement:
        # max number of frames: NES' FPS * seconds // frameskip
        env = TimeLimit(env, max_episode_steps=(60 * 360) // frameskip)
    env = StageWrapper(
        env,
        frameskip=frameskip,
        obs_space=obs_space,
        stage=0,
        truncate_if_no_improvement=truncate_if_no_improvement,
        **stage_wrapper_kwargs,
    )
    if obs_space == "screen":
        env = WarpFrame(env, crop=crop_img)

    if multi_input:
        env = MultiInputWrapper(env)
    if curriculum:
        env = TargetScreenWrapper(env)
    return env


if __name__ == "__main__":
    env = make_env()
    check_env(env)
