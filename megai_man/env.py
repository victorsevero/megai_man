from pathlib import Path

import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)
from wrappers import (
    FrameskipWrapper,
    MegaManTerminationWrapper,
    StageRewardWrapper,
    StickyActionWrapper,
    WarpFrame,
)

retro.data.Integrations.add_custom_path(
    str(Path(__file__).parent / "custom_integrations")
)


def make_venv(
    n_envs=8,
    state=retro.State.DEFAULT,
    sticky_prob=0.0,
    frameskip=1,
    damage_terminate=False,
    damage_factor=1,
    truncate_if_no_improvement=True,
    obs_space="screen",
    action_space="multi_discrete",
    render_mode="human",
    record=False,
):
    def env_fn():
        return make_env(
            state=state,
            sticky_prob=sticky_prob,
            frameskip=frameskip,
            damage_terminate=damage_terminate,
            damage_factor=damage_factor,
            truncate_if_no_improvement=truncate_if_no_improvement,
            obs_space=obs_space,
            action_space=action_space,
            render_mode=render_mode,
            record=record,
        )

    if n_envs == 1:
        venv = DummyVecEnv([env_fn])
    else:
        venv = SubprocVecEnv([env_fn] * n_envs)
    if obs_space == "screen":
        venv = VecFrameStack(venv, n_stack=3)
        venv = VecTransposeImage(venv)
    venv = VecMonitor(
        venv,
        info_keywords=("distance", "min_distance", "x", "y", "max_screen"),
    )
    return venv


def make_env(
    state=retro.State.DEFAULT,
    sticky_prob=0.25,
    frameskip=1,
    damage_terminate=False,
    damage_factor=1,
    truncate_if_no_improvement=True,
    obs_space="screen",
    action_space="multi_discrete",
    render_mode="human",
    record=False,
):
    assert not (
        sticky_prob and (frameskip > 1)
    ), "`sticky_prob` and `max_and_skip` can't be both different than zero"
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
    if sticky_prob > 0:
        env = StickyActionWrapper(env, action_repeat_probability=sticky_prob)
    if frameskip > 1:
        env = FrameskipWrapper(env, skip=frameskip)
    if not truncate_if_no_improvement:
        env = TimeLimit(env, max_episode_steps=4500)
    env = StageRewardWrapper(
        env,
        frameskip=frameskip,
        stage=0,
        damage_punishment=True,
        damage_factor=damage_factor,
        truncate_if_no_improvement=truncate_if_no_improvement,
    )
    env = MegaManTerminationWrapper(env, damage_terminate=damage_terminate)
    if obs_space == "screen":
        env = WarpFrame(env)
    # env = ClipRewardEnv(env)
    return env
