from pathlib import Path

import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)
from wrappers import (
    MegaManTerminationWrapper,
    StageRewardWrapper,
    StickyActionWrapper,
)

retro.data.Integrations.add_custom_path(
    str(Path(__file__).parent / "custom_integrations")
)


def make_venv(
    n_envs=8,
    state=retro.State.DEFAULT,
    sticky_prob=0.25,
    damage_terminate=False,
    damage_factor=1,
    truncate_if_no_improvement=True,
    render_mode="human",
    record=False,
):
    def env_fn():
        return make_env(
            state=state,
            sticky_prob=sticky_prob,
            damage_terminate=damage_terminate,
            damage_factor=damage_factor,
            truncate_if_no_improvement=truncate_if_no_improvement,
            render_mode=render_mode,
            record=record,
        )

    if n_envs == 1:
        venv = DummyVecEnv([env_fn])
    else:
        venv = SubprocVecEnv([env_fn] * n_envs)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv, info_keywords=("min_distance", "x", "y", "screen"))
    return venv


def make_env(
    state=retro.State.DEFAULT,
    sticky_prob=0.25,
    damage_terminate=False,
    damage_factor=1,
    truncate_if_no_improvement=True,
    render_mode="human",
    record=False,
):
    env = retro.make(
        game="MegaMan-v2-Nes",
        state=state,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.MULTI_DISCRETE,
        render_mode=render_mode,
        record=record,
    )
    if sticky_prob > 0:
        env = StickyActionWrapper(env, action_repeat_probability=sticky_prob)
    env = TimeLimit(env, max_episode_steps=4500)
    env = StageRewardWrapper(
        env,
        stage=0,
        damage_punishment=True,
        damage_factor=damage_factor,
        truncate_if_no_improvement=truncate_if_no_improvement,
    )
    env = MegaManTerminationWrapper(env, damage_terminate=damage_terminate)
    env = WarpFrame(env)
    return env
