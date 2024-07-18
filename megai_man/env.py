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

from megai_man.wrappers import (
    ActionSkipWrapper,
    FrameskipWrapper,
    StageWrapper,
    WarpFrame,
)

retro.data.Integrations.add_custom_path(
    str(Path(__file__).parent / "custom_integrations")
)


def make_venv(
    n_envs=8,
    no_enemies=False,
    state=retro.State.DEFAULT,
    frame_stack=3,
    render_mode="human",
    record=False,
    _enforce_subproc=False,
    **stage_wrapper_kwargs,
):
    frameskip = 4

    def env_fn():
        return make_env(
            no_enemies=no_enemies,
            state=state,
            frameskip=frameskip,
            render_mode=render_mode,
            record=record,
            **stage_wrapper_kwargs,
        )

    if (n_envs == 1) and not _enforce_subproc:
        venv = DummyVecEnv([env_fn])
    else:
        venv = SubprocVecEnv([env_fn] * n_envs)
    venv = VecFrameStack(venv, n_stack=frame_stack)
    venv = VecTransposeImage(venv)
    venv = VecMonitor(
        venv,
        info_keywords=("distance", "min_distance", "max_screen", "hp"),
    )
    return venv


def make_env(
    no_enemies=False,
    state=retro.State.DEFAULT,
    frameskip=1,
    render_mode="human",
    record=False,
    **stage_wrapper_kwargs,
):
    if no_enemies:
        game = "MegaMan-noEnemies-Nes"
    else:
        game = "MegaMan-v1-Nes"

    env = retro.make(
        game=game,
        state=state,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.MULTI_DISCRETE,
        render_mode=render_mode,
        record=record,
        obs_type=retro.Observations.IMAGE,
    )
    if not stage_wrapper_kwargs.get("no_enemies", False):
        env = ActionSkipWrapper(env)
    if frameskip > 1:
        env = FrameskipWrapper(env, skip=frameskip)
    # max number of frames: NES' FPS * seconds // frameskip
    env = TimeLimit(env, max_episode_steps=(60 * 360) // frameskip)
    env = StageWrapper(env, frameskip=frameskip, **stage_wrapper_kwargs)
    env = WarpFrame(env)

    return env


if __name__ == "__main__":
    env = make_env()
    check_env(env)
