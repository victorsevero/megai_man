from pathlib import Path

import gymnasium as gym
import numpy as np
import retro
from gymnasium.wrappers.time_limit import TimeLimit
from reward import StageRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

retro.data.Integrations.add_custom_path(
    str(Path(__file__).parent / "custom_integrations")
)


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac
                )
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def env_test(env):
    # just for quick use with VSCode debugger
    from time import time

    env = make_env()
    done = False
    rewards = []
    observation, info = env.reset(seed=666)
    s = time()
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
    e = time()
    print(e - s)


def main():
    def make_env():
        env = make_retro(
            game="MegaMan-v2-Nes",
            state="CutMan",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        env = WarpFrame(env)
        env = StageRewardWrapper(env, stage=0)
        return env

    venv = VecMonitor(
        VecTransposeImage(
            VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4)
        )
    )
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        tensorboard_log="logs/cutman",
        verbose=0,
        seed=666,
        device="cuda",
    )
    model.learn(
        tb_log_name="full_cutman_wo_damage_punishment",
        total_timesteps=100_000_000,
        log_interval=1,
    )


if __name__ == "__main__":
    main()
