from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import retro
from gymnasium.wrappers.time_limit import TimeLimit
from retro.rendering import SimpleImageViewer

from megai_man.wrappers import (
    ActionSkipWrapper,
    FrameskipWrapper,
    StageWrapper,
)

from .termination import *

retro.data.Integrations.add_custom_path(
    str(Path(__file__).parent.parent / "custom_integrations")
)


class SpecialWrapper(gym.Wrapper):
    metadata = {"render.modes": ["human", "rgb_array", "encoding"]}

    def __init__(self, env, terminal_condition=None):
        super().__init__(env)
        self.terminal_condition = terminal_condition

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if not done and self.terminal_condition is not None:
            terminated = self.terminal_condition.isterminal(
                reward, terminated, info
            )
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "encoding":
            if not "encoder" in kwargs:
                raise TypeError("Expected an encoder model `encoder`")
            if not "observation" in kwargs:
                raise TypeError("Expected previous observation `observation`")

            encoder = kwargs["encoder"]
            observation = kwargs["observation"]
            encoding = encoder.predict(np.expand_dims(observation, axis=0))[0]
            encoding = (encoding - encoding.min()) / (
                encoding.max() - encoding.min()
            )
            image = np.repeat(np.expand_dims(encoding, axis=-1), 3, axis=-1)
            image = np.uint8(image * 255)
            image = cv2.resize(image, (210, 210), interpolation=cv2.INTER_AREA)
            image = np.concatenate((observation, image), axis=1)

            # NOTE: stable-retro has an internal renderer
            # if self.viewer is None:
            #     self.viewer = SimpleImageViewer()

            self.viewer.imshow(image)
        else:
            return self.env.render()


class FrameStack(gym.Wrapper):
    def __init__(self, env, size, mergefn=np.max):
        super().__init__(env)
        self.size = size
        self.mergefn = mergefn
        self.buffer = deque(maxlen=self.size)

    def observe(self):
        return self.mergefn(self.buffer, axis=0)

    def reset(self, *, seed=None, options=None):
        self.buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self.buffer.append(obs)
        return self.observe(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(obs)
        return self.observe(), reward, terminated, truncated, info


class GymSpecialWrapper(SpecialWrapper):
    def __init__(self, env_id, terminal_condition=None, render_mode=None):
        super().__init__(
            gym.make(env_id + "Deterministic-v4"),
            terminal_condition,
            render_mode,
        )
        self.env_id = env_id


class MegaManSpecialWrapper(SpecialWrapper):
    def __init__(
        self,
        env_id,
        state=retro.State.DEFAULT,
        terminal_condition=None,
        render_mode=None,
        record=None,
    ):
        env = retro.make(
            env_id,
            state=state,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            obs_type=retro.Observations.IMAGE,
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode=render_mode,
            record=record,
        )
        # env = ActionSkipWrapper(env)
        frameskip = 4
        env = FrameskipWrapper(env, skip=frameskip)
        env = TimeLimit(env, max_episode_steps=(60 * 360) // frameskip)
        env = StageWrapper(
            env,
            frameskip=frameskip,
            fixed_damage_punishment=1,
            forward_factor=0.5,
            backward_factor=0.55,
        )

        super().__init__(env, terminal_condition)
        self.env_id = env_id


# Environment Wrappers
MontezumaRevenge = lambda: GymSpecialWrapper(
    "MontezumaRevenge", TerminateOnLifeLoss(6)
)
SpaceInvaders = lambda: GymSpecialWrapper(
    "SpaceInvaders", TerminateOnLifeLoss(3)
)
VideoPinball = lambda: GymSpecialWrapper(
    "VideoPinball", TerminateOnLifeLoss(3)
)
Asteroids = lambda: GymSpecialWrapper("Asteroids", TerminateOnLifeLoss(4))
BankHeist = lambda: GymSpecialWrapper("BankHeist", TerminateOnLifeLoss(4))
Centipede = lambda: GymSpecialWrapper("Centipede", TerminateOnLifeLoss(3))
Breakout = lambda: GymSpecialWrapper("Breakout", TerminateOnLifeLoss(3))
MsPacman = lambda: GymSpecialWrapper("MsPacman", TerminateOnLifeLoss(3))
Freeway = lambda: GymSpecialWrapper("Freeway")
Pitfall = lambda: GymSpecialWrapper(
    "Pitfall",
    TerminalConditionGroup(
        [TerminateOnNegativeReward(), TerminateOnLifeLoss(3)]
    ),
)
AirRaid = lambda: GymSpecialWrapper("AirRaid", TerminateOnLifeLoss(1))
Alien = lambda: GymSpecialWrapper("Alien", TerminateOnLifeLoss(3))
Qbert = lambda: GymSpecialWrapper("Qbert", TerminateOnLifeLoss(4))
Pong = lambda: GymSpecialWrapper("Pong", TerminateOnNegativeReward())

MegaMan = lambda: MegaManSpecialWrapper("MegaMan-v2-Nes")

name2env = {
    "MontezumaRevenge": MontezumaRevenge,
    "SpaceInvaders": SpaceInvaders,
    "VideoPinball": VideoPinball,
    "Asteroids": Asteroids,
    "BankHeist": BankHeist,
    "Centipede": Centipede,
    "Breakout": Breakout,
    "MsPacman": MsPacman,
    "Freeway": Freeway,
    "Pitfall": Pitfall,
    "AirRaid": AirRaid,
    "Alien": Alien,
    "Qbert": Qbert,
    "Pong": Pong,
}
