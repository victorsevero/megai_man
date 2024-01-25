from pathlib import Path

import gymnasium as gym
import numpy as np


class MegaManTerminationWrapper(gym.Wrapper):
    def __init__(self, env, damage_terminate=False):
        super().__init__(env)
        self.damage_terminate = damage_terminate

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = self.unwrapped.data["lives"]
        if self.damage_terminate:
            self.prev_health = self.unwrapped.data["health"]
        return obs, info

    def step(self, action):
        observation, reward, _, truncated, info = self.env.step(action)
        return observation, reward, self.terminated(), truncated, info

    def terminated(self):
        data = self.unwrapped.data
        if self.damage_terminate:
            health_condition = data["health"] < self.prev_health
        else:
            health_condition = data["health"] == 0
        life_lost = data["lives"] < self.prev_lives
        self.prev_lives = data["lives"]
        # fully damaged or suddenly lost one life
        return health_condition or life_lost


class StickyActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat_probability: float):
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability

    def reset(self, **kwargs):
        self._sticky_action = np.zeros_like(self.action_space.sample())
        return self.env.reset(**kwargs)

    def step(self, action: int):
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


class StageRewardWrapper(gym.RewardWrapper):
    # not quite frames: this wrapper is on top of stochastic frameskip
    MAX_NUMBER_OF_FRAMES_WITHOUT_IMPROVEMENT = 60 * 10  # fps * seconds

    def __init__(
        self,
        env,
        stage=0,
        damage_punishment=True,
        damage_factor=1,
        truncate_if_no_improvement=True,
    ):
        super().__init__(env)
        self.reward_calculator = StageReward(
            stage, damage_punishment, damage_factor
        )
        self.damage_factor = damage_factor
        self.truncate_if_no_improvement = truncate_if_no_improvement
        # TODO: set self.reward_range?
        # self.min_distance = self.reward_calculator.min_distance

    def reset(self, **kwargs):
        self.reward_calculator.reset()
        observation, info = self.env.reset(**kwargs)
        return observation, self.info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        return (
            observation,
            self.reward(reward),
            terminated,
            self.truncated(truncated)
            if self.truncate_if_no_improvement
            else truncated,
            self.info(info),
        )

    def reward(self, _):
        reward = self.reward_calculator.get_stage_reward(self.unwrapped.data)
        self.min_distance = self.reward_calculator.min_distance
        return reward

    def truncated(self, truncated):
        return truncated or (
            self.reward_calculator.frames_since_last_improvement
            >= self.MAX_NUMBER_OF_FRAMES_WITHOUT_IMPROVEMENT
        )

    def info(self, info):
        info["min_distance"] = self.reward_calculator.min_distance
        info["distance"] = self.reward_calculator.prev_distance
        info["max_screen"] = self.reward_calculator.max_screen
        return info


class StageReward:
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 240
    TILE_SIZE = 16
    MEGA_MAN_SPRITE_OFFSET_Y = 11  # distance from his RAM position to his feet

    SCREENS_OFFSETS_CUTMAN = [
        {"x": 0, "y": 8},
        {"x": 1, "y": 8},
        {"x": 2, "y": 8},
        {"x": 3, "y": 8},
        {"x": 3, "y": 7},
        {"x": 3, "y": 6},
        {"x": 3, "y": 5},
        {"x": 3, "y": 4},
        {"x": 4, "y": 4},
        {"x": 5, "y": 4},
        {"x": 5, "y": 3},
        {"x": 5, "y": 2},
        {"x": 5, "y": 1},
        {"x": 5, "y": 0},
        {"x": 6, "y": 0},
        {"x": 7, "y": 0},
        {"x": 7, "y": 1},
        {"x": 7, "y": 2},
        {"x": 7, "y": 3},
        {"x": 8, "y": 3},
        {"x": 9, "y": 3},
        {"x": 10, "y": 3},
        {"x": 11, "y": 3},
        {"x": 12, "y": 3},
    ]

    def __init__(self, stage=0, damage_punishment=True, damage_factor=1):
        self.damage_punishment = damage_punishment
        self.damage_factor = damage_factor
        self.distance_map = self._get_distance_map(stage)
        self.screen_offset_map = self._get_screen_offset_map(stage)

    def reset(self):
        self.prev_distance = -1  # we don't know distance at start
        self.prev_health = 28
        self.boss_filled_health = False
        self.prev_boss_health = 28
        self.min_distance = self.distance_map.max()
        self.max_screen = 0
        self.frames_since_last_improvement = 0

    def get_stage_reward(self, data):
        if self._is_in_boss_room(data):
            reward = self.boss_reward(data)
        else:
            reward = self.wavefront_expansion_reward(data)

        if self.damage_punishment:
            health = data["health"]
            damage = self.prev_health - health
            self.prev_health = health
            reward -= self.damage_factor * damage

        return reward

    def wavefront_expansion_reward(self, data):
        if data["camera_state"] != 0:
            return 0

        screen = data["screen"]
        if screen > self.max_screen:
            self.max_screen = screen
        screen_offset = self.screen_offset_map[screen]

        x = (
            self.SCREEN_WIDTH * screen_offset["x"] + data["x"]
        ) // self.TILE_SIZE
        y = (
            self.SCREEN_HEIGHT * screen_offset["y"]
            + data["y"]
            + self.MEGA_MAN_SPRITE_OFFSET_Y
        ) // self.TILE_SIZE

        try:
            distance = self.distance_map[y][x]
        except IndexError:
            # out of bounds of the map, probably falling into a pit
            distance = self.prev_distance + 1

        # some tiles were incorrectly mapped to -1, let's hope this is enough
        if distance == -1:
            distance = self.prev_distance

        if distance < self.min_distance:
            self.min_distance = distance
            self.frames_since_last_improvement = 0
        else:
            self.frames_since_last_improvement += 1

        if self.prev_distance == -1:
            distance_diff = 0
        else:
            distance_diff = self.prev_distance - distance

        self.prev_distance = distance

        return distance_diff

    def boss_reward(self, data):
        boss_health = data["boss_health"]
        if self.boss_filled_health:
            damage = self.prev_boss_health - boss_health
            self.prev_boss_health = boss_health
            return damage
        elif data["boss_health"] == 28:
            self.boss_filled_health = True
        return 0

    def _is_in_boss_room(self, data):
        return data["screen"] == len(self.screen_offset_map) - 1

    @staticmethod
    def _get_distance_map(stage):
        path_dir = Path(__file__).parent / "custom_integrations/MegaMan-v2-Nes"

        if stage == 0:
            filename = "cutman.npy"
        else:
            # TODO: add other stages
            raise ValueError()
        return np.load(str(path_dir / filename))

    def _get_screen_offset_map(self, stage):
        if stage == 0:
            return self.SCREENS_OFFSETS_CUTMAN
        else:
            # TODO: add other stages
            raise ValueError()
