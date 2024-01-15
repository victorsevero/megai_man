from pathlib import Path

import gymnasium as gym
import numpy as np

screens_offsets_cutman = [
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


class StageRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, stage=0):
        super().__init__(env)
        self.reward_calculator = StageReward(stage)
        self.data = self.unwrapped.data
        # TODO: set self.reward_range?

    def reward(self, reward):
        return self.reward_calculator.get_stage_reward(self.data.lookup_all())


class StageReward:
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 240
    TILE_SIZE = 16

    def __init__(self, stage=0):
        self.prev_distance = -1  # we don't know distance at start
        self.boss_filled_health = False
        self.prev_boss_health = 28  # bosses' health at battle start
        self.distance_map = self._get_distance_map(stage)
        self.screen_offset_map = self._get_screen_offset_map(stage)

    def get_stage_reward(self, data):
        if data["screen"] == len(self.screen_offset_map) - 1:
            reward = self.boss_reward(data)
        else:
            reward = self.wavefront_expansion_reward(data)
        # TODO: add damage and MAYBE time discount
        return reward

    def wavefront_expansion_reward(self, data):
        if data["camera_state"] != 0:
            return 0

        screen = data["screen"]
        screen_offset = self.screen_offset_map[screen]

        x = int(
            (self.SCREEN_WIDTH * screen_offset["x"] + data["x"])
            / self.TILE_SIZE
        )
        y = int(
            (self.SCREEN_HEIGHT * screen_offset["y"] + data["y"])
            / self.TILE_SIZE
        )

        distance = self.distance_map[y][x]

        if self.prev_distance == -1:
            distance_diff = 0
        else:
            distance_diff = distance - self.prev_distance

        self.prev_distance = distance

        # return -distance_diff / self.distance_map.max()
        return -distance_diff

    def boss_reward(self, data):
        boss_health = data["boss_health"]
        if self.boss_filled_health:
            damage = self.prev_boss_health - boss_health
            self.prev_boss_health = boss_health
            return damage
        elif data["boss_health"] == 28:
            self.boss_filled_health = True
        return 0

    @staticmethod
    def _get_distance_map(stage):
        path_dir = Path("custom_integrations/MegaMan-v2-Nes")

        if stage == 0:
            filename = "cutman.npy"
        else:
            # TODO: add other stages
            raise ValueError()
        return np.load(str(path_dir / filename))

    @staticmethod
    def _get_screen_offset_map(stage):
        if stage == 0:
            return screens_offsets_cutman
        else:
            # TODO: add other stages
            raise ValueError()


if __name__ == "__main__":
    data = {
        "health": 28,
        "boss_health": 28,
        "x": 90,
        "y": 180,
        "screen": 0,
        "camera_state": 0,
        "stage": 0,
    }
    stage_reward = StageReward()
    reward = stage_reward.get_stage_reward(data)
