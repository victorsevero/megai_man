from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

# https://datacrystal.romhacking.net/wiki/Mega_Man_(NES)/RAM_map
SPIKE_VALUE = 3


class FrameskipWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        with open("megai_man/hit_visibility_frames.yaml") as f:
            self.visibility_dict = yaml.safe_load(f)

    def step(self, action):
        total_reward = 0.0
        visible = False
        i = 0
        while i < 4 or not visible:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if done:
                break
            if i == 4:
                visibility = self.visibility_dict[
                    self.unwrapped.data["blink_counter"]
                ]
                if visibility in ("v", "w"):
                    visible = True
            else:
                i += 1

        return obs, total_reward, terminated, truncated, info


class WarpFrame(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84,
    ):
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(
            env.observation_space, spaces.Box
        ), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self.observation(obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        return obs[:, :, np.newaxis]


class ActionSkipWrapper(gym.ActionWrapper):
    def __init__(self, env):
        self.B_frame_count = 0
        super().__init__(env)

    def action(self, action):
        # if holding B, will shoot every other frame
        if self.B_frame_count >= 1:
            action = action.copy()
            action[0] = 0
            self.B_frame_count = 0
        elif action[0]:
            self.B_frame_count += 1
        else:
            self.B_frame_count = 0
        return action


class StageWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        frameskip,
        stage=0,
        screen=None,
        damage_punishment=0,
        forward_factor=1,
        backward_factor=1,
        time_punishment_factor=0,
        no_enemies=False,
        no_boss=True,
        distance_only_on_ground=False,
    ):
        super().__init__(env)
        self.reward_calculator = StageReward(
            stage=stage,
            damage_punishment=damage_punishment,
            forward_factor=forward_factor,
            backward_factor=backward_factor,
            time_punishment_factor=time_punishment_factor / frameskip,
            only_on_ground=distance_only_on_ground,
        )
        self.target_screen = screen
        self.frameskip = frameskip
        # max number of frames: NES' FPS * seconds // frameskip
        self.max_number_of_frames_without_improvement = (60 * 60) // frameskip

        self.no_enemies = no_enemies
        self.no_boss = no_boss

        self._set_enabled_enemies_vars()
        # -2 because we don't need to count the boss chamber
        self.screen_with_enemies = (
            len(self.reward_calculator.SCREENS_OFFSETS_CUTMAN) - 2
        )
        self.last_screen = (
            len(self.reward_calculator.SCREENS_OFFSETS_CUTMAN) - 1
        )

    def reset(self, *, seed=None, options=None):
        self.reward_calculator.reset()
        observation, _ = self.env.reset(seed=seed, options=options)
        self._remove_enemies_if_needed()

        self.reward_calculator.update_position(self.unwrapped.data)
        self.reward_calculator.prev_distance = (
            self.reward_calculator.distance_map[
                self.reward_calculator.y,
                self.reward_calculator.x,
            ]
        )

        self.prev_lives = self.unwrapped.data["lives"]

        return observation, self.info()

    def step(self, action):
        observation, *_ = self.env.step(action)
        self._remove_enemies_if_needed()
        if self.unwrapped.data["camera_y"] != 0:
            while self.unwrapped.data["camera_y"] != 0:
                observation, *_ = self.env.step(
                    np.zeros(self.action_space.shape, dtype=np.int64)
                )
                self._remove_enemies_if_needed()
            for _ in range(int(np.ceil(35 / self.frameskip))):
                observation, *_ = self.env.step(
                    np.zeros(self.action_space.shape, dtype=np.int64)
                )
                self._remove_enemies_if_needed()

        return (
            observation,
            self.reward(),
            self.terminated(),
            self.truncated(),
            self.info(),
        )

    def reward(self):
        if self.no_boss and (
            self.unwrapped.data["screen"]
            == len(self.reward_calculator.screen_offset_map) - 1
        ):
            return 1
        if (
            self.unwrapped.data["touching_obj_top"] == SPIKE_VALUE
            or self.unwrapped.data["touching_obj_side"] == SPIKE_VALUE
        ):
            return -10

        reward = self.reward_calculator.get_stage_reward(self.unwrapped.data)
        self.min_distance = self.reward_calculator.min_distance

        if self.reward_calculator.new_screen:
            reward += 1

        return reward

    def terminated(self):
        data = self.unwrapped.data

        # health condition
        if data["health"] == 0:
            return True

        # life lost
        if data["lives"] < self.prev_lives:
            return True

        # touched spike
        if (
            data["touching_obj_top"] == SPIKE_VALUE
            or data["touching_obj_side"] == SPIKE_VALUE
        ):
            return True

        if (
            self.no_enemies
            # backed a whole screen
            and data["screen"] < self.reward_calculator.max_screen
            # current screen is below max screen
            and self.reward_calculator.SCREENS_OFFSETS_CUTMAN[data["screen"]][
                "y"
            ]
            > self.reward_calculator.SCREENS_OFFSETS_CUTMAN[
                self.reward_calculator.max_screen
            ]["y"]
        ):
            return True

        # reached boss chamber
        if self.no_boss and (
            data["screen"] == len(self.reward_calculator.screen_offset_map) - 1
        ):
            self.reward_calculator.update_position(data)
            return True

        return False

    def truncated(self):
        return (
            self.reward_calculator.frames_since_last_improvement
            >= self.max_number_of_frames_without_improvement
        )

    def info(self):
        return {
            "min_distance": self.reward_calculator.min_distance,
            "distance": self.reward_calculator.prev_distance,
            "max_screen": self.reward_calculator.max_screen,
            "hp": self.unwrapped.data["health"],
            "x": self.unwrapped.data["x"],
            "y": self.unwrapped.data["y"],
            "screen": self.unwrapped.data["screen"],
            "camera_x": self.unwrapped.data["camera_x"],
            "camera_y": self.unwrapped.data["camera_y"],
            "camera_screen": self.unwrapped.data["camera_screen"],
        }

    def get_state(self):
        return self.env.unwrapped.em.get_state()

    def action_masks(self):
        # NOTE: REALLY IMPORTANT! Make the same changes in sb3-contrib as in
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49#issuecomment-2126473226
        # TODO: mask L/R when on ladder and already facing direction
        if self.no_enemies:
            # no need for shooting when there are no enemies
            mask = [True] * sum(self.action_space.nvec)
            mask[1] = False  # B button press
            return mask
        else:
            return [True] * sum(self.action_space.nvec)

    def set_screen_with_enemies(self, screen):
        self.screen_with_enemies = screen

    def _set_enabled_enemies_vars(self):
        for i in range(1, 0x20):
            self.unwrapped.data.set_variable(
                f"enemy{i}_enabled",
                {"address": 0x600 + i, "type": "|u1"},
            )

    def _remove_enemies_if_needed(self):
        if self.unwrapped.data["screen"] < self.screen_with_enemies:
            for i in range(1, 0x20):
                self.unwrapped.data.set_value(f"enemy{i}_enabled", 0xF8)


class StageReward:
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 240
    TILE_SIZE = 16

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

    def __init__(
        self,
        stage=0,
        damage_punishment=0,
        forward_factor=1,
        backward_factor=1,
        time_punishment_factor=0,
        only_on_ground=False,
    ):
        self.damage_punishment = damage_punishment
        self.forward_factor = forward_factor
        self.backward_factor = backward_factor
        self.time_punishment_factor = time_punishment_factor
        self.distance_map = self._get_distance_map(stage)
        self.screen_offset_map = self._get_screen_offset_map(stage)
        self.only_on_ground = only_on_ground

    def reset(self):
        self.prev_distance = -1  # we don't know distance at start
        self.prev_lives = None
        self.prev_health = 28
        self.boss_filled_health = False
        self.prev_boss_health = 28
        self.min_distance = self.distance_map.max()
        self.max_screen = 0
        self.frames_since_last_improvement = 0
        self.new_screen = False

    def get_stage_reward(self, data):
        if self._is_in_boss_room(data):
            reward = self.boss_reward(data)
        else:
            reward = self.wavefront_expansion_reward(data)

        health = data["health"]
        damage = self.prev_health - health
        damage = max(damage, 0)  # no reward for healing
        self.prev_health = health
        if damage:
            reward -= self.damage_punishment

        return reward - self.time_punishment_factor

    def wavefront_expansion_reward(self, data):
        # vertically moving to a new screen
        if (data["camera_y"] != 0) or (
            # touching spike
            (data["touching_obj_top"] == SPIKE_VALUE)
            or (data["touching_obj_side"] == SPIKE_VALUE)
        ):
            return 0

        self.update_position(data)

        if self.only_on_ground and data["touching_obj_top"] == 0:
            self.frames_since_last_improvement += 1
            return 0

        try:
            distance = self.distance_map[self.y, self.x]
        except IndexError:
            # out of bounds of the map, probably falling into a pit
            distance = self.prev_distance

        # some tiles were incorrectly mapped to -1, let's hope this is enough
        invalid_distance = distance == -1
        if invalid_distance:
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

        # if invalid_distance:
        #     return self.backward_factor * 1

        if distance_diff >= 0:
            return self.forward_factor * distance_diff
        else:
            return self.backward_factor * distance_diff

    def boss_reward(self, data):
        boss_health = data["boss_health"]
        if self.boss_filled_health:
            damage = self.prev_boss_health - boss_health
            self.prev_boss_health = boss_health
            return damage
        elif data["boss_health"] == 28:
            self.boss_filled_health = True
        return 0

    def update_position(self, data):
        screen = data["screen"]
        if screen > self.max_screen:
            self.new_screen = True
            self.max_screen = screen
        else:
            self.new_screen = False

        self.x, self.y = self.get_global_xy(screen, data["x"], data["y"])

    def get_global_xy(self, screen, x, y):
        screen_offset = self.screen_offset_map[screen]
        x = (self.SCREEN_WIDTH * screen_offset["x"] + x) // self.TILE_SIZE
        y = (
            self.SCREEN_HEIGHT * screen_offset["y"]
            + min(y, self.SCREEN_HEIGHT - 1)
        ) // self.TILE_SIZE
        return x, y

    def _is_in_boss_room(self, data):
        return data["screen"] == len(self.screen_offset_map) - 1

    @staticmethod
    def _get_distance_map(stage):
        path_dir = Path(__file__).parent / "custom_integrations"

        if stage == 0:
            filename = "cutman.npy"
        else:
            # TODO: add other stages
            raise ValueError(f"Invalid stage `{stage}`")
        return np.load(str(path_dir / filename))

    def _get_screen_offset_map(self, stage):
        if stage == 0:
            return self.SCREENS_OFFSETS_CUTMAN
        else:
            # TODO: add other stages
            raise ValueError()
