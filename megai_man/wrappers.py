import itertools
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

from megai_man.utils import ActionMapper

# https://datacrystal.romhacking.net/wiki/Mega_Man_(NES)/RAM_map
SPIKE_VALUE = 3


class VecRemoveVectorStacks(VecEnvWrapper):
    VECTOR_SIZE = 1

    def __init__(self, venv: VecEnv):
        super().__init__(venv)
        observation_space = venv.observation_space
        observation_space.spaces["vector"] = self.unwrapped.observation_space[
            "vector"
        ]

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        observations, infos = self.observations(observations, infos)
        return observations, rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        observation = self.venv.reset()
        observation, _ = self.observations(observation, {})
        return observation

    def observations(self, obs, infos):
        obs["vector"] = obs["vector"][..., -self.VECTOR_SIZE :]
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"]["vector"] = info[
                    "terminal_observation"
                ]["vector"][-self.VECTOR_SIZE :]
        return obs, infos


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
        crop: bool = True,
    ):
        super().__init__(env)
        self.width = width
        self.height = height
        self.crop = crop
        self.crop_window_y, self.crop_window_x = (224, 224)
        self.half_crop_y = self.crop_window_y // 2
        self.half_crop_x = self.crop_window_x // 2
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
            self.observation(obs, info),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs, info), info

    def observation(self, obs, info):
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        if self.crop:
            obs_height, obs_width = obs.shape
            y = info["y"]
            x_camera_offset = (
                info["camera_screen"] * obs_width + info["camera_x"]
            )
            x = info["screen"] * obs_width + info["x"] - x_camera_offset

            # offsets to better centralize cropping area on Mega Man
            x -= 9
            y -= 7

            # horizontal padding
            left_x = x - self.half_crop_x
            right_x = x + self.half_crop_x - obs_width
            if left_x < 0:
                before_x = -left_x
                after_x = 0
            elif right_x > 0:
                before_x = 0
                after_x = right_x
            else:
                before_x = 0
                after_x = 0

            # vertical padding
            up_y = y - self.half_crop_y
            down_y = y + self.half_crop_y - obs_height
            if up_y < 0:
                before_y = -up_y
                after_y = 0
            elif down_y > 0:
                before_y = 0
                after_y = down_y
            else:
                before_y = 0
                after_y = 0

            obs = np.pad(obs, ((before_y, after_y), (before_x, after_x)))
            obs = obs[
                y
                - self.half_crop_y
                + before_y : y
                + self.half_crop_y
                + before_y,
                x
                - self.half_crop_x
                + before_x : x
                + self.half_crop_x
                + before_x,
            ]

        obs = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
            # interpolation=cv2.INTER_NEAREST,
        )

        return obs[:, :, np.newaxis]


class VecImageScaling(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        observation_space = spaces.Box(
            0.0,
            1.0,
            shape=venv.observation_space.shape,
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=observation_space)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        return observations / 255, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        return observations / 255


class MultiInputWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(84, 84, 1),
                    # shape=(224, 240, 1),
                    dtype=np.uint8,
                ),
                "vector": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        self.last_A = 0
        self.calculator = self.env.get_wrapper_attr("reward_calculator")
        self.max_distance = self.calculator.distance_map.max()

    def reset(self, **kwargs):
        self.last_A = 0
        observation, info = self.env.reset(**kwargs)
        observation = self.observation(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        observation = self.observation(observation)
        # if isinstance(self.action_space, spaces.MultiDiscrete):
        #     self.last_A = action[2]
        # else:
        #     self.last_A = "A" in self.env.unwrapped.get_action_meaning(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def observation(self, obs):
        vector = [0]
        # if hasattr(self.calculator, "x"):
        #     try:
        #         if (
        #             self.calculator.distance_map[
        #                 self.calculator.y, self.calculator.x + 1
        #             ]
        #             == self.calculator.prev_distance - 1
        #         ):
        #             vector[0] = 1
        #     except IndexError:
        #         pass
        #     try:
        #         if (
        #             self.calculator.distance_map[
        #                 self.calculator.y, self.calculator.x - 1
        #             ]
        #             == self.calculator.prev_distance - 1
        #         ):
        #             vector[0] = -1
        #     except IndexError:
        #         pass

        #     try:
        #         if (
        #             self.calculator.distance_map[
        #                 self.calculator.y - 1, self.calculator.x
        #             ]
        #             == self.calculator.prev_distance - 1
        #         ):
        #             vector[1] = 1
        #     except IndexError:
        #         pass
        #     try:
        #         if (
        #             self.calculator.distance_map[
        #                 self.calculator.y + 1, self.calculator.x
        #             ]
        #             == self.calculator.prev_distance - 1
        #         ):
        #             vector[1] = -1
        #     except IndexError:
        #         pass

        # vector[2] = self.last_A

        # farthest point achieved to avoid breaking Markov Property
        # since some rewards are given based on reaching new screens
        vector[0] = 1 - self.calculator.min_distance / self.max_distance

        # # set vector[1] to HP and normalize it
        # vector[1] = self.env.unwrapped.data["health"] / 28

        # # set vector[2] to invincibility frame counter and normalize it
        # vector[2] = self.env.unwrapped.data["blink_counter"] / 111

        return {
            "image": obs,
            "vector": np.array(vector, dtype=np.float32),
        }


class TargetScreenWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.get_wrapper_attr("target_screen")

    @property
    def target_screen(self):
        return self._target_screen

    @target_screen.setter
    def target_screen(self, new_screen):
        self._target_screen = new_screen
        env = self.env
        while not isinstance(env, StageWrapper):
            env = env.env
        env.target_screen = new_screen


class ActionSkipWrapper(gym.ActionWrapper):
    def __init__(self, env):
        self.B_frame_count = 0
        self.A_frame_count = 0
        super().__init__(env)

        if isinstance(self.action_space, spaces.Discrete):
            self.action_mapper = ActionMapper(self.env, "discrete")

    def action(self, action):
        if isinstance(self.action_space, spaces.MultiDiscrete):
            # if holding B, will shoot every other frame
            if self.B_frame_count >= 1:
                action = action.copy()
                action[0] = 0
                self.B_frame_count = 0
            elif action[0]:
                self.B_frame_count += 1
            else:
                self.B_frame_count = 0

            # if holding A, will jump every 20 frames
            # if self.A_frame_count >= 19:
            #     action = action.copy()
            #     action[3] = 0
            #     self.A_frame_count = 0
            # elif action[3]:
            #     self.A_frame_count += 1
            # else:
            #     self.A_frame_count = 0
        else:
            buttons = self.env.unwrapped.get_action_meaning(action)
            pressing_b = "B" in buttons
            if self.B_frame_count >= 1 and pressing_b:
                buttons.remove("B")
                action = self.action_mapper.map_action(buttons)
                self.B_frame_count = 0
            elif pressing_b:
                self.B_frame_count += 1
            else:
                self.B_frame_count = 0

        return action


class StageWrapper(gym.Wrapper):
    # https://bisqwit.iki.fi/jutut/megamansource/maincode.txt: FindFreeObject
    FREE_OBJECT = 0xF8

    def __init__(
        self,
        env,
        frameskip,
        obs_space="screen",
        stage=0,
        screen=None,
        damage_terminate=False,
        damage_factor=1,
        fixed_damage_punishment=0,
        forward_factor=1,
        backward_factor=1,
        time_punishment_factor=0,
        truncate_if_no_improvement=True,
        no_enemies=False,
        screen_rewards=False,
        score_reward=0,
        distance_only_on_ground=False,
        term_back_screen=False,
    ):
        super().__init__(env)
        self.reward_calculator = StageReward(
            stage,
            damage_factor,
            fixed_damage_punishment,
            forward_factor,
            backward_factor,
            time_punishment_factor / frameskip,
            distance_only_on_ground,
        )
        self.target_screen = screen
        self.damage_terminate = damage_terminate
        self.truncate_if_no_improvement = truncate_if_no_improvement
        self.frameskip = frameskip
        # max number of frames: NES' FPS * seconds // frameskip
        self.max_number_of_frames_without_improvement = (60 * 60) // frameskip

        if obs_space == "ram":
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(354,),
            )
        self.obs_space = obs_space

        self.no_enemies = no_enemies

        self.screen_rewards = screen_rewards
        self.score_reward = score_reward
        self.term_back_screen = term_back_screen

    def reset(self, **kwargs):
        self.reward_calculator.reset()
        observation, info = self.env.reset(**kwargs)

        self.reward_calculator.update_position(self.unwrapped.data)
        self.reward_calculator.prev_distance = (
            self.reward_calculator.distance_map[
                self.reward_calculator.y,
                self.reward_calculator.x,
            ]
        )

        self.prev_lives = self.unwrapped.data["lives"]
        if self.damage_terminate:
            self.prev_health = self.unwrapped.data["health"]

        if self.screen_rewards:
            self.prev_screen = self.unwrapped.data["screen"]

        if self.score_reward > 0:
            self.prev_score = 0

        return self.observation(observation), self.info(info)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        if self.unwrapped.data["camera_y"] != 0:
            while self.unwrapped.data["camera_y"] != 0:
                observation, _, terminated, truncated, info = self.env.step(
                    np.zeros(self.action_space.shape, dtype=np.int64)
                )
            for _ in range(int(np.ceil(35 / self.frameskip))):
                observation, _, terminated, truncated, info = self.env.step(
                    np.zeros(self.action_space.shape, dtype=np.int64)
                )

        return (
            self.observation(observation),
            self.reward(action),
            self.terminated(terminated),
            self.truncated(truncated)
            if self.truncate_if_no_improvement
            else truncated,
            self.info(info),
        )

    def observation(self, obs):
        if self.obs_space == "screen":
            return obs

        screen_scale = len(self.reward_calculator.screen_offset_map) - 1
        rel_pos_scale = 0xFF
        # x_scale = (
        #     max(
        #         screen_offset_map,
        #         key=lambda d: d["x"],
        #     )["x"]
        #     * offset_width
        #     - 1
        # )
        # y_scale = (
        #     max(
        #         screen_offset_map,
        #         key=lambda d: d["y"],
        #     )["y"]
        #     * offset_height
        #     - 1
        # )
        x_scale = self.reward_calculator.distance_map.shape[1] - 1
        y_scale = self.reward_calculator.distance_map.shape[0] - 1
        x_speed_scale = 1.5
        y_speed_scale = 9
        health_scale = 28
        general_scale = 0xFF

        player_x, player_y = self.reward_calculator.get_global_xy(
            obs[0x460],
            obs[0x480],
            obs[0x600],
        )
        variables = [
            # health
            obs[0x6A] / health_scale,
            # screen count
            obs[0x460] / screen_scale,
            # X position
            player_x / x_scale,
            # Y position
            player_y / y_scale,
            # X speed, composed by two bytes: X_hi and X_lo
            (obs[0x4C0] + obs[0x4E0] / general_scale) / x_speed_scale,
            # Y speed, composed by two bytes: Y_hi and Y_lo.
            # TODO: I'm currently using only Y_hi, this is too much for me
            (obs[0x680] if obs[0x680] < 127 else obs[0x680] - 256 + 5)
            / y_speed_scale,
            # flags
            self._get_flag_from_byte(obs[0x420], 4),  # on ladder
            self._get_flag_from_byte(obs[0x420], 6),  # facing direction
        ]

        variables.append(self.reward_calculator.max_screen / screen_scale)

        n_bullets = 3
        for i in range(n_bullets):
            enabled = obs[0x602 + i] != 0xF8
            if enabled:
                x, y = self.reward_calculator.get_global_xy(
                    obs[0x462 + i],
                    obs[0x482 + i],
                    obs[0x602 + i],
                )
                rel_x = x - player_x
                rel_y = y - player_y
                variables += [
                    1,
                    # TODO: direction of bullet
                    rel_x / rel_pos_scale,
                    rel_y / rel_pos_scale,
                ]
            else:
                variables += [0, 0, 0]

        n_objects = 0x20
        # OLD: start at 0x01 to ignore Mega Man himself
        # NEW: actually, let's start at 0x10 to ignore the first 16 objects,
        # because there doesn't seem to be anything useful there
        # TODO: maybe vectorize all of this with numpy?
        for i in range(0x10, n_objects):
            # game sets Y_pos to 0xF8 to disable object
            enabled = obs[0x602 + i] != 0xF8
            if enabled:
                x, y = self.reward_calculator.get_global_xy(
                    obs[0x460 + i],
                    obs[0x480 + i],
                    obs[0x600 + i],
                )
                rel_x = x - player_x
                rel_y = y - player_y
                variables += [
                    # object enabled
                    1,
                    # sprite number
                    # obs[0x0400 + i] / general_scale,
                    # flags
                    *self._get_flags_from_byte(obs[0x420 + i]),
                    # unknown440
                    # *self._get_nibbles_from_byte(obs[0x0440 + i], normalize=True),
                    # screen count
                    # obs[0x460 + i] / screen_scale,
                    # X position
                    rel_x / rel_pos_scale,
                    # X speed
                    (obs[0x4C0 + i] + obs[0x4E0 + i] / general_scale)
                    / x_speed_scale,
                    # Y position
                    rel_y / rel_pos_scale,
                    # Y speed
                    (
                        obs[0x680 + i]
                        if obs[0x680 + i] < 127
                        else obs[0x680] - 256 + 5
                    )
                    / y_speed_scale,
                    # life cycle counter (wtf is that supposed to mean? no idea)
                    # obs[0x06A0 + i] / general_scale,
                    # life meter
                    # obs[0x06C0 + i] / general_scale,
                    # type
                    *self._get_flags_from_byte(obs[0x6E0 + i]),
                ]
            else:
                variables += [0] * 21

        return variables

    def reward(self, action):
        if self.screen_rewards:
            current_screen = self.unwrapped.data["screen"]
            reward = current_screen - self.prev_screen
            self.prev_screen = current_screen
            self.reward_calculator.get_stage_reward(self.unwrapped.data)
            self.min_distance = self.reward_calculator.min_distance
        else:
            reward = self.reward_calculator.get_stage_reward(
                self.unwrapped.data
            )
            self.min_distance = self.reward_calculator.min_distance

        if self.reward_calculator.new_screen:
            reward += 1

        if self.score_reward > 0:
            reward += self._get_score_reward()

        # if self.get_wrapper_attr("statename") == "NightmarePit.state":
        #     reward = int(action[2] == 1) - int(action[1] == 2)
        return reward

    def terminated(self, terminated):
        data = self.unwrapped.data

        # if (self.get_wrapper_attr("statename") == "NightmarePit.state") and (
        #     data["screen"] == 3
        # ):
        #     return True

        # finish screen
        target_screen = self.target_screen
        if target_screen is not None and data["screen"] > target_screen:
            return True

        # health condition
        if self.damage_terminate:
            if data["health"] < self.prev_health:
                return True
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
            self.term_back_screen
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

        return terminated

    def truncated(self, truncated):
        return truncated or (
            self.reward_calculator.frames_since_last_improvement
            >= self.max_number_of_frames_without_improvement
        )

    def info(self, info):
        # if not self.screen_rewards:
        info["min_distance"] = self.reward_calculator.min_distance
        info["distance"] = self.reward_calculator.prev_distance
        info["max_screen"] = self.reward_calculator.max_screen
        info["hp"] = self.unwrapped.data["health"]
        info["x"] = self.unwrapped.data["x"]
        info["y"] = self.unwrapped.data["y"]
        info["screen"] = self.unwrapped.data["screen"]
        info["camera_x"] = self.unwrapped.data["camera_x"]
        info["camera_y"] = self.unwrapped.data["camera_y"]
        info["camera_screen"] = self.unwrapped.data["camera_screen"]
        return info

    def get_state(self):
        return self.env.unwrapped.em.get_state()

    def set_state(self, state):
        self.env.unwrapped.em.set_state(state)
        self.prev_screen = self.unwrapped.data["screen"]

    def action_masks(self):
        # NOTE: REALLY IMPORTANT! Make the same changes in sb3-contrib as in
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49#issuecomment-2126473226
        # TODO: mask L/R when on ladder and already facing direction
        if self.no_enemies:
            # remove shooting when there are no enemies
            if isinstance(self.action_space, spaces.Discrete):
                return NotImplementedError("No masking for Discrete actions")
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                mask = [True] * sum(self.action_space.nvec)
                mask[1] = False  # B button press
                return mask
            else:
                raise NotImplementedError(
                    f"No masking for {self.action_space}"
                )
        else:
            return [True] * sum(self.action_space.nvec)

    def _get_score_reward(self):
        score = 0
        for i in range(5):
            score += self.env.unwrapped.get_ram()[0x74 + i] * 10**i

        reward = self.score_reward if score > self.prev_score else 0
        self.prev_score = score
        return reward

    # def _get_kill_reward(self):
    #     n_objects = self.env.unwrapped.ram[0x0054]
    #     ram = self.env.unwrapped.get_ram()
    #     y_pos = np.array([ram[0x600 + i] for i in range(n_objects)])
    #     alive_count = y_pos[y_pos != self.FREE_OBJECT].sum()
    #     enemies_alive = [ram[0x007B + 4 * i] / alive_scale for i in range(8)]

    @staticmethod
    def _get_flags_from_byte(byte):
        return [int(bool(byte & (1 << i))) for i in range(8)]

    @staticmethod
    def _get_nibbles_from_byte(byte, normalize=True):
        if normalize:
            return [(byte >> 4) / 0x0F, (byte & 0x0F) / 0x0F]
        return [byte >> 4, byte & 0x0F]

    @staticmethod
    def _get_flag_from_byte(byte, bit):
        return int(bool(byte & (1 << bit)))


class StageReward:
    SCREEN_WIDTH = 256
    SCREEN_HEIGHT = 240
    TILE_SIZE = 16
    # won't use this for now, it's buggy in some edge cases, we'll see
    # MEGA_MAN_SPRITE_OFFSET_Y = 11  # distance from his RAM position to his feet

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
        damage_factor=1,
        fixed_damage_punishment=0,
        forward_factor=1,
        backward_factor=1,
        time_punishment_factor=0,
        only_on_ground=False,
    ):
        self.damage_factor = damage_factor
        self.fixed_damage_punishment = fixed_damage_punishment
        self.forward_factor = forward_factor
        self.backward_factor = backward_factor
        self.time_punishment_factor = time_punishment_factor
        self.distance_map = self._get_distance_map(stage)
        self.screen_offset_map = self._get_screen_offset_map(stage)
        assert not fixed_damage_punishment or (
            damage_factor == 1
        ), "Not possible to set `damage_factor` if `fixed_damage_punishment` != 0"
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
        if self.fixed_damage_punishment:
            if damage:
                reward -= self.fixed_damage_punishment
        else:
            reward -= self.damage_factor * damage

        return reward - self.time_punishment_factor

    def wavefront_expansion_reward(self, data):
        # this didn't end well, but it might be useful in the future:
        # if self.prev_lives is not None and data["lives"] < self.prev_lives:
        #     return -5
        # else:
        #     self.prev_lives = data["lives"]

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

        if distance_diff >= 0:
            return self.forward_factor * distance_diff
        else:
            # high pits were discouraging the agent to try jumping over them,
            # so we only penalize the agent for 9 backward tiles at max
            # return self.backward_factor * max(distance_diff, -9)
            # NOTE: this ended in infinite positive loops as expected :)
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
        path_dir = Path(__file__).parent / "custom_integrations/MegaMan-v2-Nes"

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
