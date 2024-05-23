from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

# https://datacrystal.romhacking.net/wiki/Mega_Man_(NES)/RAM_map
SPIKE_VALUE = 3


class VecRemoveVectorStacks(VecEnvWrapper):
    VECTOR_SIZE = 3

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

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if done:
                break

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
                    dtype=np.uint8,
                ),
                "vector": gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(3,),
                    dtype=np.int16,
                ),
            }
        )
        self.last_A = 0

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
        if isinstance(self.action_space, spaces.MultiDiscrete):
            self.last_A = action[3]
        else:
            # TODO: implement A/B alternation for spaces.Discrete
            pass
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def observation(self, obs):
        calculator = self.env.get_wrapper_attr("reward_calculator")
        vector = [0, 0, 0]
        if hasattr(calculator, "x"):
            try:
                if (
                    calculator.distance_map[calculator.y, calculator.x + 1]
                    == calculator.prev_distance - 1
                ):
                    vector[0] = 1
            except IndexError:
                pass
            try:
                if (
                    calculator.distance_map[calculator.y, calculator.x - 1]
                    == calculator.prev_distance - 1
                ):
                    vector[0] = -1
            except IndexError:
                pass

            try:
                if (
                    calculator.distance_map[calculator.y - 1, calculator.x]
                    == calculator.prev_distance - 1
                ):
                    vector[1] = 1
            except IndexError:
                pass
            try:
                if (
                    calculator.distance_map[calculator.y + 1, calculator.x]
                    == calculator.prev_distance - 1
                ):
                    vector[1] = -1
            except IndexError:
                pass

        if isinstance(self.action_space, spaces.MultiDiscrete):
            vector[2] = self.last_A
        else:
            # TODO: implement A/B alternation for spaces.Discrete
            pass

        return {
            "image": obs,
            "vector": np.array(vector, dtype=np.int16),
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
            # TODO: implement A/B alternation for spaces.Discrete
            pass

        return action


class StageWrapper(gym.Wrapper):
    NO_ENEMIES_CHEATCODE = "NNLOGO"

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
    ):
        super().__init__(env)
        self.reward_calculator = StageReward(
            stage,
            damage_factor,
            fixed_damage_punishment,
            forward_factor,
            backward_factor,
            time_punishment_factor / frameskip,
        )
        self.target_screen = screen
        self.damage_terminate = damage_terminate
        self.truncate_if_no_improvement = truncate_if_no_improvement
        # max number of frames: NES' FPS * seconds // frameskip
        self.max_number_of_frames_without_improvement = (60 * 60) // frameskip

        if obs_space == "ram":
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(53,))
        self.obs_space = obs_space

        self.no_enemies = no_enemies

    def reset(self, **kwargs):
        if self.no_enemies:
            self.unwrapped.em.clear_cheats()
        self.reward_calculator.reset()
        observation, info = self.env.reset(**kwargs)
        if self.no_enemies:
            # TODO: NOT WORKING, BREAKS THE GAME WHEN RESET
            for _ in range(1):
                observation, *_, info = self.env.step(
                    np.zeros(self.action_space.shape, dtype=np.int64)
                )
            self.unwrapped.em.add_cheat(self.NO_ENEMIES_CHEATCODE)
        self.prev_lives = self.unwrapped.data["lives"]
        if self.damage_terminate:
            self.prev_health = self.unwrapped.data["health"]
        return self.observation(observation), self.info(info)

    def step(self, action):
        while True:
            observation, _, terminated, truncated, info = self.env.step(action)
            if self.unwrapped.data["camera_y"] == 0:
                break

        if (terminated or truncated) and self.no_enemies:
            self.env.unwrapped.em.clear_cheats()

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

        screen_scale = 25
        pos_scale = 255
        x_speed_scale = 1.5
        y_speed_scale = 9
        facing_scale = 64
        type_scale = 255  # no idea about this one
        alive_scale = 255  # this one too

        variables = [
            # screen count
            obs[0x460] / screen_scale,
            # X position
            obs[0x0480] / pos_scale,
            # Y position
            obs[0x0600] / pos_scale,
            # X speed, composed by two bytes: X_hi and X_lo
            (obs[0x04C0] + obs[0x04E0] / pos_scale) / x_speed_scale,
            # Y speed, composed by two bytes: Y_hi and Y_lo.
            # TODO: I'm currently using only Y_hi, this is too much for me
            (obs[0x0680] if obs[0x0680] < 127 else obs[0x0680] - 256 + 5)
            / y_speed_scale,
            # which side Mega Man is facing
            obs[0x009B] / facing_scale,
            # bullets on screen
            obs[0x0060] - 192 / 20,
        ]

        n_objects = 8
        for i in range(n_objects):
            variables += [
                # i'th object's screen count
                obs[0x0470 + i] / screen_scale,
                # i'th object's X position
                obs[0x0490 + i] / pos_scale,
                # i'th object's Y position
                obs[0x0610 + i] / pos_scale,
                # i'th object's type
                obs[0x06F0 + i] / type_scale,
                # i'th object is alive? I'm not sure about this variable, but
                # it seems to indicate if each object is alive/rendered or not
                obs[0x007B + i] / alive_scale,
            ]

        n_bullets = 3
        for i in range(n_bullets):
            variables += [
                # i'th bullet's X position
                obs[0x0482 + i] / pos_scale,
                # i'th bullet's Y position
                obs[0x0602 + i] / pos_scale,
            ]

        return variables

    def reward(self, action):
        reward = self.reward_calculator.get_stage_reward(self.unwrapped.data)

        self.min_distance = self.reward_calculator.min_distance

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

        return terminated

    def truncated(self, truncated):
        return truncated or (
            self.reward_calculator.frames_since_last_improvement
            >= self.max_number_of_frames_without_improvement
        )

    def info(self, info):
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

    def reset(self):
        self.prev_distance = -1  # we don't know distance at start
        self.prev_lives = None
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

    def wavefront_expansion_reward(self, data, nightmare=False):
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

        screen = data["screen"]
        if screen > self.max_screen:
            self.max_screen = screen

        screen_offset = self.screen_offset_map[screen]

        self.x = (
            self.SCREEN_WIDTH * screen_offset["x"] + data["x"]
        ) // self.TILE_SIZE
        self.y = (
            self.SCREEN_HEIGHT * screen_offset["y"]
            # + min(data["y"] + self.MEGA_MAN_SPRITE_OFFSET_Y, 255)
            + min(data["y"], 239)
            # + data["y"]
        ) // self.TILE_SIZE

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
            # POST MORTEM: this ended in infinite positive loops as expected :)
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
