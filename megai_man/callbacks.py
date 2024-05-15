from pathlib import Path
from time import time

import numpy as np
import torch as th
from PIL import Image
from rllte.xplore.reward import RE3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean


class StageLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if (
            len(self.model.ep_info_buffer) > 0
            and len(self.model.ep_info_buffer[0]) > 0
        ):
            # TODO: log min/max rewards?
            # self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            hps = [ep_info["hp"] for ep_info in self.model.ep_info_buffer]
            self.logger.record("rollout/final_hp_min", min(hps))
            self.logger.record("rollout/final_hp_mean", safe_mean(hps))

            min_distances = [
                ep_info["min_distance"]
                for ep_info in self.model.ep_info_buffer
            ]
            self.logger.record(
                "rollout/min_distance_min",
                min(min_distances),
            )
            self.logger.record(
                "rollout/min_distance_mean",
                safe_mean(min_distances),
            )

            final_distances = [
                ep_info["distance"] for ep_info in self.model.ep_info_buffer
            ]
            self.logger.record(
                "rollout/final_distance_min",
                min(final_distances),
            )
            self.logger.record(
                "rollout/final_distance_mean",
                safe_mean(final_distances),
            )
            self.logger.record(
                "rollout/max_screen",
                max(
                    [
                        ep_info["max_screen"]
                        for ep_info in self.model.ep_info_buffer
                    ]
                ),
            )


class CurriculumCallback(BaseCallback):
    def __init__(self, freq: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.freq = freq
        self.screen = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.screen += 1
            self.training_env.set_attr("target_screen", self.screen)
        self.logger.record("rollout/curriculum_screen", self.screen)

        return True


class RE3Callback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.buffer = None
        self.device = "cuda"

    def init_callback(self, model):
        super().init_callback(model)
        assert isinstance(
            self.model, OnPolicyAlgorithm
        ), "support for off-policy algorithms will be added soon!!!"
        self.buffer = self.model.rollout_buffer
        self.irs = RE3(self.training_env, device=self.device)

    def _on_step(self) -> bool:
        observations = self.locals["obs_tensor"]
        actions = th.as_tensor(self.locals["actions"], device=self.device)
        rewards = th.as_tensor(self.locals["rewards"], device=self.device)
        dones = th.as_tensor(self.locals["dones"], device=self.device)
        next_observations = th.as_tensor(
            self.locals["new_obs"],
            device=self.device,
        )

        self.irs.watch(
            observations, actions, rewards, dones, dones, next_observations
        )
        return True

    def _on_rollout_end(self):
        obs = th.as_tensor(self.buffer.observations, device=self.device)
        actions = th.as_tensor(self.buffer.actions, device=self.device)
        rewards = th.as_tensor(self.buffer.rewards, device=self.device)
        dones = th.as_tensor(self.buffer.episode_starts, device=self.device)
        intrinsic_rewards = self.irs.compute(
            samples=dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                terminateds=dones,
                truncateds=dones,
                next_observations=obs,
            )
        )
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()


class TrainingStatsLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.model.rollout_buffer

        rewards = rollout_buffer.rewards
        rewards = np.hstack(rewards)
        self.logger.record("dists/rewards", th.from_numpy(rewards))

        value_targets = rollout_buffer.returns
        value_targets = np.hstack(value_targets)
        self.logger.record("dists/value_targets", th.from_numpy(value_targets))

        values = rollout_buffer.values
        values = np.hstack(values)
        self.logger.record("dists/values", th.from_numpy(values))

        advs = rollout_buffer.advantages
        advs = np.hstack(advs)
        self.logger.record("dists/advantages", th.from_numpy(advs))


class StopTrainingOnTimeBudget(BaseCallback):
    def __init__(self, budget: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.budget = budget

    def on_training_start(self, locals_, globals_):
        super().on_training_start(locals_, globals_)
        self.start = time()

    def _on_step(self) -> bool:
        continue_training = (time() - self.start) < self.budget

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because time budget of {self.budget} seconds was reached."
            )

        return continue_training
