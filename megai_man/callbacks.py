import os
from pathlib import Path
from time import time

import gymnasium as gym
import numpy as np
import torch as th
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import sync_envs_normalization

from megai_man.utils import evaluate_policy


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


class CurriculumNoEnemiesCallback(BaseCallback):
    parent: EvalCallback

    def __init__(self, first_enemy_screen: int = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.first_enemy_screen = first_enemy_screen

    def _on_training_start(self) -> None:
        self.last_screen = self.training_env.get_attr("last_screen")[0]
        self.screen_with_enemies = self.last_screen - 1
        if self.first_enemy_screen is not None:
            self.screen_with_enemies = self.first_enemy_screen

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "``CurriculumNoEnemiesCallback`` callback must be used with an ``EvalCallback``"
        max_screen = self.parent.episode_last_infos[0]["max_screen"]
        if (self.screen_with_enemies > 0) and (max_screen == self.last_screen):
            self.screen_with_enemies -= 1
            self.parent.training_env.env_method(
                "set_screen_with_enemies",
                self.screen_with_enemies,
            )
            self.parent.eval_env.env_method(
                "set_screen_with_enemies",
                self.screen_with_enemies,
            )

        self.logger.record(
            "eval/curriculum_enemies_screen",
            self.screen_with_enemies,
        )
        self.logger.record("eval/max_screen", max_screen)
        return True


class MaskableEvalCallback(EvalCallback):
    """
    Modified from sb3-contrib. Added `episode_last_infos` attribute.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    :param use_masking: Whether to use invalid action masks during evaluation
    """

    def __init__(self, *args, use_masking: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_masking = use_masking

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            # Note that evaluate_policy() has been patched to support masking
            (
                episode_rewards,
                episode_lengths,
                self.episode_last_infos,
            ) = evaluate_policy(
                self.model,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=self.use_masking,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(
                episode_rewards
            )
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
                )
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps",
                self.num_timesteps,
                exclude="tensorboard",
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


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


class CuriosityCallback(BaseCallback):
    def __init__(self, RewClass, verbose=0):
        super().__init__(verbose)
        self.buffer = None
        self.device = "cuda"
        self.RewClass = RewClass

    def init_callback(self, model):
        assert isinstance(
            model, OnPolicyAlgorithm
        ), "support for off-policy algorithms will be added soon!!!"
        self.buffer = model.rollout_buffer
        self.irs = self.RewClass(model.env, device=self.device)
        super().init_callback(model)

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
            observations,
            actions,
            rewards,
            dones,
            dones,
            next_observations,
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

        self.logger.record(
            "rollout/intrinsic_rewards",
            safe_mean(intrinsic_rewards.cpu().numpy()),
        )


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

        # this was never really useful for me
        # advs = rollout_buffer.advantages
        # advs = np.hstack(advs)
        # self.logger.record("dists/advantages", th.from_numpy(advs))


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
