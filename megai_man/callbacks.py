from pathlib import Path
from time import time

from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from wrappers import StageReward


class MinDistanceCallback(BaseCallback):
    def __init__(self, verbose=0, show_image=False):
        super().__init__(verbose)
        self.show_image = show_image
        if self.show_image:
            self.bg = Image.open(
                Path("images/bg") / "MegaManMapCutManBG.png"
            ).convert("RGBA")
            sprite = Image.open(Path("images") / "mm_sprite.png").convert(
                "RGBA"
            )
            self.sprite = sprite.resize([int(1.5 * x) for x in sprite.size])

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
            self.logger.record("rollout/final_hp_max", max(hps))
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
