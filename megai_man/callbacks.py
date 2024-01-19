from pathlib import Path

from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
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
            self.logger.record(
                "rollout/min_distance",
                min(
                    [
                        ep_info["min_distance"]
                        for ep_info in self.model.ep_info_buffer
                    ]
                ),
            )
            if self.show_image:
                # TODO: PLEASE REFACTOR THIS PLEASE
                x = self.model.ep_info_buffer[-1]["x"]
                y = self.model.ep_info_buffer[-1]["y"]
                screen = self.model.ep_info_buffer[-1]["screen"]
                x = int(
                    (
                        StageReward.SCREEN_WIDTH
                        * StageReward.SCREENS_OFFSETS_CUTMAN[screen]["x"]
                        + x
                    )
                )
                y = int(
                    (
                        StageReward.SCREEN_HEIGHT
                        * StageReward.SCREENS_OFFSETS_CUTMAN[screen]["y"]
                        + y
                    )
                )

                bg = self.bg.copy()
                bg.paste(self.sprite, (x, y), self.sprite)
                bg.show()

    # @staticmethod
    # def get_image(value_grid):
    #     custom_cmap = get_custom_cmap(value_grid.max())
    #     heatmap_arr = custom_cmap(value_grid, bytes=True)

    #     # fix walls
    #     heatmap_arr[value_grid == -1] = (0, 0, 0, 255)

    #     heatmap = Image.fromarray(heatmap_arr, mode="RGBA")
    #     heatmap = heatmap.resize(
    #         tuple(16 * x for x in value_grid.shape[::-1]),
    #         Image.Resampling.NEAREST,
    #     )
    #     heatmap = draw_grid(heatmap)
    #     return heatmap
