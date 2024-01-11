import os
from pathlib import Path

import cv2
import numpy as np


def get_tiles_paths(prefix):
    dir_path = "images/tiles"
    paths = os.listdir("images/tiles")
    paths = [
        str(Path(dir_path) / path)
        for path in paths
        if path.startswith(prefix)
        and not path.endswith(("-s.png", "-full.png"))
    ]
    return sorted(paths)


def exact_match(img1, img2):
    if img1.shape != img2.shape:
        return False
    diff = cv2.subtract(img1, img2)
    return np.all(diff == 0)


img_path = "images/bg/MegaManMapCutManBG.png"
img = cv2.imread(img_path)
width = img.shape[1]
height = img.shape[0]
assert (
    width % 16 == 0 and height % 16 == 0
), f"{img_path} is not made of 16x16 tiles"

tiles = []
for tile_path in get_tiles_paths("cut"):
    tile = cv2.imread(tile_path)
    w, h = tile.shape[1::-1]
    assert w == h == 16, f"{tile_path} is not 16x16"
    tiles.append(tile)

i = 0
locs = []
collision_map = np.empty((height // 16, width // 16), dtype=str)
collision_map[:, :] = "w"

for x in range(0, height, 16):
    for y in range(0, width, 16):
        cropped_tile = img[x : x + 16, y : y + 16]
        match = False
        for tile in tiles:
            if exact_match(tile, cropped_tile):
                i += 1
                locs.append((x, y))
                img[x : x + 16, y : y + 16] = [0, 0, 0]
                match = True
        if not match and np.any(cropped_tile):
            img[x : x + 16, y : y + 16] = [255, 255, 255]
            collision_map[x // 16, y // 16] = ""

        if x // 16 == 131 and y // 16 == 0:
            img[x : x + 16, y : y + 16] = [0, 255, 0]
            collision_map[x // 16, y // 16] = "s"
        elif x // 16 == 54 and y // 16 == 193:
            img[x : x + 16, y : y + 16] = [0, 0, 255]
            collision_map[x // 16, y // 16] = "e"


# cv2.imshow("Result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite("output.png", img)
print(f"Number of tiles found: {i}")
