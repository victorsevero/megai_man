import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


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


def get_tiles(prefix):
    tiles = []
    for tile_path in get_tiles_paths(prefix):
        tile = cv2.imread(tile_path)
        w, h = tile.shape[1::-1]
        assert w == h == 16, f"{tile_path} is not 16x16"
        tiles.append(tile)
    return tiles


def get_collision_map(img, tiles, start, end):
    width = img.shape[1]
    height = img.shape[0]
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

            if x // 16 == start[1] and y // 16 == start[0]:
                img[x : x + 16, y : y + 16] = [0, 255, 0]
                collision_map[x // 16, y // 16] = "s"
            elif x // 16 == end[1] and y // 16 == end[0]:
                img[x : x + 16, y : y + 16] = [0, 0, 255]
                collision_map[x // 16, y // 16] = "e"

    # cv2.imshow("Result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("output.png", img)
    print(f"Number of tiles found: {i}")
    return collision_map


def wavefront_expansion(grid):
    end_point = np.argwhere(grid == "e")[0]
    value_grid = np.zeros_like(grid, dtype=int)

    queue = [end_point]

    # up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.pop(0)
        for dir_x, dir_y in directions:
            neighbor = current[0] + dir_x, current[1] + dir_y
            if (
                0 <= neighbor[0] < grid.shape[0]
                and 0 <= neighbor[1] < grid.shape[1]
                and grid[neighbor] in ("", "s")
                and value_grid[neighbor] == 0
            ):
                value_grid[neighbor] = value_grid[(*current,)] + 1
                queue.append(neighbor)

    value_grid[grid == "w"] = -1

    return value_grid


def get_custom_cmap(N):
    colors = ["red", "yellow", "green"]
    custom_cmap = LinearSegmentedColormap.from_list("ryg", colors, N=N)
    return custom_cmap


def draw_grid(img):
    # inefficient
    arr = np.array(img)
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if x > 0 and x % 16 == 0 or y > 0 and y % 16 == 0:
                arr[x, y] = [127, 127, 127, 255]
    return Image.fromarray(arr, mode="RGBA")


if __name__ == "__main__":
    img_path = "images/bg/MegaManMapCutManBG.png"
    tiles_prefix = "cut"
    start = (0, 131)
    end = (193, 54)

    img = cv2.imread(img_path)
    assert (
        img.shape[0] % 16 == 0 and img.shape[1] % 16 == 0
    ), f"{img_path} is not made of 16x16 tiles"
    tiles = get_tiles(tiles_prefix)
    collision_map = get_collision_map(img, tiles, start, end)
    value_grid = wavefront_expansion(collision_map)
    np.save("cutman.npy", value_grid)

    custom_cmap = get_custom_cmap(value_grid.max())
    heatmap_arr = custom_cmap(value_grid, bytes=True)

    # fix walls
    heatmap_arr[value_grid == -1] = (0, 0, 0, 255)

    heatmap = Image.fromarray(heatmap_arr, mode="RGBA")
    heatmap = heatmap.resize(
        tuple(16 * x for x in value_grid.shape[::-1]),
        Image.Resampling.NEAREST,
    )
    heatmap = draw_grid(heatmap)
    heatmap.save("heatmap.png")
