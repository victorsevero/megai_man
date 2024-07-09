import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# arbitrary high level for infinite pits
INFINITE_PIT_HEIGHT = 1_000


def get_wall_tiles_paths(prefix):
    dir_path = "images/tiles"
    paths = os.listdir("images/tiles")
    paths = [
        str(Path(dir_path) / path)
        for path in paths
        if path.startswith(prefix)
        and not path.endswith(("-l.png", "-p.png", "-full.png"))
    ]
    return sorted(paths)


def get_ladder_tiles_paths(prefix):
    dir_path = "images/tiles"
    paths = os.listdir("images/tiles")
    paths = [
        str(Path(dir_path) / path)
        for path in paths
        if path.startswith(prefix) and ("-l" in path)
    ]
    return sorted(paths)


def get_spike_tiles_paths(prefix):
    dir_path = "images/tiles"
    paths = os.listdir("images/tiles")
    paths = [
        str(Path(dir_path) / path)
        for path in paths
        if path.startswith(prefix) and ("-p" in path)
    ]
    return sorted(paths)


def exact_match(img1, img2):
    if img1.shape != img2.shape:
        return False
    diff = cv2.subtract(img1, img2)
    return np.all(diff == 0)


# 1960, 698 = 122, 43 = x, y
def get_tiles(prefix):
    wall_tiles = []
    for tile_path in get_wall_tiles_paths(prefix):
        tile = cv2.imread(tile_path)
        w, h = tile.shape[1::-1]
        assert w == h == 16, f"{tile_path} is not 16x16"
        wall_tiles.append(tile)

    ladder_tiles = []
    for tile_path in get_ladder_tiles_paths(prefix):
        tile = cv2.imread(tile_path)
        w, h = tile.shape[1::-1]
        assert w == h == 16, f"{tile_path} is not 16x16"
        ladder_tiles.append(tile)

    spike_tiles = []
    for tile_path in get_spike_tiles_paths(prefix):
        tile = cv2.imread(tile_path)
        w, h = tile.shape[1::-1]
        assert w == h == 16, f"{tile_path} is not 16x16"
        spike_tiles.append(tile)

    return wall_tiles, ladder_tiles, spike_tiles


def get_collision_map(img, wall_tiles, ladder_tiles, spike_tiles, start, end):
    width = img.shape[1]
    height = img.shape[0]
    collision_map = np.empty((height // 16, width // 16), dtype=str)
    collision_map[:, :] = "w"

    for x in range(0, height, 16):
        for y in range(0, width, 16):
            cropped_tile = img[x : x + 16, y : y + 16]

            wall_match = False
            for tile in wall_tiles:
                if exact_match(tile, cropped_tile):
                    img[x : x + 16, y : y + 16] = [0, 0, 0]
                    wall_match = True
                    break

            if not wall_match:
                ladder_match = False
                for tile in ladder_tiles:
                    if exact_match(tile, cropped_tile):
                        img[x : x + 16, y : y + 16] = [127, 127, 127]
                        collision_map[x // 16, y // 16] = "l"
                        ladder_match = True
                        break

            if not (wall_match or ladder_match):
                spike_match = False
                for tile in spike_tiles:
                    if exact_match(tile, cropped_tile):
                        img[x : x + 16, y : y + 16] = [223, 223, 223]
                        # P as in sPike; S is taken for Start
                        collision_map[x // 16, y // 16] = "p"
                        spike_match = True
                        break

            if not (wall_match or ladder_match or spike_match) and np.any(
                cropped_tile
            ):
                img[x : x + 16, y : y + 16] = [255, 255, 255]
                collision_map[x // 16, y // 16] = ""
            if (y // 16, x // 16) == start:
                img[x : x + 16, y : y + 16] = [0, 255, 0]
                collision_map[x // 16, y // 16] = "s"
            elif (y // 16, x // 16) == end:
                img[x : x + 16, y : y + 16] = [0, 0, 255]
                collision_map[x // 16, y // 16] = "e"

    cv2.imwrite("output.png", img)
    return collision_map


def find_chokepoints(grid):
    valid = ("", "l")
    choke_points = []
    rows, cols = grid.shape
    for y in range(1, rows - 1):
        for x in range(cols - 1):
            if grid[y, x] in valid and grid[y, x + 1] in valid:
                if (
                    grid[y - 1, x] not in valid
                    or grid[y - 1, x + 1] not in valid
                ) and (
                    grid[y + 1, x] not in valid
                    or grid[y + 1, x + 1] not in valid
                ):
                    choke_points.append(((y, x), (y, x + 1)))
    return choke_points


def get_relative_height(grid, node):
    node_y, node_x = node

    if grid[node_y, node_x] == "w":
        return -1
    # TODO: if it still doesn't work, change mid-ladder heights
    if grid[node_y, node_x] == "l":
        return 0

    height = 0
    for y in range(node_y + 1, grid.shape[0]):
        if grid[y, node_x] in ("w", "l"):
            return height
        height += 1

    return INFINITE_PIT_HEIGHT


def is_inside_NxN_square(node1, node2, N=3):
    return (abs(node1[0] - node2[0]) <= N) and (abs(node1[1] - node2[1]) <= N)


def get_distance(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


def is_neighborhood_close_enough(
    current,
    neighbor,
    value_grid,
    label_grid,
    max_distance=3,
):
    neighbor_y, neighbor_x = neighbor

    y_range = [neighbor_y, neighbor_y + max_distance + 1]
    y_range = [max(min(y, value_grid.shape[0]), 0) for y in y_range]
    x_range = [neighbor_x - max_distance, neighbor_x + max_distance + 1]
    x_range = [max(min(x, value_grid.shape[1]), 0) for x in x_range]
    for y in range(*y_range):
        for x in range(*x_range):
            if value_grid[y, x] != 0:
                continue
            if label_grid[y, x] != "":
                continue
            if (y, x) == neighbor:
                continue
            if (
                is_inside_NxN_square((y, x), current, N=max_distance)
                and get_relative_height(label_grid, (y, x)) == 0
                and get_distance((y, x), current) < 2 * max_distance
            ):
                return True

    return False


def is_above_spike(node, label_grid, n=3):
    y = node[0]
    x = node[1]
    try:
        for i in range(n):
            if label_grid[y, x + i] == "p":
                return True
    except IndexError:
        return False
    return False


def is_valid_path(current, neighbor, value_grid, label_grid):
    max_distance = 3
    current_y = current[0]
    neighbor_y = neighbor[0]
    neighbor_height = get_relative_height(label_grid, neighbor)

    return (
        (neighbor_height == INFINITE_PIT_HEIGHT)
        or (current_y > neighbor_y)
        or ((current_y < neighbor_y) and ((neighbor_height < max_distance)))
        or (neighbor_height == 0)
        or (
            is_neighborhood_close_enough(
                current,
                neighbor,
                value_grid,
                label_grid,
                max_distance,
            )
        )
    ) and not is_above_spike(neighbor, label_grid, n=3)


def wavefront_expansion(grid, allow_chokepoints=False, allow_any_jump=False):
    end_point = np.argwhere(grid == "e")[0]
    value_grid = np.zeros_like(grid, dtype=int)

    if not allow_chokepoints:
        chokepoints = find_chokepoints(grid)
    else:
        chokepoints = []

    queue = [(*end_point,)]

    # up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.pop(0)
        for dir_y, dir_x in directions:
            neighbor = (current[0] + dir_y, current[1] + dir_x)
            if (
                0 <= neighbor[0] < grid.shape[0]
                and 0 <= neighbor[1] < grid.shape[1]
                and grid[neighbor] in ("", "s", "l")
                and value_grid[neighbor] == 0
                and (
                    allow_chokepoints
                    or not tuple(sorted((current, neighbor))) in chokepoints
                )
                and (
                    allow_any_jump
                    or is_valid_path(current, neighbor, value_grid, grid)
                )
            ):
                value_grid[neighbor] = value_grid[current] + 1
                queue.append((*neighbor,))

    # walls and chokepoints
    value_grid[value_grid == 0] = -1
    # fix end point
    value_grid[grid == "e"] = 0

    return value_grid


def get_custom_cmap(N):
    colors = ["red", "yellow", "green", "blue"]
    custom_cmap = LinearSegmentedColormap.from_list("ryg", colors, N=N)
    return custom_cmap


def draw_grid(img):
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
    wall_tiles, ladder_tiles, spike_tiles = get_tiles(tiles_prefix)
    collision_map = get_collision_map(
        img,
        wall_tiles,
        ladder_tiles,
        spike_tiles,
        start,
        end,
    )
    value_grid = wavefront_expansion(collision_map)
    np.save(
        Path("megai_man/custom_integrations/MegaMan-v2-Nes/") / "cutman.npy",
        value_grid,
    )

    # custom_cmap = get_custom_cmap(value_grid.max())
    custom_cmap = plt.get_cmap("gist_rainbow", value_grid.max())
    heatmap_arr = custom_cmap(value_grid, bytes=True)

    # fix walls
    heatmap_arr[value_grid == -1] = (0, 0, 0, 255)
    # fix ladders
    heatmap_arr[collision_map == "l"] = (0x42, 0x28, 0x0E, 255)
    # fix spikes
    heatmap_arr[collision_map == "p"] = (223, 223, 223, 255)

    heatmap = Image.fromarray(heatmap_arr, mode="RGBA")
    heatmap = heatmap.resize(
        tuple(16 * x for x in value_grid.shape[::-1]),
        Image.Resampling.NEAREST,
    )
    heatmap = draw_grid(heatmap)
    heatmap.save("heatmap.png")
