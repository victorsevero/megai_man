# https://github.com/ryanrudes/VQVAE-Clean/blob/main/explore.py
# from multiprocessing import Pool

import cv2
from goexplore.algorithm import GoExplore
from goexplore.utils import *
from goexplore.wrappers import *
from gymnasium.utils.env_checker import check_env
from rich import print

env = MegaMan()
# check_env(env)


iterations = 100_000

# env = make_env(
#     n_envs=1,
#     state="CutMan",
#     frameskip=4,
#     frame_stack=2,
#     truncate_if_no_improvement=False,
#     obs_space="screen",
#     action_space="multi_discrete",
#     crop_img=True,
#     invincible=False,
#     render_mode="human",
#     record=False,
#     multi_input=False,
#     curriculum=False,
#     fixed_damage_punishment=1,
#     forward_factor=0.5,
#     backward_factor=0.55,
#     time_punishment_factor=0,
#     no_enemies=False,
#     _enforce_subproc=False,
# )
# env = Pong()
goexplore = GoExplore(env)

width = 11
height = 8
interpolation = cv2.INTER_AREA
grayscale = True
intensities = 8

cellfn = makecellfn(width, height, interpolation, grayscale, intensities)
goexplore.initialize(method="ram", cellfn=cellfn)

while goexplore.highscore == 0:
    goexplore.run(render=True)
    print(goexplore.report() + ", " + goexplore.status())
