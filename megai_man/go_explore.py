"""https://github.com/ryanrudes/VQVAE-Clean/blob/main/explore.py"""

import cv2
from goexplore.algorithm import GoExplore
from goexplore.utils import *
from goexplore.wrappers import *
from rich import print

env = MegaMan(render_mode="human", record=False)

goexplore = GoExplore(env)

width = 11
height = 8
interpolation = cv2.INTER_AREA
grayscale = True
intensities = 8

cellfn = makecellfn(width, height, interpolation, grayscale, intensities)
goexplore.initialize(method="ram", cellfn=cellfn)

# goexplore.load("models/goxpl")

for _ in range(1000):
    goexplore.run(render=True)
    print(goexplore.report() + ", " + goexplore.status())
goexplore.save("models/goxpl")
goexplore.close()


env = MegaMan(render_mode="human", record="goxpl")
goexplore.env = env
obs, _ = env.reset(seed=goexplore.seed)
done = False
episode_reward = 0

while not done:
    obs, reward, terminated, truncated, info = goexplore.act()
    done = terminated or truncated
    episode_reward += reward
print(f"Reward: {episode_reward:.2f}")
goexplore.close()
