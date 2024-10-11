# Reinforcement Learning for Mega Man (NES)

This repository contain a full reproducible pipeline on how to train a Deep Reinforcement Learning agent to complete the Cut Man stage of the first game of the Mega Man series from NES.

I started it with the intention to make a full run of the game using AI only (except for some parts where I intended to hardcode a few commands, like selecting a stage). It was much harder than I expected. I always worked alone on it and I don't have the time nor the energy to make a full run of the game by myself.

All being said, contributions are very welcome! I'd be glad to answer any questions and work on solutions with other people. If you're interested, contact me on [`stable-retro` chat from Farama Foundation Discord Server](https://discord.gg/pJtSNbfr49) (my handler is `@el_sevs`).

## Environment specifications

### Observations

The observation space is a `Box(0, 255, (3, 84, 84), uint8)` gymnasium space instance.

Observation preprocessing is done in this order:
1. We use "classical" frame skipping (as in [[1]](https://doi.org/10.48550/arxiv.1312.5602)) and an additional dynamic frame skipping is applied when the character is not visible on the screen (when it just took damage) until we have a visual clue of where the character is
2. The frames are warped to 84x84 greyscale images
3. We stack 3 frames (as in [[1]](https://doi.org/10.48550/arxiv.1312.5602)) to give the model a sense of movement: we show to the agent the current frame and the past two frames after frame skipping

### Actions

The action space is a `MultiDiscrete([2 5 2])` gymnasium space instance, which means it has 3 axes: one for jumping, one for directionals; and one for shooting (for details, check `megai_man/custom_integrations/MegaMan-v1-Nes/scenario.json`).

Shooting action is masked during pre-training (when the agent is learning on the environment without enemies) since it only needs it when there are shootable objects on the screen. We decrease entropy that way for faster learning.

### Rewards

This is where things get really messy. There are two types of reward that act exclusively to one another: stage reward and boss reward. Stage reward is applied during most of the stage and boss reward is applied as soon as the player enters the boss chamber.

* Stage reward is a composition of two main factors: distance progress and damage taken. A distance map was made using `megai_man/collision_mapper.py`, which applies Wavefront Expansion with heuristics to calculate which paths the agent can take from the start of the stage to the boss chamber (the end of the stage) and how good is each path. The agent doesn't have direct access to it, but it's used to give positive or negative rewards depending on how much closer the agent is getting to the goal. There are lots of details that don't fit here (but were important to achieve stage completion), but the most important thing is that the positive reward it gets from going forwards is smaller in magnitude than the negative reward it gets from going backwards. This is critical to progress (the agent gets stuck in an infinite loop going forwards and backwards without it). This idea was taken from [[8]](https://github.com/DarkAutumn/triforce).

Boss reward **[NOT YET IMPLEMENTED]** is as simple as a positive reward for damaging the enemy and a negative reward for taking damage.

### Termination and Truncation

Episode will be terminated if any of these conditions apply:
* Player health drops to 0;
* A life was lost;
* Player touched a spike;
* During pre-training, the agent falls from a higher to a lower previous screen;
* Player reached boss chamber (this was done to clearly separate these two situations and train them separately).

Episode will be truncated if any of these conditions apply:
* After `(60 * 60) // frameskip_count = 900 timesteps` (roughly equivalent to 60 seconds of real gameplay) since the last time the agent got to the closest point to the boss chamber it ever got in that episode;
* After `(60 * 360) // frameskip = 5400 timesteps` (roughly equivalent to 360 seconds of real gameplay) since the start of the episode.

## How to reproduce

1. Extract ROM image from your own Mega Man (US version) original NES cartridge and save it as `megai_man/custom_integrations/MegaMan-v1-Nes/rom.nes`
2. Create a virtual environment and install dependencies from `pyproject.toml` with [Poetry](https://python-poetry.org/) or install dependencies directly from `requirements.txt` with [pip](https://packaging.python.org/en/latest/key_projects/#pip)
3. Make a modified version of the ROM using [Rock and Roll Level Editor](https://www.romhacking.net/utilities/246/). You need to remove all the enemies from Cut Man's stage for the pre-training step. Save the new ROM as `megai_man/custom_integrations/MegaMan-noEnemies-Nes/rom.nes`
4. Run `megai_man/train.py` for at least 25M steps (I ran it for 27M)
5. Rename the resulting model to `models/no_enemies_complete.zip`
6. Run `megai_man/train_pretrained.py` for at least 30M steps. The best model will be stored at `models/cutman_pretrained_noTermBackScreen_gamma95_10spikepunish_enemies_curriculum_best/best_model.zip`
7. You can see it in action by running `megai_man/test.py`. If you want to record a movie, change the `record` parameter of the `make_venv` function to `record="."`. The concept of movie here is taken from the Tool Assisted Speedrun community, as it's only a button press record, not the actual frames from the game. It will be stored as a `bk2` file
8. If you want to render the movie into an actual video, run `megai_man/playback_movie.py MegaMan-v1-Nes-CutMan-000000.bk2`

Results of any training step will show up inside `logs` directory. You can visualize them with TensorBoard by running `tensorboard --logdir logs` inside project root directory.

PS: you might get away with the third step (modifying the original ROM) by skipping to step 6, which will remove enemies from early screens (and progressively add more enemies as training goes on) by directly modifying RAM as the AI plays it. I didn't test it this way and, even if it works, it will probably be much slower since there will be no action masking. This could be a future improvement. Feel free to contribute!


## Credits

Images in `images/bg` were ripped by Rick N. Bruns ([taken from here](https://nesmaps.com/maps/MegaMan/MegaManBG.html)).

Tilesets in `images/tiles` were ripped by Mister Mike ([taken from here](https://www.spriters-resource.com/fullview/260/)).

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (n.d.). Playing Atari with Deep Reinforcement Learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1312.5602
2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy optimization Algorithms. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1707.06347
3. Berner, C., Brockman, G., Chan, B., Cheung, V., Debiak, P., Dennison, C., Farhi, D., Fischer, Q., Hashme, S., Hesse, C., Józefowicz, R., Gray, S., Olsson, C., Pachocki, J., Petrov, M., De Oliveira Pinto, H. P., Raiman, J., Salimans, T., Schlatter, J., . . . Zhang, S. (2019). Dota 2 with Large Scale Deep Reinforcement Learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1912.06680
4. Anssi, K., Christian, S., & Ville, H. (2020). Action space shaping in deep reinforcement learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2004.00980
5. Andrychowicz, M., Raichuk, A., Stańczyk, P., Orsini, M., Girgin, S., Marinier, R., Hussenot, L., Geist, M., Pietquin, O., Michalski, M., Gelly, S., & Bachem, O. (2020). What matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2006.05990
6. Huang, S., Fernand Julien Dossa, R., Raffin, A., Kanervisto, A., & Wang, W. (2022, March 25). The 37 Implementation Details of Proximal Policy Optimization. The ICLR Blog Track. Retrieved October 10, 2024, from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
7. Jones, A. (n.d.). Debugging reinforcement learning systems. andy jones. Retrieved October 10, 2024, from https://andyljones.com/posts/rl-debugging.html
8. DarkAutumn/triforce: A deep learning agent for The Legend of Zelda (nes). (n.d.). GitHub. https://github.com/DarkAutumn/triforce
9. Mega Man (NES)/RAM map - Data Crystal. (2024). In Mega Man (NES)/RAM map. Retrieved October 10, 2024, from https://datacrystal.tcrf.net/wiki/Mega_Man_(NES)/RAM_map
10. Yliluoma, J. (2013, October 6). Rockman / Mega Man Source Code: Main disassembly (banks 5,6,7). Joel Yliluoma. Retrieved October 10, 2024, from https://bisqwit.iki.fi/jutut/megamansource/maincode.txt
