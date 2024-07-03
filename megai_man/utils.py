import numpy as np
import pygame


class ActionMapper:
    def __init__(self, env, action_space="multi_discrete"):
        if action_space == "multi_discrete":
            sizes = env.action_space.nvec
            ranges = [np.arange(size) for size in sizes]
            mesh = np.meshgrid(*ranges, indexing="ij")
            actions_arrs = np.stack(mesh, axis=-1).reshape(-1, len(sizes))
        else:
            actions_arrs = np.arange(env.action_space.n)

        self.actions = {}
        for action_arr in actions_arrs:
            buttons = tuple(sorted(env.get_action_meaning(action_arr)))
            self.actions[buttons] = action_arr

    def map_keys(self, keys):
        buttons = []
        if keys[pygame.K_UP]:
            buttons.append("UP")
        elif keys[pygame.K_DOWN]:
            buttons.append("DOWN")
        if keys[pygame.K_LEFT]:
            buttons.append("LEFT")
        elif keys[pygame.K_RIGHT]:
            buttons.append("RIGHT")
        if keys[pygame.K_x]:
            buttons.append("A")
        if keys[pygame.K_z]:
            buttons.append("B")

        return self.map_action(buttons), buttons

    def map_action(self, buttons):
        return self.actions[tuple(sorted(buttons))]
