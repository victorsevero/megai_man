import numpy as np
import pygame
from env import make_venv


def play():
    pygame.init()

    env = make_venv(
        n_envs=1,
        state="CutMan",
        sticky_prob=0.0,
        damage_terminate=False,
        truncate_if_no_improvement=False,
        obs_space="screen",
        action_space="multi_discrete",
        render_mode=None,
        record=".",
    )
    action_mapper = ActionMapper(env)

    obs = env.reset()
    done = False
    current_length = 0

    # FPS things
    desired_fps = 60
    clock = pygame.time.Clock()

    last_obs = obs[0, -1]
    height, width = last_obs.shape
    resize_factor = 10
    screen = pygame.display.set_mode(
        (width * resize_factor, height * resize_factor)
    )

    while not done:
        last_obs = obs[0, -1]
        last_obs_resized = np.repeat(
            np.repeat(last_obs, resize_factor, axis=0),
            resize_factor,
            axis=1,
        ).T
        last_obs_rgb = np.stack((last_obs_resized,) * 3, axis=-1)
        surface = pygame.surfarray.make_surface(last_obs_rgb)

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(desired_fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                    break
                elif event.key == pygame.K_SPACE:
                    wait = True
                    while wait:
                        for event in pygame.event.get():
                            if (
                                event.type == pygame.KEYDOWN
                                and event.key == pygame.K_SPACE
                            ):
                                wait = False

        keys = pygame.key.get_pressed()

        if done:
            break

        action, buttons = action_mapper.map_keys(keys)
        obs, rewards, dones, infos = env.step([action])
        done = dones[0]
        current_length += 1
        print(f"Action: {buttons}")
        print(f"Reward: {rewards[0]}")
        if "terminal_observation" in infos[0]:
            del infos[0]["terminal_observation"]
        print(f"Infos: {infos[0]}\n")


class ActionMapper:
    def __init__(self, env):
        env_ = env.unwrapped.envs[0].unwrapped

        sizes = env_.action_space.nvec
        ranges = [np.arange(size) for size in sizes]
        mesh = np.meshgrid(*ranges, indexing="ij")
        actions_arrs = np.stack(mesh, axis=-1).reshape(-1, len(sizes))

        self.actions = {}
        for action_arr in actions_arrs:
            buttons = tuple(sorted(env_.get_action_meaning(action_arr)))
            self.actions[buttons] = action_arr

    def map_keys(self, keys):
        buttons = []
        if keys[pygame.K_UP]:
            buttons.append("UP")
        if keys[pygame.K_DOWN]:
            buttons.append("DOWN")
        if keys[pygame.K_LEFT]:
            buttons.append("LEFT")
        if keys[pygame.K_RIGHT]:
            buttons.append("RIGHT")
        if keys[pygame.K_x]:
            buttons.append("A")
        if keys[pygame.K_z]:
            buttons.append("B")

        return self._map_action(buttons), buttons

    def _map_action(self, buttons):
        return self.actions[tuple(sorted(buttons))]


if __name__ == "__main__":
    play()
