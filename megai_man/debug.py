import numpy as np
import pygame
from env import make_venv


class Debugger:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font_size = 24
        self.font = pygame.font.SysFont("opensans", self.font_size)
        self.env = make_venv(
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
        self.action_mapper = ActionMapper(self.env)
        self.desired_fps = 60
        self.clock = pygame.time.Clock()
        self._setup_screen()

    def _setup_screen(self):
        obs = self.env.reset()
        last_obs = obs[0, -1]
        height, width = last_obs.shape
        resize_factor = 10
        text_area_width = 200
        self.screen = pygame.display.set_mode(
            (width * resize_factor + text_area_width, height * resize_factor)
        )
        self.width = width
        self.height = height
        self.resize_factor = resize_factor

    def render_observation(self, obs):
        last_obs_resized = np.repeat(
            np.repeat(obs, self.resize_factor, axis=0),
            self.resize_factor,
            axis=1,
        ).T
        last_obs_rgb = np.stack((last_obs_resized,) * 3, axis=-1)
        surface = pygame.surfarray.make_surface(last_obs_rgb)
        self.screen.blit(surface, (0, 0))

    def display_info(self, infos, rewards):
        info_text_x = f"X: {infos.get('x', 'N/A')}"
        info_text_y = f"Y: {infos.get('y', 'N/A')}"
        reward_text = f"Reward: {rewards}"

        info_surface_x = self.font.render(info_text_x, True, (255, 255, 255))
        info_surface_y = self.font.render(info_text_y, True, (255, 255, 255))
        reward_surface = self.font.render(reward_text, True, (255, 255, 255))

        padding = 10
        x_pos = self.width * self.resize_factor + padding
        y_pos = padding

        self.screen.blit(info_surface_x, (x_pos, y_pos))
        self.screen.blit(info_surface_y, (x_pos, y_pos + self.font_size + 5))
        self.screen.blit(
            reward_surface, (x_pos, y_pos + 2 * (self.font_size + 5))
        )

    def handle_events(self):
        paused = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    return False
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_SPACE
                ):
                    paused = not paused
                    if paused:
                        self.display_pause_message(
                            "Game Paused - Press SPACE to resume"
                        )
                    else:
                        return True
            if not paused:
                break
            self.clock.tick(10)
        return True

    def display_pause_message(self, message):
        overlay_rect = pygame.Rect(
            left=0,
            top=0,
            width=self.width * self.resize_factor,
            height=self.height * self.resize_factor,
        )
        overlay = pygame.Surface(
            (overlay_rect.width, overlay_rect.height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 128))

        pause_text = self.font.render(message, True, (255, 255, 255))
        text_rect = pause_text.get_rect(
            center=(overlay_rect.width / 2, overlay_rect.height / 2)
        )

        self.screen.blit(overlay, overlay_rect.topleft)
        self.screen.blit(
            pause_text,
            (overlay_rect.left + text_rect.x, overlay_rect.top + text_rect.y),
        )
        pygame.display.flip()

    def run(self):
        obs = self.env.reset()
        done = False
        rewards = [None]
        infos = [{}]

        while not done:
            self.screen.fill((0, 0, 0))
            self.render_observation(obs[0, -1])
            self.display_info(infos[0], rewards[0])
            pygame.display.flip()
            self.clock.tick(self.desired_fps)

            if not self.handle_events():
                break

            keys = pygame.key.get_pressed()
            action, buttons = self.action_mapper.map_keys(keys)
            obs, rewards, dones, infos = self.env.step([action])
            done = dones[0]
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
    debugger = Debugger()
    debugger.run()
