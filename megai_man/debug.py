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
        self.debug_info_start_y = 10
        self.debug_messages = []

    def _setup_screen(self):
        obs = self.env.reset()
        last_obs = obs[0, -1]
        height, width = last_obs.shape
        resize_factor = 10
        self.text_area_width = 200
        self.screen = pygame.display.set_mode(
            (
                width * resize_factor + self.text_area_width,
                height * resize_factor,
            )
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
        self.debug_messages.append((infos, rewards))

        max_messages = (self.height * self.resize_factor) // (
            self.font_size * 3 + 20
        )

        self.debug_messages = self.debug_messages[-max_messages:]

        debug_area_rect = pygame.Rect(
            self.width * self.resize_factor,
            0,
            self.text_area_width,
            self.height * self.resize_factor,
        )
        self.screen.fill((0, 0, 0), debug_area_rect)

        y_pos = self.debug_info_start_y

        for info, reward in self.debug_messages:
            info_lines = [
                f"X: {info.get('x', 'N/A')}",
                f"Y: {info.get('y', 'N/A')}",
                f"Reward: {reward}",
            ]
            self.render_text(
                info_lines,
                self.width * self.resize_factor + 10,
                y_pos,
            )
            y_pos += self.font_size * len(info_lines) + 15

    def render_text(self, text_lines, x, y):
        box_height = self.font_size * len(text_lines) + 10
        box_width = self.text_area_width - 10 - 1
        background_color = (0, 0, 0)
        text_color = (255, 255, 255)
        border_color = (255, 255, 255)

        box_rect = pygame.Rect(x, y, box_width, box_height)
        pygame.draw.rect(self.screen, background_color, box_rect)

        border_thickness = 2
        pygame.draw.rect(self.screen, border_color, box_rect, border_thickness)

        for i, line in enumerate(text_lines):
            text_surface = self.font.render(line, True, text_color)
            self.screen.blit(text_surface, (x + 5, y + 5 + i * self.font_size))

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
                        self.display_pause_message()
                    else:
                        return True
            if not paused:
                break
            self.clock.tick(10)
        return True

    def display_pause_message(self):
        overlay_rect = pygame.Rect(
            0,
            0,
            self.width * self.resize_factor,
            self.height * self.resize_factor,
        )
        overlay = pygame.Surface(
            (overlay_rect.width, overlay_rect.height),
            pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 128))

        pause_text = self.font.render(
            "Game Paused - Press SPACE to resume",
            True,
            (255, 255, 255),
        )
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
        rewards = ["N/A"]
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
