import numpy as np
import pygame
import torch
from env import make_venv
from stable_baselines3 import PPO


class Debugger:
    def __init__(self, model=None, record=False):
        pygame.init()
        pygame.font.init()
        self.font_size = 18
        self.font = pygame.font.SysFont("opensans", self.font_size)
        self.small_font = pygame.font.SysFont("opensans", 12)
        frameskip = 1
        self.env = make_venv(
            n_envs=1,
            state="CutMan",
            sticky_prob=0.0,
            frameskip=frameskip,
            damage_terminate=False,
            damage_factor=1 / 10,
            truncate_if_no_improvement=False,
            obs_space="screen",
            action_space="multi_discrete",
            render_mode=None,
            record=record,
        )
        self.retro_env = self.env.unwrapped.envs[0].unwrapped
        self.model = model
        if model is not None:
            self.model = PPO.load(model, env=self.env)
        self.action_mapper = ActionMapper(self.retro_env)
        self.desired_fps = 60 // frameskip
        self.clock = pygame.time.Clock()
        self._setup_screen()
        self.debug_info_start_y = 10
        self.debug_messages = []
        self.box_text = (
            "Step: {step}, Action: {action}, Reward: {reward}, VF: {vf}\n"
            "{probs}\n"
            "X: {x}, Y: {y}\n"
        )
        self.n_lines = len(self.box_text.split("\n"))
        self.pad = 5
        self.graph_color = (255, 255, 255)

    def _setup_screen(self):
        width, height = 240, 224
        resize_factor = 5

        obs = self.env.reset()
        last_obs = obs[0, -1]
        env_height, env_width = last_obs.shape

        self.text_area_width = 800
        self.screen = pygame.display.set_mode(
            (
                env_width * resize_factor
                + width * resize_factor
                + self.text_area_width,
                height * resize_factor,
            )
        )
        self.env_width = env_width
        self.env_height = env_height
        self.width = width
        self.height = height
        self.resize_factor = resize_factor

    def render_screen(self, obs):
        last_obs_resized = np.repeat(
            np.repeat(obs, self.resize_factor, axis=0),
            self.resize_factor,
            axis=1,
        ).T
        last_obs_rgb = np.stack((last_obs_resized,) * 3, axis=-1)
        surface = pygame.surfarray.make_surface(last_obs_rgb)
        self.screen.blit(surface, (0, 0))

        screen = self._get_screen()
        screen = np.repeat(
            np.repeat(screen, self.resize_factor, axis=0),
            self.resize_factor,
            axis=1,
        ).T
        screen = np.transpose(screen, (1, 2, 0))
        surface = pygame.surfarray.make_surface(screen)
        self.screen.blit(surface, (self.env_width * self.resize_factor, 0))

    def display_info(self, action, probs, vf, infos, reward):
        self.debug_messages.append(
            (self.step, action, probs, vf, infos, reward)
        )

        max_messages = (self.height * self.resize_factor) // (
            (self.font_size + self.pad + 1) * self.n_lines
        )

        self.debug_messages = self.debug_messages[-max_messages:]

        debug_area_rect = pygame.Rect(
            (self.env_width + self.width) * self.resize_factor,
            0,
            self.text_area_width,
            self.height * self.resize_factor,
        )
        self.screen.fill((0, 0, 0), debug_area_rect)

        y_pos = self.debug_info_start_y

        for step, action, probs, vf, info, reward in self.debug_messages:
            info_lines = self.box_text.format(
                step=step,
                action=action,
                probs=probs,
                vf=vf,
                x=info.get("x", "N/A"),
                y=info.get("y", "N/A"),
                reward=reward,
            ).split("\n")

            self.render_text(
                info_lines,
                (self.env_width + self.width) * self.resize_factor
                + 2 * self.pad,
                y_pos,
            )
            y_pos += (self.font_size + self.pad) * len(info_lines)

    def render_text(self, text_lines, x, y):
        box_height = self.font_size * len(text_lines) + 2 * self.pad
        box_width = self.text_area_width - 2 * self.pad - 1
        background_color = (0, 0, 0)
        text_color = (255, 255, 255)
        border_color = (255, 255, 255)

        box_rect = pygame.Rect(x, y, box_width, box_height)
        pygame.draw.rect(self.screen, background_color, box_rect)

        border_thickness = 2
        pygame.draw.rect(self.screen, border_color, box_rect, border_thickness)

        for i, line in enumerate(text_lines):
            text_surface = self.font.render(line, True, text_color)
            self.screen.blit(
                text_surface,
                (x + self.pad, y + i * self.font_size),
            )

    # def display_plots(self):
    #     if self.step % 10 == 0:
    #         px = 1 / plt.rcParams["figure.dpi"]
    #         size = (84 * self.resize_factor * px, 84 * self.resize_factor * px)
    #         fig = plt.figure(figsize=size)
    #         ax = fig.gca()
    #         sns.lineplot(data=self.cum_rewards, ax=ax)
    #         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #         ax.grid(False)
    #         plt.title("Cumulative Rewards")
    #         plt.tight_layout()

    #         canvas = agg.FigureCanvasAgg(fig)
    #         canvas.draw()
    #         renderer = canvas.get_renderer()
    #         raw_data = renderer.buffer_rgba()
    #         size = canvas.get_width_height()

    #         graph_surface = pygame.image.frombuffer(raw_data, size, "RGBA")
    #         self.screen.blit(graph_surface, (0, 84 * self.resize_factor + 1))

    def display_plots(self):
        if len(self.cum_rewards) < 2:
            return

        window_size = 100
        displayed_rewards = self.cum_rewards[-window_size:]
        start_index = max(0, len(self.cum_rewards) - window_size)

        left_margin = 30
        bottom_margin = 20
        top_margin = 10
        right_margin = 10

        graph_rect = pygame.Rect(
            0,
            84 * self.resize_factor + 1,
            84 * self.resize_factor,
            84 // 2 * self.resize_factor,
        )
        adjusted_rect = pygame.Rect(
            graph_rect.left + left_margin,
            graph_rect.top + top_margin,
            graph_rect.width - (left_margin + right_margin),
            graph_rect.height - (top_margin + bottom_margin),
        )

        self.screen.fill((0, 0, 0), graph_rect)
        pygame.draw.line(
            self.screen,
            (128, 128, 128),
            (adjusted_rect.left, adjusted_rect.bottom),
            (adjusted_rect.right, adjusted_rect.bottom),
        )
        pygame.draw.line(
            self.screen,
            (128, 128, 128),
            (adjusted_rect.left, adjusted_rect.top),
            (adjusted_rect.left, adjusted_rect.bottom),
        )

        reward_range = max(displayed_rewards) - min(displayed_rewards) or 1

        num_labels = 3
        x_labels = [
            str(i)
            for i in range(
                start_index,
                start_index + len(displayed_rewards) + 1,
                max(1, len(displayed_rewards) // (num_labels - 1)),
            )
        ]
        y_label_values = np.linspace(
            min(displayed_rewards),
            max(displayed_rewards),
            num=num_labels,
        )

        for i, label in enumerate(x_labels):
            text_surface = self.small_font.render(
                label,
                True,
                self.graph_color,
            )
            x_pos = (
                adjusted_rect.left
                + (i / (len(x_labels) - 1)) * adjusted_rect.width
                - text_surface.get_width() / 2
            )
            self.screen.blit(text_surface, (x_pos, adjusted_rect.bottom + 5))

        for value in y_label_values:
            if reward_range < 5:
                label = f"{value:0.1f}"
            else:
                label = f"{value:0.0f}"
            text_surface = self.small_font.render(
                label,
                True,
                self.graph_color,
            )
            text_y = (
                adjusted_rect.bottom
                - ((value - min(displayed_rewards)) / reward_range)
                * adjusted_rect.height
                - text_surface.get_height() / 2
            )
            self.screen.blit(
                text_surface,
                (adjusted_rect.left - text_surface.get_width() - 5, text_y),
            )

        for i in range(len(displayed_rewards) - 1):
            start_pos = (
                adjusted_rect.left
                + (i / (len(displayed_rewards) - 1)) * adjusted_rect.width,
                adjusted_rect.bottom
                - (
                    (displayed_rewards[i] - min(displayed_rewards))
                    / reward_range
                )
                * adjusted_rect.height,
            )
            end_pos = (
                adjusted_rect.left
                + ((i + 1) / (len(displayed_rewards) - 1))
                * adjusted_rect.width,
                adjusted_rect.bottom
                - (
                    (displayed_rewards[i + 1] - min(displayed_rewards))
                    / reward_range
                )
                * adjusted_rect.height,
            )
            pygame.draw.line(
                self.screen,
                self.graph_color,
                start_pos,
                end_pos,
                2,
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
            (self.env_width + self.width) * self.resize_factor,
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
            center=(
                self.env_width * self.resize_factor
                + self.width * self.resize_factor / 2,
                overlay_rect.height / 2,
            )
        )

        self.screen.blit(overlay, overlay_rect.topleft)
        self.screen.blit(
            pause_text,
            (
                overlay_rect.left + text_rect.x,
                overlay_rect.top + text_rect.y,
            ),
        )
        pygame.display.flip()

    def run(self):
        obs = self.env.reset()
        buttons = ["N/A"]
        action_probs = "N/A"
        vf = "N/A"
        done = False
        rewards = ["N/A"]
        cum_reward = 0
        self.cum_rewards = [cum_reward]
        infos = [{}]
        self.step = 0

        while not done:
            self.render_screen(obs[0, -1])
            self.display_info(
                ", ".join(buttons),
                action_probs,
                vf,
                infos[0],
                rewards[0],
            )
            self.display_plots()
            pygame.display.flip()
            self.clock.tick(self.desired_fps)

            if not self.handle_events():
                break

            if self.model is None:
                keys = pygame.key.get_pressed()
                action, buttons = self.action_mapper.map_keys(keys)
                vf = "N/A"
            else:
                action = self.model.predict(obs, deterministic=True)[0][0]
                buttons = self.retro_env.get_action_meaning(action)
                action_probs = self._get_action_probs(obs)
                vf = self._get_model_vf(obs)
            obs, rewards, dones, infos = self.env.step([action])
            cum_reward += rewards[0]
            self.cum_rewards.append(cum_reward)
            done = dones[0]
            self.step += 1

    def _get_screen(self):
        return self.retro_env.img

    def _get_action_probs(self, obs):
        distribution = self.model.policy.get_distribution(
            torch.from_numpy(obs).to("cuda")
        )
        probs = [
            x.probs.detach().cpu().numpy()[0]
            for x in distribution.distribution
        ]
        # TODO: make this programatically, I'm too lazy for this right now
        probs = {
            "B": {"Y": probs[0][1], "N": probs[0][0]},
            "A": {"Y": probs[3][1], "N": probs[3][0]},
            "HOR": {"L": probs[2][1], "R": probs[2][2], "N": probs[2][0]},
            "VER": {"U": probs[1][1], "D": probs[1][2], "N": probs[1][0]},
        }
        return self.dict_to_custom_string(probs)

    @staticmethod
    def dict_to_custom_string(d):
        def format_dict(inner_d):
            return (
                "{"
                + ", ".join(f"{k}: {v:.0%}" for k, v in inner_d.items())
                + "}"
            )

        lines = []
        for key, value in d.items():
            formatted_value = format_dict(value)
            line = f"{key}: {formatted_value}"
            lines.append(line)

        return ", ".join(lines)

    def _get_model_vf(self, obs):
        value_function = (
            self.model.policy.predict_values(torch.from_numpy(obs).to("cuda"))[
                0, 0
            ]
            .detach()
            .cpu()
            .numpy()
        )
        return f"{value_function:.2f}"


class ActionMapper:
    def __init__(self, env):
        sizes = env.action_space.nvec
        ranges = [np.arange(size) for size in sizes]
        mesh = np.meshgrid(*ranges, indexing="ij")
        actions_arrs = np.stack(mesh, axis=-1).reshape(-1, len(sizes))

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


if __name__ == "__main__":
    model_name = "envfix4_crop_nsteps1024_ec0.05"
    debugger = Debugger(model=f"models/{model_name}")
    # debugger = Debugger()
    debugger.run()
