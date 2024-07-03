import cv2
import matplotlib.cm as cm
import numpy as np
import pygame
import torch
from env import make_venv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.preprocessing import preprocess_obs
from torch import nn

from megai_man.utils import ActionMapper


class Debugger:
    def __init__(
        self,
        model=None,
        record=False,
        graph=False,
        record_grayscale_obs=False,
        frame_by_frame=False,
        deterministic=True,
        grad_cam=False,
    ):
        pygame.init()
        pygame.font.init()
        self.font_size = 18
        self.font = pygame.font.SysFont("opensans", self.font_size)
        self.small_font = pygame.font.SysFont("opensans", 12)
        frameskip = 4
        self.action_space = "multi_discrete"
        if model is not None and "/dqn_" in model:
            ModelClass = DQN
        else:
            ModelClass = PPO
        self.env = make_venv(
            n_envs=1,
            state="CutMan",
            screen=None,
            frameskip=frameskip,
            truncate_if_no_improvement=True,
            obs_space="ram",
            action_space=self.action_space,
            crop_img=False,
            invincible=False,
            no_enemies=False,
            render_mode=None,
            record=record,
            damage_terminate=False,
            fixed_damage_punishment=0.05,
            forward_factor=0.05,
            backward_factor=0.055,
            screen_rewards=False,
            score_reward=0.00,
            distance_only_on_ground=True,
            term_back_screen=True,
        )
        self.retro_env = self.env.unwrapped.envs[0].unwrapped
        self.model = model
        if model is not None:
            self.model = ModelClass.load(model, env=self.env)
        self.graph = graph
        self.frame_by_frame = frame_by_frame
        self.deterministic = deterministic
        self.action_mapper = ActionMapper(self.retro_env, self.action_space)
        self.desired_fps = 60 // frameskip
        self.clock = pygame.time.Clock()
        self._setup_screen()
        self.debug_info_start_y = 10
        self.debug_messages = []
        self.box_text = (
            "Step: {step}, Action: {action}, Reward: {reward:+2.2f}, VF: {vf}\n"
            "{probs}\n"
            "Screen: {screen}, X: {x}, Y: {y}, Distance: {distance}\n"
        )
        self.n_lines = len(self.box_text.split("\n"))
        self.pad = 5
        self.graph_color = (255, 255, 255)

    def _setup_screen(self):
        width, height = 240, 224
        resize_factor = 4
        env_resize_boost = 10

        obs_space = self.env.observation_space
        env_height = env_width = int(np.ceil(np.sqrt(obs_space.shape[0])))

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
        self.env_resize_boost = env_resize_boost

    def render_screen(self, obs):
        new_size = self.env_width * self.env_height
        pad_size = new_size - obs.shape[0]
        obs = np.pad(obs, (0, pad_size))
        obs = obs.reshape(self.env_height, self.env_width)
        env_screen = np.stack((obs.T,) * 3, axis=-1)
        env_screen = ((env_screen / 2 + 0.5) * 255).astype(np.uint8)

        surface = pygame.surfarray.make_surface(env_screen)
        surface = pygame.transform.scale_by(
            surface,
            self.resize_factor + self.env_resize_boost,
        )
        self.screen.blit(surface, (0, 0))

        screen = self._get_screen()
        screen = np.transpose(screen.T, (1, 2, 0))
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale_by(surface, self.resize_factor)

        self.screen.blit(
            surface,
            (self.env_width * (self.resize_factor + self.env_resize_boost), 0),
        )

    def display_info(self, action, probs, vf, infos, reward):
        self.debug_messages.append(
            (self.step, action, probs, vf, infos, reward)
        )

        max_messages = (self.height * self.resize_factor) // (
            (self.font_size + self.pad + 1) * self.n_lines
        )

        self.debug_messages = self.debug_messages[-max_messages:]

        debug_area_rect = pygame.Rect(
            self.env_width * (self.resize_factor + self.env_resize_boost)
            + self.width * self.resize_factor,
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
                screen=info.get("screen", "N/A"),
                x=info.get("x", "N/A"),
                y=info.get("y", "N/A"),
                distance=info.get("distance", "N/A"),
                reward=reward,
            ).split("\n")

            self.render_text(
                info_lines,
                self.env_width * (self.resize_factor + self.env_resize_boost)
                + self.width * self.resize_factor
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
            pygame.display.flip()

    def handle_events(self):
        paused = self.frame_by_frame
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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.frame_by_frame = not self.frame_by_frame
            if not paused:
                break
            self.clock.tick(10)
        return True

    def display_pause_message(self):
        overlay_rect = pygame.Rect(
            0,
            0,
            self.env_width * (self.resize_factor + self.env_resize_boost)
            + self.width * self.resize_factor,
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
                self.env_width * (self.resize_factor + self.env_resize_boost)
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
        rewards = [0]
        cum_reward = 0
        self.cum_rewards = [cum_reward]
        infos = [{}]
        self.step = 0

        while not done:
            if self.model is None and not self.handle_events():
                break

            if self.model is None:
                keys = pygame.key.get_pressed()
                action, buttons = self.action_mapper.map_keys(keys)
                vf = "N/A"
            else:
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                )
                action = action[0]
                buttons = self.retro_env.get_action_meaning(action)
                if self.action_space == "multi_discrete":
                    action_probs = self._get_action_probs(obs)
                else:
                    # TODO: implement this for discrete
                    action_probs = "N/A"
                vf = self._get_model_vf(obs)

            self.render_screen(obs[0])
            self.display_info(
                ", ".join(buttons),
                action_probs,
                vf,
                infos[0],
                rewards[0],
            )
            if self.graph:
                self.display_plots()
            pygame.display.flip()
            self.clock.tick(self.desired_fps)

            if self.model is not None and not self.handle_events():
                break

            obs, rewards, dones, infos = self.env.step([action])
            cum_reward += rewards[0]
            self.cum_rewards.append(cum_reward)
            done = dones[0]
            self.step += 1

        self.frame_by_frame = True
        self.handle_events()

    def _get_screen(self):
        return self.retro_env.em.get_screen()

    def _get_action_probs(self, obs):
        cuda_obs, _ = self.model.policy.obs_to_tensor(obs)
        distribution = self.model.policy.get_distribution(cuda_obs)
        probs = [
            x.probs.detach().cpu().numpy()[0]
            for x in distribution.distribution
        ]
        # TODO: make this programatically, I'm too lazy for this right now
        probs = {
            "B": {"Y": probs[0][1], "N": probs[0][0]},
            "A": {"Y": probs[2][1], "N": probs[2][0]},
            "D": {
                "U": probs[1][1],
                "D": probs[1][2],
                "L": probs[1][3],
                "R": probs[1][4],
                "N": probs[1][0],
            },
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
        cuda_obs, _ = self.model.policy.obs_to_tensor(obs)
        if isinstance(self.model, DQN):
            value_function = self.model.q_net(cuda_obs)[0].max()
        else:
            value_function = self.model.policy.predict_values(cuda_obs)[0, 0]
        value_function = value_function.detach().cpu().numpy()
        return f"{value_function:.2f}"

    def _grad_cam(self, obs):
        self.model.policy.train()

        cnn = self.model.policy.pi_features_extractor.cnn[:5]
        classifier = nn.Sequential(
            self.model.policy.pi_features_extractor.cnn[5:],
            self.model.policy.pi_features_extractor.linear,
            self.model.policy.lstm_actor,
            TensorExtractor(),
            self.model.policy.action_net,
        )
        input_tensor = preprocess_obs(
            torch.from_numpy(obs).cuda(),
            self.model.observation_space,
            normalize_images=self.model.policy.normalize_images,
        )
        input_tensor.requires_grad = True

        cnn_output = cnn(input_tensor)
        preds = classifier(cnn_output)
        top_pred_index = torch.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

        self.model.policy.zero_grad()
        top_class_channel.backward(retain_graph=True)
        # top_class_channel.backward()
        grads = input_tensor.grad
        pooled_grads = torch.mean(grads, dim=[2, 3]).cpu().numpy()
        # NOTE: won't work for 2+ channels or batch size > 1

        self.model.policy.eval()

        cnn_output = cnn_output.detach().cpu().numpy()[0]
        cnn_output *= pooled_grads

        heatmap = np.mean(cnn_output, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :4]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap *= 0.4
        self.heatmap = np.uint8(255 * jet_heatmap)


class TensorExtractor(nn.Module):
    def forward(self, x):
        tensor, *_ = x
        return tensor


if __name__ == "__main__":
    # model = (
    #     "checkpoints/"
    #     "sevs_steps2048_batch64_lr3.0e-04_epochs10_clip0.2_ecoef1e-03_gamma0.99_vf0.5__fs4_stack4_rews0.05+screen1_dmg0.12_groundonly_termbackscreen2_spikefix6_scen4_skipB_multinput5_default_visible"
    #     "_5000000_steps"
    # )
    # model = (
    #     "models/"
    #     "sevs_steps512_batch64_lr3.0e-04_epochs10_clip0.2_ecoef1e-03_gamma0.99_vf0.5__fs4_stack4_rews0.05+screen1_dmg0.12_groundonly_termbackscreen2_spikefix6_scen4_skipB_multinput5_default_visible"
    #     ".zip"
    # )
    model = (
        "models/"
        "ram2.3_steps512_batch64_lr1.0e-04_epochs10_clip0.2_ecoef1e-04_nn256x2_relu_twoFEs__rews0.05+screen1_dmg0.05_groundrew_termbackscreen2"
        "_best/best_model"
    )
    debugger = Debugger(
        # model=model,
        frame_by_frame=False,
        # graph=True,
    )
    debugger.run()
