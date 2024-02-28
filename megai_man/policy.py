import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class FineTunedArch(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.conv1 = nn.Conv2d(
            n_input_channels,
            64,
            kernel_size=8,
            stride=4,
            padding=0,
        )
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.elu3 = nn.ELU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=6272, out_features=512, bias=True)
        self.elu4 = nn.ELU()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
