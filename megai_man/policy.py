import gymnasium as gym
import torch as th
from sb3_contrib.ppo_recurrent import MultiInputLstmPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
)
from torch import nn


class FineTunedArch(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                64,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(in_features=n_flatten, out_features=512),
            nn.ELU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class DeepNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                16,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(in_features=n_flatten, out_features=features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class WideNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                64,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(in_features=n_flatten, out_features=features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    MLP_OUTPUT_DIM = 3

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim=512):
        super(CustomCombinedExtractor, self).__init__(
            observation_space, cnn_output_dim + self.MLP_OUTPUT_DIM
        )

        self.extractors = nn.ModuleDict()
        total_output_dim = 0

        for key, subspace in observation_space.spaces.items():
            if (
                isinstance(subspace, gym.spaces.Box)
                and len(subspace.shape) == 3
            ):
                # CNN for image input
                self.extractors[key] = NatureCNN(
                    subspace,
                    features_dim=cnn_output_dim,
                )
                total_output_dim += cnn_output_dim
            elif (
                isinstance(subspace, gym.spaces.Box)
                and len(subspace.shape) == 1
            ):
                # "MLP" for vector input
                # self.extractors[key] = nn.Sequential(
                #     nn.Linear(self.MLP_OUTPUT_DIM, self.MLP_OUTPUT_DIM),
                #     nn.ReLU(),
                # )
                # self.extractors[key] = nn.Identity()
                self.extractors[key] = nn.Flatten()
                total_output_dim += self.MLP_OUTPUT_DIM
            else:
                raise ValueError(f"Unsupported observation space: {subspace}")

        self._features_dim = total_output_dim

    def forward(self, observations: dict) -> th.Tensor:
        return th.cat(
            [self.extractors[key](observations[key]) for key in observations],
            dim=1,
        )


class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=256),
        )


class CustomMultiInputLstmPolicy(MultiInputLstmPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=256),
        )
