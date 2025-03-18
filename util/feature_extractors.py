import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PaperCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for segmentation inputs, following the architecture from:
    'Sim-to-Real via Sim-to-Seg: End-to-end Off-road Autonomous Driving Without Real Data'
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(PaperCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # Segmentation map channels (e.g., 1 for grayscale, 3 for RGB)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),  # Stride 2 for slight downsampling
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the number of features outputted by the CNN
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)
