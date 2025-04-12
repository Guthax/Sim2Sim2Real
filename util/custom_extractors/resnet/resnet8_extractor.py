from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

import torch as th

class ResnetExtractor(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for name, param in self.cnn.named_parameters():
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False
            else:
                param.requires_grad = True  # layer3, layer4 will be finetuned
        # Remove the final fc layer, keep everything else
        self.backbone = nn.Sequential(*list(self.cnn.children())[:-1])  # [B, 512, 1, 1]

        # SB3 expects a flat feature vector
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = ResNet18_Weights.IMAGENET1K_V1.transforms()(observations)
        x = self.backbone(x)               # [B, 512, 1, 1]
        x = th.flatten(x, 1)                       # [B, 512]
        return x