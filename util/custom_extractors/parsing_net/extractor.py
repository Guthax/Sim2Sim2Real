from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import numpy as np
from util.custom_extractors.parsing_net.parsing_net import parsingNet  # Adjust this import path as needed


class ParsingNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, size=(120, 160), pretrained=True, backbone='18'):
        super().__init__(observation_space, features_dim=1)  # Placeholder for features_dim

        # Initialize parsingNet with the given size
        cls_num_per_lane = 18

        self.parsing_net = parsingNet(pretrained=pretrained, backbone=backbone,
                                      cls_dim = (201,cls_num_per_lane, 4),
                                      use_aux=False)
        state_dict = torch.load("util/custom_extractors/parsing_net/tusimple_18.pth", map_location='cuda')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.parsing_net.load_state_dict(compatible_state_dict, strict=False)

        # Disable gradient tracking for backbone
        for param in self.parsing_net.parameters():
            param.requires_grad = False

        # Infer the output feature size dynamically based on input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape).float()  # NHWC to NCHW
            _, _, fea = self.parsing_net.forward(dummy_input)
            pooled = self.parsing_net.pool(fea)
            self._features_dim = pooled.view(1, -1).shape[1]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #x = observations.permute(0, 3, 1, 2).float()  # Convert NHWC to NCHW
        x = observations
        with torch.no_grad():
            _, _, fea = self.parsing_net.forward(x)
            fea = self.parsing_net.pool(fea).view(x.size(0), -1)
        return fea
