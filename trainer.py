import os
from typing import List

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from utils import TensorboardCallback


class Trainer:
    def __init__(self, algorithm: BaseAlgorithm):
        self.algorithm = algorithm

    def train(self, name, num_timesteps, num_checkpoints, save_path, additional_callbacks: List = None, test_freq = -1):
        cp = [CheckpointCallback(
            save_freq= num_timesteps // num_checkpoints,
            save_path=save_path,
            name_prefix=f"{name}_model_trained")]

        if additional_callbacks:
            cp = additional_callbacks + cp

        model = self.algorithm.learn(total_timesteps=num_timesteps, callback=cp, progress_bar=True)
        model.save(os.path.join(save_path, f"{name}_model_trained_{num_timesteps}"))