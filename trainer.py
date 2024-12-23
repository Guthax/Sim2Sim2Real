import os

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from gym import Env
from stable_baselines3.common.callbacks import CheckpointCallback


class Trainer:
    def __init__(self, gym_env: Env, algorithm: BaseAlgorithm, testing_env: gym.Env = None):
        self.training_environment = gym_env
        self.algorithm = algorithm

        self.testing_environment = testing_env

    def train(self, name, num_timesteps, num_checkpoints, save_path, test_freq = -1):
        cp = CheckpointCallback(
            save_freq= num_timesteps // num_checkpoints,
            save_path=save_path,
            name_prefix=f"{name}_model_trained")

        #tb = TensorboardCallback(1)
        
        model = self.algorithm.learn(total_timesteps=num_timesteps, callback=[cp])
        model.save(os.path.join(save_path, f"{name}_model_trained_{num_timesteps}"))