import os

import numpy as np
import torch.cuda
from contracts.library.extensions import kwarg
from stable_baselines3 import PPO

import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluator import Evaluator
from trainer import Trainer
from utils import TensorboardCallback, write_json


class Scenario:
    environments: dict

    def __init__(self, config: dict):
        self.config = config
        self.environments = {}

        self._init_environments()
        self._init_algorithm()

    def _init_algorithm(self, model_path = None):
        if not model_path:
            first_env = self.environments[next(iter(self.environments))]
            self.algorithm = PPO(self.config['algorithm_policy_network'], first_env, verbose=2,
                                 device='cuda' if torch.cuda.is_available() else 'cpu',
                                 **self.config["algorithm_hyperparams"])
        else:
            self.algorithm = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    def _init_environments(self):

        for key, value in self.config["environments"].items():
            base_env = value["base_env"](**value["arguments"])
            wrappers = value["wrappers"]

            current_env = base_env
            for wrapper in wrappers:
                current_env = wrapper(current_env)

            if current_env.observation_space == self.config["observation_space"] and current_env.action_space == self.config["action_space"]:
                self.environments[key] = current_env
            else:
                raise NotCompatibleEnvironmentException()


    def train_on_environment(self, env_name: str,
                             model_name:str,
                             save_path:str,
                             log_dir:str,
                             num_timesteps:int,
                             num_checkpoints:int):
        training_env = self.environments[env_name]
        self.algorithm.set_env(training_env, True)

        trainer = Trainer(self.algorithm)

        num_timesteps = num_timesteps


        model_name = model_name
        log_model_dir = os.path.join(log_dir, model_name)

        new_logger = configure(log_model_dir, ["stdout", "csv", "tensorboard"])
        self.algorithm.set_logger(new_logger)
        write_json(self.config, os.path.join(log_model_dir, 'config.json'))

        tb = [TensorboardCallback(1)]

        trainer.train(model_name, num_timesteps, num_checkpoints, save_path, tb)


    def evaluate_on_environment(self, env_name, model_path, render_grad_cam=True):
        self._init_algorithm(model_path)

        eval_env = self.environments[env_name]
        self.algorithm.set_env(eval_env, True)

        evaluator = Evaluator(self.algorithm.env, self.algorithm, apply_grad_cam=render_grad_cam)
        evaluator.evaluate()

class NotCompatibleEnvironmentException(Exception):
    def __init__(self):
        super().__init__()



