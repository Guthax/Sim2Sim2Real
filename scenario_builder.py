import os

import torch.cuda
from stable_baselines3 import PPO

import gymnasium as gym
from stable_baselines3.common.logger import configure

from simulators.carla.carla_env.envs.collect_data_rl_env import observation_space
from trainer import Trainer
from utils import TensorboardCallback


class Scenario:
    environments: dict

    def __init__(self, config: dict):
        self.algorithm = PPO(config['algorithm_policy_network'],
                        observation_space=config["observation_space"],
                        env=None, verbose=2, device='cuda' if torch.cuda.is_available() else 'cpu',
                        **config["algorithm_hyperparams"])

        self.environments = {}

    def add_environment(self, key: str, env: gym.Env) -> None:
        self.environments[key] = env

    def train_on_environment(self, env_name: str):
        training_env = self.environments[env_name]
        self.algorithm.env = training_env

        trainer = Trainer(self.algorithm)

        num_timesteps = 1000000
        num_checkpoints = 5

        log_dir = "../../tensorboard"

        model_name = "duckie_only_rgb"
        log_model_dir = os.path.join(log_dir, model_name)

        new_logger = configure(log_model_dir, ["stdout", "csv", "tensorboard"])
        self.algorithm.set_logger(new_logger)
        #write_json(CONFIG, os.path.join(log_model_dir, 'config.json'))

        tb = [TensorboardCallback(1)]

        save_path = "../../results"
        trainer.train(model_name, num_timesteps, num_checkpoints, save_path, tb)


def make_scenario_from_config(config: dict):
    scenario = Scenario(config)


    for key, value in config["environments"].items():
        base_env = value["base_env"]
        wrappers = value["wrappers"]

        current_env = base_env
        for wrapper in wrappers:
            current_env = wrapper(current_env)

        if current_env.observation_space == config["observation_space"] and current_env.action_space == config["action_space"]:
            scenario.add_environment(key, current_env)
        else:
            raise NotCompatibleEnvironmentException()

    return scenario


class NotCompatibleEnvironmentException(Exception):
    def __init__(self):
        super().__init__()
