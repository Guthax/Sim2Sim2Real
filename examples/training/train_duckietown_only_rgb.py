import os

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

import config
from envs.duckietown.base.duckietown import DuckietownBaseDynamics

config.set_config("TEST")

from simulators.duckietown.wrappers import ResizeWrapper, CropWrapper
from trainer import Trainer
from stable_baselines3 import PPO

from utils import TensorboardCallback, write_json


def train():

    from config import CONFIG

    config = CONFIG

    env = DuckietownBaseDynamics(render_img=True)
    env = ResizeWrapper(env)
    env = CropWrapper(env)
    algorithm = PPO('MultiInputPolicy', env, verbose=2, device='cuda', **config["algorithm_params"])

    trainer = Trainer(env, algorithm)

    num_timesteps = 1000000
    num_checkpoints = 5

    log_dir = "../../tensorboard"

    model_name = "duckie_only_rgb"
    log_model_dir = os.path.join(log_dir, model_name)

    new_logger = configure(log_model_dir, ["stdout", "csv", "tensorboard"])
    algorithm.set_logger(new_logger)
    write_json(CONFIG, os.path.join(log_model_dir, 'config.json'))


    tb = [TensorboardCallback(1)]

    save_path = "../../results"
    trainer.train(model_name, num_timesteps, num_checkpoints, save_path, tb)

train()