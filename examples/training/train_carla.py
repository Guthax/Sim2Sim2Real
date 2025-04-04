import os

import numpy as np
from gymnasium.vector.utils import spaces
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

import config
from envs.carla.carla_env import CarlaBaseEnv
from envs.carla.carla_steering_only_env import CarlaSteeringEnv
from simulators.carla.carla_env import SelfCarlaEnv
from utils import TensorboardCallback, write_json

config.set_config("TEST")
from envs.carla.carla_route_env import CarlaRouteEnv
from simulators.carla.carla_env.rewards import reward_functions
from simulators.carla.carla_env.state_commons import create_encode_state_fn, load_vae
from trainer import Trainer
from stable_baselines3 import PPO


def train():

    from config import CONFIG

    configuration = CONFIG

    observation_space = spaces.Dict({
        "camera_rgb": spaces.Box(low=0, high=255, shape=(300, 600, 3), dtype=np.uint8)
    })

    action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    # CarlaUE4.exe -quality-level=Low -benchmark -fps=15 -prefernvidia -dx11 -carla-world-port=2000
    """
    env = CarlaRouteEnv(obs_res=configuration["obs_res"], host="localhost", port=2000,
                        reward_fn=reward_functions[configuration["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=15, action_smoothing=configuration["action_smoothing"],
                        action_space_type='continuous', activate_spectator=False, activate_render=True)
    """
    #env = CarlaBaseEnv(observation_space, action_space, reward_functions[configuration["reward_fn"]],
    #    obs_res=configuration["obs_res"], host="localhost", port=2000,activate_spectator=False, activate_render=True)
    env = SelfCarlaEnv()

    algorithm = PPO('CnnPolicy', env, verbose=2, device='cuda', **configuration["algorithm_params"])
    #algorithm = PPO.load("F:\Github\Sim2Sim2Real\\results\carla_only_rgb_steering_model_trained_600000_steps", env=env, device='cuda')

    trainer = Trainer(env, algorithm)

    num_timesteps = 1000000
    num_checkpoints = 10

    log_dir = "../../tensorboard"

    model_name = "scenario_2"
    log_model_dir = os.path.join(log_dir, model_name)

    new_logger = configure(log_model_dir, ["stdout", "csv", "tensorboard"])
    algorithm.set_logger(new_logger)
    write_json(CONFIG, os.path.join(log_model_dir, 'config.json'))

    tb = [TensorboardCallback(1)]


    save_path = "../../results"
    trainer.train(model_name, num_timesteps, num_checkpoints, save_path, tb)

train()