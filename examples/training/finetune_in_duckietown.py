from stable_baselines3.common.callbacks import CheckpointCallback

from config import CONFIGS
from simulators.duckietown.wrappers import MultiInputWrapper, ResizeWrapper
from trainer import Trainer
from stable_baselines3 import PPO

from utils import TensorboardCallback


def train():

    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnvNoDomainRand(render_img=True)
    env = ResizeWrapper(env)
    env = MultiInputWrapper(env)
    #algorithm = PPO('MultiInputPolicy', env, verbose=2, device='cuda', **config["algorithm_params"])
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/carla_only_rgb_steering_model_trained_200000_steps', env=env)
    trainer = Trainer(env, algorithm)

    num_timesteps = 10000
    num_checkpoints = 5

    tb = [TensorboardCallback(1)]

    save_path = "../../results"
    trainer.train("carla_to_duckie_finetuned", num_timesteps, 2, save_path, tb)

train()