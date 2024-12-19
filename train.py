from stable_baselines3.common.callbacks import CheckpointCallback

from config import CONFIGS
from simulators.duckietown.wrappers import MultiInputWrapper
from trainer import Trainer
from stable_baselines3 import PPO

def train():

    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnvNoDomainRand(render_img=True)
    env = MultiInputWrapper(env)
    algorithm = PPO('MultiInputPolicy', env, verbose=2, device='cpu', **config["algorithm_params"])

    trainer = Trainer(env, algorithm)

    num_timesteps = 1000
    num_checkpoints = 2

    save_path = "/home/jurriaan/Documents/Programming/Sim2Sim2Real/results"
    trainer.train(num_timesteps, [], 5, save_path)

train()