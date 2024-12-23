from stable_baselines3.common.callbacks import CheckpointCallback

from config import CONFIGS
from simulators.duckietown.wrappers import MultiInputWrapper, ResizeWrapper
from trainer import Trainer
from stable_baselines3 import PPO

def train():

    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnvNoDomainRand(render_img=True)
    env = ResizeWrapper(env)
    env = MultiInputWrapper(env)
    algorithm = PPO('MultiInputPolicy', env, verbose=2, device='cuda', **config["algorithm_params"])

    trainer = Trainer(env, algorithm)

    num_timesteps = 2000
    num_checkpoints = 5

    save_path = "../../results"
    trainer.train("duckietown_new_rew", num_timesteps, num_checkpoints, save_path)

train()