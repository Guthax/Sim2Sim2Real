from stable_baselines3 import PPO

from config import CONFIGS
from envs.duckietown.duckietown_env import DuckietownEnv
from evaluator import Evaluator
from simulators.duckietown.wrappers import ResizeWrapper, CarlaToDuckietownActionWrapper, \
    CannyWrapper, CropWrapper


def evaluate():
    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnv()
    env = ResizeWrapper(env)
    env = CropWrapper(env)
    #env = CarlaToDuckietownActionWrapper(env)
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckie_basic_model_trained_1000000_steps.zip')


    evaluator = Evaluator(env, algorithm)

    evaluator.evaluate()

evaluate()