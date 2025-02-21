from stable_baselines3 import PPO

from config import CONFIGS
from evaluator import Evaluator
from simulators.duckietown.wrappers import ResizeWrapper, CarlaToDuckietownActionWrapper, \
    CropWrapper, CannyWrapper


def evaluate():
    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnvNoDomainRand(render_img=False)
    env = ResizeWrapper(env)
    env = CropWrapper(env)
    env = CannyWrapper(env)
    #env = CarlaToDuckietownActionWrapper(env)
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/carla_only_rgb_steering_model_trained_200000_steps')

    evaluator = Evaluator(env, algorithm)

    evaluator.evaluate()

evaluate()