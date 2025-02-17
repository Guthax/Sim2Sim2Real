from stable_baselines3 import PPO

from config import CONFIGS
from evaluator import Evaluator
from simulators.duckietown.wrappers import ResizeWrapper, MultiInputWrapper, CarlaToDuckietownActionWrapper, CropWrapper


def evaluate():
    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnvNoDomainRand(render_img=True)
    env = ResizeWrapper(env)
    env = MultiInputWrapper(env)
    env = CropWrapper(env)
    #env = CarlaToDuckietownActionWrapper(env)
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckie_domain_rand_cropped_model_trained_2000000_steps')

    evaluator = Evaluator(env, algorithm)

    evaluator.evaluate()

evaluate()