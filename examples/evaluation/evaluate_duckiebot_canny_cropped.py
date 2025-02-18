from stable_baselines3 import PPO

from config import CONFIGS
from evaluator import Evaluator
from simulators.duckietown.wrappers import ResizeWrapper, MultiInputWrapper, CarlaToDuckietownActionWrapper, \
    CropWrapper, CannyWrapper


def evaluate():
    from envs.duckietown.duckietown_env_no_domain_rand import DuckietownEnvNoDomainRand
    config = CONFIGS["TEST"]

    env = DuckietownEnvNoDomainRand(render_img=True)
    env = ResizeWrapper(env)
    env = MultiInputWrapper(env)
    env = CropWrapper(env)
    env = CannyWrapper(env)
    #env = CarlaToDuckietownActionWrapper(env)
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckie_domain_rand_canny_cropped_model_trained_2000000')

    evaluator = Evaluator(env, algorithm)

    evaluator.evaluate()

evaluate()