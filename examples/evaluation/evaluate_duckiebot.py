from stable_baselines3 import PPO

from config import CONFIGS
from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from envs.duckietown.duckie_town_direct_velocities import DuckietownDirectVelocities
from evaluator import Evaluator
from simulators.duckietown.wrappers import ResizeWrapper, CannyWrapper, \
    CropWrapper

def evaluate():
    env = DuckietownBaseDynamics(render_img=True)
    env = ResizeWrapper(env)
    env = CropWrapper(env)
    env = CannyWrapper(env)
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckie_domain_rand_canny_cropped_model_trained_800000_steps')

    evaluator = Evaluator(env, algorithm)

    evaluator.evaluate()

evaluate()