from stable_baselines3 import PPO

from config import CONFIGS
from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from evaluator import Evaluator
from simulators.duckietown.wrappers import ResizeWrapper, CannyWrapper, \
    CropWrapper



def evaluate():
    env = DuckietownBaseDynamics(render_img=True   , randomize_maps_on_reset=True)
    env = ResizeWrapper(env)
    env = CropWrapper(env)
    algorithm = PPO.load('/home/jurriaan/Documents/Programming/Sim2Sim2Real/results/duckie_only_rgb_domain_rand_model_trained_1000000_steps.zip')
    print(algorithm.policy)
    evaluator = Evaluator(env, algorithm, apply_grad_cam=True)

    evaluator.evaluate()

evaluate()