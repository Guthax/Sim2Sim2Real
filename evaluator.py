import gym
from stable_baselines3.common.base_class import BaseAlgorithm


class Evaluator:
    evaluation_environment: gym.Env

    def __init__(self, eval_env: gym.Env, algorithm: BaseAlgorithm):
        self.evaluation_environment = eval_env
        self.algorithm = algorithm

    def evaluate(self, ):
        state =  self.evaluation_environment.reset()
        done = False
        while True:
            while not done:
                action, _states = self.algorithm.predict(state, deterministic=True)
                state, reward, done, info = self.evaluation_environment.step(action)
                print(done)

            done = False
            obs = self.evaluation_environment.reset()
