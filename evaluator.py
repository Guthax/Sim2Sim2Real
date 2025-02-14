import cv2
import gym
from stable_baselines3.common.base_class import BaseAlgorithm


class Evaluator:
    evaluation_environment: gym.Env

    def __init__(self, eval_env: gym.Env, algorithm: BaseAlgorithm):
        self.evaluation_environment = eval_env
        self.algorithm = algorithm

        self.evaluation_timesteps = 100000
        self.episode_length = 1500

    def evaluate(self, ):
        state =  self.evaluation_environment.reset()
        done = False
        timesteps = 0
        current_episode_length = 0
        total_reward = 0

        while timesteps < self.evaluation_timesteps or self.evaluation_timesteps == -1:
            while not done and current_episode_length < self.episode_length:
                action, _states = self.algorithm.predict(state, deterministic=True)
                print(f"Action: {action}")
                state, reward, done, info = self.evaluation_environment.step(action)
                total_reward += reward
                timesteps += 1
                current_episode_length += 1
            done = False
            obs = self.evaluation_environment.reset()
            current_episode_length = 0

        avg_reward = total_reward / timesteps
        print(f"AVERAGE REWARD: {avg_reward}")