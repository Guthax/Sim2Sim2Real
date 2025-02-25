from abc import ABC, abstractmethod

import gym


class BaseEnv(gym.Env, ABC):
    def __init__(self, observation_space, action_space, reward_function):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_function = reward_function

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass




