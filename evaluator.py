import gym


class Evaluator:
    evaluation_environment: gym.Env

    def __init__(self, eval_env: gym.Env):
        self.evaluation_environment = eval_env