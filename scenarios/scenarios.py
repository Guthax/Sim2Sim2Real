import gym


class Scenario:
    base_env: gym.Env
    wrappers: [gym.Wrapper]
    