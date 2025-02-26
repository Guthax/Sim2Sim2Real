import numpy as np

from config.algorithm_configs import algorithm_params
import gymnasium as gym

from config.environment_configs import environment_configs

_CONFIG_BASIC = {
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32),
    "environments": {
        "duckietown": environment_configs["duckietown_rgb"],
    }

}