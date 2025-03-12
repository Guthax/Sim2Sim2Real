import numpy as np

from config.algorithm_configs import algorithm_params
import gymnasium as gym

from config.environment_configs import environment_configs

_CONFIG_CARLA = {
    "name": "carla",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(60, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla"],
    }
}


_CONFIG_DUCKIE = {
    "name": "duckie",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(60, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckietown"],
    }
}



configs = [_CONFIG_CARLA, _CONFIG_DUCKIE]

def get_config_by_name(name: str):
    return next((item for item in configs if item['name'] == name), None)