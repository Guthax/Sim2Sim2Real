import numpy as np

from config.algorithm_configs import algorithm_params
import gymnasium as gym

from config.environment_configs import environment_configs

_CONFIG_CARLA_RGB = {
    "name": "carla",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla"],
    }
}

_CONFIG_CARLA_CANNY = {
    "name": "carla_canny",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(60, 160, 1), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_canny"],
    }
}

_CONFIG_CARLA_SEG = {
    "name": "carla_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_seg"],
    }
}

_CONFIG_CARLA_RGB_SEG = {
    "name": "carla_rgb_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
                "camera_seg": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_rgb_seg"],
    }
}


_CONFIG_DUCKIE = {
    "name": "duckie",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckietown"],
    }
}

_CONFIG_DUCKIE_RGB = {
    "name": "duckie_rgb",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckietown"],
    }
}

"""
_CONFIG_DUCKIE_SEG = {
    "name": "duckie_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["du"],
    }
}"""

_CONFIG_DUCKIE_RGB_SEG = {
    "name": "duckie_rgb_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
                "camera_seg": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckie_rgb_seg"],
    }
}

configs = [_CONFIG_CARLA_RGB,_CONFIG_DUCKIE_RGB_SEG, _CONFIG_CARLA_RGB_SEG, _CONFIG_CARLA_SEG, _CONFIG_CARLA_CANNY,_CONFIG_DUCKIE_RGB]

def get_config_by_name(name: str):
    return next((item for item in configs if item['name'] == name), None)
