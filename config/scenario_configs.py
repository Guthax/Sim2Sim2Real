import numpy as np

from config.algorithm_configs import algorithm_params
import gymnasium as gym

from config.environment_configs import environment_configs

_CONFIG_PPO_CNN_CARLA_RGB_STEER = {
    "name": "carla_rgb_test",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_rgb"],
    }
}


NEW = {
    "name": "test",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_rgb"],
    }
}


_CONFIG_PPO_CNN_DUCKIETOWN_RGB_STEER = {
    "name": "duckietown_rgb_test",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckietown": environment_configs["duckietown_rgb"],
    }
}

_CONFIG_PPO_CNN_DUCKIETOWN_RGB_STEER_RAND = {
    "name": "duckietown_rgb_test_rand",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckietown": environment_configs["duckietown_rgb_domain_rand"],
    }
}



_CONFIG_PPO_MULTI_DUCKIETOWN_RGB_CANNY_STEER = {
    "name": "duckietown_rgb_canny_test",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
            "camera_rgb": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
            "camera_canny": gym.spaces.Box(low=0, high=255, shape=(80, 160), dtype=np.uint8)
    }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckietown": environment_configs["duckietown_rgb_canny"],
    }
}

_CONFIG_PPO_MULTI_CARLA_DUCKIETOWN_RGB_STEER = {
    "name": "scenario_1",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32),
    "environments": {
        "carla": environment_configs["carla_rgb"],
        "duckietown": environment_configs["duckietown_rgb"],
    }
}


configs = [_CONFIG_PPO_CNN_CARLA_RGB_STEER, NEW, _CONFIG_PPO_CNN_DUCKIETOWN_RGB_STEER, _CONFIG_PPO_CNN_DUCKIETOWN_RGB_STEER_RAND, _CONFIG_PPO_MULTI_DUCKIETOWN_RGB_CANNY_STEER, _CONFIG_PPO_MULTI_CARLA_DUCKIETOWN_RGB_STEER]

def get_config_by_name(name: str):
    return next((item for item in configs if item['name'] == name), None)