import numpy as np

from config.algorithm_configs import algorithm_params
import gymnasium as gym

from config.environment_configs import environment_configs

_CONFIG_CARLA_RGB = {
    "name": "carla_rgb",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=1, shape=(3, 80, 160), dtype=np.float32),
                "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla"],
    }
}

_CONFIG_CARLA_RGB_NO_CROP = {
    "name": "carla_rgb_no_crop",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=1, shape=(3, 120, 160), dtype=np.float32),
                "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_rgb_no_crop"],
    }
}

_CONFIG_CARLA_RGB_FEATURE = {
    "name": "carla_feature",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO_FEATURE"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla"],
    }
}


_CONFIG_CARLA_ENCODER = {
    "name": "carla_encoder",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
            "encoding": gym.spaces.Box(low=-10, high=10, shape=(95,), dtype=np.float32),
            "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
        }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_encoder"],
    }
}

_CONFIG_CARLA_GRAY = {
    "name": "carla_gray",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
            "camera_gray": gym.spaces.Box(low=0, high=1.0, shape=(1,80, 160), dtype=np.float32),
            "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
        }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_gray"],
    }
}

_CONFIG_CARLA_GRAY_NO_CROP = {
    "name": "carla_gray_no_crop",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=1, shape=(3, 120, 160), dtype=np.float32),
                "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_gray_no_crop"],
    }
}

_CONFIG_CARLA_CANNY = {
    "name": "carla_canny",
    "algorithm": "PPO",
    "algorithm_policy_network": "CnnPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_canny"],
    }
}

_CONFIG_CARLA_SEG = {
    "name": "carla_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_seg": gym.spaces.Box(low=0, high=1, shape=(3, 80, 160), dtype=np.uint8),
                "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
            }),
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
                "camera_seg": gym.spaces.Box(low=0, high=1, shape=(120, 160, 3), dtype=np.uint8),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "carla": environment_configs["carla_rgb_seg"],
    }
}

_CONFIG_DUCKIE_RGB = {
    "name": "duckie_rgb",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=1, shape=(3, 80, 160), dtype=np.float32),
                "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckie_rgb"],
    }
}


_CONFIG_DUCKIE_GRAY = {
    "name": "duckie_gray",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
            "camera_gray": gym.spaces.Box(low=0, high=1.0, shape=(1,80, 160), dtype=np.float32),
            "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
        }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckie_gray"],
    }
}

_CONFIG_DUCKIE_SEG = {
    "name": "duckie_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_seg": gym.spaces.Box(low=0, high=1, shape=(3,80, 160), dtype=np.uint8),
                "vehicle_dynamics": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckie_seg"],
    }
}

_CONFIG_DUCKIE_RGB_SEG = {
    "name": "duckie_rgb_seg",
    "algorithm": "PPO",
    "algorithm_policy_network": "MultiInputPolicy",
    "algorithm_hyperparams": algorithm_params["PPO"],
    "observation_space": gym.spaces.Dict({
                "camera_rgb": gym.spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8),
                "camera_seg": gym.spaces.Box(low=0, high=1, shape=(120, 160, 3), dtype=np.uint8),
            }),
    "action_space": gym.spaces.Box(np.float32(-1), high=np.float32(1)),
    "environments": {
        "duckie": environment_configs["duckie_rgb_seg"],
    }
}


configs = [_CONFIG_CARLA_RGB, _CONFIG_CARLA_RGB_SEG,_CONFIG_CARLA_GRAY,_CONFIG_CARLA_GRAY_NO_CROP, _CONFIG_CARLA_SEG, _CONFIG_CARLA_CANNY, _CONFIG_CARLA_RGB_FEATURE,_CONFIG_CARLA_RGB_NO_CROP,
           _CONFIG_DUCKIE_RGB, _CONFIG_DUCKIE_GRAY, _CONFIG_DUCKIE_SEG, _CONFIG_DUCKIE_RGB_SEG,
           _CONFIG_CARLA_ENCODER]

def get_config_by_name(name: str):
    return next((item for item in configs if item['name'] == name), None)
