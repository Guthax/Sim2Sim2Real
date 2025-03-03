from gymnasium.wrappers import TimeLimit

from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from simulators.carla.carla_env import SelfCarlaEnv
from util.general_wrappers import ResizeWrapper, CropWrapper, CannyWrapper

environment_configs = {
    "carla_rgb": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True
        ),
        "wrappers": [ResizeWrapper, CropWrapper, TimeLimit]
    },

    "duckietown_rgb": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            randomize_maps_on_reset=True,
        ),
        "wrappers": [ResizeWrapper, CropWrapper, TimeLimit]
    },

    "duckietown_rgb_domain_rand": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            domain_rand=True,
            camera_rand=True,
            randomize_maps_on_reset=True,
        ),
        "wrappers": [ResizeWrapper, CropWrapper, TimeLimit]
    },

    "duckietown_rgb_canny": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            domain_rand=True,
            camera_rand=True,
            randomize_maps_on_reset=True
        ),
        "wrappers": [ResizeWrapper, CropWrapper, CannyWrapper, TimeLimit]
    }
}