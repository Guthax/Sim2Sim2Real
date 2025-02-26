from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from util.general_wrappers import ResizeWrapper, CropWrapper, CannyWrapper

environment_configs = {
    "duckietown_rgb": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=False
        ),
        "wrappers": [ResizeWrapper, CropWrapper]
    },

    "duckietown_rgb_canny": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=False
        ),
        "wrappers": [ResizeWrapper, CropWrapper, CannyWrapper]
    }
}