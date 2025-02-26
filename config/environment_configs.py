from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from util.general_wrappers import ResizeWrapper

environment_configs = {
    "duckietown_rgb": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=False
        ),
        "wrappers": [ResizeWrapper]
    }
}