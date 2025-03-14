from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecFrameStack

from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from simulators.carla.carla_env import SelfCarlaEnv
from util.general_wrappers import ResizeWrapper, CropWrapper, CannyWrapper, LaneMarkingWrapper, \
    SegmentationFilterWrapper

environment_configs = {
    "carla": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=160, dst_height=120)),
            (CropWrapper, dict(crop_height_start=60, crop_height_end=120)),
            (TimeLimit, dict(max_episode_steps=2000))
        ]
    },
    "carla_canny": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=160, dst_height=120)),
            (CropWrapper, dict(crop_height_start=60, crop_height_end=120)),
            (CannyWrapper, None),
            (TimeLimit, dict(max_episode_steps=2000))
        ]
    },
    "carla_lane_detect": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=128, dst_height=128)),
            (SegmentationFilterWrapper, None),
            (TimeLimit, dict(max_episode_steps=2000))
        ]
    },
    "duckietown": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            randomize_maps_on_reset=False,
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=160, dst_height=120)),
            (CropWrapper, dict(crop_height_start=60, crop_height_end=120)),
            (TimeLimit, dict(max_episode_steps=2000))
        ]
    }
}