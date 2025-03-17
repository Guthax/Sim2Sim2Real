from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecFrameStack

from envs.duckietown.base.duckietown import DuckietownBaseDynamics
from simulators.carla.carla_env import SelfCarlaEnv
from util.general_wrappers import ResizeWrapper, CropWrapper, CannyWrapper,SegmentationFilterWrapper, DuckieClipWrapper

environment_configs = {
    "carla": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=160, dst_height=120)),
            (TimeLimit, dict(max_episode_steps=2000))
        ]
    },
    "carla_seg": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False,
            rgb_camera=False,
            seg_camera=True,
        ),
        "wrappers": [
            (SegmentationFilterWrapper, None)
        ]
    },
    "carla_rgb_seg": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False,
            rgb_camera=True,
            seg_camera=True,
        ),
        "wrappers": [
            (SegmentationFilterWrapper, None)
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
            render=False
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=256, dst_height=256)),
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
    },
    "duckietown_lane_detect": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            randomize_maps_on_reset=True,
        ),
        "wrappers": [
            (ResizeWrapper, dict(dst_width=256, dst_height=256)),
            (DuckieClipWrapper, None),
            (TimeLimit, dict(max_episode_steps=2000))
        ]
    },
}
