from gymnasium.wrappers import TimeLimit

from envs.duckietown.duckietown import DuckietownBaseDynamics
from simulators.carla.carla_env import SelfCarlaEnv
from util.general_wrappers import ResizeWrapper, CropWrapper, CannyWrapper, SegmentationFilterWrapper, \
    DuckieClipWrapper, OneHotEncodeSegWrapper

environment_configs = {
    "carla": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
        ]
    },
    "carla_crop": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (CropWrapper, None)
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
            (OneHotEncodeSegWrapper, None)
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
            (CannyWrapper, None),
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
        ]
    },

    "duckie_seg": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            rgb_camera=False,
            seg_camera=True,
            randomize_maps_on_reset=False,
        ),
        "wrappers": [
            (DuckieClipWrapper, None),
            (OneHotEncodeSegWrapper, None)
        ]
    },
    "duckie_rgb_seg": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,

            rgb_camera=True,
            seg_camera=True,
            randomize_maps_on_reset=False,
        ),
        "wrappers": [
            (DuckieClipWrapper, None)
        ]
    },
}
