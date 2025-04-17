from gymnasium.wrappers import TimeLimit

from envs.duckietown.duckietown import DuckietownBaseDynamics
from simulators.carla.carla_env import SelfCarlaEnv
from util.general_wrappers import NormalizeWrapper, ChannelFirstWrapper, ResizeWrapper, CropWrapper, CannyWrapper, SegmentationFilterWrapper, \
    DuckieClipWrapper, OneHotEncodeSegWrapper, GrayscaleWrapper, EncoderWrapper

environment_configs = {
    "carla": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=True,
            seg_camera=False,
            layered_mapping=False,
        ),
        "wrappers": [
            (TimeLimit, dict(max_episode_steps=300)),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None)
        ]
    },
    "carla_gray": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (GrayscaleWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
        ]
    },
    "carla_crop": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (CropWrapper, None),
            #(TimeLimit, dict(max_episode_steps=1000))
        ]
    },
    "carla_seg": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=False,
            seg_camera=True,
        ),
        "wrappers": [
            (OneHotEncodeSegWrapper, None),
            (ChannelFirstWrapper, None),
        ]
    },
    "carla_encoder": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=False,
            seg_camera=True,
            layered_mapping=False,
            convert_segmentation=False
        ),
        "wrappers": [
            (CropWrapper, dict(keys=["camera_seg"])),
            (EncoderWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))

        ]
    },
    "carla_rgb_seg": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=True,
            seg_camera=True,
        ),
        "wrappers": [
            (OneHotEncodeSegWrapper, None)
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
            (TimeLimit, dict(max_episode_steps=1000)),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
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
            (OneHotEncodeSegWrapper, None),
            (ChannelFirstWrapper, None)
        ]
    },
    "duckie_gray": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            rgb_camera=True,
            seg_camera=False,
            randomize_maps_on_reset=False,
        ),
        "wrappers": [
            (GrayscaleWrapper, None),
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
            (DuckieClipWrapper, None),
            (OneHotEncodeSegWrapper, None)
        ]
    },
}
