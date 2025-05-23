from gymnasium.wrappers import TimeLimit

from envs.duckietown.duckietown import DuckietownBaseDynamics
from simulators.carla.carla_env import SelfCarlaEnv
from util.general_wrappers import NormalizeWrapper, ChannelFirstWrapper, ResizeWrapper, CropWrapper, CannyWrapper, \
    SegmentationFilterWrapper, \
    DuckieClipWrapper, OneHotEncodeSegWrapper, GrayscaleWrapper, EncoderWrapper, OneHotEncodeClassLabelWrapper, \
    ClassEmbeddingWrapper

environment_configs = {
    "carla": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (CropWrapper, dict(keys=["camera_rgb"])),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
        ]
    },
    "carla_rgb_no_crop": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
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
            (CropWrapper, dict(keys=["camera_rgb"])),
            (GrayscaleWrapper, None),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
        ]
    },
    "carla_gray_no_crop": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=False,
            rgb_camera=True,
            seg_camera=False,
        ),
        "wrappers": [
            (GrayscaleWrapper, None),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
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
            #(CropWrapper, dict(keys=["camera_seg"])),
            (OneHotEncodeSegWrapper, None),
            (ChannelFirstWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
        ]
    },

    "carla_seg_encode": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=False,
            seg_camera=True,
        ),
        "wrappers": [
            (CropWrapper, dict(keys=["camera_seg"])),
            (ClassEmbeddingWrapper, None),
            (TimeLimit, dict(max_episode_steps=300))
        ]
    },
    "carla_encoder": {
        "base_env": SelfCarlaEnv,
        "arguments": dict(
            render=True,
            rgb_camera=False,
            seg_camera=True,
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
    "duckie_rgb": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            randomize_maps_on_reset=False,
        ),
        "wrappers": [
            (CropWrapper, dict(keys=["camera_rgb"])),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            #(TimeLimit, dict(max_episode_steps=300)),
        ]
    },
    "duckie_rgb_no_crop": {
        "base_env": DuckietownBaseDynamics,
        "arguments": dict(
            render_img=True,
            randomize_maps_on_reset=False,
        ),
        "wrappers": [
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            #(TimeLimit, dict(max_episode_steps=1000)),
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
            (CropWrapper, dict(keys=["camera_seg"])),
            (DuckieClipWrapper, None),
            (ClassEmbeddingWrapper, None),
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
            #(CropWrapper, dict(keys=["camera_rgb"])),
            (GrayscaleWrapper, None),
            (ChannelFirstWrapper, None),
            (NormalizeWrapper, None),
            #(TimeLimit, dict(max_episode_steps=600))
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
