# coding=utf-8
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt


class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [0, 1.0]
        # Turn right
        elif action == 1:
            vels = [1, 0]
        # Go forward
        elif action == 2:
            vels = [0.5, 0.5]
        else:
            assert False, "unknown action"
        return np.array(vels)

    def reverse_action(self, action):
        raise NotImplementedError()


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self, env, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def action(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

    def reverse_action(self, action):
        raise NotImplementedError()


class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=160, resize_h=120):
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.observation_space = spaces.Dict({
            "camera_rgb": spaces.Box(low=0, high=255, shape=(resize_h, resize_w, 3), dtype=np.uint8)
        })

    def observation(self, observation):
        img, _ = observation["camera_rgb"]
        obs = {
            "camera_rgb": cv2.resize(img, dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)
        }
        return obs
    """
    def reset(self, seed = None, options = None):
        obs = gym.ObservationWrapper.reset(self, seed, options)
        img = obs["camera_rgb"]
        obs = {
            "camera_rgb": cv2.resize(img, dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)
        }
        return obs
    """


class CannyWrapper(gym.ObservationWrapper):
    def __init__(self, env=None,):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Dict({
            "camera_rgb": spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8),
            "camera_canny": spaces.Box(low=0, high=255, shape=(80, 160), dtype=np.uint8)
        })

    def region_of_interest(self, img):
        height, width = img.shape[:2]
        mask = np.zeros_like(img)

        # Define a triangular region of interest
        roi_vertices = np.array([[
            (0, height), (width, height), (width // 2, height // 2)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detect_lanes(self, img):
        # Load the image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 100, 300)

        # Apply region of interest mask
        #roi = self.region_of_interest(edges)

        return edges

    def observation(self, observation):
        img_rgb, _= observation["camera_rgb"]
        canny = self.detect_lanes(img_rgb)

        dict = {
            "camera_rgb": img_rgb,
            "camera_canny": canny
        }

        #cv2.imshow("test", canny)
        #cv2.waitKey(1)
        return dict

class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None,):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Dict({
            "camera_rgb": spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8)
        })

    def observation(self, observation):
        img = observation["camera_rgb"]
        img = img[40:120, :160, :]
        obs = {
            "camera_rgb" : img
        }
        return obs

class UndistortWrapper(gym.ObservationWrapper):
    """
    To Undo the Fish eye transformation - undistorts the image with plumbbob distortion
    Using the default configuration parameters on the duckietown/Software repo
    https://github.com/duckietown/Software/blob/master18/catkin_ws/src/
    ...05-teleop/pi_camera/include/pi_camera/camera_info.py
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

        assert env.unwrapped.distortion, "Distortion is false, no need for this wrapper"

        # Set a variable in the unwrapped env so images don't get distorted
        self.env.unwrapped.undistort = True

        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix = [
            305.5718893575089,
            0,
            303.0797142544728,
            0,
            308.8338858195428,
            231.8845403702499,
            0,
            0,
            1,
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0]
        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # P - Projection Matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        projection_matrix = [
            220.2460277141687,
            0,
            301.8668918355899,
            0,
            0,
            238.6758484095299,
            227.0880056118307,
            0,
            0,
            0,
            1,
            0,
        ]
        self.projection_matrix = np.reshape(projection_matrix, (3, 4))

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

    def observation(self, observation):
        return self._undistort(observation)

    def _undistort(self, observation):
        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coefs,
                self.rectification_matrix,
                self.projection_matrix,
                (W, H),
                cv2.CV_32FC1,
            )

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)
