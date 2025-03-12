import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=160, resize_h=120):
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.observation_space = spaces.Box(low=0, high=255, shape=(resize_h, resize_w, 3), dtype=np.uint8)

    def observation(self, observation):
        img = observation
        obs =  cv2.resize(img, dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)
        return obs


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
        img_rgb = observation
        canny = self.detect_lanes(img_rgb)

        dict = {
            "camera_rgb": img_rgb,
            "camera_canny": canny
        }
        cv2.imshow("camera_canny", canny)
        cv2.waitKey(1)

        return dict

class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(low=0, high=255, shape=(60, 160, 3), dtype=np.uint8)

    def observation(self, observation):
        cropped = observation[60:120, :160, :]
        return cropped