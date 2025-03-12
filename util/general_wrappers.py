import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType
from stable_baselines3.common.utils import obs_as_tensor


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, dst_width=160, dst_height=120):
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = dst_height
        self.resize_w = dst_width
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.resize_h, self.resize_w, 3), dtype=np.uint8)

    def observation(self, observation):
        img = observation
        obs =  cv2.resize(img, dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)
        return obs


class LaneMarkingWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 160, 1), dtype=np.uint8)

    def observation(self, observation):
        img_rgb = observation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Define yellow color range in HSV (tuned for both CARLA and Duckietown)
        yellow_lower = np.array([5, 20, 20])  # Adjusted for lighting variations
        yellow_upper = np.array([35, 255, 255])

        # Create a binary mask for yellow lanes
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Apply morphological operations to close gaps in dashed lines
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask_closed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask_closed =  np.expand_dims(yellow_mask_closed, axis=2)
        cv2.imshow("Lane marking", yellow_mask)
        cv2.waitKey(1)
        return yellow_mask_closed


class CannyWrapper(gym.ObservationWrapper):
    def __init__(self, env=None,):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 160, 1), dtype=np.uint8)

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

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow("gray", gray)
        cv2.waitKey(1)
        kernel_size = 5

        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        low_t = 0
        high_t = 200
        edges = cv2.Canny(blur, low_t, high_t)

        #region = region_selection(edges)
        # Applying hough transform to get straight lines from our image
        # and find the lane lines
        # Will explain Hough Transform in detail in further steps
        #hough = hough_transform(region)
        # lastly we draw the lines on our resulting frame and return it as output
        #result = lane_lines(image, hough)
        return edges

    def observation(self, observation):
        img_rgb = observation
        canny = self.detect_lanes(img_rgb)
        canny = np.expand_dims(canny, axis=2)
        cv2.imshow("test", canny)
        cv2.waitKey(1)
        return canny

class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, crop_height_start= 60, crop_height_end=120, crop_width_start=0, crop_width_end=160, channels=3):
        gym.ObservationWrapper.__init__(self, env)

        self.crop_h_start =crop_height_start
        self.crop_h_end = crop_height_end
        self.crop_w_start = crop_width_start
        self.crop_w_end = crop_width_end
        self.num_channels = channels
        self.observation_space = spaces.Box(low=0, high=255, shape=(crop_height_end - crop_height_start, crop_width_end - crop_width_start, channels), dtype=np.uint8)

    def observation(self, observation):
        cropped = observation[self.crop_h_start:self.crop_h_end, self.crop_w_start:self.crop_w_end, :]
        return cropped